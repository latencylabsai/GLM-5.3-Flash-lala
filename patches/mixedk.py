r"""Load and run mixed-K EXL3 MoE layers on ExLlamaV3 1.6.3.

Provenance: this module ports the mixed-K approach of
``vcruz305/exllamav3`` (MIT) at ``909ad85``, file
``exllamav3/modules/block_sparse_mlp.py``, commits ``785f206``, ``04c8e0d``,
``554a8c9``, ``69df32d``, ``e3e6878``, ``bb912e8``, ``329e051`` and
``ecb5013``. It patches upstream ExLlamaV3 ``v1.6.3`` (MIT). Copyright (c)
2024 turboderp and the fork authors; the MIT license of ExLlamaV3 applies.

Problem: stock 1.6.3 ``BlockSparseMLP.load_local`` makes one
``MultiLinear`` for each expert projection, and ``MultiLinear`` asserts one
K for all experts. A layer with a different K for each expert fails at load,
and the converter fails when it reloads a quantized layer.

``install()`` patches ``BlockSparseMLP`` at run time:

1. ``load_local`` finds a layer whose experts differ in K or codebook. It
   loads that layer as the dense per-expert path does (no ``MultiLinear``,
   no BC graph). Then it groups the experts by (K gate, K up, K down) and
   makes the pointer tables of each group.
2. Decode (1 to 8 tokens, ``decode: coop``, or ``decode: auto`` without
   the unified kernel): a ``bc`` stand-in runs one
   stock ``exl3_moe_coop`` call for each K group. The groups have
   contiguous ranges in a permuted expert order, and each call adds the
   output of the previous call (``sh_out``). Nothing reads the device on
   the host, so a decode step does not sync.
3. More tokens: the fused branch of ``forward`` calls ``ext.exl3_moe`` once
   for the layer. A proxy for ``ext`` in ``block_sparse_mlp`` sends that call
   to one stock ``exl3_moe`` launch for each K group. The proxy partitions
   the sorted tokens on the device.
4. With ``unified=True`` and an available ``experimental.exl3_mixedk``
   extension, a layer that the unified kernel takes (integer K, or
   half-integer K 1.5, 2.5, 3.5 with the mul1 codebook) uses one
   ``exl3_moe_mixedk`` launch for the fused branch when the layer has at least
   ``unified_min_rows`` routed rows. ``decode: auto`` (the default) then
   sends decode to the fused branch too; ``decode: fused`` always does.
5. An expert with more rows than the fused kernel holds uses the expert
   ``Linear`` objects, which support all K values.
6. A layer with one K for all experts uses the stock code without change.

Limits: the fused path needs one codebook (``mcg`` or ``mul1``) for all
experts of the layer, and the stock activation and bias limits. Other
mixed layers use the dense per-expert path. The coop decode path needs
K gate = K up in each group; else decode uses the fused branch. The CPU
expert offload of 1.6.3 is not checked with mixed-K layers.

TabbyAPI: the patch must run in each process that loads the model. For one
process, start TabbyAPI through this module from the TabbyAPI folder::

    PYTHONPATH=/path/to/lab python -m experimental.exl3.mixedk \
        --config configs/experimental/exl3/mixedk_patch_20261020.yaml \
        main.py -- --config config.yml

Tensor-parallel workers start new Python processes. For tensor parallelism,
put a ``sitecustomize.py`` that calls ``mixedk.install()`` on the
``PYTHONPATH`` of the workers.
"""

import argparse
import dataclasses
import importlib
import pathlib
import runpy
import sys
import types

import torch
import yaml

SUPPORTED_VERSION = "1.6.3"
BSM_MODULE = "exllamav3.modules.block_sparse_mlp"
MIXEDK_EXTENSION = "experimental.exl3_mixedk"
CONFIG_SECTION = "patch"
GUARD_ATTRIBUTE = "mixedk_guard"
LAYER_ATTRIBUTE = "mixedk_layer"
FUSED_CODEBOOKS = ((True, False), (False, True))
DECODE_PATHS = ("auto", "coop", "fused")
PREFILL_PATHS = ("grouped", "per_expert")
# Activation codes of exl3_moe_coop (MOE_COOP_ACT_* in exl3_moe_coop.cuh).
COOP_GATED_ACTIVATIONS = {"silu": 0, "gelu": 1}
COOP_RELU2 = 2

# Positions in the ExLlamaV3 1.6.3 ``ext.exl3_moe`` argument list.
ARG_EXPERT_COUNT = 2
ARG_TOKEN_SORTED = 3
ARG_WEIGHT_SORTED = 4
ARG_K = slice(10, 13)
ARG_POINTERS = slice(13, 22)
ARG_UP_TRELLIS = 16
ARG_CODEBOOKS = slice(22, 28)
ARG_NUM_ACTIVE = 29
ARG_FUSED_BASE = 31
ARG_COUNT = 35


@dataclasses.dataclass
class Settings:
    """Runtime switches of the patch.

    Attributes:
        unified: Use the unified kernel when it is available.
        unified_min_rows: Routed rows (tokens times top-k) below which the
            grouped path runs instead of the unified kernel.
        decode: The path of 1 to 8 tokens: "coop" (one ``exl3_moe_coop``
            call for each K group), "fused" (the fused branch) or "auto"
            ("fused" for a layer with unified tables, else "coop").
        prefill: "grouped" (the fused branch with the K groups or the
            unified kernel) or "per_expert" (the expert linears, with a
            host sync) for more than 8 tokens.
        verbose: Print one line for each mixed-K layer at load.
    """

    unified: bool = False
    unified_min_rows: int = 0
    decode: str = "auto"
    prefill: str = "grouped"
    verbose: bool = True


class Runtime:
    """Hold the installed state of the patch."""

    def __init__(self):
        self.settings = Settings()
        self.originals = {}
        self.proxy: "ExtProxy | None" = None
        self.extension: types.ModuleType | None = None
        # Last load line of each mixed-K layer key, for probes and logs.
        self.loaded: dict[str, str] = {}

    def active_proxy(self) -> "ExtProxy":
        """Return the installed ``ext`` proxy.

        Raises:
            RuntimeError: The patch is not installed.
        """
        if self.proxy is None:
            raise RuntimeError("mixedk: the patch is not installed")
        return self.proxy


RUNTIME = Runtime()


def quant_key(linear):
    """Return the (K, mcg, mul1) key of a loaded EXL3 ``Linear``."""
    inner = linear.inner
    return inner.K, inner.mcg, inner.mul1


def mixed_layout(module):
    """Return True when the quantized experts of a layer differ in K.

    Args:
        module: A loaded ``BlockSparseMLP``.
    """
    linears = module.gates + module.ups + module.downs
    if not linears or any(lin.quant_type != "exl3" for lin in linears):
        return False
    projections = (module.gates, module.ups, module.downs)
    return any(
        len({quant_key(lin) for lin in group}) > 1
        for group in projections
        if group
    )


def pointer_table(tensors, device):
    """Return an int64 tensor with the data pointers of ``tensors``."""
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        dtype=torch.long,
        device=device,
    )


def projection_pointers(linears, device):
    """Return the (trellis, suh, svh) pointer tables of EXL3 linears."""
    inners = [lin.inner for lin in linears]
    return (
        pointer_table([inner.trellis for inner in inners], device),
        pointer_table([inner.suh for inner in inners], device),
        pointer_table([inner.svh for inner in inners], device),
    )


class MixedMultiLinear:
    """Stand in for ``MultiLinear`` over experts with different K.

    ``K`` is None: a stock kernel call with this object fails loudly. Only
    the ``ext`` proxy uses the pointer tables of all experts.
    """

    def __init__(self, device, linears):
        """Make the pointer tables of all experts of one projection.

        Args:
            device: The CUDA device of the layer.
            linears: The EXL3 ``Linear`` objects, one for each expert.
        """
        self.device = device
        self.linears = linears
        self.num_linears = len(linears)
        self.in_features = linears[0].in_features
        self.out_features = linears[0].out_features
        self.K = None  # pylint: disable=invalid-name
        self.ks = [lin.inner.K for lin in linears]
        pointers = projection_pointers(linears, device)
        self.ptrs_trellis, self.ptrs_suh, self.ptrs_svh = pointers
        self.mcg = linears[0].inner.mcg
        self.mul1 = linears[0].inner.mul1

    def q_cb(self):
        """Return the (mcg, mul1) codebook flags."""
        return self.mcg, self.mul1

    def unload(self):
        """Release nothing; the experts own the tensors."""


@dataclasses.dataclass
class KGroup:
    """One group of experts with the same (K gate, K up, K down).

    Attributes:
        experts: The local expert indices, in ascending order.
        ks: The (K gate, K up, K down) bitrates.
        pointers: The nine pointer tables in ``exl3_moe`` order.
    """

    experts: list
    ks: tuple
    pointers: tuple


class MixedLayer:
    """Dispatch tables of one mixed-K ``BlockSparseMLP`` layer."""

    def __init__(self, module, unified):
        """Group the experts of ``module`` by K and make the tables.

        Args:
            module: A loaded mixed-K ``BlockSparseMLP``.
            unified: The tables from ``exl3_mixedk.build_tables``, or None.
        """
        device = module.device
        gates = module.gates if module.gated else module.ups
        num_experts = len(module.ups)
        buckets = {}
        for index in range(num_experts):
            ks = (
                gates[index].inner.K,
                module.ups[index].inner.K,
                module.downs[index].inner.K,
            )
            buckets.setdefault(ks, []).append(index)
        self.groups = []
        for ks, experts in sorted(buckets.items()):
            pointers = ()
            for linears in (gates, module.ups, module.downs):
                chosen = [linears[index] for index in experts]
                pointers += projection_pointers(chosen, device)
            self.groups.append(KGroup(experts, ks, pointers))
        self.key = module.multi_up.ptrs_trellis
        self.unified = unified
        self.width = max(len(group.experts) for group in self.groups) + 1
        self.expert_ids = torch.arange(
            num_experts + 1, dtype=torch.long, device=device
        )
        # lut[g, e]: local index of expert e in group g, else the sentinel.
        lut = torch.empty((len(self.groups), num_experts + 1), dtype=torch.long)
        # members[g, j]: global index of local expert j, padded with E.
        members = torch.full(
            (len(self.groups), self.width), num_experts, dtype=torch.long
        )
        for row, group in enumerate(self.groups):
            lut[row].fill_(len(group.experts))
            for local, expert in enumerate(group.experts):
                lut[row, expert] = local
                members[row, local] = expert
        self.lut = lut.to(device)
        self.members = members.to(device)
        self.bin_offset = (
            torch.arange(len(self.groups), device=device).unsqueeze(1)
            * self.width
        )
        # Decode order: the groups one after the other, each a contiguous
        # range [start, start + size) of permuted local experts.
        self.starts = []
        first = 0
        if module.num_local_experts != module.num_experts:
            first = module.routing_first
        # permuted[g]: permuted index of global expert g; others get E.
        permuted = torch.full((module.num_experts,), num_experts)
        position = 0
        for group in self.groups:
            self.starts.append(position)
            for expert in group.experts:
                permuted[first + expert] = position
                position += 1
        self.permuted = permuted.to(device)

    def describe(self):
        """Return a short text with the groups of the layer."""
        parts = [
            f"{'/'.join(str(k) for k in group.ks)}:{len(group.experts)}"
            for group in self.groups
        ]
        return f"{len(self.groups)} K groups (gate/up/down:experts) " + (
            " ".join(parts)
        )

    def dispatch(self, ext, args):
        """Run one stock ``exl3_moe`` call of the layer.

        Args:
            ext: The real ``exllamav3_ext`` module.
            args: The positional arguments of the stock call.
        """
        if args[ARG_EXPERT_COUNT].shape[0] != self.expert_ids.shape[0]:
            raise ValueError("mixedk: expert_count has an unexpected size")
        rows = args[ARG_TOKEN_SORTED].shape[0]
        settings = RUNTIME.settings
        if self.unified is not None and rows >= settings.unified_min_rows:
            self.run_unified(args)
        else:
            self.run_grouped(ext, args)

    def run_grouped(self, ext, args):
        """Run one stock ``exl3_moe`` launch for each K group."""
        rows = args[ARG_TOKEN_SORTED].shape[0]
        if rows == 0:
            return
        # Expert of each sorted slot; output_size avoids a host sync.
        expert_id = torch.repeat_interleave(
            self.expert_ids, args[ARG_EXPERT_COUNT], output_size=rows
        )
        local = self.lut.index_select(1, expert_id)
        # A stable sort keeps the slot order inside each expert, so the
        # deterministic slot bases of the stock caller stay valid.
        order = torch.argsort(local, dim=1, stable=True)
        counts = torch.bincount(
            (local + self.bin_offset).flatten(),
            minlength=len(self.groups) * self.width,
        ).view(len(self.groups), self.width)
        tokens = args[ARG_TOKEN_SORTED][order]
        weights = args[ARG_WEIGHT_SORTED][order]
        fused_base = args[ARG_FUSED_BASE]
        bases = fused_base[self.members] if fused_base is not None else None
        # The launch width comes from num_active. An upper bound of the
        # active experts of a group gives each expert more SMs than -1.
        bound = args[ARG_NUM_ACTIVE] if args[ARG_NUM_ACTIVE] > 0 else rows
        for row, group in enumerate(self.groups):
            size = len(group.experts)
            call = list(args)
            call[ARG_EXPERT_COUNT] = counts[row, : size + 1]
            call[ARG_TOKEN_SORTED] = tokens[row]
            call[ARG_WEIGHT_SORTED] = weights[row]
            call[ARG_K] = group.ks
            call[ARG_POINTERS] = group.pointers
            call[ARG_NUM_ACTIVE] = min(bound, size)
            if bases is not None:
                call[ARG_FUSED_BASE] = bases[row, : size + 1]
            ext.exl3_moe(*call)

    def run_unified(self, args):
        """Run one ``exl3_moe_mixedk`` launch for all experts."""
        tables = self.unified
        call = list(args)
        call[ARG_K] = tables["k"]
        call[ARG_POINTERS] = tables["pointers"]
        call[ARG_CODEBOOKS] = tables["codebooks"]
        tables["kernel"](*call, tables["k_max"], None)


def expert_mlp(module, index, x):
    """Return the output of one expert for rows ``x`` (stock torch path)."""
    up = module.ups[index].forward(x, {})
    if module.gated:
        gate = module.gates[index].forward(x, {})
        act = up
        if module.interm_dtype != torch.half:
            act = torch.empty_like(up, dtype=torch.half)
        module.activation_fn_call(gate, up, act, module.act_limit)
    else:
        act = module.gateless_act(up)
        if act.dtype != torch.half:
            act = act.half()
    return module.downs[index].forward(act, {})


class MixedDecode:
    """Stand in for ``BC_BlockSparseMLP`` of a mixed-K layer.

    ``run_bszN`` runs one stock ``exl3_moe_coop`` call for each K group.
    The single-expert methods use the expert ``Linear`` objects.
    """

    def __init__(self, module, layer, bsm, act):
        """Find the stock scratch tensors of ``module``.

        Args:
            module: A loaded mixed-K ``BlockSparseMLP``.
            layer: The ``MixedLayer`` of ``module``.
            bsm: The ``exllamav3.modules.block_sparse_mlp`` module.
            act: The ``exl3_moe_coop`` activation code.
        """
        self.module = module
        self.layer = layer
        self.act = act
        device = module.device
        hidden_in = module.ups[0].in_features
        interm = module.intermediate_size_padded
        hidden_out = module.downs[0].out_features
        slots = bsm.MAX_BSZN * module.num_experts_per_tok
        cache = bsm.g_tensor_cache
        # The same cache keys and shapes as the stock load_local.
        self.had_u = cache.get(
            device, (slots, hidden_in), torch.half, "moe1_temp_hidden_u"
        )
        counters = (
            slots * (interm // 128)
            + bsm.MAX_BSZN * (hidden_out // 128)
            + 2 * slots
            + 3
        )
        self.ctr = cache.get(device, (counters,), torch.int, "moe1_coop_ctr")
        shape = (bsm.MAX_BSZN, module.expert_size)
        self.partial = [
            cache.get(device, shape, torch.float, f"mixedk_partial_{index}")
            for index in range(2)
        ]
        self.hidden_in = hidden_in

    def run_bszN(self, y, selected_experts, routing_weights):  # pylint: disable=invalid-name
        """Write the routed sum of 1 to 8 tokens to ``out_bszn``."""
        module = self.module
        cfg = module.experts_cfg
        ext = RUNTIME.active_proxy().ext
        local = self.layer.permuted[selected_experts]
        groups = self.layer.groups
        previous = None
        for index, group in enumerate(groups):
            if index == len(groups) - 1:
                out = cfg.out_bszn
            else:
                out = self.partial[index % 2]
            start = self.layer.starts[index]
            ext.exl3_moe_coop(
                y,
                local,
                routing_weights,
                start,
                start + len(group.experts),
                self.hidden_in,
                *group.pointers,
                None,
                None,
                None,
                *group.ks,
                module.multi_up.mcg,
                module.multi_up.mul1,
                self.act,
                module.act_limit,
                module.gated,
                cfg.yh,
                self.had_u,
                cfg.interm_g,
                cfg.interm_u,
                cfg.interm_a,
                cfg.out_d,
                self.ctr,
                out,
                previous,
                None,
            )
            previous = out

    def run_single_expert(self, x, index):
        """Write the output of one expert to ``out_d2`` (graph path)."""
        out = expert_mlp(self.module, index, x)
        self.module.experts_cfg.out_d2[: x.shape[0]].copy_(out)

    def run_single_expert_dq(self, x, index, *buffers):
        """Write the output of one expert to the last buffer (DQ path)."""
        buffers[-1].copy_(expert_mlp(self.module, index, x))


def coop_activation(module, layer):
    """Return the ``exl3_moe_coop`` activation code, or None.

    None means that the coop decode path does not apply to the layer.
    """
    decode = RUNTIME.settings.decode
    if decode == "fused" or (decode == "auto" and layer.unified is not None):
        return None
    if module.gated:
        if any(group.ks[0] != group.ks[1] for group in layer.groups):
            return None
        return COOP_GATED_ACTIVATIONS.get(module.activation_fn)
    return COOP_RELU2 if module.activation_fn == "relu2" else None


class ExtProxy:
    """Proxy for ``exllamav3_ext`` in ``block_sparse_mlp``.

    The proxy sends an ``exl3_moe`` call of a registered mixed-K layer to
    that layer. All other calls go to the real extension without change.
    """

    def __init__(self, ext):
        """Wrap the real extension module ``ext``."""
        self.ext = ext
        self.layers = {}

    def __getattr__(self, name):
        """Return (and keep) an attribute of the real extension."""
        value = getattr(self.ext, name)
        setattr(self, name, value)
        return value

    def register(self, layer):
        """Send the ``exl3_moe`` calls of ``layer`` to it."""
        self.layers[id(layer.key)] = layer

    def unregister(self, layer):
        """Remove ``layer``."""
        self.layers.pop(id(layer.key), None)

    def exl3_moe(self, *args):
        """Run the stock kernel, or the dispatch of a mixed-K layer."""
        if len(args) == ARG_COUNT:
            layer = self.layers.get(id(args[ARG_UP_TRELLIS]))
            if layer is not None and layer.key is args[ARG_UP_TRELLIS]:
                layer.dispatch(self.ext, args)
                return
        self.ext.exl3_moe(*args)


def unified_tables(module):
    """Return the unified-kernel tables of ``module``, or None.

    The tables are None when the config disables the unified path, when the
    extension is not available, or when the unified kernel cannot take a K
    of the layer (``build_tables`` raises ``ValueError``, for example for a
    half-integer K with the mcg codebook). Then grouped dispatch runs.
    The K arrays use the encoding of ``exl3_mixedk`` (16 + KA for
    K = KA + 0.5), and ``k_max`` is the largest K rounded up.
    """
    if not RUNTIME.settings.unified:
        return None
    gates = module.gates if module.gated else module.ups
    package = RUNTIME.extension or load_extension()
    if package is None:
        return None
    RUNTIME.extension = package
    try:
        built = package.build_tables(
            [lin.inner for lin in gates],
            [lin.inner for lin in module.ups],
            [lin.inner for lin in module.downs],
            module.device,
        )
    except ValueError as error:
        report(module, f"grouped dispatch: {error}")
        return None
    package.prepare(module.device)
    return {
        "k": tuple(built.k_args()),
        "pointers": tuple(built.ptr_args()),
        "codebooks": tuple(built.codebook_flags()),
        "k_max": int(built.k_max),
        "kernel": package.exl3_moe_mixedk,
    }


def load_extension():
    """Return the ``experimental.exl3_mixedk`` package, or None.

    None means that the package is absent or the kernel does not build.
    """
    try:
        package = importlib.import_module(MIXEDK_EXTENSION)
    except ImportError as error:
        print(f" !! mixedk: no unified kernel: {error}", flush=True)
        return None
    if not package.available():
        print(
            f" !! mixedk: unified kernel not available: {package.last_error()}",
            flush=True,
        )
        return None
    return package


def attach(module, bsm):
    """Prepare the fused grouped path of a mixed-K layer.

    The stock ``load_local`` already ran without ``MultiLinear`` objects.
    When the layer cannot use the fused path, it keeps the dense
    per-expert path.

    Args:
        module: A loaded mixed-K ``BlockSparseMLP``.
        bsm: The ``exllamav3.modules.block_sparse_mlp`` module.
    """
    linears = (module.gates if module.gated else []) + module.ups
    linears += module.downs
    codebooks = {(lin.inner.mcg, lin.inner.mul1) for lin in linears}
    if not module.support_quant_paths or len(codebooks) != 1:
        report(module, "dense per-expert path")
        return
    if next(iter(codebooks)) not in FUSED_CODEBOOKS:
        report(module, "dense per-expert path")
        return
    device = module.device
    if module.gated:
        module.multi_gate = MixedMultiLinear(device, module.gates)
    module.multi_up = MixedMultiLinear(device, module.ups)
    module.multi_down = MixedMultiLinear(device, module.downs)
    layer = MixedLayer(module, unified_tables(module))
    RUNTIME.active_proxy().register(layer)
    setattr(module, LAYER_ATTRIBUTE, layer)
    # The batched reconstruct tier needs one K for each projection.
    module.batch_recon = False
    # More tokens than the decode path takes go to the fused branch.
    module.f_threshold = 0
    act = coop_activation(module, layer)
    if act is not None:
        module.bc = MixedDecode(module, layer, bsm, act)
    decode = "coop" if act is not None else "fused branch"
    if RUNTIME.settings.prefill == "per_expert":
        report(module, f"per-expert, decode {decode}, {layer.describe()}")
        return
    module.support_fused = True
    module.mtile_ok = bsm.MTILE and bool(module.multi_up.mul1)
    module.fused_rows = (
        bsm.FUSED_ROWS_WIDE if module.mtile_ok else bsm.TEMP_ROWS_FUSED
    )
    rows = module.fused_rows
    hidden = module.expert_size
    interm = module.intermediate_size_padded
    groups = RUNTIME.active_proxy().exl3_moe_max_concurrency(
        torch.device(device).index
    )
    cache = bsm.g_tensor_cache
    module.fused_mode_buffers = bsm.FusedBuffers(
        temp_state_g=cache.get(
            device, (groups, rows, hidden), torch.half, "moe2_temp_state_g"
        ),
        temp_state_u=cache.get(
            device, (groups, rows, hidden), torch.half, "moe2_temp_state_u"
        ),
        temp_intermediate_g=cache.get(
            device,
            (groups, rows, interm),
            torch.half,
            "moe2_temp_intermediate_g",
        ),
        temp_intermediate_u=cache.get(
            device,
            (groups, rows, interm),
            torch.half,
            "moe2_temp_intermediate_u",
        ),
    )
    fused = "unified kernel" if layer.unified is not None else "grouped"
    report(module, f"{fused}, decode {decode}, {layer.describe()}")


def detach(module, bsm):
    """Remove the mixed-K state of ``module`` before the stock unload."""
    layer = getattr(module, LAYER_ATTRIBUTE, None)
    if layer is None:
        return
    RUNTIME.active_proxy().unregister(layer)
    setattr(module, LAYER_ATTRIBUTE, None)
    module.support_fused = False
    module.mtile_ok = False
    module.fused_rows = bsm.TEMP_ROWS_FUSED
    module.f_threshold = min(
        module.num_experts // module.num_experts_per_tok, 4
    )


def report(module, text):
    """Print (and keep) one load line for a mixed-K layer."""
    RUNTIME.loaded[str(module.key)] = text
    if RUNTIME.settings.verbose:
        print(f" -- Mixed-K experts in {module.key}: {text}", flush=True)


def patch_class(cls, bsm):
    """Wrap ``load_local`` and ``unload`` of ``BlockSparseMLP``."""
    original_load = cls.load_local
    original_unload = cls.unload

    def load_local(self, **kwargs):
        if not mixed_layout(self):
            return original_load(self, **kwargs)
        params = self.config.infer_params
        requested = params.no_reconstruct
        # Stock load_local makes MultiLinear objects only without this flag.
        params.no_reconstruct = True
        try:
            result = original_load(self, **kwargs)
        finally:
            params.no_reconstruct = requested
        if not requested:
            attach(self, bsm)
        return result

    def unload(self):
        detach(self, bsm)
        return original_unload(self)

    setattr(load_local, GUARD_ATTRIBUTE, True)
    cls.load_local = load_local
    cls.unload = unload
    RUNTIME.originals.update(
        {"load_local": original_load, "unload": original_unload}
    )


def exllamav3_version():
    """Return the version text of the installed ExLlamaV3 package."""
    return importlib.import_module("exllamav3.version").__version__


def install(
    unified=False,
    unified_min_rows=0,
    decode="auto",
    prefill="grouped",
    verbose=True,
):
    """Patch ExLlamaV3 1.6.3 ``BlockSparseMLP`` for mixed-K experts.

    A second call changes only the settings.

    Args:
        unified: Use ``experimental.exl3_mixedk`` for the layers that it
            takes (integer K, or half-integer K with mul1).
        unified_min_rows: Routed rows below which the grouped path runs.
        decode: "auto", "coop" or "fused"; the path of 1 to 8 tokens.
        prefill: "grouped" or "per_expert"; the path of more tokens.
        verbose: Print one line for each mixed-K layer at load.

    Returns:
        True when the patch is active. False for another ExLlamaV3 version;
        then the stock code runs without change.
    """
    if decode not in DECODE_PATHS or prefill not in PREFILL_PATHS:
        raise ValueError(
            f"mixedk: decode must be in {DECODE_PATHS} and prefill must be "
            f"in {PREFILL_PATHS}"
        )
    RUNTIME.settings = Settings(
        unified, unified_min_rows, decode, prefill, verbose
    )
    version = exllamav3_version()
    if not version.startswith(SUPPORTED_VERSION):
        print(
            f" !! mixedk: ExLlamaV3 {version} is not {SUPPORTED_VERSION}; "
            "the patch is not installed",
            flush=True,
        )
        return False
    bsm = importlib.import_module(BSM_MODULE)
    cls = bsm.BlockSparseMLP
    if getattr(cls.load_local, GUARD_ATTRIBUTE, False):
        return True
    RUNTIME.proxy = ExtProxy(bsm.ext)
    RUNTIME.originals["ext"] = bsm.ext
    setattr(bsm, "ext", RUNTIME.proxy)
    patch_class(cls, bsm)
    return True


def uninstall():
    """Restore the stock ``BlockSparseMLP``; unload mixed models first."""
    if not RUNTIME.originals:
        return
    bsm = importlib.import_module(BSM_MODULE)
    setattr(bsm, "ext", RUNTIME.originals["ext"])
    bsm.BlockSparseMLP.load_local = RUNTIME.originals["load_local"]
    bsm.BlockSparseMLP.unload = RUNTIME.originals["unload"]
    RUNTIME.originals = {}
    RUNTIME.proxy = None


def install_from_config(path):
    """Install the patch with the ``patch`` section of a YAML config."""
    config = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    return install(**config.get(CONFIG_SECTION, {}))


def main(argv=None):
    """Install the patch, then run a script (for example TabbyAPI)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="YAML config with a patch section")
    parser.add_argument("script", help="Python script to run as __main__")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    # Under "python -m" this file is __main__; keep the patch state in the
    # module that other code imports.
    canonical = importlib.import_module("experimental.exl3.mixedk")
    if args.config:
        canonical.install_from_config(args.config)
    else:
        canonical.install()
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    script = pathlib.Path(args.script).resolve()
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
