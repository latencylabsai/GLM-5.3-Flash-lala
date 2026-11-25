"""Repack the complete Lala checkpoint for Sushi without requantization."""

import argparse
import collections
import hashlib
import json
import pathlib
import shutil
import struct

from glm_lala import repack_experts

STAMP = {"k": "3", "codebook": "mul1", "window": "16"}
CHUNK_BYTES = 8 * 1024 * 1024


def json_file(path, value):
    """Write an indented JSON record."""
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_native(source):
    """Read the native configuration, quantization data and index."""
    config = json.loads((source / "config.json").read_text())
    quant = json.loads((source / "quantization_config.json").read_text())
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    if quant["quant_method"] != "exl3" or quant["codebook"] != "mul1":
        raise ValueError("This converter requires Lala's native EXL3 format.")
    return config, quant["tensor_storage"], index


def layer_plan(source, prefix, experts, tensors, index, headers):
    """Plan a grouped expert bank and the router's index permutation."""
    groups = repack_experts.route_groups(tensors, prefix, experts)
    packed, ranges = repack_experts.plan(source, prefix, groups, index, headers)
    remap = [None] * experts
    for packed_index, expert in enumerate(
        expert for _, ids in groups for expert in ids
    ):
        remap[expert] = packed_index
    extra = struct.pack(f"<{experts}I", *remap)
    end = max(tensor["data_offsets"][1] for tensor in packed.values())
    packed[f"{prefix}.exl3_route_remap"] = {
        "dtype": "U32",
        "shape": [experts],
        "data_offsets": [end, end + len(extra)],
    }
    return packed, ranges, extra, groups


def flat_plan(source, names, index, headers, compressions):
    """Keep the native tensor bytes and add the EXL3 bank dimension."""
    packed, ranges = {}, []
    cursor = 0
    for name in sorted(names):
        shard = source / index[name]
        header, payload_start = headers[shard]
        original = header[name]
        begin, end = original["data_offsets"]
        size = end - begin
        shape = original["shape"]
        dtype = original["dtype"]
        if name in compressions:
            part = name.rsplit(".", 1)[-1]
            expected, target = repack_experts.KINDS[part]
            if dtype != expected:
                raise ValueError(f"Incorrect stored dtype for {name}.")
            dtype, shape = target, [1, *shape]
        ranges.append((name, shard, payload_start + begin, size))
        packed[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    return packed, ranges


def check_payload(path, receipt, extra):
    """Read the repacked file and check every source tensor hash."""
    _, start = repack_experts.read_header(path)
    with path.open("rb") as target:
        target.seek(start)
        for item in receipt:
            digest = hashlib.sha256()
            remaining = item["bytes"]
            while remaining:
                data = target.read(min(remaining, CHUNK_BYTES))
                if not data:
                    raise ValueError(f"Short tensor in {path}.")
                digest.update(data)
                remaining -= len(data)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Payload changed for {item['tensor']}.")
        if target.read(len(extra) + 1) != extra:
            raise ValueError(f"The remap or final offset differs for {path}.")


def serving_weight_map(weight_map, layers):
    """Index the GLM target tensors and retain MTP bytes in the shards."""
    prefix = "model.language_model.layers."
    return {
        key: value
        for key, value in weight_map.items()
        if not key.startswith(prefix)
        or int(key[len(prefix) :].partition(".")[0]) < layers
    }


def save_metadata(source, output, config, weight_map):
    """Preserve the native metadata and write the runtime index."""
    replacements = {"config.json", "model.safetensors.index.json"}
    for path in source.iterdir():
        preserve = (
            path.suffix in {".json", ".jinja", ".model"}
            or path.name == "README.md"
            or path.name.startswith(("LICENSE", "NOTICE"))
        )
        if path.is_file() and preserve:
            if path.name in replacements:
                shutil.copyfile(path, output / f"native-{path.name}")
            else:
                shutil.copyfile(path, output / path.name)
    config["expert_quant"] = {
        "format": "exl3",
        "codebook": "mul1",
        "k": int(STAMP["k"]),
        "window": int(STAMP["window"]),
    }
    config["sushi_pack"] = {"format": "glm5-next-hf-names-1"}
    json_file(output / "config.json", config)
    layers = config["text_config"]["num_hidden_layers"]
    json_file(
        output / "model.safetensors.index.json",
        {"weight_map": serving_weight_map(weight_map, layers)},
    )


def compressed_keys(tensors, index):
    """Find every EXL3 bank, including vision omitted from tensor_storage."""
    # The native tensor_storage manifest omits vision projections. Discover
    # EXL3 banks from the indexed payload too; keep every payload byte intact.
    compressed_names = {
        name for name, tensor in tensors.items() if tensor.get("quant_format") == "exl3"
    }
    for key in index:
        if key.endswith(".trellis"):
            name = key.removesuffix(".trellis")
            required = {f"{name}.{part}" for part in (*repack_experts.PARTS, "mul1")}
            if not required <= index.keys():
                raise ValueError(f"Incomplete native EXL3 projection: {name}")
            compressed_names.add(name)
    return {
        f"{name}.{part}" for name in compressed_names for part in repack_experts.PARTS
    }


def convert(source, output, check_only):
    """Check every input key and write the selected storage layout."""
    config, tensors, index = read_native(source)
    model = config["text_config"]
    files = sorted(set(index.values()))
    headers = {
        source / name: repack_experts.read_header(source / name) for name in files
    }
    weight_map, layers, written = {}, [], set()
    manifest = {}
    if not check_only:
        output.mkdir(parents=True)

    def emit(name, packed, ranges, extra=b"", indexed=True):
        keys = {item[0] for item in ranges}
        if len(keys) != len(ranges) or written & keys:
            raise ValueError(f"Duplicate source tensor in {name}.")
        written.update(keys)
        if indexed:
            weight_map.update(dict.fromkeys(packed, name))
        if not check_only:
            path = output / name
            receipt = repack_experts.write_pack(path, packed, ranges, extra, STAMP)
            check_payload(path, receipt, extra)
            manifest[name] = {
                "bytes": path.stat().st_size,
                "source_tensors": receipt,
            }
        print(f"{name} covers {len(keys)} source tensors.", flush=True)

    for layer in range(model["first_k_dense_replace"], model["num_hidden_layers"]):
        prefix = f"model.language_model.layers.{layer}.mlp"
        packed, ranges, extra, groups = layer_plan(
            source, prefix, model["n_routed_experts"], tensors, index, headers
        )
        name = f"model-exl3-L{layer:02d}.safetensors"
        emit(name, packed, ranges, extra)
        layers.append({"layer": layer, "file": name, "groups": groups})
    markers = [name for name in index if name.endswith(".mul1")]
    emit(
        "native-mul1-markers.safetensors",
        *flat_plan(source, markers, index, headers, set()),
        indexed=False,
    )
    remaining = collections.defaultdict(list)
    for key, shard in index.items():
        if key not in written:
            remaining[shard].append(key)
    compressed = compressed_keys(tensors, index)
    for i, keys in enumerate(remaining.values()):
        emit(
            f"model-trunk-{i:02d}.safetensors",
            *flat_plan(source, keys, index, headers, compressed),
        )
    if written != set(index):
        raise ValueError("The output does not cover the complete native index.")
    if not check_only:
        save_metadata(source, output, config, weight_map)
        download_receipt = source / "lala-download.json"
        snapshot = None
        if download_receipt.is_file():
            download = json.loads(download_receipt.read_text())
            snapshot = {
                "repo": download.get("repo"),
                "revision": download.get("revision"),
                "receipt_sha256": hashlib.sha256(
                    download_receipt.read_bytes()
                ).hexdigest(),
                "weights_verified": bool(download.get("all_selected_files_verified"))
                and not download.get("metadata_only", True),
            }
        json_file(
            output / "PROVENANCE.json",
            {
                "scope": "Byte-preserving Lala EXL3 checkpoint repack",
                "source_download": snapshot,
                "source_index_sha256": hashlib.sha256(
                    (source / "model.safetensors.index.json").read_bytes()
                ).hexdigest(),
                "tensor_count": len(written),
                "layers": layers,
                "files": manifest,
            },
        )
        (output / "COMPLETE").touch()
    print(f"The layout covers {len(written)} native tensors.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    convert(args.source, args.output, args.check_only)
