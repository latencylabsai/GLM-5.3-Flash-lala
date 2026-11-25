"""Pack the native Lala MTP layer for the Sushi runtime."""

import argparse
import hashlib
import json
import pathlib

from glm_lala import repack as repack_lala
from glm_lala import repack_experts


def plan(source):
    """Return the packed layout and the source byte ranges."""
    config, _, index = repack_lala.read_native(source)
    model = config["text_config"]
    layer = model["num_hidden_layers"]
    if model["num_nextn_predict_layers"] != 1:
        raise ValueError("The source must contain one MTP layer.")
    prefix = f"model.language_model.layers.{layer}"
    names = [name for name in index if name.startswith(prefix + ".")]
    ffn = f"{prefix}.mlp"
    headers = {
        source / file: repack_experts.read_header(source / file)
        for file in sorted({index[name] for name in names})
    }
    rates = []
    for projection in repack_experts.PROJECTIONS:
        key = f"{ffn}.experts.0.{projection}.trellis"
        entry = headers[source / index[key]][0][key]
        rates.append(entry["shape"][-1] / 16)
    ids = list(range(model["n_routed_experts"]))
    for expert in ids:
        for projection, rate in zip(repack_experts.PROJECTIONS, rates, strict=True):
            key = f"{ffn}.experts.{expert}.{projection}.trellis"
            if headers[source / index[key]][0][key]["shape"][-1] / 16 != rate:
                raise ValueError("Native MTP expert rates are not uniform.")
    bank, expert_ranges = repack_experts.plan(
        source, ffn, [(rates, ids)], index, headers
    )
    bank = {name.replace(".g0.", "."): entry for name, entry in bank.items()}
    trunk_names = [
        name
        for name in names
        if f"{ffn}.experts." not in name or name.endswith(".mul1")
    ]
    compressed = {
        f"{name.removesuffix('.trellis')}.{part}"
        for name in trunk_names
        if name.endswith(".trellis")
        for part in repack_experts.PARTS
    }
    trunk, trunk_ranges = repack_lala.flat_plan(
        source, trunk_names, index, headers, compressed
    )
    offset = max(value["data_offsets"][1] for value in bank.values())
    for entry in trunk.values():
        begin, end = entry["data_offsets"]
        entry["data_offsets"] = [begin + offset, end + offset]
    ranges = expert_ranges + trunk_ranges
    if {row[0] for row in ranges} != set(names) or len(ranges) != len(names):
        raise ValueError("The pack must cover each MTP tensor once.")
    return bank | trunk, ranges, rates, tuple(sorted({index[k] for k in names}))


def main():
    """Validate the layout or write a verified MTP pack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    packed, ranges, rates, shards = plan(args.source)
    receipt = {
        "source_index_sha256": hashlib.sha256(
            (args.source / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "source_shards": shards,
        "source_tensor_count": len(ranges),
        "output_tensor_count": len(packed),
        "expert_rates_bpw": rates,
        "tensor_bytes": sum(row[3] for row in ranges),
    }
    print(json.dumps(receipt, indent=2), flush=True)
    if args.check_only:
        return
    args.output.mkdir(parents=True)
    path = args.output / "mtp.safetensors"
    checks = repack_experts.write_pack(path, packed, ranges, stamp=repack_lala.STAMP)
    repack_lala.check_payload(path, checks, b"")
    receipt["source_tensors"] = checks
    repack_lala.json_file(args.output / "PROVENANCE.json", receipt)
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
