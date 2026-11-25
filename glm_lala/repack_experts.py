"""Write one Lala expert layer as byte-preserving Sushi rate groups."""

import argparse
import collections
import hashlib
import json
import pathlib
import struct

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("trellis", "suh", "svh")
KINDS = {
    "trellis": ("I16", "U16"),
    "suh": ("F16", "F16"),
    "svh": ("F16", "F16"),
}


def read_header(path):
    """Return the tensor header and the start of its payload."""
    with path.open("rb") as source:
        size = struct.unpack("<Q", source.read(8))[0]
        if size > 100_000_000:
            raise ValueError(f"The tensor header exceeds 100 MB in {path}.")
        return json.loads(source.read(size)), 8 + size


def route_groups(tensors, prefix, expert_count, members=None):
    """Select experts by the rates of their three projections."""
    groups = collections.defaultdict(list)
    for expert in range(expert_count):
        names = [f"{prefix}.experts.{expert}.{proj}" for proj in PROJECTIONS]
        rates = tuple(tensors[name]["bits_per_weight"] for name in names)
        if any(tensors[name]["mul1_multiplier"] != 2212286765 for name in names):
            raise ValueError(f"Expert {expert} uses a different codebook.")
        groups[rates].append(expert)
    return [
        (rates, ids[:members] if members else ids)
        for rates, ids in sorted(groups.items())
    ]


def plan(directory, prefix, groups, weight_map=None, headers=None):
    """Plan tensor shapes and source byte ranges for each rate group."""
    if headers is None:
        headers = {}
    if weight_map is None:
        index_path = directory / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text())["weight_map"]
    packed, ranges = {}, []
    cursor = 0
    for group, (rates, experts) in enumerate(groups):
        for projection, rate in zip(PROJECTIONS, rates, strict=True):
            for part in PARTS:
                source_type, output_type = KINDS[part]
                group_start = cursor
                shape = None
                for expert in experts:
                    key = f"{prefix}.experts.{expert}.{projection}.{part}"
                    shard = directory / weight_map[key]
                    if shard not in headers:
                        headers[shard] = read_header(shard)
                    header, start = headers[shard]
                    entry = header[key]
                    begin, end = entry["data_offsets"]
                    expected = entry["shape"]
                    if (
                        entry["dtype"] != source_type
                        or (shape is not None and shape != expected)
                        or (part == "trellis" and expected[-1] != 16 * rate)
                    ):
                        raise ValueError(f"The tensor geometry changed for {key}.")
                    shape = expected
                    ranges.append((key, shard, start + begin, end - begin))
                    cursor += end - begin
                if shape is None:
                    raise ValueError(f"Expert group {group} is empty.")
                new = f"{prefix}.switch_mlp.{projection}.g{group}.{part}"
                packed[new] = {
                    "dtype": output_type,
                    "shape": [len(experts), *shape],
                    "data_offsets": [group_start, cursor],
                }
    return packed, ranges


def write_pack(output, packed, ranges, extra=b"", stamp=None):
    """Copy and hash each source tensor in the planned order."""
    if stamp is not None:
        packed = {"__metadata__": stamp, **packed}
    header = json.dumps(packed, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    digests = []
    with output.open("xb") as target:
        target.write(struct.pack("<Q", len(header)))
        target.write(header)
        for key, shard, offset, length in ranges:
            digest = hashlib.sha256()
            with shard.open("rb") as source:
                source.seek(offset)
                remaining = length
                while remaining:
                    data = source.read(min(8 * 1024 * 1024, remaining))
                    if not data:
                        raise ValueError(f"The tensor payload is incomplete for {key}.")
                    target.write(data)
                    digest.update(data)
                    remaining -= len(data)
            digests.append(
                {
                    "tensor": key,
                    "bytes": length,
                    "sha256": digest.hexdigest(),
                }
            )
        target.write(extra)
    return digests


def main():
    """Repack a routed layer and record its original expert identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--members-per-group", type=int)
    args = parser.parse_args()
    config = json.loads((args.source / "config.json").read_text())["text_config"]
    metadata = json.loads((args.source / "quantization_config.json").read_text())
    if metadata["codebook"] != "mul1":
        raise ValueError("The repacker requires the Lala mul1 codebook.")
    prefix = f"model.language_model.layers.{args.layer}.mlp"
    groups = route_groups(
        metadata["tensor_storage"],
        prefix,
        config["n_routed_experts"],
        args.members_per_group,
    )
    packed, ranges = plan(args.source, prefix, groups)
    digests = write_pack(args.output, packed, ranges)
    receipt = {"layer": args.layer, "groups": groups, "source_tensors": digests}
    args.output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
