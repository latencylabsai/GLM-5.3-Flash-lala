"""GPU regression: router outputs must agree for serial and MTP row counts."""

import os

os.environ["LALA_EXL3_CONSISTENT"] = "1"

import torch
import exllamav3_ext as ext


def main():
    assert getattr(ext, "lala_consistent_decode_version", None) == 1
    torch.manual_seed(8127)
    device = "cuda"
    for hidden in (1024, 4096):
        gate_t = torch.randn(288, hidden, device=device, dtype=torch.half) / hidden**0.5
        gate = gate_t.T.contiguous()
        x = torch.randn(8, hidden, device=device, dtype=torch.half)
        bias = torch.randn(288, device=device, dtype=torch.half) * 0.01

        def route(rows):
            scores = torch.empty((len(rows), 288), device=device, dtype=torch.half)
            selected = torch.empty((len(rows), 8), device=device, dtype=torch.long)
            weights = torch.empty_like(selected, dtype=torch.half)
            ext.routing_ds3_nogroup(
                rows.contiguous(), gate, scores, bias, selected, weights,
                2.5, gate_t, 0, None, None,
            )
            return scores, selected, weights

        references = [route(x[i : i + 1]) for i in range(8)]
        for width in range(1, 9):
            actual = route(x[:width])
            for i in range(width):
                for got, expected in zip(actual, references[i]):
                    assert torch.equal(got[i : i + 1], expected), (hidden, width, i)
    print("Router logits, selections, and weights agree exactly for 1–8 rows.")


if __name__ == "__main__":
    main()
