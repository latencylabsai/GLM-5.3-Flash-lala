"""Record CPU token IDs already emitted by the engine during a benchmark."""

import hashlib
import json
import os
import runpy
import sys

from glm_lala.spark_server import configure_decode

# Engine modules cache environment settings at import time. The benchmark must
# apply exactly the production profile before installing its observation hook.
configure_decode()

import torch  # Load the CUDA extension's shared-library dependencies first.
from exllamav3.generator.job import Job
from exllamav3.util.backend import INT8_GEMV_MODE

from glm_lala import cli

assert INT8_GEMV_MODE == 0, "Benchmark imported the engine before the serving policy"

original = Job.receive_sample
trace = open(os.environ["LALA_TRACE"], "w", buffering=1)


def receive(self, logits, next_token, next_k_tokens, next_k_probs, next_prob, results, *args, **kwargs):
    before = len(results)
    result = original(
        self, logits, next_token, next_k_tokens, next_k_probs, next_prob,
        results, *args, **kwargs,
    )
    if not hasattr(self, "_lala_output_ids"):
        self._lala_output_ids = []
    for event in results[before:]:
        ids = event.get("token_ids")
        if ids is not None:
            self._lala_output_ids.extend(ids.flatten().tolist())
        if event.get("eos"):
            ids = self._lala_output_ids
            trace.write(json.dumps({
                "token_ids": ids,
                "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "accepted_draft_tokens": event.get("accepted_draft_tokens", 0),
                "rejected_draft_tokens": event.get("rejected_draft_tokens", 0),
            }) + "\n")
    return result


Job.receive_sample = receive


def execute(file, command, env):
    """Keep the benchmark hook when the launcher enters the normal server."""
    os.environ.update(env)
    sys.path[:0] = env.get("PYTHONPATH", "").split(os.pathsep)
    sys.argv = ["spark_server", *command[3:]]
    runpy.run_module("glm_lala.spark_server", run_name="__main__")


cli.os.execvpe = execute
cli.main()
