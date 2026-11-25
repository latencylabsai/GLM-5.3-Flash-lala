"""Download the checkpoint and launch existing serving backends."""

import argparse
import fnmatch
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOCK = pathlib.Path(__file__).with_name("runtime-lock.json")
MODEL_REPO = "latencylabs/GLM-5.3-Flash-lala"

# Native MTP uses Sushi's shared grouped-verification kernel switches.
MAC_ACCELERATION = {
    "SUSHI_GLM_WIDE_LOOKUP": "1",
    "SUSHI_GLM_LOOKUP15": "1",
    "SUSHI_GLM_WIDE_SHARED": "1",
    "SUSHI_GLM_PREPARED_VERIFY": "1",
    "SUSHI_GLM_BATCH_MLA": "1",
    "SUSHI_GLM_ROUTER_COLUMNS": "1",
    "SUSHI_GLM_WIDE_FUSED_HC": "1",
    "SUSHI_GLM_BATCH_FP16_GEMV": "1",
    "SUSHI_GLM_DENSE_VERIFY": "1",
    "SUSHI_GLM_ADAPTIVE_LOOKUP": "1",
    "SUSHI_GLM_LOOKUP_SHORT_SUFFIX": "1",
    "SUSHI_GLM_PREFIX_COMMIT": "1",
    "SUSHI_GLM_VERIFY_EXPERT_GEMM": "1",
    "SUSHI_GLM_SHARED_ATTENTION": "1",
    "SUSHI_GLM_LOOKUP_ROWS": "32",
}


def download(args):
    """Resolve and record the exact HF revision without executing model code."""
    from huggingface_hub import HfApi, snapshot_download

    lock = json.loads(LOCK.read_text())
    revision = args.revision or lock["model"]["revision"]
    if not revision:
        raise ValueError(
            "The release model revision is not pinned yet. Supply --revision "
            "explicitly; the resolved commit will be recorded."
        )
    info = HfApi().model_info(MODEL_REPO, revision=revision, files_metadata=True)
    destination = args.output.resolve()
    state_path = destination / "lala-download-state.json"
    receipt_path = destination / "lala-download.json"
    if destination.exists() and any(destination.iterdir()):
        prior_path = state_path if state_path.is_file() else receipt_path
        prior = json.loads(prior_path.read_text()) if prior_path.is_file() else {}
        if prior.get("repo") != MODEL_REPO or prior.get("revision") != info.sha:
            raise ValueError(
                "Use an empty output directory, or resume the same recorded snapshot."
            )
    destination.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"repo": MODEL_REPO, "revision": info.sha}) + "\n")
    receipt_path.unlink(missing_ok=True)
    patterns = [
        "*.json",
        "*.jinja",
        "*.model",
        "README.md",
        "LICENSE*",
        "NOTICE*",
        "runtime/mixedk.py",
        "runtime/LICENSE",
    ]
    if not args.metadata_only:
        patterns.append("*.safetensors")
    snapshot_download(
        MODEL_REPO,
        revision=info.sha,
        local_dir=destination,
        allow_patterns=patterns,
    )
    verified = []
    for entry in info.siblings:
        if not any(
            fnmatch.fnmatchcase(entry.rfilename, pattern) for pattern in patterns
        ):
            continue
        path = destination / entry.rfilename
        size = path.stat().st_size
        if entry.size is not None and size != entry.size:
            raise ValueError(f"Download size differs: {entry.rfilename}")
        digest = hashlib.sha256()
        blob = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as source:
            while chunk := source.read(16 * 1024 * 1024):
                digest.update(chunk)
                if not entry.lfs:
                    blob.update(chunk)
        actual = digest.hexdigest() if entry.lfs else blob.hexdigest()
        expected = entry.lfs.sha256 if entry.lfs else entry.blob_id
        if not expected or actual != expected:
            raise ValueError(f"Hugging Face checksum differs: {entry.rfilename}")
        verified.append(
            {
                "file": entry.rfilename,
                "bytes": size,
                "sha256": digest.hexdigest(),
                "hf_hash": expected,
                "hf_hash_type": "sha256" if entry.lfs else "git-blob-sha1",
            }
        )
        print(f"Verified {entry.rfilename} ({size:,} bytes)", flush=True)
    if not args.metadata_only:
        index = json.loads((destination / "model.safetensors.index.json").read_text())
        verified_names = {entry["file"] for entry in verified}
        if not set(index["weight_map"].values()) <= verified_names:
            raise ValueError("The verified snapshot is missing indexed weight shards.")
    if info.sha == lock["model"]["revision"]:
        for name, key in [
            ("model.safetensors.index.json", "native_index_sha256"),
            ("quantization_config.json", "quantization_config_sha256"),
        ]:
            digest = hashlib.sha256((destination / name).read_bytes()).hexdigest()
            if digest != lock["model"][key]:
                raise ValueError(f"Pinned model metadata differs: {name}")
    receipt = {
        "repo": MODEL_REPO,
        "revision": info.sha,
        "metadata_only": args.metadata_only,
        "verified_files": verified,
        "all_selected_files_verified": True,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def mac_command(args):
    """Use Sushi's existing OpenAI-compatible server."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("The Mac backend requires Apple Silicon macOS.")
    model = args.model.resolve()
    config = json.loads((model / "config.json").read_text())
    if not config.get("sushi_pack"):
        raise ValueError(
            "Sushi requires the byte-preserving pack. Run "
            "python -m glm_lala.repack --source NATIVE --output PACK first."
        )
    binary = args.binary or ROOT / ".runtime/sushi/zig-out/bin/sushi"
    command = [
        str(binary),
        "serve",
        "--model",
        str(model),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ctx-size",
        str(args.context),
        "--max-tokens",
        str(args.max_tokens),
        "--kv-quant",
        "16",
        "--max-concurrent",
        "1",
        "--prefix-cache-entries",
        "0",
        "--vision",
        "--no-mtp",
        "--no-update-check",
    ]
    # Clear inherited tuning switches so serial mode is a clean reference.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SUSHI_GLM_")}
    command += ["--no-drafter"]  # Never discover or load a separate assistant.
    if args.serial and args.mtp_pack:
        raise ValueError("--serial cannot be combined with Mac MTP options.")
    if not args.serial:
        mtp_pack = args.mtp_pack or model.parent / "mtp" / "mtp.safetensors"
        if not mtp_pack.is_file():
            raise ValueError(
                "Native MTP is enabled by default. Run glm_lala.repack_mtp into "
                "the sibling mtp directory, supply --mtp-pack, or use --serial."
            )
        # Old runtimes must fail instead of silently running serial.
        command += ["--glm-native-mtp"]
        env.update(MAC_ACCELERATION)
        env["SUSHI_GLM_MTP_PACK"] = str(mtp_pack.resolve())
        env["SUSHI_GLM_MTP_HISTORY"] = "1"
        command += ["--no-mtp-lookup"]
    env["SUSHI_COMPLETION_TOKEN_SHA"] = "1"
    return command, env


def spark_command(args):
    """Publish TabbyAPI only on the requested host interface."""
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("The Spark backend requires Linux on aarch64.")
    if args.mtp_pack:
        raise ValueError(
            "--mtp-pack is a Mac option."
        )
    if args.serial and args.spark_draft not in (None, "serial"):
        raise ValueError("--serial conflicts with --spark-draft mtp.")
    draft_mode = "serial" if args.serial else (args.spark_draft or "mtp")
    if args.context % 256:
        raise ValueError("Spark context size must be a multiple of 256.")
    if args.recurrent_cache_mb < 0:
        raise ValueError("--recurrent-cache-mb must be 0 or more.")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "This local Spark profile disables authentication. Configure "
            "TabbyAPI authentication before publishing a shared endpoint."
        )
    model = args.model.resolve()
    if not (model / "quantization_config.json").is_file():
        raise ValueError("Spark requires the original native EXL3 checkpoint.")
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    model_store = runtime / f"spark-models-{args.port}"
    config = {
        "network": {
            "host": args.host,
            "port": args.port,
            "disable_auth": True,
            "allowed_origins": [],
        },
        "model": {
            "model_dir": str(model_store),
            "model_name": "lala",
            "backend": "exllamav3",
            "max_seq_len": args.context,
            "cache_size": args.context,
            "cache_mode": "FP16",
            "max_batch_size": 1,
            "chunk_size": 2048,
            "tensor_parallel": False,
            "vision": True,
        },
        "draft_model": {"draft_mode": "disabled"},
        "memory": {
            # Prefix reuse needs stored recurrent-state checkpoints.
            "sysmem_recurrent_cache": args.recurrent_cache_mb,
            "sysmem_kv_cache": 0,
        },
        "sampling": {
            "override_preset": "safe_defaults",
            "max_tokens": {"override": args.max_tokens, "force": False},
        },
    }
    if draft_mode != "serial":
        config["draft_model"] = {
            "draft_mode": "mtp",
            "draft_num_tokens": 4,
            "draft_cache_mode": "FP16",
            "hybrid_draft_tokens": 0,
        }
    config_path = runtime / f"spark-{args.port}.json"
    python = runtime / "spark-venv/bin/python"
    if not python.is_file():
        raise ValueError("Run bash scripts/setup-spark.sh first.")
    model_store.mkdir(exist_ok=True)
    model_link = model_store / "lala"
    if model_link.is_symlink():
        if model_link.resolve() != model:
            model_link.unlink()
    elif model_link.exists():
        raise ValueError(f"Expected a managed model symlink at {model_link}")
    if not model_link.is_symlink():
        model_link.symlink_to(model, target_is_directory=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    env = os.environ.copy()
    env.pop("LALA_SUFFIX_TOKENS", None)
    build_env_path = runtime / "spark-build-env.json"
    if build_env_path.is_file():
        build_env = json.loads(build_env_path.read_text())
        for key in ("CPATH", "CUDA_HOME"):
            if isinstance(build_env.get(key), str):
                env.setdefault(key, build_env[key])
    env.update(
        TABBY_DIR=str(runtime / "tabby-native"),
        PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "patches")]),
        EXL3_INT8_GEMV="0",
    )
    return [
        str(python),
        "-m",
        "glm_lala.spark_server",
        "--config",
        str(config_path),
    ], env


def serve(args):
    """Replace the launcher with the selected existing API server."""
    if not 1 <= args.port <= 65535 or args.max_tokens < 1:
        raise ValueError("Port or output token limit is invalid.")
    if args.context <= args.max_tokens:
        raise ValueError("Context must exceed the output token limit.")
    command, env = spark_command(args) if args.backend == "spark" else mac_command(args)
    if args.print_command:
        import shlex

        print(shlex.join(command))
        print(
            json.dumps(
                {k: v for k, v in env.items() if k.startswith("SUSHI_")}, indent=2
            )
        )
        return
    os.execvpe(command[0], command, env)


def doctor(_args):
    """Print local prerequisites without loading a model."""
    memory = None
    if platform.system() == "Darwin":
        try:
            memory = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    stderr=subprocess.DEVNULL,
                )
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    elif platform.system() == "Linux":
        try:
            info = dict(
                line.split(":", 1)
                for line in pathlib.Path("/proc/meminfo").read_text().splitlines()
            )
            memory = int(info["MemTotal"].split()[0]) * 1024
        except (OSError, KeyError, ValueError):
            pass
    cuda = pathlib.Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    print(
        json.dumps(
            {
                "system": platform.system(),
                "architecture": platform.machine(),
                "memory_gib": memory / 1024**3 if memory else None,
                "git": shutil.which("git"),
                "python": sys.version.split()[0],
                "disk_free_gib": shutil.disk_usage(ROOT).free / 1024**3,
                "target_context_tokens": 262144,
                "spark_python": str(ROOT / ".runtime/spark-venv/bin/python"),
                "spark_installed": (ROOT / ".runtime/spark-installed.txt").is_file(),
                "cuda_compiler": str(cuda / "bin/nvcc")
                if (cuda / "bin/nvcc").is_file()
                else None,
                "sushi_built": (ROOT / ".runtime/sushi/zig-out/bin/sushi").is_file(),
                "runtime_lock": json.loads(LOCK.read_text()),
            },
            indent=2,
        )
    )


def main():
    """Parse commands for model management and native serving."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download", help="Download a pinned HF snapshot")
    get.add_argument("--revision")
    get.add_argument("--metadata-only", action="store_true")
    get.add_argument("--output", type=pathlib.Path, required=True)
    get.set_defaults(run=download)
    check = commands.add_parser("doctor", help="Inspect host prerequisites")
    check.set_defaults(run=doctor)
    server = commands.add_parser("serve", help="Launch the existing native server")
    server.add_argument("--backend", choices=["mac", "spark"], required=True)
    server.add_argument(
        "--spark-draft",
        choices=["serial", "mtp"],
        help="Spark drafting mode (default: mtp)",
    )
    server.add_argument("--serial", action="store_true", help="Disable native MTP")
    server.add_argument("--model", type=pathlib.Path, required=True)
    server.add_argument("--binary", type=pathlib.Path)
    server.add_argument(
        "--mtp-pack", type=pathlib.Path, help="Standalone native Mac MTP weights"
    )
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--context", type=int, default=262144)
    server.add_argument("--max-tokens", type=int, default=4096)
    server.add_argument(
        "--recurrent-cache-mb",
        type=int,
        default=1024,
        help="Spark recurrent-state store for prefix reuse, in MiB (0 disables)",
    )
    server.add_argument("--print-command", action="store_true")
    server.set_defaults(run=serve)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"glm-lala: {error}\n")


if __name__ == "__main__":
    main()
