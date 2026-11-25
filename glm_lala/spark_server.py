"""Install the native mixed-K hook before entering the existing TabbyAPI."""

import os
import pathlib
import runpy
import sys


def configure_decode():
    """Apply the serving policy before any engine imports or kernel launches."""
    # Set before importing the engine: CUDA dispatch caches these settings.
    # Both serial and MTP must use the same projection/router/reduction policy.
    os.environ.update(
        EXL3_INT8_GEMV="0",
        LALA_EXL3_CONSISTENT="1",
        EXL3_MOE_COOP_WIDE="1",
        EXL3_MOE_COOP_KSPLIT="1",
    )


def main():
    """Run TabbyAPI in the pinned Spark runtime."""
    configure_decode()
    import mixedk
    import exllamav3_ext
    from exllamav3.version import __version__

    if getattr(exllamav3_ext, "lala_consistent_decode_version", None) != 1:
        raise RuntimeError(
            "ExLlamaV3 is missing the Lala decode correction. "
            "Rebuild with bash scripts/setup-spark.sh."
        )
    if __version__ != "1.6.2":
        raise RuntimeError(f"Expected validated ExLlamaV3 1.6.2, got {__version__}")
    print("Lala decode policy v1: consistent small-row kernels, FP32 recurrent state, FP16 KV.", flush=True)
    mixedk.SUPPORTED_VERSION = "1.6.2"
    if not mixedk.install(unified=False, decode="coop", prefill="grouped"):
        raise RuntimeError("Native mixed-K installation failed.")
    tabby = pathlib.Path(os.environ.get("TABBY_DIR", "/opt/tabbyAPI"))
    sys.path.insert(0, str(tabby))
    os.chdir(tabby)
    sys.argv = [str(tabby / "main.py"), *sys.argv[1:]]
    runpy.run_path(str(tabby / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
