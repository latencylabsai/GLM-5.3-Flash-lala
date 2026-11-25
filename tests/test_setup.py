"""Exercise setup routing with stand-ins for costly downloads and builds."""

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


class SetupTests(unittest.TestCase):
    def run_setup(self, system, machine):
        with tempfile.TemporaryDirectory(prefix="lala setup ") as directory:
            root = pathlib.Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy(ROOT / "scripts/setup.sh", scripts / "setup.sh")
            binaries = root / "bin"
            binaries.mkdir()
            log = root / "calls.txt"

            def executable(path, body):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("#!/bin/bash\nset -e\n" + body)
                path.chmod(0o755)

            executable(
                binaries / "uname",
                f'if [ "$1" = -s ]; then echo {system}; else echo {machine}; fi\n',
            )
            executable(binaries / "uv", 'echo "uv $*" >> "$SETUP_LOG"\n')
            for backend in ["mac", "spark"]:
                executable(
                    scripts / f"setup-{backend}.sh",
                    f'echo build-{backend} >> "$SETUP_LOG"\n',
                )
            python = root / (
                ".venv/bin/python"
                if system == "Darwin"
                else ".runtime/spark-venv/bin/python"
            )
            executable(python, 'printf "%s\\n" "$*" >> "$SETUP_LOG"\n')
            env = dict(
                os.environ, PATH=f"{binaries}:{os.environ['PATH']}", SETUP_LOG=str(log)
            )
            subprocess.run(
                ["bash", str(scripts / "setup.sh")],
                env=env,
                check=True,
                capture_output=True,
            )
            calls = log.read_text()
            self.assertIn("glm_lala.cli download --output", calls)
            self.assertNotIn("glm_lala.cli serve", calls)
            subprocess.run(
                ["bash", str(root / ".runtime/serve.sh"), "--serial"],
                env=env,
                check=True,
            )
            self.assertIn("--serial", log.read_text())
            return calls, (root / ".runtime/serve.sh").read_text()

    def test_mac_prepares_both_packs(self):
        calls, launch = self.run_setup("Darwin", "arm64")
        self.assertIn("glm_lala.repack --source", calls)
        self.assertIn("glm_lala.repack_mtp --source", calls)
        self.assertIn("--backend mac", launch)
        self.assertNotIn("--serial", launch)

    def test_spark_uses_native_checkpoint(self):
        calls, launch = self.run_setup("Linux", "aarch64")
        self.assertIn("build-spark", calls)
        self.assertNotIn("repack", calls)
        self.assertIn("--backend spark", launch)
