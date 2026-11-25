"""Stage only hash-verified libraries from the pinned Sushi release."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent.parent
LIBRARIES = ("libmlx.dylib", "libmlxc.dylib", "libjaccl.dylib",
             "libwebp.dylib", "libsharpyuv.dylib", "mlx.metallib")


def stage(archive, destination, pin, mac):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != pin["sha256"]:
        raise ValueError("Sushi release archive SHA-256 mismatch")
    payload = {}
    with tarfile.open(archive) as bundle:
        prefix = "sushi-macos-arm64/"
        guest = json.load(bundle.extractfile(prefix + "guest.json"))
        if guest["commit"] != mac["revision"] or guest["version"] != pin["version"]:
            raise ValueError("Sushi release metadata does not match the source pin")
        for name in LIBRARIES:
            member = bundle.getmember(prefix + "lib/" + name)
            if not member.isfile():
                raise ValueError(f"Expected a regular library file: {name}")
            payload["lib/" + name] = bundle.extractfile(member).read()
        for name in ("LICENSE", "LICENSE-APACHE-2.0", "NOTICE", "guest.json"):
            payload[name] = bundle.extractfile(prefix + name).read()
    payload[".version"] = (
        f'mlx={mac["mlx"]} mlxc={mac["mlx_c"]} target=26.2\n'
    ).encode()
    receipt = {
        **pin,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
    }
    payload["RELEASE.json"] = (json.dumps(receipt, indent=2) + "\n").encode()
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or any(
            not (destination / name).is_file()
            or (destination / name).is_symlink()
            or (destination / name).read_bytes() != data
            for name, data in payload.items()
        ):
            raise ValueError("Existing MLX stage differs; set SUSHI_RUNTIME to a new directory")
        print("Pinned Sushi libraries already verified.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "stage"
        for name, data in payload.items():
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        staged.rename(destination)
    print(f"Verified Sushi {pin['version']} libraries staged at {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    args = parser.parse_args()
    mac = json.loads((ROOT / "glm_lala/runtime-lock.json").read_text())["mac"]
    pin = mac["library_release"]
    # Each setup verifies a fresh download, independent of any existing lab stage.
    with tempfile.TemporaryDirectory(prefix="lala-sushi-release-") as temporary:
        archive = Path(temporary) / "sushi.tar.gz"
        subprocess.run(["curl", "-fL", "--retry", "3", "--proto", "=https",
                        "--proto-redir", "=https", "-o", str(archive), pin["url"]], check=True)
        stage(archive, args.runtime / "lib/mlx", pin, mac)


if __name__ == "__main__":
    main()
