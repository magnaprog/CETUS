"""Rebuild the v0.2.0 release files from the extracted Titan image collection."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ARCHIVE = "titan-images.tar.zst"
NOTICE = "RELEASE_NOTICE.md"
CHECKSUMS = "SHA256SUMS.release"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pack(root, names, output):
    with output.open("xb") as stream:
        with subprocess.Popen(
            ["zstd", "--no-progress", "-q", "-T4", "-4", "-c"],
            stdin=subprocess.PIPE, stdout=stream,
        ) as compressor:
            with tarfile.open(fileobj=compressor.stdin, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                for name in names:
                    path = root / name
                    entry = tarfile.TarInfo(name)  # Root owner, zero timestamp, empty owner names.
                    entry.mode = 0o644
                    entry.size = path.stat().st_size
                    with path.open("rb") as member:
                        archive.addfile(entry, member)
            compressor.stdin.close()
            if compressor.wait():
                raise RuntimeError("zstd compression failed")


def build(source, destination):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("Choose a new output directory outside the image collection")

    subprocess.run([
        sys.executable, "-I", "-B", str(HERE / "inputs/verify_inputs.py"), "--root", str(source)
    ], check=True)
    names = sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file())

    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        assets = Path(temporary) / "assets"
        assets.mkdir()
        pack(source, names, assets / ARCHIVE)
        shutil.copyfile(HERE / NOTICE, assets / NOTICE)
        checksums = "".join(f"{digest(assets / name)}  {name}\n" for name in (ARCHIVE, NOTICE))
        if checksums != (HERE / CHECKSUMS).read_text(encoding="ascii"):
            raise ValueError("The rebuilt files differ from the v0.2.0 release checksums")
        shutil.copyfile(HERE / CHECKSUMS, assets / CHECKSUMS)
        assets.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="extracted Titan image collection")
    parser.add_argument("destination", type=Path, help="new directory for the three release files")
    args = parser.parse_args()
    build(args.source, args.destination)
