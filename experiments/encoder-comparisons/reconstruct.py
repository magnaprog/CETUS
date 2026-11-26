"""Recalculate the four CETUS encoder comparisons from their saved results."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import stat
import subprocess
import sys
import zipfile


PACKAGE = Path(__file__).resolve().parent
ARCHIVES = ("scientific", "followup", "initialization", "reassessment")
VOTE_MANIFEST_SHA256 = "be98e9950e19641889e8e8234a43b3b2913648848407df38796fc58b9e556e69"


def read_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(archive, record, destination):
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
            raise ValueError(f"Archive SHA-256 differs: {archive.name}")
        stream.seek(0)
        with zipfile.ZipFile(stream) as packed:
            members = packed.infolist()
            names = {item.filename for item in members}
            if len(members) != record["archive_entries"] or len(names) != len(members):
                raise ValueError(f"Archive member count or uniqueness differs: {archive.name}")
            for item in members:
                name = item.filename
                path = PurePosixPath(name)
                if (not path.parts or path.is_absolute() or ".." in path.parts
                        or str(path) != name or "\\" in name or PureWindowsPath(name).drive
                        or stat.S_ISLNK(item.external_attr >> 16)):
                    raise ValueError(f"Invalid member {name!r} in {archive.name}")
            destination.mkdir()
            packed.extractall(destination)


def check_files(directory):
    checksum_list = directory / "SHA256SUMS"
    expected = checksum_list.read_text().splitlines()
    actual = [
        f"{sha256(path)}  {path.relative_to(directory).as_posix()}"
        for path in directory.rglob("*") if path.is_file() and path != checksum_list
    ]
    if sorted(expected) != sorted(actual):
        raise ValueError(f"Extracted files differ from {checksum_list}")


def run(directory, script, *args, stdout=None):
    subprocess.run([sys.executable, "-B", script, *args], cwd=directory,
                   stdout=stdout, check=True)


def replay(output):
    provenance = read_json(PACKAGE / "provenance.json")
    runtime = provenance["reconstruction"]["runtime"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.pop("PYTHONOPTIMIZE", None)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

    import numpy

    if (platform.python_version(), numpy.__version__) != (runtime["python"], runtime["numpy"]):
        raise ValueError(f"Requires Python {runtime['python']} and NumPy {runtime['numpy']}; "
                         f"found {platform.python_version()} and {numpy.__version__}")

    root = PACKAGE.parents[1] if PACKAGE.parent.name == "experiments" else PACKAGE
    if output.is_symlink() or output.resolve().is_relative_to(root):
        raise ValueError("Choose a new output directory outside the checkout")
    work = output.resolve()
    work.mkdir()
    reports = work / "reports"
    reports.mkdir()
    archives = provenance["archives"]
    for name in ARCHIVES:
        filename = f"{name}-supplement.zip"
        unpack(PACKAGE / filename, archives[filename], work / name)
    for name in ("scientific", "followup", "initialization"):
        check_files(work / name)

    run(work / "scientific", "scripts/replay_metrics.py", "--output", reports / "scientific.json")
    run(work / "scientific", "scripts/prepare_tables.py", "--output", "paper/tables", "--check")

    coverage = work / "followup/coverage"
    scientific_zip = PACKAGE / "scientific-supplement.zip"
    run(coverage, "analyze_sar_coverage.py", "--package", scientific_zip,
        "--sha256", archives[scientific_zip.name]["sha256"], "--output", reports / "coverage.json")
    saved = read_json(coverage / "results.json")
    replayed = read_json(reports / "coverage.json")
    del saved["execution"], replayed["execution"]
    if saved != replayed:
        raise ValueError("Coverage results differ from the saved follow-up report")

    run(work / "followup/label-support", "replay_vote_inputs.py", "data",
        "--report", reports / "votes.json", "--expect-manifest-sha256", VOTE_MANIFEST_SHA256)
    run(work / "initialization", "reconstruct.py", "--output", reports / "initialization.json")
    # The reassessment checker verifies its own MANIFEST.json and prints the report.
    with (reports / "reassessment.json").open("x") as report:
        run(work / "reassessment", "check.py", "--initialization-zip",
            PACKAGE / "initialization-supplement.zip", stdout=report)

    for name in ARCHIVES:
        filename = f"{name}-supplement.zip"
        if sha256(PACKAGE / filename) != archives[filename]["sha256"]:
            raise ValueError(f"Input changed during replay: {filename}")
    print(f"All six checks passed. Reports are in {reports}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="New directory outside the checkout for extracted inputs and reports")
    args = parser.parse_args()
    try:
        replay(args.output)
    except (OSError, ImportError, KeyError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as err:
        parser.exit(1, f"Replay failed: {err}\n")


if __name__ == "__main__":
    main()
