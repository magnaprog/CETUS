"""Check the public file inventory, or refresh its file sizes and checksums."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = "release/inventory.json"
PROVENANCE = "release/provenance.json"
SUMS = "SHA256SUMS"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tracked_files():
    listing = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT)
    for entry in listing.decode().split("\0"):
        if not entry:
            continue
        header, name = entry.split("\t", 1)
        mode, _, stage = header.split()
        if stage != "0" or mode not in {"100644", "100755"}:
            raise ValueError(f"Unmerged or unsupported Git entry: {name}")
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise ValueError(f"Expected a regular tracked file: {name}")
        yield name


def check_metadata(index, records):
    def require(name, expected=None):
        if name not in records:
            raise ValueError(f"Missing inventory target: {name}")
        if expected is not None and records[name]["sha256"] != expected:
            raise ValueError(f"Metadata checksum differs: {name}")

    software = index["software"]
    require(software["manifest"])
    source = json.loads((ROOT / software["manifest"]).read_bytes())
    if software["development_source_commit"] != source["development_source_commit"]:
        raise ValueError("Development source commits differ")
    for name, record in source["files"].items():
        require(name, record["sha256"])

    for group in index.values():
        if isinstance(group, dict):
            for key in ("guide", "source_guide", "input_guide", "manifest", "archive", "path"):
                if isinstance(group.get(key), str):
                    expected = group.get("manifest_sha256") if key == "manifest" else None
                    if key == "path":
                        expected = group.get("sha256")
                    require(group[key], expected)
    result_index = Path(index["result_index"]["path"])
    for name, record in json.loads((ROOT / result_index).read_bytes())["family_files"].items():
        require((result_index.parent / name).as_posix(), record["sha256"])
    for record in index["catalogs"].values():
        require(record["path"], record["sha256"])
    for folder in index["packages"].values():
        if not any(name.startswith(folder + "/") for name in records):
            raise ValueError(f"Package directory has no tracked files: {folder}")

    replay = software["saved_prediction_reconstruction"]
    require(replay["package"] + "/evidence/acceptance.json", replay["acceptance_sha256"])
    display = index["terrain_plots"]
    scripts = Path(display["manifest"]).parent / "scripts"
    for name, expected in display["current_renderers"].items():
        require((scripts / name).as_posix(), expected)

    provenance = json.loads((ROOT / PROVENANCE).read_bytes())
    for key in ("guide", "manifest"):
        if provenance["software"][key] != software[key]:
            raise ValueError(f"Software {key} differs between provenance and index")
    require(software["manifest"], provenance["software"]["manifest_sha256"])
    image = provenance["processed_titan_image_asset"]
    require(image["notice"], image["notice_sha256"])
    require(image["checksums"], image["checksums_sha256"])
    sums = {}
    for line in (ROOT / image["checksums"]).read_text().splitlines():
        checksum, name = line.split("  ", 1)
        if name in sums:
            raise ValueError(f"Duplicate release asset: {name}")
        sums[name] = checksum
    if sums != {image["archive"]: image["archive_sha256"], Path(image["notice"]).name: image["notice_sha256"]}:
        raise ValueError("Release asset checksums differ from provenance")
    for key, package in provenance["packages"].items():
        folder = package["path"]
        if index["packages"].get(key) != folder:
            raise ValueError(f"Unknown provenance package: {folder}")
        for name in (SUMS, "SOURCE_EXPORT"):
            key = name + "_sha256"
            if key in package:
                member = name + ".json" if name == "SOURCE_EXPORT" else name
                require(folder + "/" + member, package[key])


def replace_file(name, data):
    path = ROOT / name
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        staged = Path(tmp.name)
        try:
            tmp.write(data)
            tmp.flush()
            os.fchmod(tmp.fileno(), path.stat().st_mode & 0o777)
            os.fsync(tmp.fileno())
            os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)


def run(update):
    names = set(tracked_files())
    if not {INDEX, SUMS} <= names:
        raise ValueError(f"{INDEX} and {SUMS} must be tracked in Git")
    records = {}
    for name in sorted(names - {INDEX, SUMS}):
        with (ROOT / name).open("rb") as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
            records[name] = {"bytes": stream.tell(), "sha256": hasher.hexdigest()}

    original = (ROOT / INDEX).read_bytes()
    index = json.loads(original)
    check_metadata(index, records)
    if index["files"] != records and not update:
        changed = sorted(n for n in set(records) | set(index["files"])
                         if records.get(n) != index["files"].get(n))
        raise ValueError(f"{INDEX} differs for {', '.join(changed)}")
    index_bytes = original
    if update and index["files"] != records:
        index["files"] = records
        index_bytes = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()
    hashes = {name: item["sha256"] for name, item in records.items()}
    hashes[INDEX] = digest(index_bytes)
    checksums = "".join(f"{hashes[name]}  {name}\n" for name in sorted(hashes)).encode()
    if update:
        if index_bytes != original:
            replace_file(INDEX, index_bytes)
        if (ROOT / SUMS).read_bytes() != checksums:
            replace_file(SUMS, checksums)
    elif (ROOT / SUMS).read_bytes() != checksums:
        raise ValueError(f"{SUMS} differs from the tracked files or index")
    print(f"Verified {len(records)} indexed files and {len(hashes)} checksums.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help=f"Refresh file entries in {INDEX} and checksums in {SUMS} using tracked paths.")
    try:
        run(parser.parse_args().update)
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
