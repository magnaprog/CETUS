"""Verify the saved plots against the terrain display manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def check_group(directory: Path, expected: dict, count: int) -> None:
    actual_names = {
        path.name for path in directory.iterdir() if path.suffix in {".pdf", ".png"}
    }
    if len(expected) != count:
        raise ValueError(f"{directory.name}: expected {count} entries in the manifest")
    if actual_names != expected.keys():
        raise ValueError(f"{directory}: differing files {sorted(actual_names ^ expected.keys())}")
    for name, record in expected.items():
        path = directory / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if record != {"bytes": path.stat().st_size, "sha256": digest}:
            raise ValueError(f"{path} differs from the display manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for folder, field, count in (
        ("crater-retrieval", "expected_figures", 4),
        ("class-contrasts", "expected_class_contrasts", 4),
        ("absolute-ap", "expected_absolute_ap", 4),
        ("regional-errors", "expected_regional_errors", 4),
        ("class-retrieval", "expected_class_retrieval", 24),
    ):
        check_group(args.output_directory / folder, manifest[field], count)
    for style in ("sans", "roman"):
        check_group(
            args.output_directory / f"geography-{style}",
            manifest["expected_dataset_geography"][style],
            8,
        )


if __name__ == "__main__":
    main()
