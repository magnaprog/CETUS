"""Reconstruct the packaged class-ranking results from the unchanged input ZIP."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys


ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_package(root):
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["schema"] == "class-ranking-package-v1", "Package schema differs")
    files = manifest["files"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == set(files) | {"manifest.json", "SHA256SUMS"}, "Package inventory differs")
    for name, bound in files.items():
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and ".." not in rel.parts and str(rel) == name,
                "Invalid package member")
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root), "Package symlink differs")
        require(path.stat().st_size == bound["bytes"] and sha(path) == bound["sha256"],
                "Package file changed: " + name)
    expected_sums = {**{name: bound["sha256"] for name, bound in files.items()},
                     "manifest.json": sha(root / "manifest.json")}
    lines = (root / "SHA256SUMS").read_text().splitlines()
    require(sorted(lines) == sorted(value + "  " + name for name, value in expected_sums.items()),
            "Checksum inventory differs")
    return manifest


def compare_results(expected, actual, expected_curves, actual_curves):
    """Compare saved answers only; do not compute another AP estimator."""
    import numpy as np

    for result in (expected, actual):
        require(len(result["metrics"]) == 360 and len(result["craters_curve_index"]) == 60,
                "Result grid differs")
        require([len(result["summaries"][k]) for k in ("regional", "contrasts", "geographic")]
                == [140, 70, 42], "Aggregation grid differs")
    require({k: v for k, v in expected.items() if k != "execution"}
            == {k: v for k, v in actual.items() if k != "execution"}, "Scientific result differs")
    require(set(expected_curves) == set(actual_curves) and len(expected_curves) == 300,
            "Curve array inventory differs")
    for name in expected_curves:
        a, b = expected_curves[name], actual_curves[name]
        require(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b),
                "Curve array differs: " + name)
    return {"status": "reconstruction_verified", "class_rows": 360, "craters_curves": 60,
            "curve_arrays": 300, "regional_rows": 140, "contrasts": 70, "geographic_rows": 42,
            "scientific_result_exact": True, "curve_arrays_exact": True,
            "excluded_result_fields": ["execution"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization-zip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = check_package(ROOT)
    require(not args.output.exists() and not args.output.is_symlink(), "Use a fresh output directory")
    output = args.output.resolve()
    require(output.parent.is_dir() and not output.is_relative_to(ROOT), "Output must be outside the package")
    require(platform.python_version() == "3.12.3", "Recorded Python 3.12.3 required")

    from scripts import analyze_class_ranking as analysis
    import numpy as np

    require(np.__version__ == "2.4.4", "Recorded NumPy 2.4.4 required")
    for module in (analysis, analysis.primary, sys.modules[analysis.write_result.__module__]):
        path = Path(module.__file__).resolve()
        require(path.parent == ROOT / "scripts", "Unexpected analysis import origin")
        require(sha(path) == manifest["files"]["scripts/" + path.name]["sha256"],
                "Analysis source changed")
    analysis.runtime_record()
    try:
        analysis.main(["--initialization-zip", str(args.initialization_zip.resolve()),
                       "--contract", str(ROOT / "contract.json"),
                       "--contract-sha256", manifest["files"]["contract.json"]["sha256"],
                       "--output", str(output)])
        expected = analysis.primary.read_json(ROOT / "results.json")
        actual = analysis.primary.read_json(output / "results.json")
        with np.load(ROOT / "craters-curves.npz", allow_pickle=False) as a, \
                np.load(output / "craters-curves.npz", allow_pickle=False) as b:
            receipt = compare_results(expected, actual, a, b)
        require(check_package(ROOT) == manifest, "Package changed during reconstruction")
        analysis.runtime_record()
        receipt.update({"input": actual["input"], "runtime": actual["execution"]["runtime"],
                        "package_manifest_sha256": sha(ROOT / "manifest.json"),
                        "outputs": {name: {"sha256": sha(output / name),
                                           "bytes": (output / name).stat().st_size}
                                    for name in ("results.json", "craters-curves.npz", "complete.json")}})
        analysis.write_result(output / "reconstruction-check.json", receipt)
    except Exception as error:
        if output.is_dir():
            try:
                analysis.write_result(output / "reconstruction-failed.json",
                                      {"status": "failed", "type": type(error).__name__, "error": str(error)})
            except Exception:
                pass  # A failed diagnostic must not replace the original failure.
        raise
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
