"""Draw all six classes from accepted retrieval endpoints, without rescoring.

Each class retains five regions and both model families. Every tied-score group
is drawn as a vector point. All six classes use the same display spacing for
larger symbols that identify encoder states or training seeds.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import matplotlib
import numpy as np

from scripts import plot_crater_retrieval as base
from scripts.plot_class_ranking import CLASSES
from scripts.display_io import write_completion


require = base.require
ARCHIVE_LIMIT = 120 * 1024 * 1024
REPORT_LIMIT = 8 * 1024 * 1024


def bound_bytes(path, expected, limit):
    require(path.stat().st_size <= limit, "Input exceeds size bound")
    data = path.read_bytes()
    require(len(data) <= limit, "Input exceeds size bound")
    require(hashlib.sha256(data).hexdigest() == expected, "Input SHA256 differs")
    return data


def decode(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_curves(export_path, export_sha256, curve_path, curve_sha256, accepted, accepted_sha256):
    report = decode(bound_bytes(export_path, export_sha256, REPORT_LIMIT))
    require(report["schema_version"] == "class-retrieval-export-v1"
            and report["status"] == "computed_pending_publication", "Export schema/status differs")
    require(report["accepted_results"]["sha256"] == accepted_sha256,
            "Export belongs to a different accepted report")
    require(report["class_order"] == list(CLASSES)
            and canonical(report["fold_counts"]) == canonical(accepted["fold_counts"]),
            "Export class order/support differs")
    # Canonical row comparisons retain scalar types, irrespective of list order.
    require(sorted(map(canonical, report["metrics"])) == sorted(map(canonical, accepted["metrics"])),
            "Export metrics differ from accepted report")
    metrics = {(tuple(row["cell"]), row["class_index"]): row for row in accepted["metrics"]}
    expected = {(cell, c) for cell in base.expected_cells() for c in range(6)}
    entries = report["curve_index"]
    require(isinstance(entries, list) and len(entries) == 360, "Expected 360 curves")
    seen, keys = set(), set()
    for entry in entries:
        cell, c = entry["cell"], entry["class_index"]
        require(type(cell) is list and len(cell) == 4 and type(cell[0]) is str
                and type(cell[1]) is int and type(cell[2]) is str and type(cell[3]) is int
                and type(c) is int and 0 <= c < 6, "Invalid class/cell identity")
        identity = (tuple(cell), c)
        require(identity in expected and identity not in seen, "Duplicate/unexpected curve identity")
        kind, fold, arm, seed = cell
        prefix = f"{kind}__fold_{fold}__{arm}__seed_{seed}__class_{c}"
        names = {name: f"{prefix}_{name}" for name in base.FIELDS}
        require(entry["prefix"] == prefix and entry["arrays"] == names
                and entry["class"] == CLASSES[c] and type(entry["region"]) is int
                and entry["region"] == fold + 1, "Curve label/array mapping differs")
        require(type(entry["score_groups"]) is int
                and entry["score_groups"] == metrics[identity]["distinct_scores"],
                "Curve group count differs")
        seen.add(identity)
        keys.update(names.values())
    require(seen == expected and len(keys) == 1800, "Incomplete curve grid")
    inventory = report["array_inventory"]
    require(set(inventory) == keys, "Incomplete array inventory")
    data = bound_bytes(curve_path, curve_sha256, ARCHIVE_LIMIT)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        require(len(members) == 1800 and {m.filename for m in members} == {k + ".npy" for k in keys},
                "Expected exactly 1800 unique arrays")
        require(all(0 < m.file_size <= base.MEMBER_LIMIT for m in members)
                and sum(m.file_size for m in members) <= ARCHIVE_LIMIT, "Array size bound exceeded")
    curves = []
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        for entry in sorted(entries, key=lambda e: (e["class_index"], e["cell"])):
            arrays = {name: archive[key] for name, key in entry["arrays"].items()}
            for name, value in arrays.items():
                require(canonical(inventory[entry["arrays"][name]]) == canonical({
                    "dtype": value.dtype.str, "shape": list(value.shape), "bytes": value.nbytes}),
                    "Saved array inventory differs")
            cell, c = entry["cell"], entry["class_index"]
            curve = base.validated_curve(arrays, metrics[tuple(cell), c], entry["prefix"], cell)
            curves.append({**curve, "class_index": c, "class": CLASSES[c]})
    return curves


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "export", "curves"):
        parser.add_argument("--" + name, required=True, type=Path)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(not output.is_relative_to(base.ROOT), "Use a private output directory outside the checkout")
    require(not args.output.exists() and not args.output.is_symlink(), "Use a fresh output directory")
    accepted = base.read_report(args.report, args.report_sha256)
    curves = read_curves(args.export, args.export_sha256, args.curves, args.curves_sha256,
                         accepted, args.report_sha256)
    binding = {"report_sha256": args.report_sha256, "export_sha256": args.export_sha256,
               "curves_sha256": args.curves_sha256, "python_version": sys.version,
               "numpy": np.__version__, "matplotlib": matplotlib.__version__,
               "sources_sha256": {str(p.relative_to(base.ROOT)): base.digest(p) for p in (
                   Path(__file__), Path(base.__file__), base.ROOT / "scripts/plot_class_ranking.py",
                   base.ROOT / "scripts/display_io.py", base.ROOT / "scripts/prepare_cvpr_tables.py",
                   Path(base.sans.__file__),
                   Path(base.roman.__file__), base.sans.FONT_FILE,
                   base.roman.FONT_DIR / "NimbusRoman-Regular.ttf")}}
    output.mkdir(parents=True, exist_ok=False)
    sheets, names = {}, []
    for c, name in enumerate(CLASSES):
        selected = [curve for curve in curves if curve["class_index"] == c]
        require(len(selected) == 60, "A class sheet is incomplete")
        sheets[name] = {}
        for variant in ("cvpr", "full"):
            figure, evidence = base.draw_figure(selected, variant, class_name=name.title(), dense=True)
            stem = f"{name}-retrieval-{variant}"
            with matplotlib.rc_context({"savefig.dpi": base.RASTER_DPI}):
                base.roman.save_figure(figure, output / stem)
            sheets[name][variant] = evidence
            names.extend(stem + "." + ext for ext in ("pdf", "png"))
            del figure
    write_completion(output / "display.json", {
        **binding, "schema_version": "class-retrieval-display-v1", "class_order": list(CLASSES),
        "rendering": "All tied-score endpoints are vector points. Every class adds identity markers at the same deterministic display spacing on existing endpoints. No observed connectors or thinning. Dashed uniform-order expectation is separate.",
        "curves": [{**base.display_mapping(c), "class_index": c["class_index"], "class": c["class"]}
                   for c in curves], "sheets": sheets})
    names.append("display.json")
    require(all((output / name).stat().st_size > 0 for name in names), "Missing/empty render output")
    write_completion(output / "complete.json", {
        **binding, "schema_version": "class-retrieval-display-v1", "status": "rendered",
        "curves_per_variant": 360, "sheets_per_variant": 6,
        "outputs": {name: {"sha256": base.digest(output / name), "bytes": (output / name).stat().st_size}
                    for name in names}})


if __name__ == "__main__":
    main()
