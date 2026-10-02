"""Post hoc descriptive M1/M2/M3 analysis of accepted released predictions.

CPU metadata only: no training, model/checkpoint loading, inference or network.
Equal-fold scores remain primary. Neither pooling nor exposure identifies causal
leakage, independent landforms, geological correctness or global Titan accuracy.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts import export_repaired_release as base

MODELS, CLASSES, METRICS = base.MODELS, base.CLASSES, base.METRICS
FT_MODELS = MODELS[:3]
FROZEN = "longitude-review-20260929"
AUX = "longitude-auxiliary-20260929"
ADAPT = "longitude-adaptation-controls-20260930"
# Accepted public bytes at CETUS 6f139c9, not native/private catalog hashes.
PINS = {
    f"{FROZEN}/catalogs/titan_catalog.json": "8152226f876e0cb5fad105e0b00a37ef5b62d156d7f6eb3a85b77c549a402967",
    f"{FROZEN}/splits/titan_contiguous_fold_0.json": "b69e2fa3551dde125edf31a5601c52ccae755cb6169ed57d145b16f6983f85ab",
    f"{FROZEN}/splits/titan_contiguous_fold_1.json": "58ad14b783c0d6ffd2851cb622f1d0d4fd1438978aeb146e32e5dcb8d55a8adc",
    f"{FROZEN}/splits/titan_contiguous_fold_2.json": "2eee98044e7a636a6726800328b0361991432e9ea272eaee7873b3faa2ee28d2",
    f"{FROZEN}/splits/titan_contiguous_fold_3.json": "2dbd6117de1c299655116af827ee95998839c67da443bf2a03729b7e4e2fab75",
    f"{FROZEN}/splits/titan_contiguous_fold_4.json": "1df555f0f95c16e3fb44243ce7ecca41225e4e588353148f25725c68f1cb4271",
    f"{FROZEN}/evidence/crater_support.json": "8da6f2e74fd7542211892513a8092642f5398b2954a910d450c4dff9c57f990e",
    f"{AUX}/results/predictions.json.gz": "fa6f8a56a67c8f3a36020724b6aa0f4a32a933598089ca45f6bbbe692c2636b9",
    f"{ADAPT}/membership.json.gz": "d4e84a6c250c38b6630a30aee71017704244f5a850ef31b69dfe47df9b0dae30",
    f"{ADAPT}/results/predictions.json.gz": "5acb19f30812541689e52407cfbdd10e923c396524ad3bc7b9d4741d7ac57c7c",
    f"{ADAPT}/results/adaptation.json": "370f8e9f8eb1bdb8d2ba0bc4532a2ec0e1e1001750e95a601a18d58d78d0816b",
}
SCOPE = {
    "status": "post hoc descriptive; equal-fold metrics remain primary",
    "limits": "No causal leakage effect, independent replication, geological or global accuracy claim.",
    "units": "JSON/CSV scores and differences are fractions unless named _pp; TeX uses percent/points.",
    "m1": "Pool fold confusion counts within each adaptation seed, calculate metrics, then average seeds; LR pooled once.",
    "m2": "Common LR only; center distance strictly below exact split threshold from any reinstated training center. Average five draws, retain all four models/five folds.",
    "m3": "Map polygon parts are not physical craters; one-tile leverage changes the selection metric, not a demonstrated counterfactual selected epoch.",
    "confusion": "Rows are reference classes, columns predicted classes. Pooled LR counts test each tile once.",
    "empty_subsets": "Zero support yields null error rates, not zero error.",
    "display": "Only TeX is rounded (two decimals); JSON and CSV retain binary64 precision and integer counts.",
}


def index(rows, fields, expected=None):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        base.require(key not in result, f"Duplicate join key {fields}: {key}")
        result[key] = row
    if expected is not None:
        base.require(set(result) == set(expected), f"Incomplete/extra join keys: {fields}")
    return result


def load_bound(path, expected):
    base.require(base.sha256(path) == expected, f"Input hash differs: {path}")
    value = base.read_json(path)  # Rejects duplicate JSON keys and nonfinite data.
    base.require(base.sha256(path) == expected, f"Input changed while reading: {path}")
    return value


def confusion(labels, predictions):
    y, p = np.asarray(labels), np.asarray(predictions)
    base.require(y.ndim == 1 and y.shape == p.shape and y.dtype.kind in "iu"
                 and p.dtype.kind in "iu" and ((0 <= y) & (y < 6)).all()
                 and ((0 <= p) & (p < 6)).all(), "Invalid labels/prediction vector")
    return np.bincount(6 * y + p, minlength=36).reshape(6, 6)


def score(counts):
    return {"confusion_counts": np.asarray(counts).tolist(), **base.metrics(counts)}


def mean_scores(rows):
    return {level: {metric: np.mean([r[level][metric] for r in rows], axis=0).tolist()
                    for metric in METRICS} for level in ("macro", "per_class")}


def aggregate(ft_counts, lr_counts):
    """FT array is seed × fold × true × predicted; never pool seeds."""
    ft, lr = np.asarray(ft_counts), np.asarray(lr_counts)
    base.require(ft.ndim == 4 and lr.shape == ft.shape[1:] and ft.shape[-2:] == (6, 6),
                 "Confusion axes differ")
    ft_equal = mean_scores([mean_scores([score(cm) for cm in seed]) for seed in ft])
    lr_equal = mean_scores([score(cm) for cm in lr])
    pooled_seeds = [score(seed.sum(axis=0)) for seed in ft]
    pooled_lr = score(lr.sum(axis=0))
    pooled_ft = mean_scores(pooled_seeds)
    differences = {}
    for name, a, b in (("equal_fold_primary", ft_equal, lr_equal),
                       ("pooled_test_secondary", pooled_ft, pooled_lr)):
        differences[name] = {level: {k: (np.asarray(a[level][k]) - b[level][k]).tolist()
                                    for k in METRICS} for level in ("macro", "per_class")}
    return dict(equal_fold_primary=dict(adaptation=ft_equal, lr=lr_equal),
                pooled_test_secondary=dict(adaptation=pooled_ft, lr=pooled_lr),
                pooled_adaptation_seeds=pooled_seeds, ft_minus_lr=differences)


def nearest_distances(test_coordinates, boundary_coordinates, radius_m):
    """Haversine great-circle center distance, float64, bounded temporary arrays."""
    a, b = np.radians(test_coordinates), np.radians(boundary_coordinates)
    base.require(a.ndim == b.ndim == 2 and a.shape[1] == b.shape[1] == 2
                 and len(b) > 0 and np.isfinite(a).all() and np.isfinite(b).all()
                 and np.isfinite(radius_m) and radius_m > 0, "Invalid exposure geometry")
    result = []
    for block in np.array_split(a, max(1, (len(a) + 511) // 512)):
        h = (np.sin((block[:, None, 0] - b[None, :, 0]) / 2) ** 2
             + np.cos(block[:, None, 0]) * np.cos(b[None, :, 0])
             * np.sin((block[:, None, 1] - b[None, :, 1]) / 2) ** 2)
        result.extend((2 * radius_m * np.arcsin(np.sqrt(np.clip(h, 0, 1)))).min(axis=1))
    return np.asarray(result)


def error_rows(labels, baseline, draws, mask):
    """Per-draw ordinary errors and their mean; no macro score on local subsets."""
    y, original, draws, mask = map(np.asarray, (labels, baseline, draws, mask))
    base.require(y.ndim == 1 and original.shape == mask.shape == y.shape
                 and mask.dtype.kind == "b" and draws.ndim == 2
                 and draws.shape[1] == len(y) and len(draws) > 0, "Error subset shape")
    n = int(mask.sum())
    support = dict(zip(CLASSES, np.bincount(y[mask], minlength=6).tolist()))
    old = int((original[mask] != y[mask]).sum())
    counts = (draws[:, mask] != y[mask]).sum(axis=1)
    rows = []
    for draw, new in [*enumerate(counts.tolist()), ("mean", float(counts.mean()))]:
        rows.append(dict(draw=draw, n=n, **support, baseline_errors=old, substitution_errors=new,
                         baseline_error_rate=old / n if n else None,
                         substitution_error_rate=new / n if n else None,
                         difference=(new - old) / n if n else None))
    return rows


def exposure(test_coordinates, boundary_coordinates, radius_m, threshold_m):
    base.require(np.isfinite(threshold_m) and threshold_m > 0, "Invalid separation threshold")
    distances = nearest_distances(test_coordinates, boundary_coordinates, radius_m)
    return distances, distances < threshold_m


def single_crater_leverage(counts):
    """Hold all other validation predictions fixed; do not reselect an epoch."""
    cm = np.asarray(counts)
    base.require(cm.shape == (6, 6) and cm[5].sum() == 1, "Single-Craters selection support")
    recall = base.metrics(cm)["macro"]["recall"]
    correct = int(cm[5, 5])
    return dict(correct=correct, validation_macro_recall=recall,
                if_incorrect=recall-correct/6, if_correct=recall+(1-correct)/6, leverage=1/6)


def check_membership(catalog, splits, membership, auxiliary):
    base.require(catalog["class_names"] == auxiliary["class_names"] == list(CLASSES), "Class order")
    entries = {key[0]: value for key, value in index(catalog["tiles"], ("tile_id",)).items()}
    ids = list(entries)
    base.require(all(type(t["label"]) is int and 0 <= t["label"] < 6 for t in entries.values()), "Catalog labels")
    members = index(membership["folds"], ("fold",), [(f,) for f in range(5)])
    aux = index(auxiliary["folds"], ("fold",), members)
    union = set()
    for f, split in enumerate(splits):
        mem, row = members[f,], aux[f,]
        roles = split["assignments"]
        base.require(split["policy"]["test_fold"] == f and set(roles) == set(ids), "Split identity/coverage")
        base.require(set(roles.values()) <= set(base.ROLES), "Unknown split role")
        for role, field in (("train", "retained_train_ids"), ("val", "validation_ids"), ("test", "test_ids")):
            base.require(mem[field] == [t for t in ids if roles[t] == role], f"Exact ordered role join: {field}")
        for role in ("test", "validation"):
            base.require(mem[role + "_labels"] == [entries[t]["label"] for t in mem[role + "_ids"]], "Membership labels")
        base.require(row["test_ids"] == mem["test_ids"] and row["true_labels"] == mem["test_labels"], "LR test ordering/labels")
        base.require(set(row["common"]) == set(MODELS), "LR model coverage")
        base.require(not union.intersection(mem["test_ids"]), "Test folds overlap")
        union.update(mem["test_ids"])
        reasons = split["policy"]["exclusion_reasons"]
        base.require(set(reasons) == {t for t in ids if roles[t] == "buffer_excluded"}, "Exclusion coverage")
        boundary = sorted(t for t, r in reasons.items() if r == "train_buffer_excluded")
        base.require(mem["boundary_ids"] == boundary, "Reinstated training membership")
        draws = index(mem["draws"], ("draw_seed",), [(s,) for s in range(5)])
        for d in draws.values():
            removed = d["removed_ids"]
            base.require(len(removed) == len(set(removed)) == len(boundary)
                         and set(removed) <= set(mem["eligible_removal_ids"]), "Removal membership")
            selected = (set(mem["retained_train_ids"]) - set(removed)) | set(boundary)
            base.require(d["training_ids"] == [t for t in ids if t in selected]
                         and base.digest(d["training_ids"]) == d["training_ids_sha256"], "Replacement order/hash")
            base.require(Counter(entries[t]["label"] for t in removed)
                         == Counter(entries[t]["label"] for t in boundary), "Replacement class counts")
    expected = {t for t in ids if any(s["assignments"][t] == "test" for s in splits)}
    base.require(union == expected, "Test union differs")
    return entries, members, aux


def analyze(data):
    catalog = data[f"{FROZEN}/catalogs/titan_catalog.json"]
    splits = [data[f"{FROZEN}/splits/titan_contiguous_fold_{f}.json"] for f in range(5)]
    entries, members, aux = check_membership(catalog, splits, data[f"{ADAPT}/membership.json.gz"],
                                            data[f"{AUX}/results/predictions.json.gz"])
    base.require(len(entries) == 23380 and sum(len(m["test_ids"]) for m in members.values()) == 23246, "Accepted population")
    predictions, accepted = data[f"{ADAPT}/results/predictions.json.gz"], data[f"{ADAPT}/results/adaptation.json"]
    base.require(accepted["class_names"] == list(CLASSES) and set(accepted["models"]) == set(FT_MODELS), "Analysis model/classes")
    base.require({r["family"] for r in predictions["test"]} == {"adaptation", "buffer_lr", "buffer_sgd"}, "Prediction families")
    ft = index([r for r in predictions["test"] if r["family"] == "adaptation"],
               ("model", "fold", "seed"), itertools.product(FT_MODELS, range(5), range(5)))
    buffers = index([r for r in predictions["test"] if r["family"] == "buffer_lr"],
                    ("model", "fold", "draw_seed"), itertools.product(MODELS, range(5), range(5)))
    index([r for r in predictions["test"] if r["family"] == "buffer_sgd"],
          ("model", "fold", "draw_seed", "seed"), itertools.product(MODELS, range(5), range(5), range(5)))
    validation = index(predictions["validation"], ("model", "fold", "seed"), ft)
    tables = {name: [] for name in ("m1_scores", "m1_differences", "primary_lr_confusion", "m2_exposure", "m2_support", "m2_errors", "m3_parts", "m3_support", "m3_validation")}
    result = dict(schema_version=1, definitions=SCOPE, class_names=list(CLASSES),
                  fold_order=list(range(5)), seed_order=list(range(5)), draw_order=list(range(5)),
                  m1={}, primary_lr={}, m2=[], m3={})
    lr_counts = {m: [] for m in MODELS}
    for m in MODELS:
        for f in range(5):
            lr_counts[m].append(confusion(members[f,]["test_labels"], aux[f,]["common"][m]))
        pooled = sum(lr_counts[m])
        result["primary_lr"][m] = score(pooled)
        for i, j in itertools.product(range(6), repeat=2):
            tables["primary_lr_confusion"].append(dict(model=m, reference_class=CLASSES[i], predicted_class=CLASSES[j],
                count=int(pooled[i, j]), reference_support=int(pooled[i].sum())))
    for m in FT_MODELS:
        records = index(accepted["models"][m]["folds"], ("fold",), members)
        counts = np.zeros((5, 5, 6, 6), dtype=np.int64)
        for f in range(5):
            base.require(lr_counts[m][f].tolist() == records[f,]["common_lr"]["metrics"]["confusion_counts"], "LR saved confusion")
            seeds = index(records[f,]["seeds"], ("seed",), [(s,) for s in range(5)])
            for s in range(5):
                saved, v = seeds[s,], validation[m, f, s]
                counts[s, f] = confusion(members[f,]["test_labels"], ft[m, f, s]["predictions"])
                vc = confusion(members[f,]["validation_labels"], v["predictions"])
                base.require(counts[s, f].tolist() == saved["finetuned"]["confusion_counts"]
                             and vc.tolist() == saved["validation"]["confusion_counts"], "Saved adaptation/validation confusion")
                base.require(ft[m, f, s]["source_result_sha256"] == saved["outputs"]["finetuning_results.json"]
                             and v["source_result_sha256"] == saved["replay_sha256"], "Prediction source binding")
                if f == 0:
                    where = [i for i, y in enumerate(members[f,]["validation_labels"]) if y == 5]
                    base.require(len(where) == 1 and (vc.sum(axis=1) > 0).all(), "Single-Craters selection support")
                    tables["m3_validation"].append(dict(model=m, fold=f, seed=s, tile_id=members[f,]["validation_ids"][where[0]],
                        selected_epoch=saved["training"]["selected_epoch"], predicted_class=v["predictions"][where[0]],
                        **single_crater_leverage(vc)))
        summary = aggregate(counts, lr_counts[m])
        result["m1"][m] = summary
        for recipe, key in (("adaptation", "finetuned"), ("lr", "common_lr")):
            for level in ("macro", "per_class"):
                for metric in METRICS:
                    base.close(summary["equal_fold_primary"][recipe][level][metric],
                               accepted["models"][m]["summary"][key][level][metric]["mean"], "Primary aggregate differs")
        summaries = [(name, recipe, "mean", summary[name][recipe]) for name in
                     ("equal_fold_primary", "pooled_test_secondary") for recipe in ("adaptation", "lr")]
        summaries += [("pooled_test_secondary", "adaptation", s, row) for s, row in enumerate(summary["pooled_adaptation_seeds"])]
        for name, recipe, seed, row in summaries:
            for c, label in enumerate(("macro", *CLASSES)):
                tables["m1_scores"].append(dict(model=m, aggregation=name, recipe=recipe, seed=seed, class_name=label,
                    **{k: row["macro"][k] if c == 0 else row["per_class"][k][c-1] for k in METRICS}))
        for name, row in summary["ft_minus_lr"].items():
            for c, label in enumerate(("macro", *CLASSES)):
                tables["m1_differences"].append(dict(model=m, aggregation=name, class_name=label,
                    **{k: row["macro"][k] if c == 0 else row["per_class"][k][c-1] for k in METRICS}))
    for f, split in enumerate(splits):
        mem, policy = members[f,], split["policy"]
        coords = lambda ids: [[entries[t]["center_lat"], entries[t]["center_lon"]] for t in ids]
        distances, exposed = exposure(coords(mem["test_ids"]), coords(mem["boundary_ids"]),
                                      policy["radius_m"], policy["minimum_center_m"])
        support = dict(fold=f, threshold_m=policy["minimum_center_m"], radius_m=policy["radius_m"],
                       reinstated_train_tiles=len(mem["boundary_ids"]), test_tiles=len(distances), exposed_tiles=int(exposed.sum()))
        result["m2"].append(support)
        tables["m2_support"].append(support)
        for t, y, distance, flag in zip(mem["test_ids"], mem["test_labels"], distances, exposed):
            tables["m2_exposure"].append(dict(fold=f, tile_id=t, reference_class=y, nearest_reinstated_center_m=float(distance), exposed=bool(flag)))
        for m in MODELS:
            draws = [buffers[m, f, s]["predictions"] for s in range(5)]
            for p in draws:
                confusion(mem["test_labels"], p)  # Shape/type/class validation before subsetting.
            for name, mask in (("exposed", exposed), ("remaining", ~exposed), ("full", np.ones(len(exposed), dtype=bool))):
                tables["m2_errors"].extend(dict(model=m, fold=f, subset=name, **r) for r in
                    error_rows(mem["test_labels"], aux[f,]["common"][m], draws, mask))
    crater = data[f"{FROZEN}/evidence/crater_support.json"]
    ct = index(crater["tiles"], ("tile_id",))
    base.require({k[0] for k, r in ct.items() if r["label"] == 5} == {t for t, r in entries.items() if r["label"] == 5}, "Complete modal-Craters join")
    part_ids = {str(r["source_part_zero_based"]) for r in crater["source_parts"]}
    for (t,), row in ct.items():
        base.require(t in entries and row["label"] == entries[t]["label"]
                     and row["fold_roles"] == {str(f): s["assignments"][t] for f, s in enumerate(splits)}, "Crater catalog/role join")
        base.require(set(row["source_pixel_intersections_by_part"]) <= part_ids, "Unknown source part")
    all_parts = defaultdict(set)
    for f in range(5):
        selected = [t for t in members[f,]["test_ids"] if entries[t]["label"] == 5]
        parts = defaultdict(set)
        for t in selected:
            hits = ct[t,]["source_pixel_intersections_by_part"]
            base.require(bool(hits) and all(type(n) is int and n > 0 for n in hits.values()), "Missing crater-part support")
            for part in hits:
                parts[part].add(t)
                all_parts[part].add(t)
        for part, ids in sorted(parts.items(), key=lambda kv: int(kv[0])):
            tables["m3_parts"].append(dict(fold=f, part_id=int(part), test_tile_count=len(ids), test_tiles=len(selected), fraction=len(ids)/len(selected)))
        ranked = sorted(parts, key=lambda part: (-len(parts[part]), int(part)))
        tables["m3_support"].append(dict(fold=f, test_tiles=len(selected), map_parts=len(parts),
            largest_part_id=int(ranked[0]), largest_part_tiles=len(parts[ranked[0]]),
            largest_part_fraction=len(parts[ranked[0]])/len(selected), part_memberships=sum(map(len, parts.values()))))
    ranked = sorted(all_parts, key=lambda part: (-len(all_parts[part]), int(part)))
    result["m2_errors"] = tables["m2_errors"]
    result["m3"] = dict(support=tables["m3_support"], parts=tables["m3_parts"], validation=tables["m3_validation"],
                        top_two_part_ids=list(map(int, ranked[:2])),
                        unique_test_tiles=len(set.union(*all_parts.values())), map_parts=len(all_parts),
                        top_two_unique_test_tiles=len(set.union(*(all_parts[p] for p in ranked[:2]))))
    result["checks"] = dict(test_tiles=23246, adaptation_confusions=75, validation_confusions=75,
                            primary_lr_confusions=20, substitution_vectors=100, substitution_mean_rows=60)
    return result, tables


def write_outputs(output, result, tables):
    with (output / "analysis.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for name, rows in tables.items():
        with (output / (name + ".csv")).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    contrasts = index([r for r in tables["m2_errors"] if r["draw"] == "mean"],
                      ("model", "fold", "subset"), itertools.product(MODELS, range(5), ("exposed", "remaining", "full")))
    displays = {
        "m1_aggregation": ("Scores are percent; FT-LR differences are percentage points. Equal-fold primary; pooled secondary.",
            "lrrrr", r"Model & Primary $FT-LR$ & Pooled LR & Pooled FT & Pooled $FT-LR$",
            [[m.upper() if m == "dofa" else {"dinov2": "DINOv2", "croma": "CROMA"}[m],
              *[f"${100*v:.2f}$" for v in (r["ft_minus_lr"]["equal_fold_primary"]["macro"]["f1"],
                r["pooled_test_secondary"]["lr"]["macro"]["f1"], r["pooled_test_secondary"]["adaptation"]["macro"]["f1"],
                r["ft_minus_lr"]["pooled_test_secondary"]["macro"]["f1"])]] for m, r in result["m1"].items()]),
        "m2_exposure": ("Counts are tiles; exposed fraction is percent. Exact center-distance rule is in analysis.json.",
            "rrrrr", "Fold & Reinstated & Test & Exposed & Exposed (\\%)",
            [[r["fold"], r["reinstated_train_tiles"], r["test_tiles"], r["exposed_tiles"],
              f"{100*r['exposed_tiles']/r['test_tiles']:.2f}"] for r in result["m2"]]),
        "m2_errorcontrasts": ("Each cell is exposed / remaining ordinary error-rate change in percentage points: substitution minus baseline, mean of five draws. Positive means more errors; supports and full subsets remain in m2_errors.csv.",
            "rcccc", "Fold & DINOv2 (E/R) & DOFA (E/R) & CROMA (E/R) & Random init. (E/R)",
            [[f, *["$" + " / ".join(f"{100*contrasts[m, f, subset]['difference']:+.2f}" for subset in
                                    ("exposed", "remaining")) + "$" for m in MODELS]] for f in range(5)]),
        "m3_support": ("Counts are modal-Craters test tiles and intersecting map parts; part IDs are zero-based. Parts are not independent physical craters.",
            "rrrrr", "Fold & Craters tiles & Map parts & Largest part & Tiles in largest",
            [[r[k] for k in ("fold", "test_tiles", "map_parts", "largest_part_id", "largest_part_tiles")]
             for r in result["m3"]["support"]]),
    }
    for name, (units, columns, header, rows) in displays.items():
        lines = ["% Post hoc descriptive; see analysis.json for definitions. No global accuracy or causal claim.",
                 "% " + units, "% Only display values are rounded (two decimals).",
                 "\\begin{tabular}{" + columns + "}", "\\hline", header + r" \\", "\\hline"]
        lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
        (output / (name + ".tex")).write_text("\n".join(lines + ["\\hline", "\\end{tabular}", ""]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True, help="CETUS/release directory; exact accepted bytes required")
    parser.add_argument("--output", type=Path, required=True, help="New external output directory")
    args = parser.parse_args(argv)
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    source = {str(p.relative_to(ROOT)): base.sha256(p) for p in
              (Path(__file__).resolve(), Path(base.__file__).resolve(), ROOT / "tests/test_repaired_interpretation.py")}
    release, output = args.release_root.resolve(), args.output.resolve()
    base.require(not output.exists() and not output.is_relative_to(ROOT)
                 and not output.is_relative_to(release.parent), "Output must be new and outside source/CETUS")
    inputs = {name: dict(sha256=h, bytes=(release/name).stat().st_size) for name, h in PINS.items()}
    data = {name: load_bound(release/name, h) for name, h in PINS.items()}
    result, tables = analyze(data)
    for name, h in PINS.items():
        base.require(base.sha256(release/name) == h, "Input changed during analysis")
    for name, h in source.items():
        base.require(base.sha256(ROOT/name) == h, "Source changed during analysis")
    output.mkdir(parents=True)
    write_outputs(output, result, tables)
    execution = dict(status="complete", started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-clock, command=sys.orig_argv, cwd=str(Path.cwd()), release_root=str(release),
        output=str(output), scope=SCOPE, inputs=inputs, source_sha256=source,
        source_git_head=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        source_git_status=subprocess.check_output(["git", "-C", str(ROOT), "status", "--short"], text=True),
        runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, platform=platform.platform(),
                     environment={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")}),
        outputs={p.name: dict(sha256=base.sha256(p), bytes=p.stat().st_size) for p in sorted(output.iterdir())})
    with (output / "execution.json").open("x") as stream:
        json.dump(execution, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="complete", output=str(output), checks=result["checks"])))


if __name__ == "__main__":
    main()
