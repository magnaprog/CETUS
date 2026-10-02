"""Verify and rebuild a fixed adaptation/control release using CPU metadata only.

The acceptance SHA256 is an external trust anchor supplied by the reviewer.
No training, checkpoint/estimator loading, Git, network or raw tiles are used.
"""

import argparse
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

import numpy as np
from scripts import export_repaired_release as base
from scripts import export_repaired_auxiliary_results as auxiliary
from scripts import analyze_repaired_adaptation as adaptation
from scripts import plot_repaired_adaptation as display_a
from scripts import plot_repaired_training_substitution as display_b

require, close, digest = base.require, base.close, base.digest
ROOT = Path(__file__).resolve().parents[1]
IMPORT_SHA256 = base.sha256(__file__)
EXPORTER_SHA256 = base.sha256(ROOT / "scripts/export_repaired_adaptation_results.py")
TEST_SHA256 = {
    name: base.sha256(ROOT / name)
    for name in (
        "tests/repaired_adaptation_release_fixtures.py",
        "tests/test_repaired_adaptation_release.py",
    )
}
MODELS, FT_MODELS, METRICS = base.MODELS, adaptation.MODELS, base.METRICS
PAIRS = {(m, f) for m in MODELS for f in range(5)}
TRIPLES = {(m, f, s) for m in FT_MODELS for f in range(5) for s in range(5)}
JSON_FILES = (
    "provenance.json",
    "membership.json.gz",
    "results/adaptation.json",
    "results/adaptation_history.json.gz",
    "results/buffer_sgd.json",
    "results/buffer_lr.json",
    "results/predictions.json.gz",
    "evidence/acceptance.json",
    "evidence/runs.json",
    "evidence/boundary.json",
    *(
        f"protocols/{n}.json"
        for n in ("adaptation", "checkpoint_replay", "buffer_sgd", "buffer_lr")
    ),
)
HELPERS = {
    "scripts/export_repaired_release.py": (
        "d7dce28f65f39b08ee7bf4f355eab6d10a7a44c7edba469ffbbd6d7c34d2c349"
    ),
    "scripts/export_repaired_auxiliary_results.py": (
        "047ba533d0b2399daa1e6dd048104d19c1684a71d1fece96630e747dd103f93c"
    ),
    "scripts/analyze_repaired_adaptation.py": (
        "8b7b842e4829eb54b45bc0b741891bfbde603391ed52602a6060b4992548187b"
    ),
    "scripts/plot_repaired_adaptation.py": (
        "144859703ef0aca609c083764261f238e96e81824b847bc689c1fe8590306b50"
    ),
    "scripts/plot_repaired_training_substitution.py": (
        "79367e5fc02ae7bdcb9e607938babe7018bfabfa99d30fca3e76f9880871adca"
    ),
    "scripts/plot_repaired_common_classifier.py": (
        "479d9fccbf2bca7a3c3c2bc3ee6dd737f548af2225b518d1f773320ebdcc65b5"
    ),
    "scripts/summarize_repaired_titan.py": (
        "a166437fc65b86dec8fa49ae24f5ad46bf72b645b4103c4f6646f0ac9f421e72"
    ),
    "scripts/summarize_review_frozen.py": (
        "362730fe9b55b2359ee55fe90575cf58b9ba4ebb78f20d3c34bac9df2732f1cd"
    ),
}

PUBLIC_HELPERS = {
    **HELPERS,
    "scripts/export_repaired_release.py": (
        "146567c8b9767fbb0d7148986bcb2384c834a6a17b0abe86448434c1566bce50"
    ),
}

CONTROL_COMMIT = "54281f05cfd67453f9b699810021b1e2da20c3ec"
RECOVERY_SCHEMA = "repaired-followup-recovery/1"
RECOVERY_COUNTS = dict(
    finetuning=75, boundary=20, buffer=20, replay=75, buffer_heads=500
)


def public_helper_bytes(name, raw):
    if name == "scripts/export_repaired_release.py":
        raw = re.sub(rb"\"gpu-[a-z]+\", \"gpu-[a-z]+\",", b'"gpu-",', raw)
    require(
        hashlib.sha256(raw).hexdigest()
        == PUBLIC_HELPERS.get(name, hashlib.sha256(raw).hexdigest()),
        "Public helper projection differs",
    )
    return raw


def exact(a, b, message):
    require(digest(a) == digest(b), message)


def hash_value(value, length=64):
    require(
        isinstance(value, str) and re.fullmatch("[0-9a-f]{" + str(length) + "}", value),
        "Invalid hash identity",
    )
    return value


def grid(rows, fields, expected):
    return adaptation.index(rows, fields, expected)


def numeric_tree(actual, expected, label):
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and set(actual) == set(expected), label + " fields"
        )
        for k in expected:
            numeric_tree(actual[k], expected[k], label + "/" + k)
    elif isinstance(expected, (int, float, list)) and not isinstance(expected, bool):
        close(actual, expected, label)
    else:
        exact(actual, expected, label)


def has_absolute_unix_token(value):
    """Recognize plain Unix tokens without inferring relative or encoded paths."""
    # Leave scheme URLs and numeric fractions to their scientific fields.
    text = re.sub(r"""\b[A-Za-z][A-Za-z0-9+.-]*://[^\s<>"'`]+""", " ", value)
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    text = re.sub(
        r"(?<![\w./])" + number + r"\s*/\s*" + number + r"(?=$|\s|[,;)\]}])",
        " ",
        text,
    )
    # Exclude characters continuing a relative path or slash-separated notation.
    # Any other separator can introduce a root; its first character need not be a letter.
    return re.search(r"(?<![\w./~%+-])/\S", text) is not None


def privacy(value):
    base.check_public(value)
    # Explicit projections remain the primary defense. Reject operational keys
    # anywhere, including unrecognized nested extras supplied by a corrupt export.
    if isinstance(value, dict):
        for k, v in value.items():
            require(
                k
                not in {
                    "path",
                    "root",
                    "repository",
                    "hostname",
                    "gpu_uuid",
                    "pid",
                    "command",
                    "raw_tiles_root",
                    "buffer_fit_root",
                    "filepath",
                    "resolved_path",
                },
                "Operational field: " + k,
            )
            privacy(v)
    elif isinstance(value, list):
        for v in value:
            privacy(v)
    elif isinstance(value, str):
        require(
            not has_absolute_unix_token(value)
            and not re.search(
                r"""(?:^|[\s=(\[{:;,'"])(?:[A-Za-z]:\\|[\w.-]+@[\w.-]+:)""",
                value,
            ),
            "Operational absolute location",
        )


def score(predicted, labels, saved=None):
    require(
        isinstance(predicted, list)
        and len(predicted) == len(labels)
        and all(type(x) is int and 0 <= x < 6 for x in predicted),
        "Prediction vector or class differs",
    )
    truth = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(6 * truth + predicted, minlength=36).reshape(6, 6)
    result = {"confusion_counts": counts.tolist(), **base.metrics(counts)}
    if saved is not None:
        exact(
            saved["confusion_counts"],
            result["confusion_counts"],
            "Confusion counts differ",
        )
        for k in ("macro", "per_class"):
            numeric_tree(saved[k], result[k], "Metrics " + k)
        for k in ("support", "prediction_counts", "kappa"):
            if k in saved:
                close(saved[k], result[k], "Metrics " + k)
    return result


def average(rows):
    return {
        level: {
            m: np.mean([r[level][m] for r in rows], axis=0).tolist() for m in METRICS
        }
        for level in ("macro", "per_class")
    }


def summaries(folds):
    return {
        level: {
            m: adaptation.joint.describe([f[level][m] for f in folds]) for m in METRICS
        }
        for level in ("macro", "per_class")
    }


def protocols(data):
    out = {}
    for family in ("adaptation", "checkpoint_replay", "buffer_sgd", "buffer_lr"):
        p = data[f"protocols/{family}.json"]
        require(
            set(p)
            == {
                "original_protocol_sha256",
                "original_specification_sha256",
                "specification",
                "extract_specification_sha256",
            },
            "Protocol projection fields differ",
        )
        hash_value(p["original_protocol_sha256"])
        hash_value(p["original_specification_sha256"])
        exact(
            digest(p["specification"]),
            p["extract_specification_sha256"],
            "Projected protocol digest differs",
        )
        out[family] = p["specification"]
    return out


def parent_data(data, frozen, auxiliary_root):
    provenance = data["provenance.json"]
    for name, root in (("frozen", frozen), ("auxiliary", auxiliary_root)):
        base.verify_indexes(root)
        exact(
            base.inventory(root),
            provenance["parents"][name],
            "Pinned parent inventory differs: " + name,
        )
    catalog = base.read_json(frozen / "catalogs/titan_catalog.json")
    require(catalog["class_names"] == list(base.CLASSES), "Catalog class order differs")
    rows = catalog["tiles"]
    ids = [r["tile_id"] for r in rows]
    require(
        len(ids) == len(set(ids))
        and all(type(r["label"]) is int and 0 <= r["label"] < 6 for r in rows),
        "Catalog IDs/classes",
    )
    entries = {r["tile_id"]: r for r in rows}
    splits = [
        base.read_json(frozen / f"splits/titan_contiguous_fold_{f}.json")
        for f in range(5)
    ]
    fp = base.read_json(frozen / "results/predictions.json.gz")
    ap = base.read_json(auxiliary_root / "results/predictions.json.gz")
    native = base.read_json(frozen / "results/frozen_results.json")
    lr = base.read_json(auxiliary_root / "results/common_classifier.json")
    cp = base.read_json(auxiliary_root / "protocols/common_classifier.json")[
        "specification"
    ]
    fj = grid(fp["jobs"], ("model", "fold"), PAIRS)
    lrj = grid(lr["folds"], ("model", "fold"), PAIRS)
    require([r["fold"] for r in ap["folds"]] == list(range(5)), "Parent LR folds")
    return entries, ids, splits, fj, ap["folds"], native, lrj, cp


def membership(data, parents):
    entries, catalog_ids, splits, frozen, lr, native, _, cp = parents
    rows = data["membership.json.gz"]["folds"]
    require([r["fold"] for r in rows] == list(range(5)), "Membership folds differ")
    union = set()
    for f, row in enumerate(rows):
        split = splits[f]
        assignments = split["assignments"]
        require(set(assignments) == set(catalog_ids), "Assignment inventory differs")
        roles = {
            r: [t for t in catalog_ids if assignments[t] == r]
            for r in ("train", "val", "test")
        }
        for role, key in [
            ("train", "retained_train_ids"),
            ("val", "validation_ids"),
            ("test", "test_ids"),
        ]:
            exact(row[key], roles[role], "Membership role/order differs")
        require(not union.intersection(row["test_ids"]), "Test folds overlap")
        union.update(row["test_ids"])
        for role in ("validation", "test"):
            exact(
                row[role + "_labels"],
                [entries[t]["label"] for t in row[role + "_ids"]],
                "Truth/order differs",
            )
        norm = native["folds"][f]["normalization_provenance"]
        exact(
            row["normalization_provenance"], norm, "Fixed image normalization differs"
        )
        exact(
            cp["folds"][f]["normalization_provenance"], norm, "LR normalization differs"
        )
        anchors = norm["sampling"]["selected_tile_ids"]
        require(
            len(anchors) == len(set(anchors)) and set(anchors) <= set(roles["train"]),
            "Normalization anchors",
        )
        exact(
            norm["training_ids_sha256"],
            digest(roles["train"]),
            "Normalization training IDs",
        )
        boundary = sorted(
            t
            for t, v in split["policy"]["exclusion_reasons"].items()
            if v == "train_buffer_excluded"
        )
        exact(row["boundary_ids"], boundary, "Boundary membership differs")
        eligible = sorted(set(roles["train"]) - set(anchors))
        exact(row["eligible_removal_ids"], eligible, "Removal eligibility differs")
        counts = np.bincount(
            [entries[t]["label"] for t in roles["train"]], minlength=6
        ).tolist()
        bc = np.bincount([entries[t]["label"] for t in boundary], minlength=6).tolist()
        exact(row["class_counts"], counts, "Train support differs")
        exact(row["boundary_class_counts"], bc, "Boundary support differs")
        require(
            [d["draw_seed"] for d in row["draws"]] == list(range(5)),
            "Replacement draw membership",
        )
        for d in row["draws"]:
            rng = np.random.default_rng(d["draw_seed"])
            removed = []
            for c, n in enumerate(bc):
                pool = [t for t in eligible if entries[t]["label"] == c]
                require(len(pool) >= n, "Insufficient eligible removal support")
                if n:
                    removed.extend(
                        np.asarray(pool)[
                            rng.choice(len(pool), n, replace=False)
                        ].tolist()
                    )
            exact(d["removed_ids"], sorted(removed), "Sealed draw differs")
            chosen = (set(roles["train"]) - set(removed)) | set(boundary)
            ordered = [t for t in catalog_ids if t in chosen]
            exact(d["training_ids"], ordered, "Replacement training order differs")
            exact(
                d["training_ids_sha256"],
                digest(ordered),
                "Replacement ID digest differs",
            )
            exact(
                d["train_class_counts"],
                np.bincount(
                    [entries[t]["label"] for t in ordered], minlength=6
                ).tolist(),
                "Replacement counts",
            )
            exact(d["train_class_counts"], counts, "Replacement class support changed")
        for m in MODELS:
            j = frozen[m, f]
            exact(j["tile_ids"], row["test_ids"], "Frozen test order")
            exact(j["true_labels"], row["test_labels"], "Frozen truth")
            exact(j["seeds"], list(range(5)), "Frozen head seeds")
            require(len(j["probe_predictions"]) == 5, "Frozen reference heads")
        exact(lr[f]["test_ids"], row["test_ids"], "LR test order")
        exact(lr[f]["true_labels"], row["test_labels"], "LR truth")
    return rows, len(union)


def history_check(record, training, validation, n):
    h = record["history"]
    names = (
        "train_loss",
        "val_balanced_accuracy",
        "optimizer_steps",
        "amp_skipped_steps",
        "loss_scale",
    )
    require(set(h) == set(names), "History fields differ")
    for name in names:
        a = np.asarray(h[name])
        require(a.shape == (50,) and np.isfinite(a).all(), "History shape/nonfinite")
    steps = np.asarray(h["optimizer_steps"])
    skips = np.asarray(h["amp_skipped_steps"])
    recall = np.asarray(h["val_balanced_accuracy"])
    attempts = (n + 31) // 32
    require(
        steps.dtype.kind in "iu"
        and skips.dtype.kind in "iu"
        and (steps > 0).all()
        and (skips >= 0).all()
        and np.array_equal(steps + skips, np.full(50, attempts))
        and (np.asarray(h["loss_scale"]) > 0).all()
        and ((recall >= 0) & (recall <= 1)).all(),
        "History update or validation values differ",
    )
    selected = int(np.argmax(recall)) + 1
    expected = dict(
        selected_epoch=selected,
        epoch_indexing="one based",
        optimizer_attempts_per_epoch=attempts,
        optimizer_attempts_total=50 * attempts,
        optimizer_steps_total=int(steps.sum()),
        amp_skipped_steps_total=int(skips.sum()),
        training_example_presentations=50 * n,
    )
    exact(training, expected, "History selection/budget differs")
    close(
        validation["macro"]["recall"],
        recall[selected - 1],
        "Selected validation score differs",
    )


def analyze_public(data, parents, members):
    p = data["results/predictions.json.gz"]
    require(p["schema_version"] == "1.0.0", "Prediction schema")
    tests = {}
    for family, models, fields, expected in (
        ("adaptation", FT_MODELS, ("model", "fold", "seed"), TRIPLES),
        (
            "buffer_sgd",
            MODELS,
            ("model", "fold", "draw_seed", "seed"),
            {
                (m, f, d, s)
                for m in MODELS
                for f in range(5)
                for d in range(5)
                for s in range(5)
            },
        ),
        (
            "buffer_lr",
            MODELS,
            ("model", "fold", "draw_seed"),
            {(m, f, d) for m in MODELS for f in range(5) for d in range(5)},
        ),
    ):
        tests[family] = grid(
            [r for r in p["test"] if r["family"] == family], fields, expected
        )
    require(len(p["test"]) == 675, "Unexpected test family")
    validation = grid(p["validation"], ("model", "fold", "seed"), TRIPLES)
    histories = grid(
        data["results/adaptation_history.json.gz"]["records"],
        ("model", "fold", "seed"),
        TRIPLES,
    )
    replays = grid(
        data["evidence/acceptance.json"]["replays"], ("model", "fold", "seed"), TRIPLES
    )
    a = copy.deepcopy(data["results/adaptation.json"])
    b = data["results/buffer_sgd.json"]
    l = data["results/buffer_lr.json"]
    require(
        a["status"] == b["status"] == l["status"] == "complete", "Incomplete result"
    )
    exact(
        a["validation"],
        dict(ft_jobs=75, checkpoint_replays=75, common_lr_jobs=20),
        "Adaptation coverage",
    )
    exact(l["validation"], dict(complete_jobs=20, estimator_replays=100), "LR coverage")
    require(
        set(a["models"]) == set(FT_MODELS)
        and set(b["models"]) == set(l["models"]) == set(MODELS),
        "Result model family",
    )
    sgd = grid(b["folds"], ("model", "fold"), PAIRS)
    lr = grid(l["folds"], ("model", "fold"), PAIRS)
    _, _, _, frozen, lp, _, lr_parent, common_protocol = parents
    replacements = {"lr": {}, "sgd": {}}
    diagnostics = []
    for m in MODELS:
        buf_fold = {"sgd": [], "lr": []}
        afolds = []
        if m in FT_MODELS:
            display_b.ordered_five(a["models"][m]["folds"], "fold")
        display_b.ordered_five(b["models"][m]["folds"], "fold")
        for f, mem in enumerate(members):
            truth = mem["test_labels"]
            refs = [score(v, truth) for v in frozen[m, f]["probe_predictions"]]
            ref = score(lp[f]["common"][m], truth, lr_parent[m, f])
            srow = sgd[m, f]
            lrow = lr[m, f]
            score(lp[f]["common"][m], truth, lrow["reference"])
            require(len(srow["reference_metrics"]) == 5, "SGD reference metrics")
            for saved, v in zip(
                srow["reference_metrics"], frozen[m, f]["probe_predictions"]
            ):
                score(v, truth, saved)
            sr = grid(
                srow["runs"],
                ("draw_seed", "seed"),
                {(d, s) for d in range(5) for s in range(5)},
            )
            ld = grid(lrow["draws"], ("draw_seed",), {(d,) for d in range(5)})
            sdmeans = []
            ldmeans = []
            for d in range(5):
                srmetrics = []
                for s in range(5):
                    row = tests["buffer_sgd"][m, f, d, s]
                    v = score(row["predictions"], truth, sr[d, s]["metrics"])
                    srmetrics.append(v)
                    numeric_tree(
                        sr[d, s]["difference"],
                        {k: v["macro"][k] - refs[s]["macro"][k] for k in METRICS},
                        "SGD paired difference",
                    )
                sdmeans.append(average(srmetrics)["macro"])
                row = ld[d,]
                v = score(
                    tests["buffer_lr"][m, f, d]["predictions"], truth, row["metrics"]
                )
                ldmeans.append(v["macro"])
                numeric_tree(
                    row["difference"],
                    {
                        **{k: v["macro"][k] - ref["macro"][k] for k in METRICS},
                        "kappa": v["kappa"] - ref["kappa"],
                    },
                    "LR paired difference",
                )
                exact(
                    row["training_ids_sha256"],
                    mem["draws"][d]["training_ids_sha256"],
                    "LR draw identity",
                )
                exact(row["train_class_counts"], mem["class_counts"], "LR train counts")
                diagnostics.append(
                    dict(
                        model=m,
                        fold=f,
                        draw_seed=d,
                        **display_b.fit_diagnostics(
                            row, common_protocol["resolved_parameters"]
                        ),
                    )
                )
            for recipe, reference, draws in [
                ("sgd", average(refs)["macro"], sdmeans),
                ("lr", ref["macro"], ldmeans),
            ]:
                buf_fold[recipe].append(
                    dict(
                        fold=f,
                        draw_means=draws,
                        reference=reference,
                        replacement={
                            k: float(np.mean([v[k] for v in draws])) for k in METRICS
                        },
                    )
                )
            exact(
                srow["recipe"],
                data["protocols/buffer_sgd.json"]["specification"]["recipe"],
                "SGD recipe",
            )
            exact(
                data["protocols/buffer_lr.json"]["specification"][
                    "resolved_parameters"
                ],
                common_protocol["resolved_parameters"],
                "LR parent parameters",
            )
            exact(srow["class_counts"], mem["class_counts"], "SGD train counts")
            weights = 1 / np.asarray(mem["class_counts"], float)
            weights = (weights / weights.sum() * 6).astype(np.float32).tolist()
            close(srow["training_class_weights"], weights, "SGD class weights")
            exact(
                srow["optimization"],
                dict(
                    planned_steps_per_head=100
                    * ((len(mem["retained_train_ids"]) + 255) // 256),
                    planned_example_presentations_per_head=100
                    * len(mem["retained_train_ids"]),
                ),
                "SGD planned budget",
            )
            if m in FT_MODELS:
                saved = a["models"][m]["folds"][f]
                require(
                    [r["seed"] for r in saved["seeds"]] == list(range(5)),
                    "Adaptation seeds",
                )
                seedrows = []
                for s in range(5):
                    r = saved["seeds"][s]
                    k = (m, f, s)
                    ft = score(
                        tests["adaptation"][k]["predictions"], truth, r["finetuned"]
                    )
                    score(frozen[m, f]["probe_predictions"][s], truth, r["frozen_sgd"])
                    val = score(
                        validation[k]["predictions"],
                        mem["validation_labels"],
                        r["validation"],
                    )
                    history_check(
                        histories[k], r["training"], val, len(mem["retained_train_ids"])
                    )
                    score(
                        tests["adaptation"][k]["predictions"],
                        truth,
                        replays[k]["test_metrics"],
                    )
                    score(
                        validation[k]["predictions"],
                        mem["validation_labels"],
                        replays[k]["validation_metrics"],
                    )
                    close(
                        replays[k]["comparisons"]["validation"][
                            "expected_macro_recall"
                        ],
                        histories[k]["history"]["val_balanced_accuracy"][
                            r["training"]["selected_epoch"] - 1
                        ],
                        "Replay expected validation recall differs from selected history",
                    )
                    delta = adaptation.difference(ft, refs[s])
                    numeric_tree(r["ft_minus_sgd"], delta, "FT/SGD difference")
                    seedrows.append(
                        dict(finetuned=ft, frozen_sgd=refs[s], ft_minus_sgd=delta)
                    )
                scores = {
                    name: adaptation.score_summary([r[name] for r in seedrows])
                    for name in ("finetuned", "frozen_sgd", "ft_minus_sgd")
                }
                numeric_tree(saved["seed_summaries"], scores, "Seed summaries")
                values = {name: average([r[name] for r in seedrows]) for name in scores}
                values["common_lr"] = {k: ref[k] for k in ("macro", "per_class")}
                values["ft_minus_lr"] = adaptation.difference(values["finetuned"], ref)
                numeric_tree(saved["values"], values, "Adaptation fold values")
                score(lp[f]["common"][m], truth, saved["common_lr"]["metrics"])
                exact(
                    saved["common_lr"]["fit"],
                    lr_parent[m, f]["fit"],
                    "Reference LR diagnostics",
                )
                expected_counts = {
                    role: dict(
                        total=len(mem[key]),
                        class_counts=np.bincount(
                            [parents[0][t]["label"] for t in mem[key]], minlength=6
                        ).tolist(),
                    )
                    for role, key in [
                        ("train", "retained_train_ids"),
                        ("val", "validation_ids"),
                        ("test", "test_ids"),
                    ]
                }
                exact(saved["counts"], expected_counts, "Adaptation role supports")
                afolds.append(values)
        if m in FT_MODELS:
            expected = {
                name: summaries([r[name] for r in afolds]) for name in adaptation.GROUPS
            }
            numeric_tree(a["models"][m]["summary"], expected, "Adaptation aggregate")
        for f, saved in enumerate(b["models"][m]["folds"]):
            exact(saved["fold"], f, "SGD fold order")
            for key in ("draw_means", "reference", "replacement"):
                if key == "draw_means":
                    require(len(saved[key]) == 5, "SGD draw summary count")
                    for x, y in zip(saved[key], buf_fold["sgd"][f][key]):
                        numeric_tree(x, y, "SGD draw summary")
                else:
                    numeric_tree(
                        saved[key], buf_fold["sgd"][f][key], "SGD fold summary"
                    )
        for recipe in ("lr", "sgd"):
            result = {}
            for metric in METRICS:
                refs = [r["reference"][metric] for r in buf_fold[recipe]]
                repl = [r["replacement"][metric] for r in buf_fold[recipe]]
                saved = (
                    l["models"][m][metric]
                    if recipe == "lr"
                    else {
                        k: b["models"][m][k][metric]
                        for k in ("reference", "replacement", "difference")
                    }
                )
                result[metric] = display_b.summarize_metric(refs, repl, saved)
            replacements[recipe][m] = result
        # LR kappa is a native summary too, and must not escape verification.
        refk = [lr[m, f]["reference"]["kappa"] for f in range(5)]
        replk = [
            float(np.mean([d["metrics"]["kappa"] for d in lr[m, f]["draws"]]))
            for f in range(5)
        ]
        numeric_tree(
            l["models"][m]["kappa"],
            {
                k: adaptation.joint.describe(v)
                for k, v in [
                    ("reference", refk),
                    ("replacement", replk),
                    ("difference", np.asarray(replk) - refk),
                ]
            },
            "LR kappa summary",
        )
    support = []
    for f, mem in enumerate(members):
        support.append(
            dict(
                fold=f,
                train_class_counts=mem["class_counts"],
                test_class_counts=np.bincount(mem["test_labels"], minlength=6).tolist(),
                boundary_class_counts=mem["boundary_class_counts"],
                train_total=len(mem["retained_train_ids"]),
                test_total=len(mem["test_ids"]),
                boundary_total=len(mem["boundary_ids"]),
                validation_total=len(mem["validation_ids"]),
                split_sha256=mem["split_sha256"],
                test_ids_sha256=digest(mem["test_ids"]),
                normalization_sha256=digest(mem["normalization_provenance"]),
                replacement_training_ids_sha256=[
                    d["training_ids_sha256"] for d in mem["draws"]
                ],
            )
        )
    counts = dict(
        fits=100,
        fits_with_warnings=sum(bool(d["fit"]["fit_warnings"]) for d in diagnostics),
        convergence_warnings=sum(d["fit"]["convergence_warning"] for d in diagnostics),
        iteration_caps=sum(d["fit"]["iteration_limit_reached"] for d in diagnostics),
    )
    bdisplay = dict(
        schema_version="1.0.0",
        status="complete",
        recipes=replacements,
        caption=display_b.CAPTION,
        support_design=support,
        lr_fit_diagnostics=diagnostics,
        lr_diagnostic_counts=counts,
        units=dict(
            reference="percent",
            replacement="percent",
            difference="percentage points",
            sample_std="percentage points",
        ),
    )
    return display_a.display_values(a), bdisplay


def check_replay_environment(environment, specification):
    exact(specification["inference"]["cpu_intraop_threads"], 4, "Replay thread seal")
    expected = dict(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    exact(
        specification["execution"]["thread_environment"],
        expected,
        "Replay environment seal",
    )
    exact(environment["cpu_intraop_threads"], 4, "Actual replay CPU threads")
    interop = environment["cpu_interop_threads"]
    require(type(interop) is int and interop > 0, "Recorded replay inter-op threads")
    exact(
        {key: environment["declared_environment"][key] for key in expected},
        expected,
        "Actual replay thread environment",
    )


def evidence(data, specs, members):
    a = data["evidence/acceptance.json"]
    prov = data["provenance.json"]
    runs = data["evidence/runs.json"]
    require(
        a["schema_version"] == "1.0.0" and a["status"] == "complete",
        "Acceptance status/schema",
    )
    gate = a["full_gate"]
    exact(
        [gate["schema_version"], gate["status"]],
        [RECOVERY_SCHEMA, "pass"],
        "Recovery gate schema/status",
    )
    exact(gate["counts"], RECOVERY_COUNTS, "Recovery gate counts")
    exact(gate["sha256"], a["inputs"]["full_gate"], "Recovery gate input")
    for key in ("sha256", "inputs_sha256", "checker_sha256"):
        hash_value(gate[key])
    hash_value(gate["checker_commit"], 40)
    exact(
        gate["reports"],
        {
            family: a["inputs"][name]
            for family, name in (
                ("finetuning", "ft_audit"),
                ("buffer", "buffer_sgd_audit"),
                ("replay", "replay_audit"),
            )
        },
        "Recovery gate report joins",
    )
    require(
        set(a["processes"])
        == {
            "full_gate",
            "adaptation",
            "checkpoint_replay",
            "buffer_sgd",
            "buffer_lr",
            "analysis",
            "display_a",
            "display_b",
        },
        "Process receipt coverage",
    )
    for process in a["processes"].values():
        require(
            type(process["exit_code"]) is int and process["exit_code"] == 0,
            "Unsuccessful accepted process",
        )
        hash_value(process["receipt_sha256"])
    exact(a["inputs"], prov["input_sha256"], "Acceptance input identities")
    require(
        set(a["reports"])
        == {"adaptation", "checkpoint_replay", "buffer_sgd", "buffer_lr"},
        "Report families",
    )
    commits = prov["producer_commits"]
    require(
        set(commits)
        == set(runs)
        == {"adaptation", "checkpoint_replay", "boundary", "buffer_sgd", "buffer_lr"},
        "Run families",
    )
    for c in commits.values():
        hash_value(c, 40)
    exact(commits["checkpoint_replay"], adaptation.REPLAY_COMMIT, "Replay execution")
    exact(commits["boundary"], CONTROL_COMMIT, "Original boundary execution")
    exact(commits["buffer_sgd"], CONTROL_COMMIT, "Original SGD execution")
    report_a = a["reports"]["adaptation"]
    report_r = a["reports"]["checkpoint_replay"]
    report_s = a["reports"]["buffer_sgd"]
    report_l = a["reports"]["buffer_lr"]
    exact(
        [report_a["schema_version"], report_a["status"], report_a["validated_jobs"]],
        ["1.0.0", "pass", 75],
        "FT acceptance",
    )
    exact(
        [report_r["schema_version"], report_r["status"], report_r["verified_replays"]],
        ["1.0.0", "pass", 75],
        "Checkpoint acceptance",
    )
    exact(report_s["status"], "complete", "SGD acceptance")
    exact(
        [report_l["status"], report_l["schema_version"], report_l["validation"]],
        ["complete", "1.0.0", dict(complete_jobs=20, estimator_replays=100)],
        "LR acceptance",
    )
    for fam, r, key in [
        ("adaptation", report_a, "protocol_sha256"),
        ("checkpoint_replay", report_r, "replay_protocol_sha256"),
        ("buffer_sgd", report_s, "protocol_sha256"),
        ("buffer_lr", report_l, "protocol_sha256"),
    ]:
        exact(
            r[key],
            data[f"protocols/{fam}.json"]["original_protocol_sha256"],
            "Acceptance/protocol binding",
        )
        if fam != "checkpoint_replay":
            exact(
                r["specification_sha256"],
                data[f"protocols/{fam}.json"]["original_specification_sha256"],
                "Acceptance/specification binding",
            )
    exact(report_a["source_commit"], commits["adaptation"], "FT source")
    exact(
        report_r["plan_sha256"], a["inputs"]["replay_plan"], "Replay plan binding"
    )
    exact(
        report_r["parent_plan_sha256"], a["inputs"]["ft_plan"], "FT parent plan"
    )
    exact(
        specs["checkpoint_replay"]["producer_plan_sha256"],
        a["inputs"]["ft_plan"],
        "Replay protocol producer plan",
    )
    plan_hashes = [
        a["inputs"][key] for key in ("ft_plan", "followup_plan", "replay_plan")
    ]
    require(len(set(plan_hashes)) == 3, "Distinct FT, control and replay plans")
    for value in plan_hashes:
        hash_value(value)
    exact(
        report_r["replay_execution_commit"],
        commits["checkpoint_replay"],
        "Replay source",
    )
    exact(report_l["producer_commit"], commits["buffer_lr"], "LR source")
    exact(commits["boundary"], commits["buffer_sgd"], "Boundary/SGD source")
    for fam, field in [
        ("adaptation", "scientific_source_sha256"),
        ("checkpoint_replay", "replay_source_sha256"),
        ("buffer_sgd", "source_sha256"),
        ("buffer_lr", "scientific_source_sha256"),
    ]:
        exact(
            prov["scientific_source_sha256"][fam],
            specs[fam][field],
            "Scientific source map differs",
        )
        for h in specs[fam][field].values():
            hash_value(h)
    for fam in ("adaptation", "buffer_sgd"):
        exact(
            specs[fam]["catalog_sha256"],
            prov["native_catalog_sha256"],
            "Native catalog binding",
        )
    exact(
        specs["buffer_lr"]["catalog"]["sha256"],
        prov["native_catalog_sha256"],
        "LR native catalog",
    )
    for fam in ("adaptation", "buffer_sgd"):
        display_b.ordered_five(specs[fam]["folds"], "fold")
        for f, record in enumerate(specs[fam]["folds"]):
            exact(
                record["split_sha256"],
                members[f]["split_sha256"],
                "Native split binding",
            )
    for f, record in enumerate(specs["buffer_sgd"]["folds"]):
        for key in (
            "retained_train_ids",
            "validation_ids",
            "test_ids",
            "boundary_ids",
            "eligible_removal_ids",
            "overlap_ids",
            "class_counts",
            "boundary_class_counts",
            "normalization_provenance",
        ):
            exact(record[key], members[f][key], "SGD protocol membership " + key)
        display_b.ordered_five(record["draws"], "draw_seed")
        for d, draw in enumerate(record["draws"]):
            exact(
                draw,
                {
                    k: members[f]["draws"][d][k]
                    for k in ("draw_seed", "removed_ids", "training_ids_sha256")
                },
                "SGD sealed draws",
            )
    exact(specs["buffer_lr"]["draws"], list(range(5)), "LR sealed draw order")
    require(
        type(specs["buffer_lr"]["fits"]) is int and specs["buffer_lr"]["fits"] == 100,
        "LR sealed fit count",
    )
    exact(
        specs["buffer_lr"]["buffer_producer_commit"],
        commits["buffer_sgd"],
        "LR boundary producer",
    )
    exact(specs["buffer_lr"]["parents"], report_l["parents"], "LR parent projection")
    exact(
        report_l["auditor_sha256"],
        specs["buffer_lr"]["scientific_source_sha256"][
            "scripts/run_repaired_common_buffer.py"
        ],
        "LR sealed checker source",
    )
    for k, expected in [
        ("buffer_protocol", report_s["protocol_sha256"]),
        ("buffer_audit", a["inputs"]["buffer_sgd_audit"]),
        ("finetuning_audit", a["inputs"]["ft_audit"]),
    ]:
        exact(report_l["parents"][k]["sha256"], expected, "LR accepted parent " + k)
    run_index = {}
    for fam, rows in runs.items():
        fields = (
            ("model", "fold", "seed")
            if fam in ("adaptation", "checkpoint_replay")
            else ("model", "fold")
        )
        indexed = grid(rows, fields, TRIPLES if len(fields) == 3 else PAIRS)
        run_index[fam] = indexed
        for key, row in indexed.items():
            require(
                row["status"] == "complete" and row["producer_commit"] == commits[fam],
                "Run status/source",
            )
            hash_value(row["manifest_sha256"])
            auxiliary.safe_names(row["outputs_sha256"])
            for h in row["outputs_sha256"].values():
                hash_value(h)
            native_family = "buffer_sgd" if fam == "boundary" else fam
            required_key = "repaired_protocol" if fam == "adaptation" else "protocol"
            # Native replay inputs use replay_protocol, unlike the buffer runner.
            if fam == "checkpoint_replay":
                required_key = "replay_protocol"
            exact(
                row["inputs_sha256"][required_key],
                data[f"protocols/{native_family}.json"]["original_protocol_sha256"],
                "Run protocol input",
            )
            if fam == "checkpoint_replay":
                check_replay_environment(row["environment"], specs[fam])
    histories = grid(
        data["results/adaptation_history.json.gz"]["records"],
        ("model", "fold", "seed"),
        TRIPLES,
    )
    replay = grid(a["replays"], ("model", "fold", "seed"), TRIPLES)
    for row in (
        data["results/predictions.json.gz"]["test"]
        + data["results/predictions.json.gz"]["validation"]
    ):
        fam = row.get("family", "checkpoint_replay")
        fields = (
            ("model", "fold", "seed")
            if fam in ("adaptation", "checkpoint_replay")
            else ("model", "fold")
        )
        key = tuple(row[k] for k in fields)
        r = run_index[fam][key]
        name = {
            "adaptation": "finetuning_results.json",
            "checkpoint_replay": "replay.json",
            "buffer_lr": "common_buffer_results.json",
        }.get(fam)
        if fam == "buffer_sgd":
            name = f"draw-{row['draw_seed']}/seed-{row['seed']}.json"
        exact(
            row["source_result_sha256"],
            r["outputs_sha256"][name],
            "Prediction source hash",
        )
        if fam == "adaptation":
            exact(
                histories[key]["source_result_sha256"],
                row["source_result_sha256"],
                "History/result source",
            )
            saved = data["results/adaptation.json"]["models"][key[0]]["folds"][key[1]][
                "seeds"
            ][key[2]]
            exact(
                saved["manifest_sha256"], r["manifest_sha256"], "FT analysis manifest"
            )
            exact(saved["outputs"], r["outputs_sha256"], "FT output hashes")
    for key, r in replay.items():
        exact(
            r["producer_manifest_sha256"],
            run_index["adaptation"][key]["manifest_sha256"],
            "Replay/producer manifest",
        )
        exact(
            r["replay_manifest_sha256"],
            run_index["checkpoint_replay"][key]["manifest_sha256"],
            "Replay manifest",
        )
        exact(
            r["replay_sha256"],
            run_index["checkpoint_replay"][key]["outputs_sha256"]["replay.json"],
            "Replay result",
        )
        saved = data["results/adaptation.json"]["models"][key[0]]["folds"][key[1]][
            "seeds"
        ][key[2]]
        exact(
            saved["replay_manifest_sha256"],
            r["replay_manifest_sha256"],
            "Adaptation seed replay manifest",
        )
        exact(
            saved["replay_sha256"], r["replay_sha256"], "Adaptation seed replay result"
        )
        exact(
            r["selected_epoch"],
            saved["training"]["selected_epoch"],
            "Replay selected epoch",
        )
        c = r["comparisons"]
        v, t = c["validation"], c["test"]
        require(
            set(c) == {"validation", "test"}
            and v["passed"] is True
            and t["passed"] is True
            and v["historical_vector_available"] is False
            and 0 <= v["macro_recall_abs_error"] <= 1e-12
            and t["confusion_exact"] is True
            and t["prediction_mismatches"] == 0
            and t["mismatched_tile_ids"] == []
            and t["probability_components_over_atol"] == 0
            and 0 <= t["probability_max_abs_error"] <= 1e-6,
            "Checkpoint replay acceptance",
        )
    for fam in ("buffer_sgd", "buffer_lr"):
        for row in data[f"results/{fam}.json"]["folds"]:
            exact(
                row["manifest_sha256"],
                run_index[fam][row["model"], row["fold"]]["manifest_sha256"],
                "Buffer report/run binding",
            )
    boundary = grid(data["evidence/boundary.json"]["records"], ("model", "fold"), PAIRS)
    for key, row in boundary.items():
        exact(
            row["manifest_sha256"],
            run_index["boundary"][key]["manifest_sha256"],
            "Boundary manifest",
        )
        expected_norm = members[key[1]]["normalization_provenance"]
        exact(
            sorted(row["normalization"]),
            sorted(expected_norm),
            "Boundary normalization schema",
        )
        for k, value in expected_norm.items():
            if k in ("lower", "upper"):
                close(row["normalization"][k], value, "Boundary normalization bounds")
            else:
                exact(row["normalization"][k], value, "Boundary normalization field")
        o = row["overlap"]
        require(
            o["status"] == "pass" and o["atol"] == 1e-6 and o["rtol"] == 1e-5,
            "Boundary overlap acceptance",
        )
        exact(o["ids"], members[key[1]]["overlap_ids"], "Boundary overlap IDs")
        exact(o["ids_sha256"], digest(o["ids"]), "Boundary overlap digest")
        require(o["rows"] == len(o["ids"]), "Boundary overlap count")
    return run_index


def validate_payloads(data, frozen, auxiliary_root):
    require(set(data) == set(JSON_FILES), "Unexpected/missing scientific payload")
    for v in data.values():
        base.encoded(v)
        privacy(v)
    parents = parent_data(data, Path(frozen), Path(auxiliary_root))
    members, unique = membership(data, parents)
    specs = protocols(data)
    evidence(data, specs, members)
    a, b = analyze_public(data, parents, members)
    return (
        dict(
            status="pass",
            catalog_tiles=len(parents[0]),
            distinct_test_tiles=unique,
            test_vectors=675,
            validation_vectors=75,
            test_prediction_rows=135 * unique,
            adaptation_fits=75,
            sgd_heads=500,
            lr_fits=100,
            manifests=210,
            shared_memberships=25,
        ),
        a,
        b,
    )


def source_snapshot():
    require(base.sha256(__file__) == IMPORT_SHA256, "Imported verifier source changed")
    modules = (
        base,
        auxiliary,
        adaptation,
        display_a,
        display_b,
        adaptation.joint,
        adaptation.native,
        adaptation.common,
    )
    actual = {
        Path(m.__file__).resolve().relative_to(ROOT).as_posix(): m for m in modules
    }
    require(set(actual) == set(HELPERS), "Imported helper locations differ")
    observed = {}
    for name, h in HELPERS.items():
        observed[name] = base.sha256(actual[name].__file__)
        require(
            observed[name] in (h, PUBLIC_HELPERS[name]),
            "Pinned helper changed: " + name,
        )
    require(
        base.sha256(ROOT / "scripts/export_repaired_adaptation_results.py")
        == EXPORTER_SHA256,
        "Imported exporter source changed",
    )
    for name, h in TEST_SHA256.items():
        require(base.sha256(ROOT / name) == h, "Public test source changed")
    return {
        **observed,
        **TEST_SHA256,
        "scripts/verify_repaired_adaptation_release.py": IMPORT_SHA256,
        "scripts/export_repaired_adaptation_results.py": EXPORTER_SHA256,
    }


def unchanged(tracked):
    for p, h in tracked.items():
        require(base.sha256(p) == h, "Input/source changed: " + Path(p).name)


def guard_output(output, protected):
    p = Path(output)
    require(
        not p.exists() and not any(x.is_symlink() for x in (p, *p.parents)),
        "Output must be new without symlinks",
    )
    p = p.resolve()
    require(p.parent.is_dir(), "Output parent missing")
    for q in protected:
        q = Path(q).resolve()
        require(
            not (p.is_relative_to(q) or q.is_relative_to(p)),
            "Output overlaps protected input/source",
        )
    return p


def write_new(path, raw):
    """Publish only after the entire exclusive pending file successfully closes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name("." + path.name + ".pending")
    with pending.open("xb") as stream:
        require(stream.write(raw) == len(raw), "Short output write")
    os.link(pending, path)  # Never replaces an existing canonical file.
    # Leave the staging link until the caller has verified publication. Failure
    # before linking cannot leave a complete canonical receipt.


def write_json(path, value):
    privacy(value)
    raw = base.encoded(value)
    if str(path).endswith(".gz"):
        buf = io.BytesIO()
        with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0) as stream:
            stream.write(raw)
        raw = buf.getvalue()
    write_new(path, raw)


def clear_pending(root):
    for p in Path(root).rglob(".*.pending"):
        p.unlink()


def indexes(root):
    files = base.inventory(root, ("file_index.json", "SHA256SUMS"))
    write_json(root / "file_index.json", dict(schema_version="1.0.0", files=files))
    clear_pending(root)
    files = base.inventory(root, ("SHA256SUMS",))
    write_new(
        root / "SHA256SUMS",
        "".join(f"{r['sha256']}  {n}\n" for n, r in files.items()).encode(),
    )
    clear_pending(root)


def render(a, b, root):
    import matplotlib

    # Canonical key order also fixes CSV column order after JSON round trips.
    a, b = json.loads(base.encoded(a)), json.loads(base.encoded(b))
    ad = root / "display/adaptation"
    bd = root / "display/substitution"
    ad.mkdir(parents=True)
    bd.mkdir(parents=True)
    display_a.table_files(a, ad)
    # Match a fresh process and restore settings changed by plotting helpers.
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        display_a.figure(a, ad)
        display_b.write_tables(b, bd)
        figure = display_b.make_figure(b)
        adaptation.joint.save_figure(
            figure, bd / "repaired_training_substitution_20260930"
        )
        import matplotlib.pyplot as plt

        plt.close(figure)
    write_json(ad / "display.json", a)
    write_json(bd / "summary.json", b)
    write_new(bd / "caption.txt", (b["caption"] + "\n").encode())
    clear_pending(root)


CPU_REQUIREMENTS = (
    "numpy==2.4.4 --hash=sha256:"
    "81f4a14bee47aec54f883e0cad2d73986640c1590eb9bfaaba7ad17394481e6e\n"
    "matplotlib==3.10.9 --hash=sha256:"
    "ae20801130378b82d647ff5047c07316295b68dc054ca6b3c13519d0ea624285\n"
    "contourpy==1.3.3 --hash=sha256:"
    "4d00e655fcef08aba35ec9610536bfe90267d7ab5ba944f7032549c55a146da1\n"
    "cycler==0.12.1 --hash=sha256:"
    "85cef7cff222d8644161529808465972e51340599459b8ac3ccbac5a854e0d30\n"
    "fonttools==4.62.1 --hash=sha256:"
    "149f7d84afca659d1a97e39a4778794a2f83bf344c5ee5134e09995086cc2392\n"
    "kiwisolver==1.5.0 --hash=sha256:"
    "bb5136fb5352d3f422df33f0c879a1b0c204004324150cc3b5e3c4f310c9049f\n"
    "packaging==26.0 --hash=sha256:"
    "b36f1fef9334a5588b4166f8bcd26a14e521f2b55e6b9de3aaa80d3ff7a37529\n"
    "pillow==10.2.0 --hash=sha256:"
    "127cee571038f252a552760076407f9cff79761c3d436a12af6000cd182a9d04\n"
    "pyparsing==3.1.1 --hash=sha256:"
    "32c7c0b711493c72ff18a981d24f28aaf9c1fb7ed5e9667c9e84e3db623bdbfb\n"
    "python-dateutil==2.8.2 --hash=sha256:"
    "961d03dc3453ebbc59dbdea9e4e11c5651520a876d0f4db161e8674aae935da9\n"
    "six==1.16.0 --hash=sha256:"
    "8abb2f1d86890a2dfb989f9a77cfcfd3e47c2a354b01111771326f8aa26e0254\n"
)


CLAIMS_FILE = "display/claims.json"
CLAIMS_SCOPE = (
    "Registered printed numerals and fixed artifact values in the new adaptation "
    "and training-substitution additions. Narrative English quantities, external "
    "audit bookkeeping, inherited values, inequality predicates and the separate "
    "public frozen replay are checked by Main."
)
CLAIM_DISPLAYS = (
    "display/adaptation/display.json",
    "display/substitution/summary.json",
)
CLAIM_PROTOCOL = "protocols/adaptation.json"
CLAIM_SGD_PROTOCOL = "protocols/buffer_sgd.json"
CLAIM_ANALYSIS = "results/adaptation.json"
CLAIM_SGD_RESULTS = "results/buffer_sgd.json"
CLAIM_MEMBERSHIP = "membership.json.gz"
CLAIM_SOURCES = (
    *CLAIM_DISPLAYS,
    CLAIM_PROTOCOL,
    CLAIM_SGD_PROTOCOL,
    CLAIM_ANALYSIS,
    CLAIM_SGD_RESULTS,
    CLAIM_MEMBERSHIP,
)
TRAINING_TABLE_FIELDS = (
    "selected_epoch_min",
    "selected_epoch_max",
    "optimizer_attempts_total",
    "optimizer_steps_total",
    "amp_skipped_steps_total",
    "training_example_presentations",
)
CLAIM_ARTIFACTS = (
    "display/adaptation/repaired_adaptation_macro_20260930.tex",
    "display/adaptation/repaired_adaptation_classes_20260930.tex",
    "display/adaptation/repaired_adaptation_recipe_20260930.pdf",
    "display/substitution/repaired_training_substitution_20260930.pdf",
)


def claim_number(value, mode):
    require(type(value) in (int, float) and np.isfinite(value), "Claim must be finite")
    if mode == "coordinate":
        return None
    if mode == "integer":
        require(value == int(value), "Fractional count claim")
        return str(int(value))
    if mode == "tex_grouped_integer":
        require(value == int(value), "Fractional grouped count claim")
        return f"{int(value):,}".replace(",", "{,}")
    if mode == "unsigned_one_decimal":
        return f"{0.0 if round(value, 1) == 0 else value:.1f}"
    if mode == "tex_power_of_ten":
        powers = {0.0001: "10^{-4}", 0.001: "10^{-3}"}
        require(value in powers, "Unregistered TeX power")
        return powers[value]
    if mode == "unsigned_two_decimals":
        return display_a.tex_number(value)
    if mode == "signed_two_decimals":
        return display_b.mean_label(value)
    raise ValueError("Unregistered claim format")


def claim_catalog(a, b):
    """Enumerate fixed publication cells and permitted descriptive sources."""
    required, allowed = {}, {}

    def add(
        name,
        file,
        pointer,
        unit,
        target=None,
        mode="integer",
        calculation="scalar",
        extra=None,
    ):
        sources = [dict(file=file, pointer=pointer)]
        if extra is not None:
            sources.append(dict(file=file, pointer=extra))
        definition = dict(
            sources=sources,
            calculation=calculation,
            source_unit="count" if calculation == "boundary_percent" else unit,
            display_unit=unit,
            scale=1,
        )
        allowed[digest([sources, calculation])] = definition
        if target is not None:
            required[name] = dict(**definition, target=target, format=mode)

    def table(name, collection, index, row, column, component, unit, artifact, label):
        add(
            name,
            CLAIM_DISPLAYS[0],
            [collection, index, column],
            unit,
            dict(
                kind="table",
                file=artifact,
                label=label,
                row=row,
                column=column,
                component=component,
            ),
            "integer" if unit == "count" else "unsigned_two_decimals",
        )

    for i, model in enumerate(FT_MODELS):
        require(a["macro_rows"][i]["model"] == display_a.NAMES[i], "Macro row identity")
        for field in ("ft_f1", "ft_recall", "ft_minus_lr_f1", "ft_minus_sgd_f1"):
            for part in ("mean", "sample_std"):
                unit = (
                    "percent"
                    if part == "mean" and "minus" not in field
                    else "percentage points"
                )
                table(
                    f"macro.{model}.{field}.{part}",
                    "macro_rows",
                    i,
                    [model],
                    field + "_" + part,
                    part,
                    unit,
                    CLAIM_ARTIFACTS[0],
                    "tab:repairedadaptation",
                )
        for c, terrain in enumerate(display_a.CLASSES):
            index = i * 6 + c
            row = a["class_rows"][index]
            require(
                [row["model"], row["terrain_class"]] == [display_a.NAMES[i], terrain],
                "Class row identity",
            )
            table(
                f"class.{model}.{terrain}.support",
                "class_rows",
                index,
                [model, terrain],
                "test_support",
                "support",
                "count",
                CLAIM_ARTIFACTS[1],
                "tab:repairedadaptationclasses",
            )
            for recipe in ("ft", "lr"):
                for metric in METRICS:
                    for part in ("mean", "sample_std"):
                        field = f"{recipe}_{metric}_{part}"
                        table(
                            f"class.{model}.{terrain}.{field}",
                            "class_rows",
                            index,
                            [model, terrain],
                            field,
                            part,
                            "percent" if part == "mean" else "percentage points",
                            CLAIM_ARTIFACTS[1],
                            "tab:repairedadaptationclasses",
                        )
        for reference in display_a.REFERENCES:
            for metric in ("f1", "recall"):
                prefix = ["models", model, "summary", reference, "macro", metric]
                for fold in (*range(5), None):
                    part = "point" if fold is not None else "mean"
                    pointer = prefix + (
                        ["fold_values", fold] if fold is not None else ["mean"]
                    )
                    target = dict(
                        kind="figure",
                        file=CLAIM_ARTIFACTS[2],
                        label="fig:repairedadaptation",
                        panel=metric,
                        row=[model, reference],
                        component=part,
                        fold=fold,
                    )
                    add(
                        f"figure_a.{model}.{reference}.{metric}.{fold}",
                        CLAIM_DISPLAYS[0],
                        pointer,
                        "percentage points",
                        target,
                        "coordinate",
                    )
    for recipe in ("lr", "sgd"):
        for model in MODELS:
            prefix = ["recipes", recipe, model, "f1", "difference"]
            for part, fold in [("point", f) for f in range(5)] + [
                ("mean", None),
                ("printed_mean", None),
            ]:
                pointer = prefix + (
                    ["fold_values", fold] if fold is not None else ["mean"]
                )
                target = dict(
                    kind="figure",
                    file=CLAIM_ARTIFACTS[3],
                    label="fig:repairedbuffers",
                    panel=recipe,
                    row=[model],
                    component=part,
                    fold=fold,
                )
                add(
                    f"figure_b.{recipe}.{model}.{part}.{fold}",
                    CLAIM_DISPLAYS[1],
                    pointer,
                    "percentage points",
                    target,
                    "signed_two_decimals" if part == "printed_mean" else "coordinate",
                )
    # Additional paper registrations select only these finite source families.
    for field in ("backbone_lr", "head_lr", "weight_decay"):
        add("", CLAIM_PROTOCOL, ["specification", "recipe", field], "dimensionless")
    for field in ("probe_epochs", "probe_batch_size"):
        add("", CLAIM_SGD_PROTOCOL, ["specification", "recipe", field], "count")
    for field in ("probe_lr", "momentum"):
        add("", CLAIM_SGD_PROTOCOL, ["specification", "recipe", field], "dimensionless")
    add("", CLAIM_ANALYSIS, ["validation", "checkpoint_replays"], "count")
    # The verified cohort contains 20 disjoint lists, each with 25 SGD heads.
    sources = [
        dict(file=CLAIM_SGD_RESULTS, pointer=["folds", i, "runs"]) for i in range(20)
    ]
    allowed[digest([sources, "count"])] = dict(
        sources=sources,
        calculation="count",
        source_unit="count",
        display_unit="count",
        scale=1,
    )
    for fold in range(5):
        add(
            "",
            CLAIM_MEMBERSHIP,
            [
                "folds",
                fold,
                "normalization_provenance",
                "sampling",
                "selected_tile_ids",
            ],
            "count",
            calculation="count",
        )
        add("", CLAIM_DISPLAYS[1], ["support_design", fold, "fold"], "index")
    for field in ("epochs", "unfreeze_blocks"):
        add("", CLAIM_PROTOCOL, ["specification", "recipe", field], "count")
    add(
        "",
        CLAIM_PROTOCOL,
        ["specification", "optimization_seeds"],
        "count",
        calculation="count",
    )
    for model in FT_MODELS:
        for reference in display_a.REFERENCES:
            for metric in ("f1", "recall"):
                add(
                    "",
                    CLAIM_DISPLAYS[0],
                    [
                        "models",
                        model,
                        "summary",
                        reference,
                        "macro",
                        metric,
                        "sample_std",
                    ],
                    "percentage points",
                )
    for recipe in ("lr", "sgd"):
        for model in MODELS:
            add(
                "",
                CLAIM_DISPLAYS[1],
                ["recipes", recipe, model, "f1", "difference", "sample_std"],
                "percentage points",
            )
    for i in range(3):
        for field in (
            "jobs",
            "selected_epoch_min",
            "selected_epoch_max",
            "optimizer_attempts_total",
            "optimizer_steps_total",
            "amp_skipped_steps_total",
            "training_example_presentations",
        ):
            add("", CLAIM_DISPLAYS[0], ["training_summary", i, field], "count")
    for fold in range(5):
        for role in ("train", "val", "test"):
            add(
                "", CLAIM_DISPLAYS[0], ["support_by_fold", fold, role, "total"], "count"
            )
            for c in range(6):
                add(
                    "",
                    CLAIM_DISPLAYS[0],
                    ["support_by_fold", fold, role, "class_counts", c],
                    "count",
                )
        require(b["support_design"][fold]["fold"] == fold, "Support fold identity")
        for field in (
            "train_total",
            "test_total",
            "boundary_total",
            "validation_total",
        ):
            add("", CLAIM_DISPLAYS[1], ["support_design", fold, field], "count")
        for field in (
            "train_class_counts",
            "test_class_counts",
            "boundary_class_counts",
        ):
            for c in range(6):
                add("", CLAIM_DISPLAYS[1], ["support_design", fold, field, c], "count")
        add(
            "",
            CLAIM_DISPLAYS[1],
            ["support_design", fold, "boundary_total"],
            "percent",
            calculation="boundary_percent",
            extra=["support_design", fold, "train_total"],
        )
    for field in (
        "fits",
        "fits_with_warnings",
        "convergence_warnings",
        "iteration_caps",
    ):
        add("", CLAIM_DISPLAYS[1], ["lr_diagnostic_counts", field], "count")
    for file, pointer in (
        (CLAIM_DISPLAYS[0], ["models"]),
        (CLAIM_DISPLAYS[0], ["training_rows"]),
        (CLAIM_DISPLAYS[0], ["support_by_fold"]),
        (CLAIM_DISPLAYS[1], ["lr_fit_diagnostics"]),
    ):
        add("", file, pointer, "count", calculation="count")
    return required, allowed


def support_table_catalog():
    """The seven registered columns of the five-row inline support table."""
    result = {}
    for fold in range(5):
        fields = (
            (
                "fold",
                CLAIM_DISPLAYS[1],
                ["support_design", fold, "fold"],
                "scalar",
                "index",
                "integer",
            ),
            (
                "training",
                CLAIM_DISPLAYS[1],
                ["support_design", fold, "train_total"],
                "scalar",
                "count",
                "tex_grouped_integer",
            ),
            (
                "replaced",
                CLAIM_DISPLAYS[1],
                ["support_design", fold, "boundary_total"],
                "scalar",
                "count",
                "integer",
            ),
            (
                "percent",
                CLAIM_DISPLAYS[1],
                ["support_design", fold, "boundary_total"],
                "boundary_percent",
                "percent",
                "unsigned_two_decimals",
            ),
            (
                "boundary_craters",
                CLAIM_DISPLAYS[1],
                ["support_design", fold, "boundary_class_counts", 5],
                "scalar",
                "count",
                "integer",
            ),
            (
                "validation_craters",
                CLAIM_DISPLAYS[0],
                ["support_by_fold", fold, "val", "class_counts", 5],
                "scalar",
                "count",
                "integer",
            ),
            (
                "test_craters",
                CLAIM_DISPLAYS[1],
                ["support_design", fold, "test_class_counts", 5],
                "scalar",
                "count",
                "integer",
            ),
        )
        for field, file, pointer, calculation, unit, mode in fields:
            sources = [dict(file=file, pointer=pointer)]
            if calculation == "boundary_percent":
                sources.append(
                    dict(file=file, pointer=["support_design", fold, "train_total"])
                )
            result[f"{fold}.{field}"] = dict(
                sources=sources,
                calculation=calculation,
                source_unit="count" if calculation == "boundary_percent" else unit,
                display_unit=unit,
                scale=1,
                format=mode,
            )
    return result


def claim_value(entry, data):
    values = []
    for source in entry["sources"]:
        require(
            set(source) == {"file", "pointer"} and source["file"] in CLAIM_SOURCES,
            "Claim source outside finite registration",
        )
        value = data[source["file"]]
        require(
            isinstance(source["pointer"], list) and source["pointer"],
            "Empty claim pointer",
        )
        for key in source["pointer"]:
            if isinstance(value, list):
                require(type(key) is int and 0 <= key < len(value), "Claim list index")
            else:
                require(
                    isinstance(value, dict) and isinstance(key, str) and key in value,
                    "Claim dictionary pointer",
                )
            value = value[key]
        values.append(value)
    calculation = entry["calculation"]
    if calculation == "scalar":
        require(len(values) == 1, "Scalar source count")
        value = values[0]
    elif calculation == "count":
        require(
            values and all(isinstance(value, (list, dict)) for value in values),
            "Count sources must be registered collections",
        )
        value = sum(len(value) for value in values)
    elif calculation == "boundary_percent":
        require(
            len(values) == 2
            and all(type(v) is int and v >= 0 for v in values)
            and values[1] > 0,
            "Boundary percentage support",
        )
        value = 100 * values[0] / values[1]
    else:
        raise ValueError("Unregistered claim calculation")
    claim_number(value, entry["format"])
    return value


def validate_claims(index, root, a, b):
    """Check the fixed grid and Main's accepted finite manuscript registration."""
    require(
        set(index)
        == {
            "schema_version",
            "scope",
            "mapping_input_sha256",
            "numeric_check",
            "specification",
        },
        "Claims index schema",
    )
    exact(
        [index["schema_version"], index["scope"]],
        ["1.0.0", CLAIMS_SCOPE],
        "Claims schema/scope",
    )
    hash_value(index["mapping_input_sha256"])
    check = index["numeric_check"]
    require(
        set(check)
        == {
            "report_sha256",
            "execution_sha256",
            "checker_sha256",
            "specification_sha256",
            "status",
            "returncode",
        },
        "Numeric check receipt fields",
    )
    require(
        check["status"] == "pass"
        and type(check["returncode"]) is int
        and check["returncode"] == 0,
        "Incomplete numeric check",
    )
    for key in (
        "report_sha256",
        "execution_sha256",
        "checker_sha256",
        "specification_sha256",
    ):
        hash_value(check[key])
    spec = index["specification"]
    exact(
        digest(spec),
        check["specification_sha256"],
        "Numeric check registration binding",
    )
    validate_claims_specification(spec, root, a, b)
    privacy(index)
    return index


def validate_claims_specification(spec, root, a, b):
    """Validate finite registration before the checker report and receipt exist."""
    require(
        set(spec)
        == {
            "documents",
            "source_files_sha256",
            "artifact_files_sha256",
            "required_entry_ids",
            "entries",
        },
        "Claims registration fields",
    )
    documents = spec["documents"]
    require(
        isinstance(documents, list)
        and [d["id"] for d in documents] == ["full", "workshop"],
        "Expected ordered full/workshop documents",
    )
    for document in documents:
        require(
            set(document) == {"id", "tex_file", "tex_sha256", "pdf_file", "pdf_sha256"},
            "Document identity fields",
        )
        auxiliary.safe_names([document["tex_file"], document["pdf_file"]])
        require(
            document["tex_file"].endswith(".tex")
            and document["pdf_file"].endswith(".pdf"),
            "Document file types",
        )
        hash_value(document["tex_sha256"])
        hash_value(document["pdf_sha256"])
    for field, names in (
        ("source_files_sha256", CLAIM_SOURCES),
        ("artifact_files_sha256", CLAIM_ARTIFACTS),
    ):
        require(set(spec[field]) == set(names), "Claims source/artifact inventory")
        for name, expected in spec[field].items():
            hash_value(expected)
            exact(
                base.sha256(Path(root) / name),
                expected,
                "Claims bound file differs: " + name,
            )
    required, allowed = claim_catalog(a, b)
    entries = spec["entries"]
    ids = [row["id"] for row in entries]
    require(
        ids == sorted(set(ids)) and ids == spec["required_entry_ids"],
        "Claims coverage/order differs",
    )
    require(set(required) <= set(ids), "Missing fixed table/figure scalar")
    protocol = base.read_json(Path(root) / CLAIM_PROTOCOL)
    specification = protocol["specification"]
    exact(
        [
            specification["recipe"]["epochs"],
            specification["recipe"]["unfreeze_blocks"],
            specification["optimization_seeds"],
        ],
        [50, 2, list(range(5))],
        "Adaptation design sources differ from the accepted recipe",
    )
    data = {
        **dict(zip(CLAIM_DISPLAYS, (a, b))),
        **{
            name: base.read_json(Path(root) / name)
            for name in CLAIM_SOURCES
            if name not in CLAIM_DISPLAYS
        },
    }
    training_sources = {"full": [], "workshop": []}
    support_sources = {"full": [], "workshop": []}
    support_definitions = {
        digest([row["sources"], row["calculation"]]): row
        for row in support_table_catalog().values()
    }
    expected_training_sources = [
        digest(
            [dict(file=CLAIM_DISPLAYS[0], pointer=["training_summary", model, field])]
        )
        for model in range(3)
        for field in TRAINING_TABLE_FIELDS
    ]
    occurrences, prose_documents, support_documents = set(), set(), set()
    for entry in entries:
        require(
            set(entry)
            == {
                "id",
                "target",
                "sources",
                "calculation",
                "source_unit",
                "display_unit",
                "scale",
                "format",
                "value",
                "text",
            },
            "Claim entry schema",
        )
        require(
            isinstance(entry["id"], str)
            and re.fullmatch(r"[A-Za-z0-9_.-]+", entry["id"]),
            "Claim ID",
        )
        target = entry["target"]
        places = target["occurrences"]
        target_body = {k: v for k, v in target.items() if k != "occurrences"}
        if entry["id"] in required:
            definition = required[entry["id"]]
            exact(
                {k: entry[k] for k in definition if k != "target"},
                {k: v for k, v in definition.items() if k != "target"},
                "Fixed claim source/units/format",
            )
            exact(target_body, definition["target"], "Fixed claim target identity")
            require(
                [o["document"] for o in places] == ["full", "workshop"],
                "Fixed manuscript coverage",
            )
        else:
            require(
                set(target_body) == {"kind", "file", "label", "registration"}
                and target["kind"] in ("prose", "support_table", "training_table"),
                "Unregistered target kind",
            )
            require(
                target["file"] in [d["tex_file"] for d in documents],
                "Manuscript target file",
            )
            require(
                isinstance(target["registration"], str) and target["registration"],
                "Missing prose registration",
            )
            key = digest([entry["sources"], entry["calculation"]])
            require(key in allowed, "Unregistered scalar source/calculation")
            exact(
                {k: entry[k] for k in allowed[key]},
                allowed[key],
                "Claim source units/scale",
            )
            require(entry["format"] != "coordinate", "Manuscript scalar must print")
            if target["kind"] == "support_table":
                exact(
                    target["label"], "tab:repairedcontrolsupport", "Support table label"
                )
                require(key in support_definitions, "Support table source")
                exact(
                    entry["format"],
                    support_definitions[key]["format"],
                    "Support table format",
                )
            if target["kind"] == "training_table":
                exact(target["label"], "tab:repairedtraining", "Training table label")
                require(
                    digest(entry["sources"]) in expected_training_sources
                    and entry["calculation"] == "scalar",
                    "Training table source",
                )
                field = entry["sources"][0]["pointer"][-1]
                expected_format = (
                    "integer"
                    if field.startswith("selected_epoch_")
                    else "tex_grouped_integer"
                )
                exact(entry["format"], expected_format, "Training table format")
        require(isinstance(places, list) and places, "Missing manuscript occurrence")
        for place in places:
            require(
                set(place) == {"document", "registration", "ordinal"}
                and place["document"] in ("full", "workshop")
                and isinstance(place["registration"], str)
                and place["registration"]
                and type(place["ordinal"]) is int
                and place["ordinal"] >= 0,
                "Invalid manuscript occurrence",
            )
            if entry["id"] in required:
                exact(
                    place["registration"], target["label"], "Artifact occurrence label"
                )
                identity = digest([target_body, place])
            else:
                document = documents[0 if place["document"] == "full" else 1]
                exact(document["tex_file"], target["file"], "Occurrence document/file")
                exact(
                    place["registration"],
                    target["registration"],
                    "Prose occurrence registration",
                )
                identity = digest(place)
                if target["kind"] == "prose":
                    prose_documents.add(place["document"])
                elif target["kind"] == "support_table":
                    support_documents.add(place["document"])
                    support_sources[place["document"]].append(
                        digest([entry["sources"], entry["calculation"]])
                    )
                else:
                    training_sources[place["document"]].append(digest(entry["sources"]))
            require(
                identity not in occurrences, "Repeated manuscript numeric occurrence"
            )
            occurrences.add(identity)
        value = claim_value(entry, data)
        exact(entry["value"], value, "Claim scalar differs")
        exact(
            entry["text"],
            claim_number(value, entry["format"]),
            "Claim formatting differs",
        )
    require(
        prose_documents == {"full", "workshop"},
        "Final prose registration is incomplete",
    )
    require(
        support_documents == {"full", "workshop"},
        "Final support registration is incomplete",
    )
    for sources in support_sources.values():
        exact(
            sorted(sources),
            sorted(support_definitions),
            "Incomplete or duplicate support table grid",
        )
    for sources in training_sources.values():
        exact(
            sorted(sources),
            sorted(expected_training_sources),
            "Incomplete or duplicate training table grid",
        )
    privacy(spec)
    return spec


README = r"""# Accepted adaptation and training-substitution evidence

This package reconstructs saved-prediction metrics, nested summaries, two figures
and a finite manuscript claims index with Python, NumPy and Matplotlib. Native
checkpoint, head and estimator acceptance comes from the bound audits. The code
provided here runs CPU reconstruction of saved predictions.

Obtain the complete three sibling directories from the trusted CETUS revision
identified in [the release instructions](../../README.md):
`longitude-review-20260929`, `longitude-auxiliary-20260929`, and
`longitude-adaptation-controls-20260930`. Their exact inventories are verified.
The outer instructions also supply the independently accepted SHA256 of this
package's `evidence/acceptance.json`. Use that published value in the commands
below. A locally calculated digest checks bytes; release acceptance is external.

Create an external working directory and virtual environment using the exact
Python patch version recorded in `code/SOURCE_EXPORT.json`:

```sh
PACKAGE=/absolute/release/longitude-adaptation-controls-20260930
WORK=/absolute/cetus-reconstruction
mkdir -p "$WORK"
python3 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
```

Install the eleven pinned PyPI wheels with their exact SHA256 hashes:

```sh
python -m pip install --only-binary=:all: --require-hashes -r "$PACKAGE/code/requirements.txt"
python -m pip check
```

Install once while packages are accessible; reconstruction can then run offline.
An authenticated wheel directory can be supplied with `--no-index --find-links`.
This lock describes the tested PyPI reconstruction environment. The original
accepted display environment included four Ubuntu package builds. A separate
Linux x86_64, glibc 2.39, Python 3.12.3 PyPI trial reproduced all sixteen display
artifacts and the adaptation checksum sidecar byte for byte. That comparison
covers the recorded runtime and font settings. It does not establish byte
identity on arbitrary platforms or a complete operating-system reconstruction.
The release instructions identify the independently accepted runtime evidence.
These headless figures require no TeX installation.

Keep the environment, caches, logs and outputs outside all three input packages.
Choose writable temporary storage outside them. Then verify saved evidence:

```sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$WORK/matplotlib-cache" TMPDIR="$WORK/tmp"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR"
ACCEPTANCE_SHA=VALUE_FROM_TRUSTED_OUTER_RELEASE_INSTRUCTIONS
cd "$PACKAGE/code"
python -B -m scripts.verify_repaired_adaptation_release \
  --package "$PACKAGE" \
  --frozen-package "$PACKAGE/../longitude-review-20260929" \
  --auxiliary-package "$PACKAGE/../longitude-auxiliary-20260929" \
  --acceptance-sha256 "$ACCEPTANCE_SHA"
```

Verification reconstructs numeric tables in temporary storage. Success requires
process exit zero and `status: pass`. To rebuild the complete sibling package,
repeat that command with `--output "$WORK/rebuilt-package"`. Its parent must
exist; both `rebuilt-package` and `rebuilt-package.receipt.json` must be absent.
The output must be disjoint from inputs and source. Rebuild includes both display
directories and a complete receipt. A retained hidden pending hardlink beside
the receipt records successful publication. Preserve failed evidence and choose
a fresh output name for a retry. PDFs, PNGs, CSVs and TeX fragments are under
`display/adaptation` and `display/substitution`.

`display/claims.json` binds the final manuscript identities, Main's accepted
numeric-checker receipt, generated assets, numeric sources and registered
occurrences. Public verification checks this finite index and its values.
Manuscript coverage comes from Main's independent check of the manuscript bytes.
The index covers registered printed numerals and fixed artifact values. Main
separately checks English-word quantities, inherited values, inequality predicates,
external audit bookkeeping and the independent public frozen replay. The index
makes no claim to those registrations.
The package carries the identities of those manuscripts. The trusted acceptance
digest belongs to the outer release instructions, preserving an acyclic build.

Five adaptation seeds are averaged within each fold. Matching SGD seeds are
paired; common LR has one reference fit per model and fold. Substitution averages
five SGD heads per draw, then five draws; LR fits once per draw. Means weight five
folds equally and SD uses ddof=1. Contrasts describe complete recipes or changes
to training membership. All six map classes, signed differences and warned or
capped fits are retained. Raw scores are fractions; displayed rates are percent
and changes are percentage points. Display sources already contain these units.
Fold spread is descriptive. Labels measure agreement with the expert map.
Independent geological accuracy remains unmeasured. Histories cover all fifty
epochs. Validation vectors come from checkpoint replay.

The preserved native exporter records provenance and requires private accepted
inputs. Optional tests use synthetic fixtures; pytest is separate from the CPU
runtime. Copy code to an external testing directory before installing pytest or
running tests, so test caches preserve the sealed input inventories.
"""


def build_package(
    payloads, frozen, auxiliary_root, output, tracked=None, *, claims=None
):
    require(claims is not None, "Accepted final claims map is required")
    source = source_snapshot()
    tracked = {Path(p): h for p, h in (tracked or {}).items()}
    tracked.update({ROOT / p: h for p, h in source.items()})
    for parent in (Path(frozen), Path(auxiliary_root)):
        tracked.update(
            {parent / n: r["sha256"] for n, r in base.inventory(parent).items()}
        )
    output = guard_output(
        output,
        [
            ROOT,
            Path(frozen),
            Path(auxiliary_root),
            *[p.parent for p in tracked if not p.is_relative_to(ROOT)],
        ],
    )
    data = json.loads(base.encoded(payloads))
    data["evidence/acceptance.json"].pop("public_files_sha256", None)
    receipt_path = output.with_name(output.name + ".receipt.json")
    require(
        not receipt_path.exists()
        and not receipt_path.with_name("." + receipt_path.name + ".pending").exists(),
        "Receipt destination exists",
    )
    accepted_a = data.pop("accepted_display/adaptation.json", None)
    accepted_b = data.pop("accepted_display/substitution.json", None)
    result, a, b = validate_payloads(data, Path(frozen), Path(auxiliary_root))
    if accepted_a is not None:
        exact(a, accepted_a["display"], "Accepted adaptation display differs")
    if accepted_b is not None:
        for k in (
            "recipes",
            "lr_diagnostic_counts",
            "lr_fit_diagnostics",
            "support_design",
            "units",
        ):
            exact(b[k], accepted_b[k], "Accepted substitution display differs: " + k)
    unchanged(tracked)
    output.mkdir()
    write_new(output / ".INCOMPLETE", b"Unaccepted or incomplete export.\n")
    clear_pending(output)
    for name, value in data.items():
        if name != "evidence/acceptance.json":
            write_json(output / name, value)
    clear_pending(output)
    code = output / "code"
    (code / "scripts").mkdir(parents=True)
    exported_source = {}
    for name, h in source.items():
        raw = public_helper_bytes(name, (ROOT / name).read_bytes())
        write_new(code / name, raw)
        exported_source[name] = hashlib.sha256(raw).hexdigest()
    write_new(code / "scripts/__init__.py", b"")
    write_new(code / "tests/__init__.py", b"")
    write_new(code / "LICENSE", (ROOT / "LICENSE").read_bytes())
    import matplotlib

    runtime = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        matplotlib=matplotlib.__version__,
    )
    require(
        (runtime["numpy"], runtime["matplotlib"]) == ("2.4.4", "3.10.9"),
        "CPU requirements need NumPy 2.4.4 and Matplotlib 3.10.9",
    )
    write_new(code / "requirements.txt", CPU_REQUIREMENTS.encode())
    write_json(
        code / "SOURCE_EXPORT.json",
        dict(
            files=exported_source,
            native_helper_sha256=HELPERS,
            helper_projection=(
                "Machine-specific privacy denylist tokens replaced by shared prefix; "
                "numeric code unchanged."
            ),
            runtime=runtime,
            scope="CPU saved-prediction reconstruction; original producers distinct",
        ),
    )
    write_new(output / "README.md", README.encode())
    clear_pending(output)
    render(a, b, output)
    write_json(output / CLAIMS_FILE, validate_claims(claims, output, a, b))
    clear_pending(output)
    unchanged(tracked)
    source_snapshot()
    exported = base.inventory(output, (".INCOMPLETE",))
    data["evidence/acceptance.json"]["public_files_sha256"] = {
        n: r["sha256"] for n, r in exported.items()
    }
    write_json(output / "evidence/acceptance.json", data["evidence/acceptance.json"])
    clear_pending(output)
    (output / ".INCOMPLETE").unlink()
    try:
        indexes(output)
        # Verify canonical bytes before publishing the complete receipt.
        verified = verify_package(
            output,
            Path(frozen),
            Path(auxiliary_root),
            base.sha256(output / "evidence/acceptance.json"),
        )
        unchanged(tracked)
        source_snapshot()
        receipt = dict(
            status="complete",
            validation=verified,
            acceptance_sha256=base.sha256(output / "evidence/acceptance.json"),
            file_index_sha256=base.sha256(output / "file_index.json"),
            checksums_sha256=base.sha256(output / "SHA256SUMS"),
            runtime=runtime,
        )
        write_new(receipt_path, base.encoded(receipt))
        # No fallible cleanup after success publication. The pending hardlink is
        # excluded explicitly, retaining proof that close preceded publication.
    except BaseException:
        if not receipt_path.exists():
            (output / ".INCOMPLETE").write_text(
                "Publication failed; retain evidence.\n"
            )
        raise
    return {
        **result,
        "acceptance_sha256": receipt["acceptance_sha256"],
        "receipt_name": receipt_path.name,
    }


def verify_package(package, frozen, auxiliary_root, acceptance_sha256):
    root = Path(package)
    hash_value(acceptance_sha256)
    require(
        not any(p.name == ".git" for p in root.rglob("*")),
        "Git metadata is not a release payload",
    )
    require(not (root / ".INCOMPLETE").exists(), "Incomplete package")
    require(
        base.sha256(root / "evidence/acceptance.json") == acceptance_sha256,
        "External acceptance pin differs",
    )
    inventory = base.inventory(root)
    index = base.read_json(root / "file_index.json")
    exact(
        index["files"],
        {
            k: v
            for k, v in inventory.items()
            if k not in ("file_index.json", "SHA256SUMS")
        },
        "Package inventory differs",
    )
    lines = (root / "SHA256SUMS").read_text().splitlines()
    sums = {}
    for line in lines:
        h, n = line.split("  ", 1)
        require(n not in sums, "Duplicate checksum")
        sums[n] = h
    exact(
        sums,
        {n: r["sha256"] for n, r in inventory.items() if n != "SHA256SUMS"},
        "Package checksums differ",
    )
    # Exact fixed output inventory, not a self-reported permissive whitelist.
    expected = set(JSON_FILES) | {
        "README.md",
        "file_index.json",
        "SHA256SUMS",
        "code/SOURCE_EXPORT.json",
        "code/scripts/__init__.py",
        "code/tests/__init__.py",
        "code/LICENSE",
        "code/requirements.txt",
        CLAIMS_FILE,
    }
    expected |= {
        "code/" + n
        for n in (
            *HELPERS,
            *TEST_SHA256,
            "scripts/verify_repaired_adaptation_release.py",
            "scripts/export_repaired_adaptation_results.py",
        )
    }
    expected |= {
        f"display/adaptation/repaired_adaptation_{n}_20260930.{ext}"
        for n, ext in [
            ("macro", "csv"),
            ("classes", "csv"),
            ("training", "csv"),
            ("macro", "tex"),
            ("classes", "tex"),
            ("recipe", "pdf"),
            ("recipe", "png"),
        ]
    }
    expected |= {
        "display/adaptation/display.json",
        *(
            f"display/substitution/{n}"
            for n in (
                "summary.csv",
                "folds.csv",
                "lr_fits.csv",
                "support.csv",
                "caption.txt",
                "summary.json",
                "repaired_training_substitution_20260930.pdf",
                "repaired_training_substitution_20260930.png",
            )
        ),
    }
    exact(sorted(inventory), sorted(expected), "Unknown or missing package payload")
    sources = base.read_json(root / "code/SOURCE_EXPORT.json")
    require(
        set(sources["files"])
        == set(HELPERS)
        | set(TEST_SHA256)
        | {
            "scripts/verify_repaired_adaptation_release.py",
            "scripts/export_repaired_adaptation_results.py",
        },
        "Code inventory",
    )
    for n, h in sources["files"].items():
        require(base.sha256(root / "code" / n) == h, "Public code changed")
    exact(
        {n: sources["files"][n] for n in HELPERS},
        PUBLIC_HELPERS,
        "Public helper identities",
    )
    exact(sources["native_helper_sha256"], HELPERS, "Native helper identities")
    require(
        sources["files"]["scripts/verify_repaired_adaptation_release.py"]
        == IMPORT_SHA256,
        "Public verifier identity differs",
    )
    require(
        sources["files"]["scripts/export_repaired_adaptation_results.py"]
        == EXPORTER_SHA256,
        "Public exporter identity differs",
    )
    for name, h in TEST_SHA256.items():
        exact(sources["files"][name], h, "Public test source identity")
    import matplotlib

    exact(
        sources["runtime"],
        dict(
            python=platform.python_version(),
            numpy=np.__version__,
            matplotlib=matplotlib.__version__,
        ),
        "Pinned CPU runtime differs",
    )
    data = {n: base.read_json(root / n) for n in JSON_FILES}
    sealed = {
        n: r["sha256"]
        for n, r in inventory.items()
        if n not in ("evidence/acceptance.json", "file_index.json", "SHA256SUMS")
    }
    exact(
        data["evidence/acceptance.json"]["public_files_sha256"],
        sealed,
        "Externally bound public projection differs",
    )
    result, a, b = validate_payloads(data, Path(frozen), Path(auxiliary_root))
    exact(
        base.read_json(root / "display/adaptation/display.json"),
        a,
        "Public adaptation display values",
    )
    exact(
        base.read_json(root / "display/substitution/summary.json"),
        b,
        "Public substitution display values",
    )
    exact((root / "code/requirements.txt").read_text(), CPU_REQUIREMENTS,
          "CPU requirements differ")
    validate_claims(base.read_json(root / CLAIMS_FILE), root, a, b)
    # Reconstruct all tabular numbers even if an inventory was recomputed after
    # corruption. Rendering is reserved for the explicitly requested rebuild.
    with tempfile.TemporaryDirectory(prefix="cetus-public-tables-") as temp:
        temp = Path(temp)
        (temp / "a").mkdir()
        (temp / "b").mkdir()
        display_a.table_files(json.loads(base.encoded(a)), temp / "a")
        display_b.write_tables(json.loads(base.encoded(b)), temp / "b")
        for short, family in [("a", "adaptation"), ("b", "substitution")]:
            for table in (temp / short).iterdir():
                require(
                    table.read_bytes()
                    == (root / "display" / family / table.name).read_bytes(),
                    "Public table differs: " + table.name,
                )
    require(
        (root / "display/substitution/caption.txt").read_text() == b["caption"] + "\n",
        "Public caption differs",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "frozen-package", "auxiliary-package"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--acceptance-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        sources = source_snapshot()
        result = verify_package(
            args.package,
            args.frozen_package,
            args.auxiliary_package,
            args.acceptance_sha256,
        )
        if args.output is not None:
            source_inputs = {
                args.package / n: r["sha256"]
                for n, r in base.inventory(args.package).items()
            }
            data = {n: base.read_json(args.package / n) for n in JSON_FILES}
            build_package(
                data,
                args.frozen_package,
                args.auxiliary_package,
                args.output,
                source_inputs,
                claims=base.read_json(args.package / CLAIMS_FILE),
            )
        exact(source_snapshot(), sources, "Verifier source changed")
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        print(json.dumps(dict(status="invalid", error=str(error))))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
