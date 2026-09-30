"""Export accepted common/classical predictions and VIMS v3 as a sibling package.

CPU metadata only. No Git, network, estimator loading, fitting or catalog writes.
The fixed acceptance pins intentionally limit this exporter to these accepted runs.
"""
import argparse
from pathlib import Path, PurePosixPath
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_repaired_release as base

require, read_json, sha256 = base.require, base.read_json, base.sha256
pick, digest, close = base.pick, base.digest, base.close
MODELS, CLASSES = base.MODELS, base.CLASSES
COLUMNS = {"raw_intensity_statistics": [0, 13], "intensity_histogram": [13, 45],
           "texture_glcm_lbp_gabor_hog_wavelet": [45, 1865], "combined_classical": [0, 1865]}
COMMITS = {
    "common": "3dd19a2e8c55d07e293f77ce139ded53112a7ef3",
    "classical": "54281f05cfd67453f9b699810021b1e2da20c3ec",
    "vims": "a1d5652010883f0accf90efbbd77b17c8c5047d6",
}
AUDITS = {
    "common": "e0c007157b51ea83ab8cdfb7b661232456ecc1a6181d231dbd03104ac5473379",
    "classical": "1c9dc259986b742f80afd92ff857c2798f32ecb41178b2a8fe12cff4de041cd2",
}
PROTOCOLS = {
    "common": ("REPAIRED_COMMON_CLASSIFIER_PROTOCOL_2026-09-29.json", "48515a7583e99c0cfd9fb179600cdfeda7dab484f424b4bcbdb912570632f9ca"),
    "classical": ("REPAIRED_BASELINES_PROTOCOL_2026-09-29-v2.json", "a96d9ffc113ddec8ac637dc29726e8cc241a1dff1b69bc4d9cfe5b7d320a0961"),
}
PARENT_FILES = {
    "catalogs/titan_catalog.json": "8152226f876e0cb5fad105e0b00a37ef5b62d156d7f6eb3a85b77c549a402967",
    "splits/protocol.json": "1ce5fd9e8b135a172ce3e671b12a9a06e160e089debb9cac22bb14213f784060",
    "provenance.json": "e7ccd203f104c95f140e8604514004c1e4e667a04d973c2b89353dd22b0ac533",
    "results/frozen_results.json": "dc0333a7cc04394fb41ad4729a2049f4ba70a7a3ce77062d3744daa629921a4f",
    **{f"splits/titan_contiguous_fold_{f}.json": value for f, value in enumerate([
        "b69e2fa3551dde125edf31a5601c52ccae755cb6169ed57d145b16f6983f85ab",
        "58ad14b783c0d6ffd2851cb622f1d0d4fd1438978aeb146e32e5dcb8d55a8adc",
        "2eee98044e7a636a6726800328b0361991432e9ea272eaee7873b3faa2ee28d2",
        "2dbd6117de1c299655116af827ee95998839c67da443bf2a03729b7e4e2fab75",
        "1df555f0f95c16e3fb44243ce7ecca41225e4e588353148f25725c68f1cb4271"])},
}
VIMS_FILES = {
    "producer/vims_metadata.json": "f28957a13990e10b45568c6e2868653db5f1ce9659787e295993ccfaa05a9b48",
    "producer/source_raster.json": "a9426cfb062f3bec464980d491674f64011e41646e70caa9da84587bf876d66a",
    "producer/run_manifest.json": "18edeaef57f647eb9fab259a33ac22848f8313840efe21a205f253e7f96bf899",
    "checker/verification.json": "75d90f70a2c83bc51596e63493fd629b85effdefa1ba699bb5a039f4dd4d6a4b",
    "checker/run_manifest.json": "c2a718eb45762c62c8f83c2de83d24bf283a62902419439de34b837a94c0d3b1",
}
RESULT_NAMES = {"common": "results/common_classifier.json", "classical": "results/classical_baselines_v2.json"}
PROTOCOL_NAMES = {"common": "protocols/common_classifier.json", "classical": "protocols/classical_baselines_v2.json"}
EXTRACTS = {
    "evidence/runs.json": "46be96aa1a5efe07530dea2862c90ddc9b3380bea035bfd735f6e42255c6507e",
    "protocols/common_classifier.json": "a6d7b60e6760b2a297a8684bbcc6ec028ca6f99d97035ec24343ecd3bd7a20b8",
    "protocols/classical_baselines_v2.json": "f9ca9a0694dab3f2c0c99d6cb18288a0af87e17006b671c3db6b0c3afaea1470",
    "metadata/vims_source_v3.json": "e2750cec3634fcf0282300d2b56efecb7d13fffa1fec9ed3aae448454fff1594",
}
PAYLOADS = {"README.md", "provenance.json", *RESULT_NAMES.values(), *PROTOCOL_NAMES.values(),
            "results/predictions.json.gz", "metadata/vims_display_v3.json", "metadata/vims_source_v3.json",
            "evidence/vims_verification_v3.json", "evidence/runs.json"}
BASELINE_FIELDS = ("acceptance amendment benchmark_track catalog_sha256 class_names controls "
                   "environment_snapshot_sha256 estimator_state feature_cache feature_semantics folds "
                   "frozen_reference_audit_sha256 frozen_reference_source_commit geographic_protocol_sha256 "
                   "limitations normalization recipe reporting runtime_versions scientific_source_sha256 scope volume")
COMMON_FIELDS = ("benchmark_track class_names python_version recipe resolved_parameters "
                 "runtime_import_versions runtime_versions scientific_source_sha256")


def pinned(path, expected):
    require(Path(path).is_file() and sha256(path) == expected, f"Missing or changed accepted input: {path}")
    return read_json(path)


def safe_names(values):
    for name in values:
        p = PurePosixPath(name)
        require(name and not p.is_absolute() and ".." not in p.parts and "\\" not in name,
                "Unsafe relative artifact name")
    base.check_public(values)


def protocol_extract(protocol, family):
    spec = protocol["specification"]
    require(digest(spec) == protocol["specification_sha256"], "Original protocol specification hash differs")
    if family == "classical":
        selected = pick(spec, BASELINE_FIELDS)
    else:
        selected = pick(spec, COMMON_FIELDS)
        for key in ("catalog", "baseline_protocol", "native_audit", "adopted_plan"):
            selected[key] = pick(spec[key], "sha256")
        selected["folds"] = []
        for row in spec["folds"]:
            selected["folds"].append({**pick(row, "fold counts normalization_provenance normalization_sha256 "
                                              "train_ids_sha256 test_ids_sha256 train_labels_sha256 test_labels_sha256"),
                                      "split": pick(row["split"], "sha256")})
        selected["caches"] = [pick(c, "model fold manifest_sha256 encoder_identity files_sha256") for c in spec["caches"]]
        for cache in selected["caches"]:
            safe_names(cache["files_sha256"])
    base.check_public(selected)
    return dict(schema_version="1.0.0", original_protocol_sha256=PROTOCOLS[family][1],
                original_specification_sha256=protocol["specification_sha256"],
                specification=selected, extract_specification_sha256=digest(selected))


def parent_data(root):
    docs = {name: pinned(root / name, value) for name, value in PARENT_FILES.items()}
    catalog = docs["catalogs/titan_catalog.json"]
    require(catalog["class_names"] == list(CLASSES), "Parent class order differs")
    tiles = catalog["tiles"]
    ids = [r["tile_id"] for r in tiles]
    require(len(ids) == len(set(ids)), "Parent tile IDs are not unique")
    folds = []
    for f in range(5):
        assignment = docs[f"splits/titan_contiguous_fold_{f}.json"]["assignments"]
        require(set(assignment) == set(ids), "Parent assignment coverage differs")
        row = {"fold": f, "normalization_provenance": docs["results/frozen_results.json"]["folds"][f]["normalization_provenance"]}
        for role in ("train", "test"):
            chosen = [t for t in tiles if assignment[t["tile_id"]] == role]
            row[role + "_ids"] = [t["tile_id"] for t in chosen]
            row[role + "_labels"] = [t["label"] for t in chosen]
        folds.append(row)
    return catalog, folds, docs["provenance.json"]["source_file_sha256"]


def check_protocols(protocols, folds, parent_sources):
    common, classical = [protocols[k]["specification"] for k in ("common", "classical")]
    require(common["catalog"]["sha256"] == classical["catalog_sha256"] == base.CATALOG_SHA256,
            "Producer catalog identity differs")
    require(common["baseline_protocol"]["sha256"] == PROTOCOLS["classical"][1], "Baseline protocol identity differs")
    require(common["native_audit"]["sha256"] == classical["frozen_reference_audit_sha256"] == base.RESULT_SHA256,
            "Native acceptance identity differs")
    for spec in (common, classical):
        require(spec["class_names"] == list(CLASSES) and [x["fold"] for x in spec["folds"]] == list(range(5)),
                "Protocol classes or folds differ")
    cache_keys = [(c["model"], c["fold"]) for c in common["caches"]]
    require(len(cache_keys) == 20 and set(cache_keys) == {(m, f) for m in MODELS for f in range(5)},
            "Missing or duplicate cache identities")
    for f, bound in enumerate(folds):
        a, b = common["folds"][f], classical["folds"][f]
        require(a["split"]["sha256"] == b["split_sha256"] == parent_sources[f"split_{f}"], "Original split identity differs")
        require(a["normalization_provenance"] == bound["normalization_provenance"], "Normalization provenance differs")
        require(a["normalization_sha256"] == b["normalization_sha256"] == digest(bound["normalization_provenance"]),
                "Normalization digest differs")
        require(set(a["normalization_provenance"]["sampling"]["selected_tile_ids"]) <= set(bound["train_ids"]),
                "Normalization includes nontraining tiles")
        for role in ("train", "test"):
            ids, labels = bound[role + "_ids"], bound[role + "_labels"]
            require(a[role + "_ids_sha256"] == b[role + "_ids_sha256"] == digest(ids), "Ordered membership differs")
            require(a[role + "_labels_sha256"] == digest(labels), "Ordered labels differ")
            counts = dict(total=len(ids), class_counts=np.bincount(labels, minlength=6).tolist())
            require(a["counts"][role] == b["counts"][role] == counts, "Protocol support differs")
    for family in protocols:
        p = protocols[family]
        require(p["original_protocol_sha256"] == PROTOCOLS[family][1]
                and digest(p["specification"]) == p["extract_specification_sha256"], "Protocol extract binding differs")


def check_metrics(predicted, truth, accepted):
    pred, labels = np.asarray(predicted), np.asarray(truth)
    require(pred.shape == labels.shape and pred.dtype.kind in "iu" and np.isin(pred, range(6)).all(),
            "Invalid prediction vector")
    counts = np.zeros((6, 6), dtype=np.int64)
    np.add.at(counts, (labels, pred), 1)
    require(counts.tolist() == accepted["confusion_counts"], "Predictions differ from accepted confusion counts")
    computed = base.metrics(counts)
    for key in ("support", "prediction_counts", "kappa"):
        close(accepted[key], computed[key], f"Accepted {key} differs")
    for scope in ("macro", "per_class"):
        for metric in base.METRICS:
            close(accepted[scope][metric], computed[scope][metric], f"Accepted {scope} {metric} differs")
    return computed


def extract_prediction(record, ids, labels, accepted, rule="argmax"):
    require(record["tile_ids"] == ids and record["true_labels"] == labels
            and all(type(v) is int for v in record["true_labels"]), "Saved IDs or labels differ")
    pred = record["predictions"]
    check_metrics(pred, labels, accepted)
    require(record["confusion_matrix"] == accepted["confusion_counts"], "Saved confusion counts differ")
    require(record["probability_class_indices"] == list(range(6)) and record["prediction_rule"] == rule,
            "Probability class order or rule differs")
    prob = np.asarray(record["probabilities"])
    require(prob.shape == (len(ids), 6) and prob.dtype.kind in "iuf" and np.isfinite(prob).all()
            and (prob >= 0).all() and (prob <= 1).all(), "Invalid probability array")
    require(np.allclose(prob.sum(1, dtype=np.float64), 1, atol=1e-6, rtol=0), "Probability mass differs")
    if rule == "argmax":
        require(np.array_equal(prob[np.arange(len(ids)), pred], prob.max(1)), "Prediction is not a probability maximum")
    if "fit" in accepted:
        require(pick(record, " ".join(accepted["fit"])) == accepted["fit"], "Fit evidence differs")
    return pred


def manifest_extract(path, family, expected_hash):
    record = pinned(path, expected_hash)
    require(record["status"] == "complete" and record["git"] == dict(commit=COMMITS[family], clean=True, status=[]),
            "Incomplete or unexpected producer source")
    safe_names(record["outputs"])
    if family == "vims":
        environment = pick(record["environment"], "device python numpy rasterio gdal proj pyproj")
        require(record["source"]["git"] == record["git"], "VIMS source identity differs")
        safe_names(record["source"]["files"])
        inputs = {"catalog": record["inputs"]["catalog"]["sha256"]}
        for key in ("producer_manifest",):
            if key in record["inputs"]:
                inputs[key] = record["inputs"][key]["sha256"]
        source = {"scientific_source_sha256": record["source"]["files"]}
    else:
        environment = pick(record["environment"], "python numpy torch dependencies")
        inputs = {k: v["sha256"] for k, v in record["inputs"].items()}
        safe_names(inputs)
        source = {}
    selected = dict(manifest_sha256=expected_hash, producer_commit=COMMITS[family], status="complete",
                    environment=environment, inputs_sha256=inputs, outputs_sha256=record["outputs"], **source)
    base.check_public(selected)
    return selected


def check_aggregates(audit, series):
    require(set(audit["methods"]) == set(series), "Aggregate method coverage differs")
    for method, folds in series.items():
        for metric in (*base.METRICS, "kappa"):
            values = np.array([np.mean([r["kappa"] if metric == "kappa" else r["macro"][metric]
                                       for r in records]) for records in folds])
            record = audit["methods"][method][metric]
            close(record["fold_values"], values, "Aggregate fold values differ")
            close(record["mean"], values.mean(), "Aggregate mean differs")
            close(record["sample_std"], values.std(ddof=1), "Aggregate sample SD differs")


def verify_predictions(predictions, audits, folds):
    require(predictions["source_audit_sha256"] == AUDITS and predictions["class_names"] == list(CLASSES),
            "Prediction audit or class identity differs")
    require([r["fold"] for r in predictions["folds"]] == list(range(5)), "Prediction fold coverage differs")
    common = {(r["model"], r["fold"]): r for r in audits["common"]["folds"]}
    require(len(common) == len(audits["common"]["folds"]) == 20
            and set(common) == {(m, f) for m in MODELS for f in range(5)}, "Common fit coverage differs")
    require([r["fold"] for r in audits["classical"]["folds"]] == list(range(5)), "Classical fold coverage differs")
    series = {"common": {m: [] for m in MODELS},
              "classical": {m: [] for m in (*COLUMNS, "majority", "class_prior_random")}}
    for f, row in enumerate(predictions["folds"]):
        ids, labels = folds[f]["test_ids"], folds[f]["test_labels"]
        require(row["test_ids"] == ids and row["true_labels"] == labels, "Exported test membership or labels differ")
        require(set(row["common"]) == set(MODELS) and set(row["classical"]) == set(COLUMNS), "Prediction family coverage differs")
        for model in MODELS:
            series["common"][model].append([check_metrics(row["common"][model], labels, common[model, f])])
        classical = audits["classical"]["folds"][f]
        for method in COLUMNS:
            series["classical"][method].append([check_metrics(row["classical"][method], labels, classical["classical"][method])])
        controls = row["controls"]
        require(set(controls) == {"majority", "class_prior_random"}, "Control coverage differs")
        train_counts = np.bincount(folds[f]["train_labels"], minlength=6)
        require(controls["majority"] == [int(train_counts.argmax())] * len(ids), "Majority rule differs")
        series["classical"]["majority"].append([check_metrics(controls["majority"], labels, classical["controls"]["majority"])])
        prior = train_counts / train_counts.sum()
        draws, saved = controls["class_prior_random"], classical["controls"]["class_prior_random"]
        require([r["seed"] for r in draws] == [r["seed"] for r in saved] == list(range(5)), "Prior seed coverage differs")
        computed = []
        for draw, accepted in zip(draws, saved):
            expected = np.random.default_rng(draw["seed"]).choice(6, size=len(ids), p=prior).tolist()
            require(draw["predictions"] == expected, "Seeded class-prior predictions differ")
            computed.append(check_metrics(draw["predictions"], labels, accepted))
        series["classical"]["class_prior_random"].append(computed)
    for family in series:
        check_aggregates(audits[family], series[family])
    return sum(len(f["test_ids"]) for f in folds) * 14


def check_vims(sidecar, verification, catalog):
    require(sidecar["schema_version"] == "cetus.vims-display-metadata.v3", "VIMS schema differs")
    require(sidecar["catalog"] == verification["catalog"] and sidecar["catalog"]["sha256"] == base.CATALOG_SHA256,
            "VIMS original catalog binding differs")
    rows, tiles = sidecar["tiles"], catalog["tiles"]
    require(len(rows) == len(tiles) == verification["count"] == sidecar["catalog"]["count"], "VIMS tile coverage differs")
    available = 0
    for row, tile in zip(rows, tiles):
        require(pick(row, "tile_id center_lat center_lon") == pick(tile, "tile_id center_lat center_lon"),
                "VIMS tile identity or center differs")
        valid, value = row["has_vims_display_sample"], row["vims_display_sample"]
        require(type(valid) is bool, "VIMS availability must be boolean")
        if valid:
            a = np.asarray(value)
            require(a.shape == (3,) and a.dtype.kind in "iuf" and np.isfinite(a).all(), "Invalid available VIMS sample")
            available += 1
        else:
            require(value is None, "Unavailable VIMS value must be null")
    require(verification["status"] == "pass" and verification["availability_mismatches"] == 0
            and verification["value_mismatches"] == 0 and verification["maximum_absolute_difference"] == 0,
            "VIMS reconstruction was not accepted")
    return available


def verify_package(root, parent):
    root, parent = Path(root), Path(parent)
    base.verify_indexes(root)
    require(set(base.inventory(root)) == PAYLOADS | {"file_index.json", "SHA256SUMS"}, "Unexpected auxiliary payload")
    for name in PAYLOADS:
        base.check_public(read_json(root / name) if name.endswith((".json", ".json.gz")) else (root / name).read_text())
    catalog, folds, parent_sources = parent_data(parent)
    provenance = read_json(root / "provenance.json")
    require(provenance["parent_files_sha256"] == PARENT_FILES and provenance["producer_commits"] == COMMITS
            and provenance["original_catalog_sha256"] == base.CATALOG_SHA256, "Export provenance differs")
    audits = {k: pinned(root / RESULT_NAMES[k], AUDITS[k]) for k in AUDITS}
    for family, audit in audits.items():
        require(audit["status"] == "complete" and audit["producer_commit"] == COMMITS[family]
                and audit["protocol_sha256"] == PROTOCOLS[family][1], "Accepted family identity differs")
    protocols = {k: read_json(root / PROTOCOL_NAMES[k]) for k in PROTOCOLS}
    for name, expected_hash in EXTRACTS.items():
        pinned(root / name, expected_hash)
    check_protocols(protocols, folds, parent_sources)
    predictions = read_json(root / "results/predictions.json.gz")
    rows = verify_predictions(predictions, audits, folds)
    sidecar = pinned(root / "metadata/vims_display_v3.json", VIMS_FILES["producer/vims_metadata.json"])
    checked = pinned(root / "evidence/vims_verification_v3.json", VIMS_FILES["checker/verification.json"])
    available = check_vims(sidecar, checked, catalog)
    runs = read_json(root / "evidence/runs.json")
    expected = {(r["model"], r["fold"]): r["manifest_sha256"] for r in audits["common"]["folds"]}
    require([(r["model"], r["fold"]) for r in runs["common"]] == sorted(expected), "Run identity coverage differs")
    for family in ("common", "classical"):
        require(len(runs[family]) == (20 if family == "common" else 5), "Run coverage differs")
        if family == "classical":
            require([r["fold"] for r in runs[family]] == list(range(5)), "Classical run identities differ")
        for r in runs[family]:
            f = r["fold"]
            expected_sha = expected[r["model"], f] if family == "common" else audits[family]["folds"][f]["manifest_sha256"]
            require(r["manifest_sha256"] == expected_sha and r["producer_commit"] == COMMITS[family]
                    and r["status"] == "complete", "Run source or manifest binding differs")
            inputs = r["inputs_sha256"]
            require(inputs["catalog"] == base.CATALOG_SHA256 and inputs["protocol"] == PROTOCOLS[family][1]
                    and inputs["split"] == parent_sources[f"split_{f}"], "Run scientific inputs differ")
            versions = protocols[family]["specification"]["runtime_versions"]
            for name in ("numpy", "torch"):
                require(r["environment"][name] == versions[name], "Run runtime differs")
            for name, version in r["environment"]["dependencies"].items():
                if name in versions:
                    require(version == versions[name], "Run dependency runtime differs")
    require(set(runs["vims"]) == {"producer", "checker"}, "VIMS run coverage differs")
    for role, run in runs["vims"].items():
        require(run["manifest_sha256"] == VIMS_FILES[f"{role}/run_manifest.json"]
                and run["producer_commit"] == COMMITS["vims"] and run["status"] == "complete"
                and run["inputs_sha256"]["catalog"] == base.CATALOG_SHA256,
                "VIMS run identity differs")
    require(runs["vims"]["producer"]["outputs_sha256"]["vims_metadata.json"] == VIMS_FILES["producer/vims_metadata.json"]
            and runs["vims"]["checker"]["outputs_sha256"]["verification.json"] == VIMS_FILES["checker/verification.json"],
            "VIMS output binding differs")
    require(runs["vims"]["producer"]["environment"] == runs["vims"]["checker"]["environment"]
            and runs["vims"]["producer"]["scientific_source_sha256"] == runs["vims"]["checker"]["scientific_source_sha256"],
            "VIMS checker runtime or source differs")
    return dict(status="pass", catalog_tiles=len(catalog["tiles"]), distinct_test_tiles=rows // 14,
                prediction_vectors=70, prediction_rows=rows, common_fits=20, classical_fits=20,
                vims_available=available, vims_unavailable=len(catalog["tiles"]) - available,
                estimator_replay="upstream accepted audits only; not repeated by metadata export",
                raster_resampling="upstream accepted VIMS checker only; not repeated by metadata export")


README = """# Additional repaired Titan results and display metadata

This metadata package depends on the unchanged sibling
`release/longitude-review-20260929`. Parent hashes in `provenance.json` bind its
23,380 tile catalog, geographic protocol, five splits and frozen results.
The producer catalog has a different byte hash because the parent export
removed private paths and legacy inline splits. Tile values and order agree.
Both catalogs and queued replay inputs retain their original bytes.

The two `results` audit reports are exact accepted bytes. Common fits use
source 3dd19a2 and classical v2 uses 54281f0. Protocol extracts retain each
source and runtime identity, original protocol digests and selected scientific
fields. The original sealed protocols remain the authority for execution.
`evidence/runs.json` omits private paths, hosts, commands and logs. Its output
hashes also identify the estimator files held with the accepted run artifacts.

`results/predictions.json.gz` shares test IDs and labels once per fold. It holds
20 common classifier vectors, 20 classical vectors, five majority vectors and
25 training-prior draws. Common features have 768 columns. Classical feature
slices are 0:13, 13:45, 45:1865 and 0:1865. Both fitted families use training-only
StandardScaler and balanced LogisticRegression, LBFGS, C=1, max_iter=2000,
seed 0 and the fixed protocol settings. Warnings and iteration caps are retained
in the audits.
The raw_intensity_statistics key describes normalized display values;
histogram and texture additionally use the existing within-image percentile
transform. Physical backscatter calibration remains outside this track.

For a 6 by 6 confusion matrix, true classes index rows and predictions index
columns. Precision is TP/predicted support, recall is TP/true support, and F1
is 2 TP/(predicted support+true support). Undefined precision is zero. Average
over all six classes, then report the equal-fold mean and sample SD, ddof=1.
Average the five prior draws within a fold first. Kappa uses the confusion
marginals. Metrics are fractions. Fold SD describes variation across geographic
partitions. All predictions refer to expert map labels. Independent geological
accuracy remains unmeasured.

VIMS metadata v3 uses source a1d5652. The sidecar and checker report retain
exact accepted bytes and their original producer catalog hash. Tile IDs and
centers link them to the parent catalog. Values are bilinear samples of a
display composite. Calibrated spectral interpretation remains outside their
scope. Finite zero and signed values are allowed under the declared source
masks. Availability records successful numeric sampling. Observation coverage
and physical composition require separate evidence. The one unavailable sample
remains null. Source CRS, affine, masks and raster hashes are retained separately.
Classifier inputs are SAR features and reference-map labels. This sidecar
supplies ancillary display context. Existing catalog VIMS fields keep their
historical values.

The verifier comes from the TitanSAR development source:
`scripts/export_repaired_auxiliary_results.py`, with its helper
`scripts/export_repaired_release.py`. Their hashes appear in `provenance.json`.
Use a development checkout containing both files, with Python and NumPy
installed. Run this command from that checkout's root:

```bash
python3 -m scripts.export_repaired_auxiliary_results --verify-only PACKAGE --parent-package PARENT
```

PACKAGE is this auxiliary directory. PARENT is the existing repaired package
at `release/longitude-review-20260929`. Use absolute paths for both directories.
These two metadata directories supply the verifier's data inputs. The development
checkout supplies the program; the CETUS data package supplies the metadata.

Verification covers indexes, parent identities, memberships and prediction
arithmetic. The bundled audits record earlier accepted probability checks,
estimator replays and VIMS raster reconstruction. Fine-tuning and adaptation
results remain outside this release.
"""


def build_package(source_repo, catalog_path, parent, results_root, vims_root, common_audit, classical_audit, output):
    source_repo, catalog_path, parent, results_root, vims_root, output = map(Path,
        (source_repo, catalog_path, parent, results_root, vims_root, output))
    require(not output.exists() and not output.is_symlink(), "Output directory must be new")
    protected = [source_repo, parent.parent.parent, results_root, vims_root, Path(__file__).resolve().parents[1]]
    require(all(not output.resolve().is_relative_to(p.resolve()) and not p.resolve().is_relative_to(output.resolve())
                for p in protected), "Output overlaps a protected source or result tree")
    inputs = [catalog_path, Path(common_audit), Path(classical_audit),
              *(source_repo / "audit" / p[0] for p in PROTOCOLS.values()),
              *(vims_root / p for p in VIMS_FILES), *(parent / p for p in PARENT_FILES)]
    jobs = [("common", m, f, results_root / "repaired-common-classifier" / m / f"fold-{f}")
            for m in sorted(MODELS) for f in range(5)]
    jobs += [("classical", None, f, results_root / "repaired-baselines-v2" / f"fold-{f}") for f in range(5)]
    for family, _, _, folder in jobs:
        inputs += [folder / "run_manifest.json", folder / ("common_classifier_results.json" if family == "common" else "baseline_results.json")]
    missing = [str(p) for p in inputs if not p.is_file()]
    require(not missing, "Missing accepted evidence:\n" + "\n".join(missing))
    before = {p: sha256(p) for p in inputs}
    original = pinned(catalog_path, base.CATALOG_SHA256)
    catalog, folds, parent_sources = parent_data(parent)
    require(catalog["tiles"] == [pick(t, base.TILE_FIELDS) for t in original["tiles"]], "Native/public scientific catalog differs")
    audits = {"common": pinned(common_audit, AUDITS["common"]), "classical": pinned(classical_audit, AUDITS["classical"])}
    protocols = {k: protocol_extract(pinned(source_repo / "audit" / p[0], p[1]), k) for k, p in PROTOCOLS.items()}
    check_protocols(protocols, folds, parent_sources)
    common_rows = {(r["model"], r["fold"]): r for r in audits["common"]["folds"]}
    predictions = dict(schema_version="1.0.0", source_audit_sha256=AUDITS, class_names=list(CLASSES), folds=[
        {**pick(f, "fold test_ids"), "true_labels": f["test_labels"], "common": {}, "classical": {}, "controls": {}} for f in folds])
    runs = dict(common=[], classical=[], vims={})
    for family, model, f, folder in jobs:
        accepted = common_rows[model, f] if family == "common" else audits[family]["folds"][f]
        run = manifest_extract(folder / "run_manifest.json", family, accepted["manifest_sha256"])
        name = "common_classifier_results.json" if family == "common" else "baseline_results.json"
        result = pinned(folder / name, run["outputs_sha256"][name])
        bound = folds[f]
        require(result["train_ids"] == bound["train_ids"] and result["train_labels"] == bound["train_labels"], "Training identity differs")
        require(result["source_metadata"]["fold"] == f, "Result fold differs")
        ids, labels = bound["test_ids"], bound["test_labels"]
        row = predictions["folds"][f]
        if family == "common":
            require(result["source_metadata"]["model"] == model, "Result model differs")
            require(result["runtime_versions"] == protocols[family]["specification"]["runtime_versions"], "Common runtime differs")
            row["common"][model] = extract_prediction(result["classifier"], ids, labels, accepted)
            runs[family].append(dict(model=model, fold=f, **run))
        else:
            require(sha256(folder / name) == accepted["result_sha256"], "Classical result identity differs")
            require(set(result["classical"]) == set(COLUMNS), "Classical family coverage differs")
            for method in COLUMNS:
                row["classical"][method] = extract_prediction(result["classical"][method], ids, labels, accepted["classical"][method])
            controls = result["controls"]
            row["controls"]["majority"] = extract_prediction(controls["majority"], ids, labels, accepted["controls"]["majority"])
            require([r["seed"] for r in controls["class_prior_random"]] == list(range(5)), "Saved prior seeds differ")
            row["controls"]["class_prior_random"] = [dict(seed=seed, predictions=extract_prediction(r, ids, labels,
                accepted["controls"]["class_prior_random"][seed], "categorical_draw")) for seed, r in enumerate(controls["class_prior_random"])]
            runs[family].append(dict(fold=f, **run))
    verify_predictions(predictions, audits, folds)
    vims = {p: pinned(vims_root / p, value) for p, value in VIMS_FILES.items()}
    check_vims(vims["producer/vims_metadata.json"], vims["checker/verification.json"], catalog)
    for role in ("producer", "checker"):
        p = f"{role}/run_manifest.json"
        runs["vims"][role] = manifest_extract(vims_root / p, "vims", VIMS_FILES[p])
    raster = vims["producer/source_raster.json"]
    source_raster = pick(raster, "affine bounds coordinate_conversion crs_wkt dtypes mask_flags masks nodata_values shape")
    source_raster["files"] = {Path(k).name: v for k, v in raster["files"].items()}
    require(len(source_raster["files"]) == len(raster["files"]), "Duplicate raster basenames")
    safe_names(source_raster["files"])
    provenance = dict(schema_version="1.0.0", bundle_type="accepted_auxiliary_scientific_metadata",
        parent_release="longitude-review-20260929", parent_files_sha256=PARENT_FILES,
        original_catalog_sha256=base.CATALOG_SHA256, producer_commits=COMMITS,
        accepted_audit_sha256=AUDITS, original_vims_files_sha256=VIMS_FILES,
        exporter_source_sha256={"scripts/export_repaired_auxiliary_results.py": sha256(__file__),
                               "scripts/export_repaired_release.py": sha256(base.__file__)},
        transformations=dict(audits="Exact accepted bytes", predictions="Exact saved predicted labels; share fold IDs and omit probabilities",
                             protocols="Fixed scientific projection; original seals retained separately",
                             vims="Exact sidecar/checker; separate raster and manifest projections omit machine paths"),
        classifier_uses_vims_sidecar=False, catalog_modified=False, adaptation_included=False)
    documents = {"provenance.json": provenance, "results/predictions.json.gz": predictions,
                 "evidence/runs.json": runs, "metadata/vims_source_v3.json": source_raster,
                 **{PROTOCOL_NAMES[k]: v for k, v in protocols.items()}}
    for value in documents.values():
        base.check_public(value)
    exact = {RESULT_NAMES["common"]: Path(common_audit), RESULT_NAMES["classical"]: Path(classical_audit),
             "metadata/vims_display_v3.json": vims_root / "producer/vims_metadata.json",
             "evidence/vims_verification_v3.json": vims_root / "checker/verification.json"}
    for path in exact.values():
        base.check_public(read_json(path))
    require(all(sha256(p) == value for p, value in before.items()), "Input changed during export preparation")
    output.mkdir(parents=True)
    (output / ".INCOMPLETE").write_text("Verification pending\n")
    for name, value in documents.items():
        base.write_json(output, name, value)
    for name, path in exact.items():
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        (output / name).write_bytes(path.read_bytes())
    (output / "README.md").write_text(README)
    try:
        (output / ".INCOMPLETE").unlink()
        base.write_indexes(output, dict(schema_version="1.0.0", bundle_type=provenance["bundle_type"]))
        verification = verify_package(output, parent)
        require(all(sha256(p) == value for p, value in before.items()), "Input changed during export")
    except Exception:
        (output / ".INCOMPLETE").write_text("Verification failed\n")
        raise
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-package", type=Path, required=True)
    parser.add_argument("--verify-only", type=Path)
    for flag in ("source-repo", "catalog", "results-root", "vims-root", "common-audit", "classical-audit", "output"):
        parser.add_argument("--" + flag, type=Path)
    args = parser.parse_args()
    if args.verify_only:
        result = verify_package(args.verify_only, args.parent_package)
    else:
        required = (args.source_repo, args.catalog, args.results_root, args.vims_root, args.common_audit, args.classical_audit, args.output)
        if any(p is None for p in required):
            parser.error("All input paths and --output are required for export")
        result = build_package(args.source_repo, args.catalog, args.parent_package, args.results_root, args.vims_root,
                               args.common_audit, args.classical_audit, args.output)
    print(base.encoded(result).decode(), end="")


if __name__ == "__main__":
    main()
