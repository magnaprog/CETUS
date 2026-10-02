"""Stage accepted adaptation/control evidence; never run models or discover jobs.

--inputs is a reviewer-supplied JSON binding manifest; --inputs-sha256 must come
from its independent acceptance. All run paths and new audit hashes are explicit.
Only the finite small-JSON inventory is read. Native binary artifacts are not loaded.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scripts import export_repaired_release as base
from scripts import export_repaired_auxiliary_results as auxiliary
from scripts import verify_repaired_adaptation_release as public

ROOT = Path(__file__).resolve().parents[1]
IMPORT_SHA256 = base.sha256(__file__)
require, exact, pick = base.require, public.exact, base.pick
analysis = public.adaptation
INPUTS = {
    *analysis.PINS,
    "ft_audit",
    "replay_audit",
    "buffer_sgd_audit",
    "buffer_lr_audit",
    "buffer_sgd_protocol",
    "buffer_lr_protocol",
    "full_gate",
    "analysis",
    "display_a",
    "display_b",
    "ft_plan",
    "followup_plan",
    "replay_plan",
    "buffer_lr_plan",
}
FIXED = {
    **analysis.PINS,
    "buffer_sgd_protocol": public.display_b.PINS["buffer_protocol"],
    "ft_plan": analysis.FT_PLAN_SHA,
    "followup_plan": analysis.FOLLOWUP_PLAN_SHA,
    "replay_plan": analysis.REPLAY_PLAN_SHA,
}


def read_bound(path, expected, tracked):
    path = Path(path).absolute()
    public.hash_value(expected)
    require(
        path.suffix == ".json"
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "Only explicit nonsymlink JSON inputs",
    )
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == expected,
        "Accepted input SHA differs: " + path.name,
    )

    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, "Duplicate JSON key")
            out[k] = v
        return out

    value = json.loads(raw, object_pairs_hook=pairs)
    base.encoded(value)
    if path in tracked:
        exact(tracked[path], expected, "Conflicting input binding")
    tracked[path] = expected
    return value


def bindings(records):
    result = {}
    for name, r in records.items():
        require(
            set(r) in ({"path", "sha256"}, {"path", "files"}),
            "Input binding schema differs",
        )
        if "sha256" in r:
            result[name] = public.hash_value(r["sha256"])
        else:
            auxiliary.safe_names(r["files"])
            result[name] = {n: public.hash_value(h) for n, h in r["files"].items()}
    return result


def environment(row):
    allowed = (
        "python",
        "numpy",
        "torch",
        "cuda",
        "cuda_device_names",
        "cudnn",
        "dependencies",
        "gpu_name",
        "packages",
        "torch_version",
        "capability",
        "deterministic",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_allow_tf32",
        "matmul_allow_tf32",
        "float32_matmul_precision",
        "autocast_cuda_dtype",
        "cublas_workspace_config",
        "autocast",
        "dtype",
        "batch_size",
        "torch_threads",
        "torch_interop_threads",
        "cpu_intraop_threads",
        "cpu_interop_threads",
    )
    out = {k: copy.deepcopy(row[k]) for k in allowed if k in row}
    if "declared_environment" in row:
        out["declared_environment"] = {
            k: v
            for k, v in row["declared_environment"].items()
            if k
            in (
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        }
    public.privacy(out)
    return out


def cpu_numerics(row):
    out = pick(
        row,
        "environment torch_num_threads torch_num_interop_threads deterministic_algorithms",
    )
    out["threadpools"] = [
        {
            k: v
            for k, v in p.items()
            if k
            in (
                "user_api",
                "internal_api",
                "num_threads",
                "prefix",
                "version",
                "threading_layer",
                "architecture",
            )
        }
        for p in row["threadpools"]
    ]
    public.privacy(out)
    return out


def project_warning_paths(value, changes, location=()):
    """Only path tokens in retained warning messages receive substitution."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "fit_warnings":
                for i, warning in enumerate(child):
                    message = warning["message"]
                    clean = re.sub(
                        (
                            "(?<![A-Za-z0-9:/])/(?:home|tmp|mnt|data|scratch|"
                            "Users|opt|workspace|root)/[^\\s'\\\";,)]+"
                        ),
                        "[private path]",
                        message,
                    )
                    clean = re.sub(r"[A-Za-z]:\\[^\s'\";,)]+", "[private path]", clean)
                    if clean != message:
                        changes.append(
                            dict(
                                location=list(location + (key, i)),
                                original_warning_sha256=base.digest(warning),
                                transformation="Replace operational path token in warning message.",
                            )
                        )
                        warning["message"] = clean
            else:
                project_warning_paths(child, changes, location + (key,))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            project_warning_paths(child, changes, location + (i,))


def protocol_extract(record, family, original_sha):
    s = record["specification"]
    exact(base.digest(s), record["specification_sha256"], "Native specification digest")
    if family == "adaptation":
        fields = (
            "acceptance benchmark_track catalog_sha256 checkpoint_policy "
            "class_names encoders folds frozen_jobs frozen_source_commit "
            "geographic_protocol_file_sha256 "
            "geographic_protocol_specification_sha256 head_state_digest_format "
            "limitations normalization_policy optimization_seeds recipe "
            "reference_policy reporting scientific_source_sha256 scope "
            "verified_frozen_audit_sha256"
        )
        selected = pick(s, fields)
    elif family == "checkpoint_replay":
        selected = pick(
            s,
            (
                "acceptance auditor_commit expected_jobs inference "
                "probability_tolerance_status producer_commit producer_plan_sha256 "
                "replay_source_sha256 "
                "selection training_protocol_sha256 "
                "validation_historical_prediction_vector_available "
                "verified_frozen_audit_sha256"
            ),
        )
        selected["execution"] = pick(s["execution"], "thread_environment")
        # Native 'path' names computational steps, not a filesystem location.
        for role in ("test", "validation"):
            selected["inference"][role]["computation"] = selected["inference"][
                role
            ].pop("path")
    elif family == "buffer_sgd":
        selected = pick(
            s,
            (
                "benchmark_track catalog_sha256 class_order dofa_identifier_ghz "
                "draw_seeds encoders experiment feature_batch_size folds "
                "frozen_audit_sha256 frozen_jobs frozen_source_commit models "
                "native_spatial_protocol_sha256 normalization overlap_check "
                "probe_seeds random_encoder_seed recipe sampler scope source_sha256 "
                "training_order version"
            ),
        )
        selected["native_cpu_recipe"] = {
            **pick(s["native_cpu_recipe"], "plan_sha256 environment evidence"),
            "job_ids": [j["id"] for j in s["native_cpu_recipe"]["jobs"]],
        }
    else:
        selected = pick(
            s,
            (
                "experiment schema_version buffer_producer_commit recipe draws fits "
                "scientific_source_sha256 runtime_versions runtime_import_versions "
                "python_version resolved_parameters"
            ),
        )
        selected.update(
            parents={k: pick(v, "sha256") for k, v in s["parents"].items()},
            adopted_plan=pick(s["adopted_plan"], "sha256"),
            catalog=pick(s["catalog"], "sha256"),
            jobs=[],
        )
        for j in s["jobs"]:
            row = pick(j, "model fold")
            row["cache"] = pick(
                j["cache"], "model fold manifest_sha256 encoder_identity files_sha256"
            )
            row["reference"] = {
                k: copy.deepcopy(j["reference"][k])
                for k in ("manifest_sha256", "outputs")
                if k in j["reference"]
            }
            row["buffer_fit"] = pick(j["buffer_fit"], "manifest_sha256")
            row["boundary"] = pick(
                j["boundary"], "producer_commit manifest_sha256 outputs"
            )
            selected["jobs"].append(row)
    public.privacy(selected)
    return dict(
        original_protocol_sha256=original_sha,
        original_specification_sha256=record["specification_sha256"],
        specification=selected,
        extract_specification_sha256=base.digest(selected),
    )


def prediction(record, ids, labels, metrics):
    exact(record["tile_ids"], ids, "Native prediction ID order")
    exact(record.get("true_labels", record.get("labels")), labels, "Native truth order")
    public.score(record["predictions"], labels, metrics)
    p = np.asarray(record["probabilities"])
    require(
        p.shape == (len(ids), 6)
        and p.dtype.kind in "iuf"
        and np.isfinite(p).all()
        and ((p >= 0) & (p <= 1)).all()
        and np.allclose(p.sum(1, dtype=np.float64), 1, rtol=0, atol=1e-6),
        "Invalid probabilities",
    )
    require(
        np.array_equal(p[np.arange(len(ids)), record["predictions"]], p.max(1)),
        "Prediction max rule",
    )
    counts = np.asarray(record["confusion_matrix"])
    require(
        counts.shape == (6, 6)
        and counts.dtype.kind in "iuf"
        and all(
            type(value) in (int, float)
            for row in record["confusion_matrix"]
            for value in row
        )
        and np.array_equal(counts, metrics["confusion_counts"]),
        "Native confusion counts",
    )
    return copy.deepcopy(record["predictions"])


def check_plans(raw, commits, jobs):
    plans = [
        ("ft_plan", 75, commits["adaptation"]),
        ("followup_plan", 115, commits["buffer_sgd"]),
        ("replay_plan", 75, commits["checkpoint_replay"]),
        ("buffer_lr_plan", 20, commits["buffer_lr"]),
    ]
    by_root = {}
    plan_for_root = {}
    for name, count, commit in plans:
        plan = raw[name]
        exact(plan["code_commit"], commit, "Plan producer differs")
        workers = plan["workers"]
        require(isinstance(workers, dict) and workers, "Missing workers")
        if name == "buffer_lr_plan":
            require(1 <= len(workers) <= 4, "Common buffer worker count")
            for worker in workers.values():
                for k in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    require(worker["environment"][k] == "1", "LR numerical threads")
                require(
                    worker["environment"]["CUDA_VISIBLE_DEVICES"] == "",
                    "LR CPU environment",
                )
        rows = [j for w in workers.values() for j in w["jobs"]]
        require(
            len(rows) == count and len({j["output_dir"] for j in rows}) == count,
            "Plan job count/output uniqueness",
        )
        for j in rows:
            require(j["output_dir"] not in by_root, "Output shared across plans")
            by_root[j["output_dir"]] = j
            plan_for_root[j["output_dir"]] = name
    # Preserve the old replay slots as history; accepted runs use the new plan.
    require(len(by_root) == 285, "Plan slot inventory")
    plan_for_family = dict(
        adaptation="ft_plan",
        checkpoint_replay="replay_plan",
        boundary="followup_plan",
        buffer_sgd="followup_plan",
        buffer_lr="buffer_lr_plan",
    )
    for family, rows in jobs.items():
        for row in rows:
            root = str(Path(row["manifest"]["path"]).parent)
            require(root in by_root, "Manifest outside accepted plans")
            exact(plan_for_root[root], plan_for_family[family], "Wrong family plan")
            command = by_root[root]["command"]
            require(len(command) > 3 and command[1] == "-m", "Plan module command")
            expected_module = {
                "adaptation": "scripts.run_finetuning",
                "checkpoint_replay": "scripts.replay_repaired_checkpoints",
                "boundary": "scripts.run_repaired_buffer_control",
                "buffer_sgd": "scripts.run_repaired_buffer_control",
                "buffer_lr": "scripts.run_repaired_common_buffer",
            }[family]
            require(command[2] == expected_module, "Plan family command")
            flags = (
                (("--models", "model"), ("--seeds", "seed"))
                if family == "adaptation"
                else (("--model", "model"), ("--fold", "fold"), ("--seed", "seed"))
            )
            for flag, key in flags:
                if key in row:
                    require(
                        analysis.common.option(command, flag) == [str(row[key])],
                        "Plan identity flags",
                    )
            if family in ("boundary", "buffer_sgd", "buffer_lr"):
                require(
                    command[3] == ("extract" if family == "boundary" else "fit"),
                    "Buffer plan phase",
                )
    return by_root


def recovery_jobs(gate, pins):
    exact(
        [gate["schema_version"], gate["status"]],
        [public.RECOVERY_SCHEMA, "pass"],
        "Recovery gate schema/status",
    )
    exact(gate["counts"], public.RECOVERY_COUNTS, "Recovery gate counts")
    for family, name in (
        ("finetuning", "ft_audit"),
        ("buffer", "buffer_sgd_audit"),
        ("replay", "replay_audit"),
    ):
        exact(gate["reports"][family]["sha256"], pins[name], "Recovery gate report")
    rows = gate["accepted_jobs"]
    families = dict(
        finetuning="adaptation",
        boundary="boundary",
        buffer="buffer_sgd",
        replay="checkpoint_replay",
    )
    require(len(rows) == 190, "Recovery accepted job count")
    require({row["family"] for row in rows} == set(families), "Recovery families")
    indexed = {}
    for family, exported_family in families.items():
        fields = (
            ("model", "fold", "seed")
            if family in ("finetuning", "replay")
            else ("model", "fold")
        )
        selected = []
        for row in rows:
            if row["family"] == family:
                require(len(row["identity"]) == len(fields), "Recovery identity")
                selected.append({**row, **dict(zip(fields, row["identity"]))})
        indexed[exported_family] = public.grid(
            selected, fields, public.TRIPLES if len(fields) == 3 else public.PAIRS
        )
    return indexed


def collect(inputs_path, inputs_sha256, frozen, auxiliary_root):
    tracked = {Path(__file__): IMPORT_SHA256}
    manifest = read_bound(inputs_path, inputs_sha256, tracked)
    require(
        manifest["schema_version"] == "1.0.0"
        and set(manifest)
        == {
            "schema_version",
            "inputs",
            "jobs",
            "processes",
            "common_buffer_producer_commit",
        },
        "Export input manifest schema",
    )
    require(set(manifest["inputs"]) == INPUTS, "Accepted input inventory")
    for name, h in FIXED.items():
        exact(manifest["inputs"][name]["sha256"], h, "Fixed accepted input " + name)
    # Parent packages are fixed production inputs. Synthetic tests patch pins,
    # never a production flag permitting unrelated parents.
    base.verify_package(Path(frozen))
    auxiliary.verify_package(Path(auxiliary_root), Path(frozen))
    raw = {
        k: read_bound(r["path"], r["sha256"], tracked)
        for k, r in manifest["inputs"].items()
    }
    pins = {k: v["sha256"] for k, v in manifest["inputs"].items()}
    for name in ("ft_audit", "replay_audit"):
        require(
            raw[name]["schema_version"] == "1.0.0", "Accepted report schema: " + name
        )
    analysis.accepted_inputs(
        raw
    )  # No arrays/models; preserves native source/plan/reference joins.
    measured = analysis.analyze(raw)
    for key in measured:
        exact(
            raw["analysis"][key],
            measured[key],
            "Accepted analysis scientific payload " + key,
        )
    exact(
        raw["analysis"]["analysis_source_sha256"][
            "scripts/analyze_repaired_adaptation.py"
        ],
        public.HELPERS["scripts/analyze_repaired_adaptation.py"],
        "Analysis source",
    )
    for name, r in raw["analysis"]["inputs"].items():
        exact(r["sha256"], pins[name], "Analysis input")
    lr = raw["buffer_lr_audit"]
    sgd = raw["buffer_sgd_audit"]
    lrp = raw["buffer_lr_protocol"]
    sgdp = raw["buffer_sgd_protocol"]
    commits = dict(
        adaptation=analysis.FT_COMMIT,
        checkpoint_replay=analysis.REPLAY_COMMIT,
        boundary=public.CONTROL_COMMIT,
        buffer_sgd=public.CONTROL_COMMIT,
        buffer_lr=manifest["common_buffer_producer_commit"],
    )
    public.hash_value(commits["buffer_lr"], 40)
    require(
        lr["status"] == "complete" and lr["schema_version"] == "1.0.0",
        "LR incomplete/schema",
    )
    exact(
        lr["validation"],
        dict(complete_jobs=20, estimator_replays=100),
        "LR acceptance count",
    )
    exact(lr["producer_commit"], commits["buffer_lr"], "LR future producer")
    for k, expected in [
        ("protocol_sha256", pins["buffer_lr_protocol"]),
        ("plan_sha256", pins["buffer_lr_plan"]),
        ("specification_sha256", lrp["specification_sha256"]),
    ]:
        exact(lr[k], expected, "LR seal " + k)
    exact(lrp["specification"]["parents"], lr["parents"], "LR parent records")
    expected_parents = {
        "common_protocol": pins["common_protocol"],
        "common_audit": pins["common_audit"],
        "common_plan": analysis.joint.PLANS["common"],
        "buffer_protocol": pins["buffer_sgd_protocol"],
        "buffer_audit": pins["buffer_sgd_audit"],
        "finetuning_audit": pins["ft_audit"],
    }
    exact(
        {k: v["sha256"] for k, v in lr["parents"].items()},
        expected_parents,
        "LR six parents",
    )
    exact(
        lr["auditor_sha256"],
        lrp["specification"]["scientific_source_sha256"][
            "scripts/run_repaired_common_buffer.py"
        ],
        "LR auditor source seal",
    )
    exact(
        lrp["specification"]["resolved_parameters"],
        raw["common_protocol"]["specification"]["resolved_parameters"],
        "LR recipe",
    )
    for key in ("runtime_versions", "runtime_import_versions", "python_version"):
        exact(
            lrp["specification"][key],
            raw["common_protocol"]["specification"][key],
            "LR runtime " + key,
        )
    require(
        sgd["status"] == "complete"
        and sgd["protocol_sha256"] == pins["buffer_sgd_protocol"]
        and sgd["specification_sha256"] == sgdp["specification_sha256"],
        "SGD acceptance",
    )
    gate = raw["full_gate"]
    recovered = recovery_jobs(gate, pins)
    jobs = manifest["jobs"]
    require(set(jobs) == set(commits), "Execution family inventory")
    for family, rows in jobs.items():
        public.grid(
            rows,
            ("model", "fold", "seed")
            if family in ("adaptation", "checkpoint_replay")
            else ("model", "fold"),
            public.TRIPLES
            if family in ("adaptation", "checkpoint_replay")
            else public.PAIRS,
        )
    plan_jobs = check_plans(raw, commits, jobs)
    catalog = base.read_json(Path(frozen) / "catalogs/titan_catalog.json")
    labels = {t["tile_id"]: t["label"] for t in catalog["tiles"]}
    ids = list(labels)
    members = []
    for f in sgdp["specification"]["folds"]:
        row = copy.deepcopy(f)
        row.update(
            validation_labels=[labels[t] for t in row["validation_ids"]],
            test_labels=[labels[t] for t in row["test_ids"]],
        )
        for d in row["draws"]:
            selected = (set(row["retained_train_ids"]) - set(d["removed_ids"])) | set(
                row["boundary_ids"]
            )
            d["training_ids"] = [t for t in ids if t in selected]
            d["train_class_counts"] = copy.deepcopy(row["class_counts"])
        members.append(row)
    out = {"membership.json.gz": dict(schema_version="1.0.0", folds=members)}
    for family, name in [
        ("adaptation", "ft_protocol"),
        ("checkpoint_replay", "replay_protocol"),
        ("buffer_sgd", "buffer_sgd_protocol"),
        ("buffer_lr", "buffer_lr_protocol"),
    ]:
        out[f"protocols/{family}.json"] = protocol_extract(
            raw[name], family, pins[name]
        )
    out["results/adaptation.json"] = copy.deepcopy(raw["analysis"])
    out["results/adaptation.json"]["inputs"] = {
        k: pick(v, "sha256") for k, v in raw["analysis"]["inputs"].items()
    }
    out["results/buffer_sgd.json"] = {
        **pick(
            sgd,
            "status protocol_sha256 specification_sha256 models input_fit_manifests",
        ),
        "folds": [],
    }
    out["results/buffer_lr.json"] = {
        **pick(
            lr,
            (
                "status schema_version producer_commit protocol_sha256 "
                "specification_sha256 plan_sha256 auditor_sha256 validation folds "
                "models aggregation limitations"
            ),
        ),
        "parents": {k: pick(v, "sha256") for k, v in lr["parents"].items()},
    }
    runs = {k: [] for k in commits}
    predictions = dict(schema_version="1.0.0", test=[], validation=[])
    histories = []
    boundary = []
    ft_probabilities = {}
    accepted_indices = {
        "adaptation": public.grid(
            raw["ft_audit"]["records"], ("model", "fold", "seed"), public.TRIPLES
        ),
        "checkpoint_replay": public.grid(
            raw["replay_audit"]["jobs"], ("model", "fold", "seed"), public.TRIPLES
        ),
        "buffer_sgd": public.grid(
            sgd["input_fit_manifests"], ("model", "fold"), public.PAIRS
        ),
        "buffer_lr": public.grid(lr["folds"], ("model", "fold"), public.PAIRS),
        "boundary": {
            (r["model"], r["fold"]): r["boundary"] for r in lrp["specification"]["jobs"]
        },
    }
    for family in commits:
        fields = (
            ("model", "fold", "seed")
            if family in ("adaptation", "checkpoint_replay")
            else ("model", "fold")
        )
        for job in sorted(jobs[family], key=lambda r: tuple(r[k] for k in fields)):
            key = tuple(job[k] for k in fields)
            identity = {k: job[k] for k in fields}
            accepted = accepted_indices[family][key]
            expected = (
                accepted["replay_manifest_sha256"]
                if family == "checkpoint_replay"
                else accepted["manifest_sha256"]
            )
            exact(job["manifest"]["sha256"], expected, "Accepted manifest identity")
            mfpath = Path(job["manifest"]["path"])
            mf = read_bound(mfpath, expected, tracked)
            require(
                mf["status"] == "complete"
                and mf["git"] == dict(commit=commits[family], clean=True, status=[]),
                "Native manifest source/state",
            )
            command = plan_jobs[str(mfpath.parent)]["command"]
            require(
                mf["command"][1:] == command[3:]
                and Path(mf["command"][0]).name == command[2].split(".")[-1] + ".py",
                "Manifest/worker command",
            )
            auxiliary.safe_names(mf["outputs"])
            if family != "buffer_lr":
                joined = recovered[family][key]
                exact(
                    [
                        joined["manifest_sha256"],
                        joined["outputs"],
                        joined["output_dir"],
                    ],
                    [expected, mf["outputs"], str(mfpath.parent)],
                    "Recovery gate accepted manifest/output",
                )

            def load(name):
                return read_bound(mfpath.parent / name, mf["outputs"][name], tracked)

            m, f = key[:2]
            mem = members[f]
            role_ids = mem["test_ids"]
            truth = mem["test_labels"]
            run = dict(
                **identity,
                manifest_sha256=expected,
                producer_commit=commits[family],
                status="complete",
                outputs_sha256=copy.deepcopy(mf["outputs"]),
            )
            if family != "checkpoint_replay":
                run.update(
                    inputs_sha256=bindings(mf["inputs"]),
                    environment=environment(mf["environment"]),
                )
            if family == "adaptation":
                result = load("finetuning_results.json")
                require(set(result) == {m}, "FT result model")
                block = result[m]["configurations"]
                require(set(block) == {"frozen", "ft_2block"}, "FT configurations")
                require(
                    len(block["ft_2block"]["runs"])
                    == len(block["frozen"]["runs"])
                    == 1,
                    "FT one seed per job",
                )
                record = block["ft_2block"]["runs"][0]
                require(record["seed"] == key[2], "FT seed")
                v = prediction(record, role_ids, truth, accepted["finetuned"])
                ft_probabilities[key] = np.asarray(record["probabilities"])
                predictions["test"].append(
                    dict(
                        family=family,
                        **identity,
                        predictions=v,
                        source_result_sha256=mf["outputs"]["finetuning_results.json"],
                    )
                )
                histories.append(
                    dict(
                        **identity,
                        history=pick(
                            record["history"],
                            (
                                "train_loss val_balanced_accuracy optimizer_steps "
                                "amp_skipped_steps loss_scale"
                            ),
                        ),
                        source_result_sha256=mf["outputs"]["finetuning_results.json"],
                    )
                )
                exact(record["provenance"], accepted["provenance"], "FT provenance")
                run["scientific_provenance"] = pick(
                    accepted["provenance"],
                    (
                        "titan_catalog_sha256 earth_catalog_sha256 "
                        "titan_split_manifest_sha256 earth_split_manifest_sha256 "
                        "evaluation_scope benchmark_track normalization_provenance "
                        "model_revision weights_sha256 fold seed recipe class_counts "
                        "protocol_sha256 protocol_specification_sha256 frozen_reference "
                        "dofa_band_identifier_ghz head_initialization_seed "
                        "initial_head_state_sha256 state_digest_format unfrozen_blocks "
                        "unfrozen_parameter_names trainable_backbone_parameters "
                        "trainable_head_parameters total_backbone_parameters "
                        "training_class_weights selected_epoch epoch_indexing "
                        "optimizer_attempts_per_epoch optimizer_steps_total "
                        "amp_skipped_steps_total training_example_presentations"
                    ),
                )
            elif family == "checkpoint_replay":
                r = load("replay.json")
                require(r["status"] == "pass", "Replay status")
                exact(r["comparisons"], accepted["comparisons"], "Replay comparisons")
                exact(
                    r["inputs"]["producer_manifest_sha256"],
                    accepted["producer_manifest_sha256"],
                    "Replay FT manifest",
                )
                run.update(
                    inputs_sha256={
                        "replay_protocol": r["inputs"]["replay_protocol_sha256"],
                        "producer_manifest": r["inputs"]["producer_manifest_sha256"],
                        "frozen_audit": r["inputs"]["frozen_audit_sha256"],
                    },
                    environment=environment(r["environment"]),
                    input_binding_origin="manifest-bound replay.json",
                )
                public.check_replay_environment(
                    run["environment"], raw["replay_protocol"]["specification"]
                )
                exact(
                    r["replay_source_sha256"],
                    raw["replay_protocol"]["specification"]["replay_source_sha256"],
                    "Replay source",
                )
                # Raw replay roles contain evaluation vectors and metrics.
                for role, ri, rt, metric in [
                    ("test", role_ids, truth, accepted["test_metrics"]),
                    (
                        "validation",
                        mem["validation_ids"],
                        mem["validation_labels"],
                        accepted["validation_metrics"],
                    ),
                ]:
                    record = r[role]
                    v = prediction(record, ri, rt, metric)
                    if role == "validation":
                        predictions["validation"].append(
                            dict(
                                **identity,
                                predictions=v,
                                source_result_sha256=mf["outputs"]["replay.json"],
                            )
                        )
                    else:
                        ftrow = next(
                            x
                            for x in predictions["test"]
                            if x["family"] == "adaptation"
                            and all(x[k] == job[k] for k in fields)
                        )
                        exact(
                            v, ftrow["predictions"], "Checkpoint/test predicted classes"
                        )
                        errors = np.abs(
                            np.asarray(record["probabilities"])
                            - ft_probabilities.pop(key)
                        )
                        maximum = float(errors.max())
                        over_tolerance = int((errors > 1e-6).sum())
                        require(
                            maximum <= 1e-6 and over_tolerance == 0,
                            "Checkpoint probability tolerance exceeded",
                        )
                        exact(
                            maximum,
                            accepted["comparisons"]["test"][
                                "probability_max_abs_error"
                            ],
                            "Checkpoint probability maximum differs",
                        )
                        exact(
                            over_tolerance,
                            accepted["comparisons"]["test"][
                                "probability_components_over_atol"
                            ],
                            "Checkpoint probability component count differs",
                        )
            elif family == "buffer_sgd":
                r = load("buffer_results.json")
                exact([r["model"], r["fold"]], list(key), "SGD result identity")
                selected = {
                    **pick(
                        r,
                        (
                            "model fold recipe class_counts training_class_weights optimization "
                            "reference_metrics runs"
                        ),
                    ),
                    "manifest_sha256": expected,
                    "cpu_numerics": cpu_numerics(r["cpu_numerics"]),
                }
                out["results/buffer_sgd.json"]["folds"].append(selected)
                rr = public.grid(
                    r["runs"],
                    ("draw_seed", "seed"),
                    {(d, s) for d in range(5) for s in range(5)},
                )
                for d in range(5):
                    for s in range(5):
                        name = f"draw-{d}/seed-{s}.json"
                        record = load(name)
                        exact(
                            [record["draw_seed"], record["seed"]],
                            [d, s],
                            "SGD head identity",
                        )
                        exact(
                            record["training_ids_sha256"],
                            mem["draws"][d]["training_ids_sha256"],
                            "SGD membership digest",
                        )
                        exact(
                            record["normalization_fit_training_ids_sha256"],
                            mem["normalization_provenance"]["training_ids_sha256"],
                            "SGD image normalization membership",
                        )
                        v = prediction(record, role_ids, truth, rr[d, s]["metrics"])
                        predictions["test"].append(
                            dict(
                                family=family,
                                **identity,
                                draw_seed=d,
                                seed=s,
                                predictions=v,
                                source_result_sha256=mf["outputs"][name],
                            )
                        )
            elif family == "buffer_lr":
                r = load("common_buffer_results.json")
                exact([r["model"], r["fold"]], list(key), "LR result identity")
                exact(
                    r["protocol_sha256"],
                    pins["buffer_lr_protocol"],
                    "LR result protocol",
                )
                require(
                    [x["draw_seed"] for x in r["draws"]] == list(range(5)),
                    "LR draw order",
                )
                runtime = r["runtime"]
                run["runtime"] = pick(
                    runtime,
                    (
                        "environment runtime_versions python_version torch_threads "
                        "torch_interop_threads cuda_initialized"
                    ),
                )
                run["runtime"]["threadpools"] = [
                    {
                        k: v
                        for k, v in p.items()
                        if k
                        in (
                            "user_api",
                            "internal_api",
                            "num_threads",
                            "prefix",
                            "version",
                            "threading_layer",
                            "architecture",
                        )
                    }
                    for p in runtime["threadpools"]
                ]
                exact(
                    runtime["runtime_versions"],
                    lrp["specification"]["runtime_versions"],
                    "LR recorded runtime",
                )
                require(
                    type(runtime["torch_threads"]) is int
                    and runtime["torch_threads"] == 1
                    and type(runtime["torch_interop_threads"]) is int
                    and runtime["torch_interop_threads"] == 1
                    and runtime["cuda_initialized"] is False,
                    "LR recorded CPU threads",
                )
                require(
                    runtime["threadpools"]
                    and all(
                        type(p["num_threads"]) is int and p["num_threads"] == 1
                        for p in runtime["threadpools"]
                    ),
                    "LR recorded threadpools",
                )
                for d, record in enumerate(r["draws"]):
                    exact(
                        record["train_ids"],
                        mem["draws"][d]["training_ids"],
                        "LR raw training IDs",
                    )
                    exact(
                        record["train_labels"],
                        [labels[t] for t in record["train_ids"]],
                        "LR raw training labels",
                    )
                    checked = accepted["draws"][d]
                    c = record["classifier"]
                    exact(
                        pick(c, " ".join(checked["fit"])),
                        checked["fit"],
                        "LR preserved warnings/fit state",
                    )
                    v = auxiliary.extract_prediction(
                        c, role_ids, truth, checked["metrics"]
                    )
                    predictions["test"].append(
                        dict(
                            family=family,
                            **identity,
                            draw_seed=d,
                            predictions=v,
                            source_result_sha256=mf["outputs"][
                                "common_buffer_results.json"
                            ],
                        )
                    )
            else:
                metadata = load(f"features/{m}/titan_boundary_feats.meta.json")
                meta = {
                    k: copy.deepcopy(v)
                    for k, v in metadata.items()
                    if k
                    in (
                        "model",
                        "array",
                        "weights_source",
                        "n_features",
                        "feature_dim",
                        "schema_version",
                    )
                }
                p = metadata["provenance"]
                meta["provenance"] = pick(
                    p,
                    (
                        "experiment buffer_protocol_sha256 specification_sha256 fold "
                        "titan_catalog_sha256 native_split_sha256 native_parent "
                        "normalization_provenance membership_role boundary_ids_sha256 "
                        "model_revision weights_sha256 feature_sha256 tile_ids_sha256 "
                        "labels_sha256 raw_tile_sha256 execution_source_commit "
                        "overlap_check_sha256"
                    ),
                )
                if "band_identifier_ghz" in p:
                    meta["provenance"]["band_identifier_ghz"] = p["band_identifier_ghz"]
                boundary.append(
                    dict(
                        **identity,
                        manifest_sha256=expected,
                        metadata=meta,
                        normalization=load("normalization_replay.json"),
                        overlap=load("overlap_check.json"),
                    )
                )
            runs[family].append(run)
    out["results/predictions.json.gz"] = predictions
    out["results/adaptation_history.json.gz"] = dict(records=histories)
    out["evidence/runs.json"] = runs
    out["evidence/boundary.json"] = dict(records=boundary)
    processes = {}
    require(
        set(manifest["processes"])
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
        "Process receipts",
    )
    for name, bound in manifest["processes"].items():
        receipt = read_bound(bound["path"], bound["sha256"], tracked)
        # The external binding declares the actual exit field, avoiding guessed
        # historical receipt schemas. Only these three known spellings allowed.
        field = bound["exit_field"]
        require(
            field in ("exit_code", "returncode", "process_exit"), "Exit receipt field"
        )
        require(
            type(receipt[field]) is int and receipt[field] == 0,
            "Accepted process did not exit zero",
        )
        processes[name] = dict(exit_code=0, receipt_sha256=bound["sha256"])
    reports = dict(
        adaptation=pick(
            raw["ft_audit"],
            (
                "schema_version status validated_jobs source_commit protocol_sha256 "
                "specification_sha256 frozen_audit_sha256 auditor_sha256 "
                "helper_sha256"
            ),
        ),
        checkpoint_replay=pick(
            raw["replay_audit"],
            (
                "schema_version status verified_replays replay_execution_commit "
                "replay_protocol_sha256 plan_sha256 parent_plan_sha256 checker_sha256"
            ),
        ),
        buffer_sgd=pick(sgd, "status protocol_sha256 specification_sha256"),
        buffer_lr={
            **pick(
                lr,
                (
                    "status schema_version producer_commit protocol_sha256 "
                    "specification_sha256 plan_sha256 auditor_sha256 validation"
                ),
            ),
            "parents": {k: pick(v, "sha256") for k, v in lr["parents"].items()},
        },
    )
    out["evidence/acceptance.json"] = dict(
        schema_version="1.0.0",
        status="complete",
        reports=reports,
        replays=[
            pick(
                r,
                (
                    "model fold seed replay_manifest_sha256 producer_manifest_sha256 "
                    "replay_sha256 selected_epoch comparisons validation_metrics "
                    "test_metrics"
                ),
            )
            for r in raw["replay_audit"]["jobs"]
        ],
        processes=processes,
        inputs=pins,
        full_gate=dict(
            schema_version=gate["schema_version"],
            status=gate["status"],
            sha256=pins["full_gate"],
            inputs_sha256=gate["inputs"]["sha256"],
            checker_sha256=gate["checker_sha256"],
            checker_commit=gate["checker_source"]["commit"],
            counts=copy.deepcopy(gate["counts"]),
            reports={k: value["sha256"] for k, value in gate["reports"].items()},
        ),
    )
    out["provenance.json"] = dict(
        schema_version="1.0.0",
        native_catalog_sha256=base.CATALOG_SHA256,
        exporter_source_sha256={
            "scripts/export_repaired_adaptation_results.py": IMPORT_SHA256
        },
        parents={
            "frozen": base.inventory(Path(frozen)),
            "auxiliary": base.inventory(Path(auxiliary_root)),
        },
        producer_commits=commits,
        input_sha256=pins,
        export_input_manifest_sha256=inputs_sha256,
        scientific_source_sha256={
            family: raw[name]["specification"][field]
            for family, name, field in [
                ("adaptation", "ft_protocol", "scientific_source_sha256"),
                ("checkpoint_replay", "replay_protocol", "replay_source_sha256"),
                ("buffer_sgd", "buffer_sgd_protocol", "source_sha256"),
                ("buffer_lr", "buffer_lr_protocol", "scientific_source_sha256"),
            ]
        },
        transformations=[
            "Fixed scientific projections; native and public protocol identities remain distinct.",
            "Replay input/runtime records originate in manifest-bound replay.json.",
            "Replay inference role path strings retained verbatim as computation.",
            "Predicted labels retained; probability and model replay acceptance inherited.",
        ],
    )
    out["accepted_display/adaptation.json"] = {"display": raw["display_a"]["display"]}
    out["accepted_display/substitution.json"] = pick(
        raw["display_b"],
        "recipes lr_diagnostic_counts lr_fit_diagnostics support_design units",
    )
    changes = []
    for name, value in out.items():
        project_warning_paths(
            value, changes if not name.startswith("accepted_display/") else [], (name,)
        )
    out["provenance.json"]["warning_path_projections"] = changes
    public.unchanged(tracked)
    require(base.sha256(__file__) == IMPORT_SHA256, "Exporter source changed")
    return out, tracked


def read_claims_map(path, expected, tracked):
    """Bind Main's completed registration and independent checker execution."""
    mapping = read_bound(path, expected, tracked)
    require(
        set(mapping) == {"schema_version", "specification", "numeric_check"}
        and mapping["schema_version"] == "1.0.0",
        "Claims map schema",
    )
    require(
        set(mapping["numeric_check"]) == {"report", "execution"},
        "Claims check bindings",
    )
    rows = {}
    for name, bound in mapping["numeric_check"].items():
        require(set(bound) == {"path", "sha256"}, "Claims receipt binding fields")
        rows[name] = read_bound(bound["path"], bound["sha256"], tracked)
    report, execution = rows["report"], rows["execution"]
    require(
        report["status"] == "pass" and not report.get("pending"),
        "Final paper mapping remains pending",
    )
    public.hash_value(report["checker_sha256"])
    specification = mapping["specification"]
    exact(
        report["claims_specification_sha256"],
        base.digest(specification),
        "Checker registration differs",
    )
    require(
        type(report["claims_entries"]) is int
        and report["claims_entries"] == len(specification["entries"]),
        "Checker claim coverage differs",
    )
    exact(
        report["claims_occurrences"],
        sum(len(e["target"]["occurrences"]) for e in specification["entries"]),
        "Checker occurrence coverage differs",
    )
    require(
        type(execution["returncode"]) is int and execution["returncode"] == 0,
        "Paper checker did not exit zero",
    )
    exact(
        execution["checker_sha256"],
        report["checker_sha256"],
        "Paper checker source differs",
    )
    report_binding = mapping["numeric_check"]["report"]
    exact(
        execution["outputs"][Path(report_binding["path"]).name],
        report_binding["sha256"],
        "Paper checker report receipt differs",
    )
    return dict(
        schema_version="1.0.0",
        scope=public.CLAIMS_SCOPE,
        mapping_input_sha256=expected,
        specification=specification,
        numeric_check=dict(
            report_sha256=report_binding["sha256"],
            execution_sha256=mapping["numeric_check"]["execution"]["sha256"],
            checker_sha256=report["checker_sha256"],
            specification_sha256=report["claims_specification_sha256"],
            status="pass",
            returncode=0,
        ),
    )




def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "frozen-package", "auxiliary-package", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--claims-map", type=Path, required=True)
    parser.add_argument("--claims-map-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        payloads, tracked = collect(
            args.inputs, args.inputs_sha256, args.frozen_package, args.auxiliary_package
        )
        claims = read_claims_map(args.claims_map, args.claims_map_sha256, tracked)
        result = public.build_package(
            payloads,
            args.frozen_package,
            args.auxiliary_package,
            args.output,
            tracked,
            claims=claims,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        print(json.dumps(dict(status="invalid", error=str(error))))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
