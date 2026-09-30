"""Full synthetic grids. No mission tiles, model arrays or observed scores."""

import copy
import sys
from pathlib import Path
import numpy as np
from scripts import verify_repaired_adaptation_release as v

H = v.digest
PARAMETERS = dict(
    classifier=dict(
        C=1.0,
        class_weight="balanced",
        fit_intercept=True,
        max_iter=2000,
        solver="lbfgs",
        tol=0.0001,
    ),
    scaler=dict(copy=True, with_mean=True, with_std=True),
)


def fit(tag, warn=False):
    return dict(
        classifier_parameters=copy.deepcopy(PARAMETERS["classifier"]),
        scaler_parameters=copy.deepcopy(PARAMETERS["scaler"]),
        n_iter=[2000 if warn else 13],
        iteration_limit_reached=warn,
        convergence_warning=warn,
        fit_warnings=[
            dict(category="ConvergenceWarning", message="Synthetic iteration cap")
        ]
        if warn
        else [],
        estimator_sha256=H(tag),
    )


def predicted(labels, offset):
    return [
        (c + 1) % 6 if (i + offset) % 7 < offset % 5 else c
        for i, c in enumerate(labels)
    ]


def pack(root, files):
    root.mkdir()
    for name, value in files.items():
        v.base.write_json(root, name, value)
    v.base.write_indexes(root, dict(schema_version="synthetic"))


def synthetic(tmp):
    # Unequal test supports distinguish equal-fold means from pooled metrics.
    entries = []

    def tiles(labels):
        out = []
        for c in labels:
            tid = f"titan_{len(entries):06d}"
            entries.append(dict(tile_id=tid, label=c))
            out.append(tid)
        return out

    pool = tiles(list(range(6)) * 50)
    boundary = tiles(list(range(6)))
    tests = [tiles([c for c in range(6) for _ in range(3 + c + f)]) for f in range(5)]
    vals = [tiles(list(range(6)) * 2) for _ in range(5)]
    labels = {r["tile_id"]: r["label"] for r in entries}
    ids = list(labels)
    members = []
    splits = []
    counts = []
    fp = []
    lp = []
    lra = []
    nf = []
    for f in range(5):
        train = pool[: 180 + 6 * f]
        norm = dict(
            enabled=True,
            intensity_space="hisar_log_dn",
            denominator_epsilon=1e-8,
            lower=2.0,
            upper=199.0,
            training_ids_sha256=H(train),
            sampling=dict(
                selected_tile_ids=train[:6],
                used_tile_ids=train[:6],
                missing_tile_ids=[],
                empty_tile_ids=[],
                sample_count=24,
            ),
        )
        row = dict(
            fold=f,
            split_sha256=H(["split", f]),
            retained_train_ids=train,
            validation_ids=vals[f],
            test_ids=tests[f],
            validation_labels=[labels[t] for t in vals[f]],
            test_labels=[labels[t] for t in tests[f]],
            boundary_ids=sorted(boundary),
            eligible_removal_ids=sorted(set(train) - set(train[:6])),
            overlap_ids=train[:6],
            class_counts=[30 + f] * 6,
            boundary_class_counts=[1] * 6,
            normalization_provenance=norm,
            draws=[],
        )
        for d in range(5):
            rng = np.random.default_rng(d)
            removed = []
            for c in range(6):
                eligible = [t for t in row["eligible_removal_ids"] if labels[t] == c]
                removed.extend(
                    np.asarray(eligible)[
                        rng.choice(len(eligible), 1, replace=False)
                    ].tolist()
                )
            selected = (set(train) - set(removed)) | set(boundary)
            ordered = [t for t in ids if t in selected]
            row["draws"].append(
                dict(
                    draw_seed=d,
                    removed_ids=sorted(removed),
                    training_ids=ordered,
                    training_ids_sha256=H(ordered),
                    train_class_counts=[30 + f] * 6,
                )
            )
        members.append(row)
        assignment = {t: "selk_holdout" for t in ids}
        for role, items in [
            ("train", train),
            ("val", vals[f]),
            ("test", tests[f]),
            ("buffer_excluded", boundary),
        ]:
            assignment.update({t: role for t in items})
        splits.append(
            dict(
                assignments=assignment,
                policy=dict(
                    exclusion_reasons={t: "train_buffer_excluded" for t in boundary}
                ),
            )
        )
        counts.append(
            {
                r: dict(
                    total=len(items),
                    class_counts=np.bincount(
                        [labels[t] for t in items], minlength=6
                    ).tolist(),
                )
                for r, items in [("train", train), ("val", vals[f]), ("test", tests[f])]
            }
        )
        nf.append(dict(fold=f, normalization_provenance=norm))
        lrrow = dict(
            fold=f, test_ids=tests[f], true_labels=row["test_labels"], common={}
        )
        for mi, m in enumerate(v.MODELS):
            heads = [predicted(row["test_labels"], mi + f + s + 1) for s in range(5)]
            fp.append(
                dict(
                    model=m,
                    fold=f,
                    tile_ids=tests[f],
                    true_labels=row["test_labels"],
                    seeds=list(range(5)),
                    probe_predictions=heads,
                )
            )
            lrrow["common"][m] = predicted(row["test_labels"], f + mi + 2)
            lra.append(
                dict(
                    model=m,
                    fold=f,
                    manifest_sha256=H(["common", m, f]),
                    fit=fit(["common", m, f], f == 4),
                    **v.score(lrrow["common"][m], row["test_labels"]),
                )
            )
        lp.append(lrrow)
    common = dict(
        resolved_parameters=PARAMETERS,
        folds=[
            dict(fold=f, normalization_provenance=r["normalization_provenance"])
            for f, r in enumerate(members)
        ],
    )
    frozen = tmp / "frozen"
    aux = tmp / "auxiliary"
    pack(
        frozen,
        {
            "catalogs/titan_catalog.json": dict(
                class_names=list(v.base.CLASSES), tiles=entries
            ),
            **{
                f"splits/titan_contiguous_fold_{f}.json": s
                for f, s in enumerate(splits)
            },
            "results/predictions.json.gz": dict(jobs=fp),
            "results/frozen_results.json": dict(folds=nf),
        },
    )
    pack(
        aux,
        {
            "results/predictions.json.gz": dict(folds=lp),
            "results/common_classifier.json": dict(folds=lra),
            "protocols/common_classifier.json": dict(specification=common),
        },
    )
    commits = {
        k: H(k)[:40]
        for k in (
            "adaptation",
            "checkpoint_replay",
            "boundary",
            "buffer_sgd",
            "buffer_lr",
        )
    }
    commits["checkpoint_replay"] = v.adaptation.REPLAY_COMMIT
    commits["boundary"] = commits["buffer_sgd"] = v.CONTROL_COMMIT
    sources = {k: {"scripts/example.py": H(["source", k])} for k in commits}
    sources["buffer_lr"]["scripts/run_repaired_common_buffer.py"] = H("lrchecker")
    ph = {
        k: H(["protocol", k])
        for k in ("adaptation", "checkpoint_replay", "buffer_sgd", "buffer_lr")
    }
    spec = dict(
        adaptation=dict(
            recipe=dict(epochs=50, unfreeze_blocks=2, backbone_lr=0.0001, head_lr=0.001, weight_decay=0.01),
            optimization_seeds=list(range(5)),
            catalog_sha256=H("catalog"),
            folds=[
                dict(fold=f, split_sha256=r["split_sha256"])
                for f, r in enumerate(members)
            ],
        ),
        checkpoint_replay=dict(
            producer_plan_sha256=H("ft_plan"),
            inference=dict(cpu_intraop_threads=4),
            execution=dict(
                thread_environment=dict(
                    OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1"
                )
            ),
        ),
        buffer_sgd=dict(catalog_sha256=H("catalog"), folds=[]),
        buffer_lr=dict(
            catalog={"sha256": H("catalog")}, resolved_parameters=PARAMETERS, parents={}
        ),
    )
    for mem in members:
        r = copy.deepcopy(mem)
        r.pop("test_labels")
        r.pop("validation_labels")
        r["draws"] = [
            {k: d[k] for k in ("draw_seed", "removed_ids", "training_ids_sha256")}
            for d in r["draws"]
        ]
        spec["buffer_sgd"]["folds"].append(r)
    spec["buffer_lr"].update(
        draws=list(range(5)), fits=100, buffer_producer_commit=commits["buffer_sgd"]
    )
    spec["buffer_sgd"]["recipe"] = dict(probe_epochs=100, probe_batch_size=256, probe_lr=0.01, momentum=0.9)
    for family, field in [
        ("adaptation", "scientific_source_sha256"),
        ("checkpoint_replay", "replay_source_sha256"),
        ("buffer_sgd", "source_sha256"),
        ("buffer_lr", "scientific_source_sha256"),
    ]:
        spec[family][field] = sources[family]
    inputs = {
        name: H(name)
        for name in (
            *v.adaptation.PINS,
            "ft_audit",
            "replay_audit",
            "buffer_sgd_audit",
            "buffer_lr_audit",
            "ft_plan",
            "followup_plan",
            "replay_plan",
            "full_gate",
        )
    }
    lrparents = {
        k: dict(sha256=h)
        for k, h in dict(
            common_protocol=H("common_protocol"),
            common_audit=H("common_audit"),
            common_plan=H("common_plan"),
            buffer_protocol=ph["buffer_sgd"],
            buffer_audit=inputs["buffer_sgd_audit"],
            finetuning_audit=inputs["ft_audit"],
        ).items()
    }
    spec["buffer_lr"]["parents"] = lrparents
    protos = {
        k: dict(
            original_protocol_sha256=ph[k],
            original_specification_sha256=H(["native-spec", k]),
            specification=s,
            extract_specification_sha256=H(s),
        )
        for k, s in spec.items()
    }
    a = dict(
        status="complete",
        schema_version="1.0.0",
        class_names=list(v.base.CLASSES),
        validation=dict(ft_jobs=75, checkpoint_replays=75, common_lr_jobs=20),
        definitions={},
        limitations=["Synthetic fixture only"],
        models={},
        inputs={
            k: {"sha256": inputs[k]}
            for k in (*v.adaptation.PINS, "ft_audit", "replay_audit")
        },
        analysis_source_sha256=v.HELPERS,
        analysis_runtime=dict(python=sys.version, numpy=np.__version__),
    )
    b = dict(
        status="complete",
        protocol_sha256=ph["buffer_sgd"],
        specification_sha256=protos["buffer_sgd"]["original_specification_sha256"],
        models={},
        folds=[],
        input_fit_manifests=[],
    )
    l = dict(
        status="complete",
        schema_version="1.0.0",
        producer_commit=commits["buffer_lr"],
        protocol_sha256=ph["buffer_lr"],
        specification_sha256=protos["buffer_lr"]["original_specification_sha256"],
        plan_sha256=H("lrplan"),
        auditor_sha256=H("lrchecker"),
        validation=dict(complete_jobs=20, estimator_replays=100),
        parents=lrparents,
        folds=[],
        models={},
    )
    runs = {k: [] for k in commits}
    preds = dict(schema_version="1.0.0", test=[], validation=[])
    hist = []
    replays = []
    bnd = []

    def run(family, m, f, s=None):
        key = dict(model=m, fold=f)
        if s is not None:
            key["seed"] = s
        r = dict(
            **key,
            manifest_sha256=H([family, m, f, s, "manifest"]),
            producer_commit=commits[family],
            status="complete",
            inputs_sha256={},
            outputs_sha256={},
            environment={},
        )
        protocol = "buffer_sgd" if family == "boundary" else family
        inp = (
            "repaired_protocol"
            if family == "adaptation"
            else "replay_protocol"
            if family == "checkpoint_replay"
            else "protocol"
        )
        r["inputs_sha256"][inp] = ph[protocol]
        if family == "checkpoint_replay":
            r["environment"] = dict(
                cpu_intraop_threads=4,
                cpu_interop_threads=8,
                declared_environment=dict(
                    OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1"
                ),
            )
        runs[family].append(r)
        return r

    for mi, m in enumerate(v.MODELS):
        ssummary = []
        lsummary = []
        afolds = []
        for f, mem in enumerate(members):
            truth = mem["test_labels"]
            fj = next(r for r in fp if (r["model"], r["fold"]) == (m, f))
            refs = [v.score(p, truth) for p in fj["probe_predictions"]]
            ref = next(r for r in lra if (r["model"], r["fold"]) == (m, f))
            sr = run("buffer_sgd", m, f)
            lr = run("buffer_lr", m, f)
            br = run("boundary", m, f)
            br["outputs_sha256"] = {"overlap_check.json": H(["overlap", m, f])}
            bnd.append(
                dict(
                    model=m,
                    fold=f,
                    manifest_sha256=br["manifest_sha256"],
                    metadata={},
                    normalization=mem["normalization_provenance"],
                    overlap=dict(
                        status="pass",
                        atol=1e-6,
                        rtol=1e-5,
                        ids=mem["overlap_ids"],
                        ids_sha256=H(mem["overlap_ids"]),
                        rows=len(mem["overlap_ids"]),
                    ),
                )
            )
            lrow = dict(
                model=m,
                fold=f,
                manifest_sha256=lr["manifest_sha256"],
                reference=v.score(lp[f]["common"][m], truth),
                draws=[],
            )
            weights = 1 / np.asarray(mem["class_counts"], float)
            weights = (weights / weights.sum() * 6).astype(np.float32).tolist()
            srow = dict(
                model=m,
                fold=f,
                manifest_sha256=sr["manifest_sha256"],
                recipe=copy.deepcopy(spec["buffer_sgd"]["recipe"]),
                class_counts=mem["class_counts"],
                training_class_weights=weights,
                cpu_numerics={},
                optimization=dict(
                    planned_steps_per_head=100
                    * ((len(mem["retained_train_ids"]) + 255) // 256),
                    planned_example_presentations_per_head=100
                    * len(mem["retained_train_ids"]),
                ),
                reference_metrics=refs,
                runs=[],
            )
            sr["outputs_sha256"]["buffer_results.json"] = H(["sgd-summary", m, f])
            lr["outputs_sha256"]["common_buffer_results.json"] = H(["lr-result", m, f])
            means = []
            for d in range(5):
                scores = []
                for s in range(5):
                    p = predicted(truth, mi + 2 * f + d + 2 * s)
                    metric = v.score(p, truth)
                    scores.append(metric)
                    name = f"draw-{d}/seed-{s}.json"
                    sr["outputs_sha256"][name] = H([m, f, d, s])
                    preds["test"].append(
                        dict(
                            family="buffer_sgd",
                            model=m,
                            fold=f,
                            draw_seed=d,
                            seed=s,
                            predictions=p,
                            source_result_sha256=sr["outputs_sha256"][name],
                        )
                    )
                    srow["runs"].append(
                        dict(
                            draw_seed=d,
                            seed=s,
                            metrics=metric,
                            difference={
                                k: metric["macro"][k] - refs[s]["macro"][k]
                                for k in v.METRICS
                            },
                        )
                    )
                means.append(v.average(scores)["macro"])
                p = predicted(truth, 2 * mi + f + 2 * d)
                metric = v.score(p, truth)
                preds["test"].append(
                    dict(
                        family="buffer_lr",
                        model=m,
                        fold=f,
                        draw_seed=d,
                        predictions=p,
                        source_result_sha256=lr["outputs_sha256"][
                            "common_buffer_results.json"
                        ],
                    )
                )
                lrow["draws"].append(
                    dict(
                        draw_seed=d,
                        training_ids_sha256=mem["draws"][d]["training_ids_sha256"],
                        train_class_counts=mem["class_counts"],
                        metrics=metric,
                        difference={
                            **{
                                k: metric["macro"][k] - ref["macro"][k]
                                for k in v.METRICS
                            },
                            "kappa": metric["kappa"] - ref["kappa"],
                        },
                        fit=fit([m, f, d], mi == f == d == 0),
                        estimator_replay=dict(
                            verified=True, probability_max_abs_difference=0.0
                        ),
                    )
                )
            ssummary.append(
                dict(
                    fold=f,
                    draw_means=means,
                    reference=v.average(refs)["macro"],
                    replacement={
                        k: float(np.mean([x[k] for x in means])) for k in v.METRICS
                    },
                )
            )
            b["folds"].append(srow)
            b["input_fit_manifests"].append(
                {k: sr[k] for k in ("model", "fold", "manifest_sha256")}
            )
            l["folds"].append(lrow)
            lsummary.append(lrow)
            if m in v.FT_MODELS:
                seedrows = []
                for s in range(5):
                    fr = run("adaptation", m, f, s)
                    rr = run("checkpoint_replay", m, f, s)
                    fr["outputs_sha256"] = {
                        "finetuning_results.json": H(["ft", m, f, s]),
                        f"checkpoints/{m}/ft_2block_seed_{s}.pt": H(["ckpt", m, f, s]),
                    }
                    rr["outputs_sha256"] = {
                        "replay.json": H(["replay", m, f, s]),
                        "started.json": H(["started", m, f, s]),
                    }
                    p = predicted(truth, mi + f + s * 2)
                    ft = v.score(p, truth)
                    vp = predicted(mem["validation_labels"], s)
                    val = v.score(vp, mem["validation_labels"])
                    preds["test"].append(
                        dict(
                            family="adaptation",
                            model=m,
                            fold=f,
                            seed=s,
                            predictions=p,
                            source_result_sha256=fr["outputs_sha256"][
                                "finetuning_results.json"
                            ],
                        )
                    )
                    preds["validation"].append(
                        dict(
                            model=m,
                            fold=f,
                            seed=s,
                            predictions=vp,
                            source_result_sha256=rr["outputs_sha256"]["replay.json"],
                        )
                    )
                    attempts = (len(mem["retained_train_ids"]) + 31) // 32
                    epoch = 1 + f + s
                    history = dict(
                        train_loss=[1.0] * 50,
                        val_balanced_accuracy=[0.0] * 50,
                        optimizer_steps=[attempts] * 50,
                        amp_skipped_steps=[0] * 50,
                        loss_scale=[65536.0] * 50,
                    )
                    # Equal best values later in history test earliest-tie selection.
                    history["val_balanced_accuracy"][epoch - 1] = history[
                        "val_balanced_accuracy"
                    ][49] = val["macro"]["recall"]
                    history["optimizer_steps"][0] -= 1
                    history["amp_skipped_steps"][0] = 1
                    train = dict(
                        selected_epoch=epoch,
                        epoch_indexing="one based",
                        optimizer_attempts_per_epoch=attempts,
                        optimizer_attempts_total=50 * attempts,
                        optimizer_steps_total=50 * attempts - 1,
                        amp_skipped_steps_total=1,
                        training_example_presentations=50
                        * len(mem["retained_train_ids"]),
                    )
                    hist.append(
                        dict(
                            model=m,
                            fold=f,
                            seed=s,
                            history=history,
                            source_result_sha256=fr["outputs_sha256"][
                                "finetuning_results.json"
                            ],
                        )
                    )
                    seedrows.append(
                        dict(
                            seed=s,
                            finetuned=ft,
                            frozen_sgd=refs[s],
                            ft_minus_sgd=v.adaptation.difference(ft, refs[s]),
                            training=train,
                            validation=val,
                            manifest_sha256=fr["manifest_sha256"],
                            outputs=fr["outputs_sha256"],
                            replay_manifest_sha256=rr["manifest_sha256"],
                            replay_sha256=rr["outputs_sha256"]["replay.json"],
                        )
                    )
                    replays.append(
                        dict(
                            model=m,
                            fold=f,
                            seed=s,
                            producer_manifest_sha256=fr["manifest_sha256"],
                            replay_manifest_sha256=rr["manifest_sha256"],
                            replay_sha256=rr["outputs_sha256"]["replay.json"],
                            selected_epoch=epoch,
                            validation_metrics=val,
                            test_metrics=ft,
                            comparisons=dict(
                                validation=dict(
                                    passed=True,
                                    historical_vector_available=False,
                                    macro_recall_abs_error=0.0,
                                    expected_macro_recall=val["macro"]["recall"],
                                ),
                                test=dict(
                                    passed=True,
                                    confusion_exact=True,
                                    prediction_mismatches=0,
                                    mismatched_tile_ids=[],
                                    probability_components_over_atol=0,
                                    probability_max_abs_error=0.0,
                                ),
                            ),
                        )
                    )
                ss = {
                    k: v.adaptation.score_summary([r[k] for r in seedrows])
                    for k in ("finetuned", "frozen_sgd", "ft_minus_sgd")
                }
                values = {k: v.average([r[k] for r in seedrows]) for k in ss}
                values["common_lr"] = {k: ref[k] for k in ("macro", "per_class")}
                values["ft_minus_lr"] = v.adaptation.difference(
                    values["finetuned"], ref
                )
                afolds.append(
                    dict(
                        fold=f,
                        counts=counts[f],
                        seeds=seedrows,
                        seed_summaries=ss,
                        values=values,
                        common_lr=dict(
                            metrics=v.score(lp[f]["common"][m], truth),
                            fit=ref["fit"],
                            manifest_sha256=ref["manifest_sha256"],
                        ),
                    )
                )
        if m in v.FT_MODELS:
            a["models"][m] = dict(
                folds=afolds,
                summary={
                    name: v.summaries([f["values"][name] for f in afolds])
                    for name in v.adaptation.GROUPS
                },
            )
        b["models"][m] = dict(folds=ssummary)
        for kind in ("reference", "replacement", "difference"):
            b["models"][m][kind] = {
                k: v.adaptation.joint.describe(
                    [
                        f[kind][k]
                        if kind != "difference"
                        else f["replacement"][k] - f["reference"][k]
                        for f in ssummary
                    ]
                )
                for k in v.METRICS
            }
        l["models"][m] = {}
        for k in (*v.METRICS, "kappa"):
            ref = [
                r["reference"]["kappa"] if k == "kappa" else r["reference"]["macro"][k]
                for r in lsummary
            ]
            delta = [
                float(np.mean([d["difference"][k] for d in r["draws"]]))
                for r in lsummary
            ]
            l["models"][m][k] = {
                name: v.adaptation.joint.describe(x)
                for name, x in [
                    ("reference", ref),
                    ("difference", delta),
                    ("replacement", np.asarray(ref) + delta),
                ]
            }
    reports = dict(
        adaptation=dict(
            schema_version="1.0.0",
            status="pass",
            validated_jobs=75,
            source_commit=commits["adaptation"],
            protocol_sha256=ph["adaptation"],
            specification_sha256=protos["adaptation"]["original_specification_sha256"],
            frozen_audit_sha256=H("frozen"),
            auditor_sha256=H("ftaudit"),
            helper_sha256=H("helper"),
        ),
        checkpoint_replay=dict(
            schema_version="1.0.0",
            status="pass",
            verified_replays=75,
            replay_execution_commit=commits["checkpoint_replay"],
            replay_protocol_sha256=ph["checkpoint_replay"],
            plan_sha256=inputs["replay_plan"],
            parent_plan_sha256=inputs["ft_plan"],
            checker_sha256=H("checker"),
        ),
        buffer_sgd={
            k: b[k] for k in ("status", "protocol_sha256", "specification_sha256")
        },
        buffer_lr={
            k: l[k]
            for k in (
                "status",
                "schema_version",
                "producer_commit",
                "protocol_sha256",
                "specification_sha256",
                "plan_sha256",
                "auditor_sha256",
                "validation",
                "parents",
            )
        },
    )
    data = {
        "provenance.json": dict(
            schema_version="1.0.0",
            native_catalog_sha256=H("catalog"),
            parents={
                "frozen": v.base.inventory(frozen),
                "auxiliary": v.base.inventory(aux),
            },
            producer_commits=commits,
            input_sha256=inputs,
            scientific_source_sha256=sources,
        ),
        "membership.json.gz": dict(schema_version="1.0.0", folds=members),
        "results/adaptation.json": a,
        "results/adaptation_history.json.gz": dict(records=hist),
        "results/buffer_sgd.json": b,
        "results/buffer_lr.json": l,
        "results/predictions.json.gz": preds,
        "evidence/runs.json": runs,
        "evidence/boundary.json": dict(records=bnd),
        "evidence/acceptance.json": dict(
            schema_version="1.0.0",
            status="complete",
            reports=reports,
            replays=replays,
            inputs=inputs,
            full_gate=dict(
                schema_version=v.RECOVERY_SCHEMA,
                status="pass",
                sha256=inputs["full_gate"],
                inputs_sha256=H("recovery-inputs"),
                checker_sha256=H("recovery-checker"),
                checker_commit=H("gate-source")[:40],
                counts=copy.deepcopy(v.RECOVERY_COUNTS),
                reports={
                    "finetuning": inputs["ft_audit"],
                    "buffer": inputs["buffer_sgd_audit"],
                    "replay": inputs["replay_audit"],
                },
            ),
            processes={
                k: dict(exit_code=0, receipt_sha256=H(k))
                for k in (
                    "full_gate",
                    "adaptation",
                    "checkpoint_replay",
                    "buffer_sgd",
                    "buffer_lr",
                    "analysis",
                    "display_a",
                    "display_b",
                )
            },
        ),
        **{f"protocols/{k}.json": r for k, r in protos.items()},
    }
    return data, frozen, aux


# Claims fixtures are explicit synthetic manuscript/receipt registrations.
_RENDER_FOR_CLAIMS = v.render


def synthetic_claims(data, frozen, auxiliary):
    import tempfile
    import copy

    payloads = copy.deepcopy(data)
    payloads.pop("accepted_display/adaptation.json", None)
    payloads.pop("accepted_display/substitution.json", None)
    _, a, b = v.validate_payloads(payloads, frozen, auxiliary)
    with tempfile.TemporaryDirectory(prefix="cetus-claim-fixture-") as temp:
        root = Path(temp)
        _RENDER_FOR_CLAIMS(a, b, root)
        for name in v.CLAIM_SOURCES:
            if name not in v.CLAIM_DISPLAYS:
                v.write_json(root / name, payloads[name])
        required, _ = v.claim_catalog(a, b)
        entries = []
        sources = dict(zip(v.CLAIM_DISPLAYS, (a, b)))
        for name, definition in sorted(required.items()):
            row = dict(id=name, **copy.deepcopy(definition))
            row["target"]["occurrences"] = [
                dict(document=d, registration=row["target"]["label"], ordinal=0)
                for d in ("full", "workshop")
            ]
            row["value"] = v.claim_value(row, sources)
            row["text"] = v.claim_number(row["value"], row["format"])
            entries.append(row)
        documents = [
            dict(
                id=d,
                tex_file=d + "/main.tex",
                tex_sha256=H(d + "-tex"),
                pdf_file=d + "/paper.pdf",
                pdf_sha256=H(d + "-pdf"),
            )
            for d in ("full", "workshop")
        ]
        for document in documents:
            for kind, calculation, pointer in (
                ("prose", "count", ["training_rows"]),
                (
                    "support_table",
                    "boundary_percent",
                    ["support_design", 0, "boundary_total"],
                ),
            ):
                file = v.CLAIM_DISPLAYS[0 if kind == "prose" else 1]
                row = dict(
                    id=document["id"] + "." + kind,
                    target=dict(
                        kind=kind,
                        file=document["tex_file"],
                        label="synthetic:prose"
                        if kind == "prose"
                        else "tab:repairedcontrolsupport",
                        registration="synthetic_" + kind,
                        occurrences=[
                            dict(
                                document=document["id"],
                                registration="synthetic_" + kind,
                                ordinal=0,
                            )
                        ],
                    ),
                    sources=[dict(file=file, pointer=pointer)],
                    calculation=calculation,
                    source_unit="count",
                    display_unit="count" if kind == "prose" else "percent",
                    scale=1,
                    format="integer" if kind == "prose" else "unsigned_two_decimals",
                )
                if calculation == "boundary_percent":
                    row["sources"].append(
                        dict(file=file, pointer=["support_design", 0, "train_total"])
                    )
                row["value"] = v.claim_value(row, sources)
                row["text"] = v.claim_number(row["value"], row["format"])
                entries.append(row)
        for document in documents:
            for key, definition in v.support_table_catalog().items():
                if key == "0.percent":
                    continue
                row = dict(
                    id=f"{document['id']}.support.{key}",
                    **copy.deepcopy(definition),
                    target=dict(
                        kind="support_table",
                        file=document["tex_file"],
                        label="tab:repairedcontrolsupport",
                        registration="support." + key,
                        occurrences=[
                            dict(
                                document=document["id"],
                                registration="support." + key,
                                ordinal=0,
                            )
                        ],
                    ),
                )
                row["value"] = v.claim_value(row, sources)
                row["text"] = v.claim_number(row["value"], row["format"])
                entries.append(row)
            for model in range(3):
                for field in v.TRAINING_TABLE_FIELDS:
                    row = dict(
                        id=f"{document['id']}.training.{model}.{field}",
                        target=dict(
                            kind="training_table",
                            file=document["tex_file"],
                            label="tab:repairedtraining",
                            registration=f"training.{model}.{field}",
                            occurrences=[
                                dict(
                                    document=document["id"],
                                    registration=f"training.{model}.{field}",
                                    ordinal=0,
                                )
                            ],
                        ),
                        sources=[
                            dict(
                                file=v.CLAIM_DISPLAYS[0],
                                pointer=["training_summary", model, field],
                            )
                        ],
                        calculation="scalar",
                        source_unit="count",
                        display_unit="count",
                        scale=1,
                        format="integer"
                        if field.startswith("selected_epoch_")
                        else "tex_grouped_integer",
                    )
                    row["value"] = v.claim_value(row, sources)
                    row["text"] = v.claim_number(row["value"], row["format"])
                    entries.append(row)
        entries.sort(key=lambda row: row["id"])
        spec = dict(
            documents=documents,
            source_files_sha256={n: v.base.sha256(root / n) for n in v.CLAIM_SOURCES},
            artifact_files_sha256={
                n: v.base.sha256(root / n) for n in v.CLAIM_ARTIFACTS
            },
            required_entry_ids=[r["id"] for r in entries],
            entries=entries,
        )
        return dict(
            schema_version="1.0.0",
            scope=v.CLAIMS_SCOPE,
            mapping_input_sha256=H("synthetic-map"),
            specification=spec,
            numeric_check=dict(
                report_sha256=H("synthetic-check"),
                execution_sha256=H("synthetic-exit"),
                checker_sha256=H("synthetic-checker-source"),
                specification_sha256=v.digest(spec),
                status="pass",
                returncode=0,
            ),
        )


def build_with_claims(data, frozen, auxiliary, output, tracked=None):
    claims = synthetic_claims(data, frozen, auxiliary)
    return v.build_package(data, frozen, auxiliary, output, tracked, claims=claims)
