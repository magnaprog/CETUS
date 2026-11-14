"""Write CVPR tables directly from the accepted Titan result summaries."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "audit/common_classifier_20260930/COMMON_CLASSIFIER_RESULTS_2026-09-29.json"
ADAPTATION = ROOT / "paper/figure_data/repaired/adaptation_display.json"
SENSITIVITY = ROOT / "audit/fitting_20261106/results.json"
NAMES = {
    "dinov2": "DINOv2", "dofa": "DOFA", "croma": "CROMA",
    "random_init": "Untrained ViT", "raw_intensity_statistics": "Intensity statistics",
    "intensity_histogram": "Intensity histogram",
    "texture_glcm_lbp_gabor_hog_wavelet": "Texture",
    "combined_classical": "Combined classical",
}


def score(mean, sd=None, bold=False):
    value = f"{mean:.2f}"
    if bold:
        value = r"\mathbf{" + value + "}"
    if sd is not None:
        value += rf" \pm {sd:.2f}"
    return "$" + value + "$"


def table(columns, header, rows):
    return "\n".join([
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
        header + r" \\", r"\midrule",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", "",
    ])


def render():
    report = json.loads(COMMON.read_text())
    adaptation = json.loads(ADAPTATION.read_text())
    joint = report["joint_fixed_classifier"]
    order = joint["family_order"]
    families = joint["families"]
    metrics = ("precision", "recall", "f1")
    winners = {m: max(round(100 * families[k]["macro"][m]["mean"], 2)
                      for k in order) for m in metrics}
    rows = []
    for key in order:
        row = [NAMES[key]]
        for metric in metrics:
            values = families[key]["macro"][metric]
            mean = 100 * values["mean"]
            row.append(score(mean, 100 * values["sample_std"],
                             round(mean, 2) == winners[metric]))
        rows.append(row)
    outputs = {"features.tex": table("lrrr", "Features & Precision & Recall & F1", rows)}

    models = ("dinov2", "dofa", "croma", "random_init")
    fitting = {}
    for key in models:
        record = report["models"][key]
        fitting[key] = [record["common"]["macro"]["f1"], record["original"]["macro"]["f1"]]
    ft = {r["model"].lower(): r for r in adaptation["macro_rows"]}
    best = [max(round(100 * fitting[k][i]["mean"], 2) for k in models) for i in range(2)]
    best_ft = max(round(r["ft_f1_mean"], 2) for r in ft.values())
    rows = []
    for key in models:
        row = [NAMES[key]]
        for i, values in enumerate(fitting[key]):
            mean = 100 * values["mean"]
            row.append(score(mean, 100 * values["sample_std"], round(mean, 2) == best[i]))
        row.append(score(ft[key]["ft_f1_mean"], ft[key]["ft_f1_sample_std"],
                         round(ft[key]["ft_f1_mean"], 2) == best_ft) if key in ft else "n/a")
        rows.append(row)
    outputs["fitting.tex"] = table("lrrr", "Encoder & LR & SGD & Fine-tuning", rows)

    classes = report["classes"]
    class_rows = {(r["model"].lower(), r["terrain_class"]): r for r in adaptation["class_rows"]}
    rows = []
    for i, terrain in enumerate(classes):
        row = [terrain, f"{report['test_support'][i]:,}"]
        for model in models[:3]:
            values = class_rows[model, terrain]
            best_value = max(round(values[k + "_f1_mean"], 2) for k in ("lr", "ft"))
            for method in ("lr", "ft"):
                mean = values[method + "_f1_mean"]
                row.append(score(mean, bold=round(mean, 2) == best_value))
        rows.append(row)
    outputs["classes.tex"] = table(
        "lrrrrrrr", "Terrain & Test tiles & DINOv2 LR & DINOv2 FT & DOFA LR & DOFA FT & CROMA LR & CROMA FT", rows)

    for model in models[:3]:
        rows = []
        for terrain in classes:
            values = class_rows[model, terrain]
            for method in ("lr", "ft"):
                row = [terrain if method == "lr" else "", method.upper()]
                for metric in metrics:
                    stem = method + "_" + metric
                    row.append(score(values[stem + "_mean"], values[stem + "_sample_std"]))
                rows.append(row)
        outputs[f"classes-{model}.tex"] = table("llrrr", "Terrain & Fit & Precision & Recall & F1", rows)
    outputs.update(fitting_sensitivity_tables())
    return outputs


def fitting_sensitivity_tables():
    report = json.loads(SENSITIVITY.read_text())
    settings = {(r["model"], r["standardize"], r["regularization"]): r
                for r in report["settings"]}
    models = ("dinov2", "dofa", "croma", "random_init")
    configurations = ((False, "none"), (False, "l2"), (True, "none"), (True, "l2"))
    expected = {(m, s, r) for m in models for s, r in configurations}
    if len(settings) != len(report["settings"]) or set(settings) != expected:
        raise ValueError("Expected all sixteen unique fitting settings")
    winners = {cfg: max(round(100 * settings[m, *cfg]["test"]["macro_f1"]["mean"], 2)
                        for m in models) for cfg in configurations}
    rows, regional, numerical, classes = [], [], [], []
    terrain = ("Plains", "Dunes", "Hummocky", "Labyrinths", "Lakes", "Craters")
    for model in models:
        row = [NAMES[model]]
        for standardize, regularization in configurations:
            setting = settings[model, standardize, regularization]
            f1 = setting["test"]["macro_f1"]
            mean = 100 * f1["mean"]
            value = score(mean, 100 * f1["sample_sd"],
                          round(mean, 2) == winners[standardize, regularization])
            value += rf"\,\textsuperscript{{{len(setting['iteration_limit_folds'])}}}"
            row.append(value)
            name = ("Standardized" if standardize else "Raw") + (", none" if regularization == "none" else ", $C=1$")
            regional.append([NAMES[model], name, *[score(100 * v) for v in f1["folds_1_to_5"]]])
            cells = [r for r in report["numerical"] if r["cell"]["model"] == model
                     and r["cell"]["standardize"] == standardize
                     and r["cell"]["regularization"] == regularization]
            if len(cells) != 5 or {r["cell"]["fold"] for r in cells} != set(range(5)):
                raise ValueError("Expected all five regional numerical records")
            cells.sort(key=lambda r: r["cell"]["fold"])
            concerns = sum(r["diagnostics"]["mathematical_float64"]["gradient_reference_inf"] > 1e-4
                           or r["diagnostics"]["solver_oracle"]["gradient_reference_inf"] > 1e-4
                           for r in cells)
            if any(len(r["iterations"]) != 1 for r in cells):
                raise ValueError("Expected one multinomial iteration count per fit")
            numerical.append([NAMES[model], name, ", ".join(str(r["iterations"][0]) for r in cells),
                              str(len(setting["warning_folds"])), str(concerns)])
            classes.append([NAMES[model], name, *[score(100 * v) for v in setting["test"]["f1"]["mean"]]])
        rows.append(row)
    differences = []
    names = {"standardized_minus_raw_at_no_penalty": "Scaling, no penalty",
             "standardized_minus_raw_at_l2": "Scaling, $C=1$",
             "l2_minus_no_penalty_raw": "L2, raw",
             "l2_minus_no_penalty_standardized": "L2, standardized",
             "difference_of_differences": "Interaction"}
    for item in report["differences"]:
        values = item["test"]["macro_f1"]
        differences.append([NAMES[item["model"]], names[item["comparison"]],
                            *[score(100 * v) for v in values["folds_1_to_5"]],
                            score(100 * values["mean"], 100 * values["sample_sd"])])
    return {
        "scaling.tex": table("lrrrr", "Encoder & Raw, no penalty & Raw, $C=1$ & Standardized, no penalty & Standardized, $C=1$", rows),
        "scaling-regions.tex": table("llrrrrr", "Encoder & Scaling, penalty & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5", regional),
        "scaling-numerical.tex": table("llrrr", "Encoder & Scaling, penalty & Iterations in folds 1 to 5 & Warnings & Residual concerns", numerical),
        "scaling-classes.tex": table("llrrrrrr", "Encoder & Scaling, penalty & " + " & ".join(terrain), classes),
        "scaling-differences.tex": table("llrrrrrr", "Encoder & Difference & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5 & Mean and SD", differences),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Compare existing tables without writing")
    args = parser.parse_args()
    outputs = render()
    if args.check:
        for name, text in outputs.items():
            if (args.output / name).read_text() != text:
                raise SystemExit(f"Table differs from accepted values: {name}")
        print(f"Verified {len(outputs)} tables against accepted inputs")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (args.output / name).write_text(text)
    inputs = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (COMMON, ADAPTATION, SENSITIVITY)}
    (args.output / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    print(f"Wrote {len(outputs)} tables")


if __name__ == "__main__":
    main()
