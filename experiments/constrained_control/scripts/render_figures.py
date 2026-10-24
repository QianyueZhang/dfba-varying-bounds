#!/usr/bin/env python3
"""Render publication assets only from the complete, frozen control campaign.

No simulation, fitting, selection, or result filtering is performed here. All
180 recorded policies and all 240 prescribed sensitivity checks are required.
The two cost conventions are retained explicitly, with the frozen shared
development-corpus allocation used in the principal complete-cost display.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(os.environ.get("DFBA_CONTROL_OUTPUT",Path(__file__).resolve().parents[1]/"reference"))
RESULTS = ROOT / "results"
OUTPUT = Path(os.environ.get("DFBA_CONTROL_FIGURES",Path(__file__).resolve().parents[3]/"local_verification/control_figures"))
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
FIGURES.mkdir(parents=True,exist_ok=True)
TABLES.mkdir(parents=True,exist_ok=True)
METHODS = ["direct_single_start", "direct_multistart", "database_warmstart", "quadratic_ridge", "mlp"]
LABELS = ["Direct single start", "Direct multistart", "Database + correction", "Quadratic + correction", "Neural + correction"]
SHORT = ["D1", "D3", "DB", "QR", "NN"]
COLORS = ["#436984", "#874B72", "#DA8C35", "#2B8A78", "#4F51A2"]
SEEDS = [3101, 3102, 3103]
TARGETS = [0.5, 1.0, 2.0]


def load(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    with (TABLES / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", metadata={"Creator": "Matplotlib; complete frozen control campaign"})
    fig.savefig(FIGURES / f"{name}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def table_file(filename, caption, label, specification, header, body, notes):
    text = "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\begin{tabular}{" + specification + "}", r"\hline",
        header + r" \\", r"\hline", *[line + r" \\" for line in body],
        r"\hline", r"\end{tabular}", r"\par\smallskip",
        r"\begin{minipage}{0.98\linewidth}\footnotesize " + notes + r"\end{minipage}",
        r"\end{table}", "",
    ])
    (TABLES / filename).write_text(text)


def main():
    protocol = load(ROOT / "protocol.json")
    complete = load(RESULTS / "campaign_complete.json")
    paths = sorted((RESULTS / "policies").glob("*.json"))
    rows = [load(path) for path in paths]
    expected = {(c, t, s, m) for c in range(4) for t in TARGETS for s in SEEDS for m in METHODS}
    actual = [(r["condition_id"], r["target_ratio"], r["seed"], r["method"]) for r in rows]
    assert complete["completed_runs"] == complete["expected_runs"] == 180
    assert len(rows) == len(set(actual)) == 180 and set(actual) == expected, "Incomplete or duplicated campaign"
    assert len(protocol["conditions"]["test"]) == 4
    sensitivity = load(RESULTS / "initial_condition_sensitivity.json")
    sr = sensitivity["cases"]
    case_order = [(c, t, f"condition_{c}_target_{t:g}") for c in range(4) for t in TARGETS]
    sens_expected = {(case, m, a, b) for _, _, case in case_order for m in METHODS for a in [.95, 1.05] for b in [.95, 1.05]}
    sens_actual = [(r["case_id"], r["method"], r["ratio_multiplier"], r["glucose_multiplier"]) for r in sr]
    assert len(sr) == len(set(sens_actual)) == 240 and set(sens_actual) == sens_expected, "Incomplete sensitivity campaign"
    ratio_tol = np.log(1.01)
    volume_cap = protocol["operating_domain"]["terminal_volume_cap_L"]
    volume_slack = protocol["operating_domain"]["acceptance_volume_slack_L"]
    for r in rows:
        v = r["tight_verification"]
        assert r["condition"] == protocol["conditions"]["test"][r["condition_id"]]
        assert r["total_mechanistic_attempts"] <= 300
        if v.get("status") == "completed":
            measured = abs(v["z"] - np.log(r["target_ratio"]))
            assert np.isclose(measured, v["log_ratio_error"], rtol=0, atol=1e-12)
            computed_pass = measured <= ratio_tol and v["terminal"][0] <= volume_cap + volume_slack
            assert bool(computed_pass) == v["feasible"]
            assert np.isclose(v["biomass_amount_gDW"], v["terminal"][0] * sum(v["terminal"][1:3]), rtol=1e-12)
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "grid.linewidth": .4,
    })
    by_method = {m: sorted([r for r in rows if r["method"] == m], key=lambda r: (r["condition_id"], r["target_ratio"], r["seed"])) for m in METHODS}
    costs = complete["costs"]
    train = costs["data"]["training"]["seconds"]
    val = costs["data"]["validation"]["seconds"]
    method_data = []
    for m, label in zip(METHODS, LABELS):
        rr = by_method[m]
        learned = m in ["quadratic_ridge", "mlp"]
        additional_offline = 0 if m.startswith("direct_") else train + val + (costs["fit"][m] if learned else 0)
        required_offline = additional_offline - (val if m == "database_warmstart" else 0)
        online = sum(r["online_total_seconds"] for r in rr)
        first = [r["first_feasible_ordinary_evaluation"] for r in rr if r["first_feasible_ordinary_evaluation"] is not None]
        method_data.append({
            "method": m, "label": label, "recorded_runs": len(rr),
            "verified_feasible": sum(r["tight_verification"]["feasible"] for r in rr),
            "median_mechanistic_attempts": float(np.median([r["total_mechanistic_attempts"] for r in rr])),
            "total_mechanistic_attempts": sum(r["total_mechanistic_attempts"] for r in rr),
            "median_first_ordinary_feasible_evaluation": float(np.median(first)) if first else None,
            "median_online_seconds": float(np.median([r["online_total_seconds"] for r in rr])),
            "sum_online_seconds": online,
            "training_seconds": 0 if m.startswith("direct_") else train,
            "validation_seconds_frozen_allocation": 0 if m.startswith("direct_") else val,
            "family_selection_fit_seconds": costs["fit"][m] if learned else 0,
            "complete_seconds_frozen_allocation_36_runs": online + additional_offline,
            "complete_seconds_method_required_36_runs": online + required_offline,
        })
    write_csv("publication_method_summary.csv", method_data)
    policy_data = [{
        "condition": r["condition_id"] + 1, "target_ratio": r["target_ratio"], "seed": r["seed"], "method": r["method"],
        "verified_feasible": r["tight_verification"]["feasible"],
        "biomass_amount_gDW": r["tight_verification"].get("biomass_amount_gDW"),
        "log_ratio_error": r["tight_verification"].get("log_ratio_error"),
        "volume_L": r["tight_verification"].get("terminal", [None])[0],
        "actual_mechanistic_attempts": r["total_mechanistic_attempts"],
        "online_seconds": r["online_total_seconds"],
    } for r in sorted(rows, key=lambda r: (r["condition_id"], r["target_ratio"], METHODS.index(r["method"]), r["seed"]))]
    write_csv("publication_policy_values.csv", policy_data)

    # All twelve target cases. Axes use absolute biomass and a minimum 0.30 g
    # span to avoid visually amplifying roundoff where policies coincide.
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 6.6))
    for c, t, case in case_order:
        ax = axes[c, TARGETS.index(t)]
        rr = [r for r in rows if r["case_id"] == case]
        values = [r["tight_verification"]["biomass_amount_gDW"] for r in rr if r["tight_verification"]["feasible"]]
        center = (min(values) + max(values)) / 2 if values else 0
        span = max(1.3 * (max(values) - min(values)), .30) if values else 1
        for j, m in enumerate(METHODS):
            mm = sorted([r for r in rr if r["method"] == m], key=lambda r: r["seed"])
            good = []
            for k, r in enumerate(mm):
                v = r["tight_verification"]
                if v["feasible"]:
                    good.append(v["biomass_amount_gDW"])
                    ax.scatter(j + (k - 1) * .14, v["biomass_amount_gDW"], s=13, marker=["o", "s", "^"][k], c=COLORS[j], edgecolors="white", linewidths=.25, zorder=3)
                else:
                    ax.scatter(j + (k - 1) * .14, .045, marker="x", s=24, c=COLORS[j], transform=ax.get_xaxis_transform(), zorder=3)
            if good:
                ax.plot([j - .28, j + .28], [np.median(good)] * 2, color=COLORS[j], lw=1.1, zorder=2)
        ax.set_ylim(center - .5 * span, center + .5 * span)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_xlim(-.55, 4.55)
        ax.set_xticks(range(5), SHORT)
        ax.set_title(f"C{c + 1} | target SC/KP = {t:g}", loc="left", pad=4)
        ax.grid(axis="y", alpha=.25)
        if TARGETS.index(t) == 0:
            ax.set_ylabel("Feasible biomass (gDW)")
    handles = [Line2D([0], [0], marker="o", linestyle="none", color=co, markersize=4, label=f"{short}: {label}") for co, short, label in zip(COLORS, SHORT, LABELS)]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.01), frameon=False, columnspacing=1.2)
    fig.text(.5, .005, "Three algorithm seeds per method/case; horizontal marks: medians. All 180 selected policies pass the 1% multiplicative criterion.", ha="center", fontsize=7)
    fig.subplots_adjust(top=.89, bottom=.075, left=.095, right=.99, hspace=.62, wspace=.43)
    save_figure(fig, "extension_biomass_all_cases")

    # A compact engineering view of every case: select the largest recorded
    # feasible biomass among the same fifteen runs in each case, then report
    # that policy's volume. This is descriptive pooling, not a frontier search.
    tradeoff_rows = []
    for c, t, case in case_order:
        candidates = [r for r in rows if r["case_id"] == case and r["tight_verification"]["feasible"]]
        best = max(candidates, key=lambda r: r["tight_verification"]["biomass_amount_gDW"])
        v = best["tight_verification"]
        tradeoff_rows.append({
            "condition": c+1, "initial_SC_over_KP_ratio": best["condition"][0],
            "initial_glucose_g_L": best["condition"][1], "target_ratio": t,
            "pooled_feasible_policies": len(candidates), "pooled_recorded_policies": 15,
            "best_recorded_biomass_gDW": v["biomass_amount_gDW"],
            "same_policy_terminal_volume_L": v["terminal"][0],
            "selected_method": best["method"], "selected_seed": best["seed"],
            "same_policy_log_ratio_error": v["log_ratio_error"],
            "same_policy_minimum_glucose_g_L": v["minimum_glucose_g_L"],
            "interpretation": "Best recorded feasible result from fifteen runs; not a proven optimum or Pareto frontier",
        })
    write_csv("publication_engineering_tradeoff.csv", tradeoff_rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.65), gridspec_kw={"width_ratios": [1.15, 1]})
    condition_colors = ["#436984", "#DA8C35", "#2B8A78", "#874B72"]
    condition_markers = ["o", "s", "^", "D"]
    condition_handles = []
    for c, pair in enumerate(protocol["conditions"]["test"]):
        values = [r for r in tradeoff_rows if r["condition"] == c+1]
        for ax, key in [(axes[0], "best_recorded_biomass_gDW"), (axes[1], "same_policy_terminal_volume_L")]:
            ax.plot(TARGETS, [r[key] for r in values], color=condition_colors[c], marker=condition_markers[c], markersize=4, linewidth=1.1, alpha=.95, zorder=4-c*.1)
        condition_handles.append(Line2D([0], [0], color=condition_colors[c], marker=condition_markers[c], lw=1.1, markersize=4, label=f"C{c+1}: initial ratio {pair[0]:.3f}; glucose {pair[1]:.2f} g/L"))
    for ax in axes:
        ax.set_xticks(TARGETS, ["0.5", "1", "2"])
        ax.set_xlim(.4, 2.1)
        ax.set_xlabel("Target SC/KP ratio")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Best recorded feasible biomass (gDW)")
    axes[0].set_title("a  Biomass depends on the starting condition", loc="left")
    axes[0].set_ylim(77, 105)
    axes[1].set_ylabel("Terminal volume of the same policy (L)")
    axes[1].set_title("b  Volume used by the selected policy", loc="left")
    axes[1].axhline(1.8, color="#555555", ls="--", lw=.7, zorder=1)
    axes[1].text(.44, 1.812, "Volume cap", fontsize=7, color="#555555")
    axes[1].set_ylim(1.48, 1.86)
    fig.legend(handles=condition_handles, loc="lower center", ncol=2, bbox_to_anchor=(.5, .035), frameon=False, columnspacing=1.4)
    fig.text(.5, -.012, "Best found among 15 recorded feasible policies per case (5 methods × 3 algorithm seeds); descriptive comparison, not proven optima.", ha="center", fontsize=7)
    fig.subplots_adjust(left=.085, right=.99, top=.88, bottom=.30, wspace=.37)
    save_figure(fig, "extension_engineering_tradeoff")

    # Measured calls and complete component costs. The policy proposal and
    # final tight check are already inside online_total_seconds: no double count.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), gridspec_kw={"width_ratios": [1, 1.08]})
    ax = axes[0]
    for j, m in enumerate(METHODS):
        vals = [r["total_mechanistic_attempts"] for r in by_method[m]]
        offsets = np.linspace(-.24, .24, len(vals))
        ax.scatter(j + offsets, vals, s=10, alpha=.64, c=COLORS[j], edgecolors="none")
        ax.plot([j - .29, j + .29], [np.median(vals)] * 2, color="#172A36", lw=1.5)
    ax.axhline(300, color="#555555", ls="--", lw=.7)
    ax.set_ylim(0, 327)
    ax.set_xticks(range(5), SHORT)
    ax.set_ylabel("Mechanistic attempts per run")
    ax.set_title("a  Actual evaluations (36 runs/method)", loc="left")
    ax.grid(axis="y", alpha=.25)
    ax = axes[1]
    x = np.arange(5)
    online = np.array([r["sum_online_seconds"] for r in method_data])
    trainv = np.array([r["training_seconds"] for r in method_data])
    valv = np.array([r["validation_seconds_frozen_allocation"] for r in method_data])
    fitv = np.array([r["family_selection_fit_seconds"] for r in method_data])
    ax.bar(x, online, color=COLORS, width=.62, label="All online work")
    ax.bar(x, trainv, bottom=online, color="#C4CBD0", edgecolor="white", linewidth=.35, width=.62, label="Training labels")
    ax.bar(x, valv, bottom=online+trainv, color="#7B858D", edgecolor="white", linewidth=.35, width=.62, label="Validation labels")
    ax.bar(x, fitv, bottom=online+trainv+valv, color="#263B4A", width=.62, label="Model selection/fit")
    for j, r in enumerate(method_data):
        y = r["complete_seconds_frozen_allocation_36_runs"]
        ax.text(j, y+3, f"{y:.1f}", ha="center", fontsize=7)
    db_required = method_data[2]["complete_seconds_method_required_36_runs"]
    ax.plot([1.63, 2.37], [db_required, db_required], color="#222222", lw=1.2, ls=":", zorder=5)
    ax.set_xticks(range(5), SHORT)
    ax.set_ylabel("Seconds for 36 runs + offline work")
    ax.set_title("b  Complete cost: frozen allocation", loc="left")
    ax.set_ylim(0, max(online+trainv+valv+fitv)*1.14)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    legend_handles = [Patch(facecolor="#697D8C", label="All online work"), Patch(facecolor="#C4CBD0", label="Training labels"), Patch(facecolor="#7B858D", label="Validation labels"), Patch(facecolor="#263B4A", label="Model selection/fit"), Line2D([0], [0], color="#222222", ls=":", label="DB required-only alternative")]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, bbox_to_anchor=(.5, .025), frameon=False, columnspacing=1.1)
    fig.text(.5, -.015, "D1: single start; D3: multistart; DB: database; QR: quadratic; NN: neural. The pre-existing response table is common to all methods.", ha="center", fontsize=6.8)
    fig.subplots_adjust(left=.09, right=.99, top=.89, bottom=.28, wspace=.39)
    save_figure(fig, "extension_evaluations_complete_cost")

    # Fixed-policy sensitivity includes every prescribed corner, and is not an
    # estimated probability of robustness or independent biological evidence.
    counts = np.zeros((5, 12))
    errors = np.zeros((5, 12))
    sensitivity_data = []
    for j, m in enumerate(METHODS):
        for k, (c, t, case) in enumerate(case_order):
            subset = [r for r in sr if r["case_id"] == case and r["method"] == m]
            counts[j, k] = sum(r["feasible"] for r in subset)
            completed_errors = [r["ratio_relative_error_percent"] for r in subset if r.get("status") == "completed"]
            errors[j, k] = max(completed_errors) if completed_errors else np.nan
            sensitivity_data.append({"condition": c+1, "target_ratio": t, "method": m, "passed_corners": int(counts[j, k]), "corners": 4, "maximum_relative_ratio_error_percent": errors[j, k], "completed_corners": len(completed_errors), "fixed_policy_seed": 3101})
    write_csv("publication_sensitivity_cases.csv", sensitivity_data)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7))
    for ax, matrix, title, cmap, vmax, fmt in [
        (axes[0], counts, "a  Corners satisfying the nominal acceptance criterion (out of 4)", "Blues", 4, "d"),
        (axes[1], errors, "b  Maximum relative ratio error across the four corners (%)", "YlOrBr", max(10, np.nanmax(errors)), ".1f"),
    ]:
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
        for j in range(5):
            for k in range(12):
                v = matrix[j, k]
                value_text = str(int(v)) if fmt == "d" else f"{v:.1f}"
                ax.text(k, j, value_text, ha="center", va="center", fontsize=7, color="white" if v > vmax * .65 else "#162B3A")
        ax.set_yticks(range(5), SHORT)
        ax.set_xticks(range(12), [f"{t:g}" for _, t, _ in case_order])
        ax.set_title(title, loc="left", pad=28)
        for boundary in [2.5, 5.5, 8.5]:
            ax.axvline(boundary, color="white", lw=2)
        for c in range(4):
            ax.text(3*c+1, -.83, f"C{c+1}", ha="center", va="center", fontsize=7, clip_on=False)
        ax.set_xlabel("Target SC/KP ratio within each of four initial conditions")
        cbar = fig.colorbar(im, ax=ax, fraction=.018, pad=.02)
        if fmt == "d":
            cbar.set_ticks([0, 1, 2, 3, 4])
    fig.text(.5, .004, "Fixed policies from seed 3101; initial ratio and glucose independently multiplied by 0.95 or 1.05. Descriptive sensitivity only.", ha="center", fontsize=7)
    fig.subplots_adjust(left=.07, right=.95, top=.86, bottom=.10, hspace=.95)
    save_figure(fig, "extension_initial_condition_sensitivity")

    table_file("extension_method_summary.tex", "Recorded performance across 12 target cases and three algorithm seeds per case.", "tab:extension-method-summary", "lrrrr", r"Method & Feasible & Calls, median & Online/run (s), median & Complete (s)", [f"{SHORT[i]} & {r['verified_feasible']}/36 & {r['median_mechanistic_attempts']:.1f} & {r['median_online_seconds']:.3f} & {r['complete_seconds_frozen_allocation_36_runs']:.3f}" for i,r in enumerate(method_data)], r"D1, direct single start; D3, direct multistart; DB, database plus correction; QR, quadratic surrogate plus correction; NN, neural surrogate plus correction. Calls count distinct attempted mechanistic controls, including failed attempts, under a cap of 300 per run; the selected policy receives a separate tighter verification. Complete cost is the sum over all 36 recorded optimization runs plus the one-time offline work under the frozen shared-development-corpus allocation. Online time already includes proposal work and final verification. Feasibility uses $|\log[(X_{\rm SC}/X_{\rm KP})/r^\star]|\leq\log(1.01)$ and $V_f\leq1.8+10^{-8}$ L. Four unique initial conditions underlie the 12 target cases.")
    table_file("extension_cost_ledger.tex", "Measured component costs and both transparent offline allocation conventions (seconds).", "tab:extension-cost-ledger", "lrrrrrr", r"Method & Train & Valid. & Fit/select & Online & Frozen total & Required total", [f"{SHORT[i]} & {r['training_seconds']:.3f} & {r['validation_seconds_frozen_allocation']:.3f} & {r['family_selection_fit_seconds']:.3f} & {r['sum_online_seconds']:.3f} & {r['complete_seconds_frozen_allocation_36_runs']:.3f} & {r['complete_seconds_method_required_36_runs']:.3f}" for i,r in enumerate(method_data)], r"Frozen total follows protocol amendment 02 and allocates both training and validation label generation to DB, QR, and NN. Required total is the explicitly reported alternative: DB uses only training labels, so its unused validation-label cost is removed. The two learned methods require validation for selection. The alternative changes no measurements, policies, or outcomes. Online totals cover 36 recorded runs per method and include all proposal, optimizer, mechanistic, and final-verification work. Pre-existing response-table construction is common infrastructure, excluded equally; test-label generation, sensitivity checks, and direct-solver audits are separate research-validation work.")
    conditions = [{"condition": c+1, "initial_SC_over_KP_ratio": pair[0], "initial_glucose_g_L": pair[1]} for c, pair in enumerate(protocol["conditions"]["test"])]
    write_csv("publication_conditions.csv", conditions)
    table_file("extension_conditions.tex", "The four held-out initial conditions used for the new control benchmark.", "tab:extension-conditions", "lrr", r"Condition & Initial SC/KP ratio & Initial glucose (g L$^{-1}$)", [f"C{r['condition']} & {r['initial_SC_over_KP_ratio']:.6f} & {r['initial_glucose_g_L']:.6f}" for r in conditions], r"Each condition is paired with targets $r^\star\in\{0.5,1,2\}$, giving 12 target cases. Initial volume is 0.5 L and total biomass concentration is 0.002 gDW L$^{-1}$ in every case. Three algorithm seeds are used per method and target case. These are four simulated initial-condition clusters, not independent biological replicates.")
    case_table = []
    case_data = []
    for c, t, case in case_order:
        line = [f"C{c+1}", f"{t:g}"]
        for m in METHODS:
            rr = [r for r in rows if r["case_id"] == case and r["method"] == m]
            feasible_b = [r["tight_verification"]["biomass_amount_gDW"] for r in rr if r["tight_verification"]["feasible"]]
            median = float(np.median(feasible_b)) if feasible_b else None
            line.append(f"{median:.3f}" if median is not None else "---")
            case_data.append({"condition": c+1, "target_ratio": t, "method": m, "feasible_repeats": len(feasible_b), "median_feasible_biomass_gDW": median, "minimum_feasible_biomass_gDW": min(feasible_b) if feasible_b else None, "maximum_feasible_biomass_gDW": max(feasible_b) if feasible_b else None})
        case_table.append(" & ".join(line))
    write_csv("publication_case_biomass.csv", case_data)
    table_file("extension_case_biomass.tex", "Median feasible terminal biomass amount (gDW) for every target case.", "tab:extension-case-biomass", "lrrrrrr", r"Condition & Target & D1 & D3 & DB & QR & NN", case_table, r"All three selected policies per method and case pass the tighter verification, giving 180/180 feasible selections. Values are medians over the three recorded algorithm seeds and are rounded for display. Full precision and per-seed results are retained in the accompanying CSV files. These finite-budget values are not proven global optima; differences that disappear at the displayed precision should not be interpreted as method separation.")
    sens_table = []
    for j, m in enumerate(METHODS):
        subset = [r for r in sr if r["method"] == m]
        sens_table.append(f"{SHORT[j]} & {sum(r['feasible'] for r in subset)}/48 & {max(r['ratio_relative_error_percent'] for r in subset if r.get('status') == 'completed'):.3f}")
    table_file("extension_sensitivity_summary.tex", "Open-loop sensitivity of the fixed seed-3101 policies to prescribed initial-condition perturbations.", "tab:extension-sensitivity", "lrr", r"Method & Passing corners & Maximum relative ratio error (\%)", sens_table, r"For each of the 12 target cases, the initial SC/KP ratio and initial glucose are independently multiplied by 0.95 or 1.05, giving four corners. The feed policy is held fixed. The nominal 1\% multiplicative acceptance criterion is applied without retuning. These 48 correlated deterministic perturbations per method describe sensitivity; they do not estimate a probability of robustness. Relative ratio error is $100|(X_{\rm SC}/X_{\rm KP})/r^\star-1|$ and is distinct from the symmetric log-ratio acceptance criterion.")

    direct_paths = []
    direct_audit_path = RESULTS / "direct_checks_audit.json"
    if direct_audit_path.exists():
        direct_audit = load(direct_audit_path)
        assert direct_audit["status"] == "passed" and all(c["passed"] for c in direct_audit["checks"])
        direct = sorted(direct_audit["comparisons"], key=lambda r: ["direct_multistart", "mlp"].index(r["method"]))
        assert len(direct) == 2
        for r in direct:
            assert r["condition_id"] == 0 and r["target_ratio"] == 1 and r["seed"] == 3101
            assert r["nominal_feasible"]
        write_csv("publication_direct_verification.csv", [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in direct])
        def scientific(value):
            mantissa, exponent = f"{value:.2e}".split("e")
            return "$" + mantissa + r"\times10^{" + str(int(exponent)) + "}$"
        outcome_rows = []
        for label, key, fmt in [
            (r"Direct total biomass (gDW)", "biomass_direct_gDW", ".3g"),
            (r"Tighter-table total biomass (gDW)", "biomass_tight_table_gDW", ".3g"),
            (r"Direct minus table biomass (\%)", "biomass_relative_difference_percent", ".3g"),
            (r"Direct absolute log-ratio error", "absolute_log_ratio_error_direct", ".3g"),
            (r"Tighter-table absolute log-ratio error", "absolute_log_ratio_error_tight_table", ".3g"),
            (r"Nominal log-ratio margin", "nominal_ratio_log_margin", ".3g"),
            (r"Strict volume margin (L)", "volume_raw_margin_L", "scientific"),
            (r"Volume acceptance margin (L)", "volume_acceptance_margin_L", "scientific"),
        ]:
            outcome_rows.append(label + " & " + " & ".join(scientific(r[key]) if fmt == "scientific" else format(r[key], fmt) for r in direct))
        outcome_rows.extend([
            r"Terminal volume (L) & " + " & ".join(f"{r['terminal_direct'][0]:.3f}" for r in direct),
            r"Terminal glucose (g L$^{-1}$) & " + " & ".join(f"{r['terminal_direct'][3]:.3g}" for r in direct),
            r"Strict $V_f\leq1.8$ L & " + " & ".join("Yes" if r["volume_strict_pass"] else "No" for r in direct),
            r"Nominal acceptance with volume slack & Yes & Yes",
        ])
        table_file("extension_direct_outcomes.tex", "Prospectively selected direct-backend checks of two policies at condition C1, target ratio 1, and seed 3101.", "tab:extension-direct-outcomes", "lrr", r"Quantity & D3 & NN", outcome_rows, r"D3 is direct multistart optimization of the tabulated reactor; NN is the neural proposal followed by mechanistic correction. This table reports separate final trajectory checks using the full intracellular backend without interpolation. Strict volume margin is $1.8-V_f$; acceptance margin includes the prespecified $10^{-8}$ L slack. The D3 endpoint exceeds 1.8 L by approximately $9.51\times10^{-10}$ L and passes only the stated tolerance-based volume rule. These two policies were selected before outcomes were evaluated, regardless of method ranking. Stored digits identify numerical results, not biological precision.")
        diagnostic_rows = []
        for label, key, fmt in [
            (r"Maximum equality residual", "max_equality_residual", "scientific"),
            (r"Maximum primal--dual stopping residual", "max_primal_dual_residual", "scientific"),
            (r"Maximum explicit barrier stationarity residual", "max_explicit_barrier_stationarity_residual", "scientific"),
            (r"Minimum flux slack", "minimum_slack", "scientific"),
            (r"Reactor RHS evaluations", "rhs_evaluations", "d"),
            (r"Intracellular KKT solve attempts", "KKT_solve_attempts", "d"),
            (r"Failed KKT attempts", "failed_KKT_attempts", "d"),
            (r"Per-job concurrent wall time (s)", "wall_seconds_under_concurrency", ".3f"),
        ]:
            diagnostic_rows.append(label + " & " + " & ".join(scientific(r[key]) if fmt == "scientific" else format(r[key], fmt) for r in direct))
        table_file("extension_direct_diagnostics.tex", "Numerical diagnostics for the two new direct-backend trajectory checks.", "tab:extension-direct-diagnostics", "lrr", r"Quantity & D3 & NN", diagnostic_rows, r"The primal--dual stopping residual and the separately recomputed eliminated-barrier stationarity residual are distinct quantities. Only the former is below the solver's $10^{-11}$ stopping tolerance; the explicit barrier residual is not claimed to meet that threshold. Both trajectories retain positive flux slacks. The jobs ran concurrently after the timed policy campaign; their common launch-to-detected-completion wall time was " + f"{direct_audit['common_launch_to_detected_completion_seconds']:.3f}" + r" s, including at most one 5 s completion-poll interval. Their individual wall times must not be added to estimate concurrent elapsed time or compared as optimizer timings. Both checks use the original intracellular solver and matrices with a separate amount-balance reactor implementation; this is numerical verification of two nondepleted trajectories, not independent biological validation or certification of all 180 policies.")
        direct_paths = [direct_audit_path, RESULTS / "DIRECT_VERIFICATION_AUDIT.json", RESULTS / "direct_checks_execution.json", RESULTS / "direct_direct_multistart.json", RESULTS / "direct_mlp.json"]

    captions = {'extension_engineering_tradeoff': 'Terminal biomass amount (a) and volume (b) for the best recorded feasible policy in each of four initial conditions and three composition targets. The dashed line is the volume cap; connecting lines guide the eye.', 'extension_biomass_all_cases': 'Terminal biomass amount in the twelve target cases. Symbols identify the three algorithm seeds; horizontal marks show medians. D1: direct single start; D3: direct multistart; DB: database; QR: quadratic surrogate; NN: neural surrogate. DB, QR and NN include mechanistic correction.', 'extension_evaluations_complete_cost': 'Mechanistic evaluations per run (a) and total measured cost for 36 decisions per method (b). Stacks separate online calculation, label generation and fitting. The DB marker gives the training-only cost alternative.', 'extension_initial_condition_sensitivity': 'Fixed-policy response to four combinations of ±5% initial composition and glucose perturbations: number of corners satisfying the acceptance rules (a) and maximum relative composition error (b).'}
    (FIGURES / "captions.json").write_text(json.dumps(captions, indent=2) + "\n")
    (FIGURES / "README.md").write_text("# Publication figures from the completed control extension\n\n" + "\n\n".join(f"**{name}.pdf / .png**\n\n{caption}" for name, caption in captions.items()) + "\n\nRegenerate with `../scripts/render_publication_figures.py` using an environment with NumPy and Matplotlib. The script refuses incomplete or duplicated policy/sensitivity coverage. All numeric source values are retained under `../tables/publication_*.csv`.\n")

    # Independent agreement check against the separately calculated summaries,
    # when they are available, without trusting them as plotting inputs.
    agreement = "summary.json not yet available; raw-input checks completed"
    if (RESULTS / "summary.json").exists():
        other = {r["method"]: r for r in load(RESULTS / "summary.json")["method_summary"]}
        for r in method_data:
            o = other[r["method"]]
            assert r["verified_feasible"] == o["verified_feasible"]
            assert r["median_mechanistic_attempts"] == o["median_actual_calls"]
            assert np.isclose(r["sum_online_seconds"], o["sum_online_seconds"], atol=1e-8)
            assert np.isclose(r["complete_seconds_frozen_allocation_36_runs"], o["sum_online_seconds"] + o["offline_seconds_conservative_full_development_bundle"], atol=1e-8)
        agreement = "All method feasibility, median calls, online totals, and frozen full costs match independently generated summary.json"
    source_paths = [ROOT / "protocol.json", ROOT / "protocol_amendment_02.json", RESULTS / "campaign_complete.json", RESULTS / "initial_condition_sensitivity.json", *paths, *direct_paths]
    manifest = {
        "status": "validated complete campaign; figures and tables generated",
        "policy_rows": 180, "sensitivity_rows": 240, "unique_initial_conditions": 4,
        "target_cases": 12, "algorithm_seeds": SEEDS,
        "acceptance_log_ratio_tolerance": float(ratio_tol),
        "acceptance_volume_slack_L": volume_slack,
        "feasible_selected_policies": sum(r["tight_verification"]["feasible"] for r in rows),
        "direct_backend_checked_policies": 2 if direct_paths else 0,
        "independent_summary_agreement": agreement,
        "cost_convention": "Frozen development corpus allocation is principal; database training-only required-cost alternative is displayed explicitly.",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "generated_figures": [f"{name}.{ext}" for name in captions for ext in ["pdf", "png"]],
        "excluded": ["partial campaign results", "fabricated uncertainty intervals", "global optimality claims", "probabilistic robustness claims"],
    }
    (TABLES / "publication_asset_validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["status", "policy_rows", "sensitivity_rows", "unique_initial_conditions", "feasible_selected_policies", "independent_summary_agreement"]}, indent=2))


if __name__ == "__main__":
    main()
