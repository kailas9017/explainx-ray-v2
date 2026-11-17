"""Statistics and paper figures from research/outputs/zone_eval_details.csv.

Run after evaluate_zones.py (from backend/):  python research/analyze_results.py

Writes research/outputs/:
  results_summary.json          every number quoted in the paper
  fig_zone_hit.{png,pdf}        zone-hit accuracy, grid vs anatomical, per CAM method
  fig_per_class.{png,pdf}       per-class zone hit for the primary method
  fig_empty_cam.{png,pdf}       empty-CAM rate vs model probability
  fig_asfi_area.{png,pdf}       ASFI / activated fraction vs expert box area
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, kruskal, mannwhitneyu, spearmanr

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(RESEARCH_DIR, "outputs")
DETAILS = os.path.join(OUT_DIR, "zone_eval_details.csv")

METHOD_ORDER = ["gradcam", "gradcam++", "layercam", "scorecam"]
METHOD_NAME = {"gradcam": "Grad-CAM", "gradcam++": "Grad-CAM++", "layercam": "LayerCAM", "scorecam": "Score-CAM"}
FOCAL = ["Mass", "Nodule"]
DIFFUSE = ["Infiltrate", "Pneumonia", "Effusion", "Atelectasis"]

# Reference palette (dataviz skill): categorical slots 1-2, chrome inks.
GRID_C, ANAT_C = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRIDLINE, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRIDLINE, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar_exact(a, b):
    """Exact McNemar test on paired binary outcomes a, b."""
    b01, b10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return {"anat_only": b01, "grid_only": b10, "p_value": float(p)}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    plt.close(fig)


def zone_hit_stats(lung, subset_name):
    out = {}
    for m in METHOD_ORDER:
        g = lung[lung.method == m].sort_values("box_id")
        if g.empty:
            continue
        n = len(g)
        rec = {"n": n, "cam_empty": g.cam_empty.mean(), "pointing": g.pointing.mean(),
               "energy_in_box": g.energy_in_box.mean(), "iou_otsu": g.iou_otsu.mean()}
        for p in ("grid", "anat"):
            k = int(g[f"{p}_zone_hit"].sum())
            rec[f"{p}_zone_hit"] = k / n
            rec[f"{p}_zone_hit_ci"] = wilson(k, n)
            rec[f"{p}_top2"] = g[f"{p}_zone_top2"].mean()
            rec[f"{p}_hit_any"] = g[f"{p}_zone_hit_any"].mean()
            rec[f"{p}_cam_in_zones"] = g[f"{p}_cam_in_zones"].mean()
        rec["mcnemar_grid_vs_anat"] = mcnemar_exact(g.grid_zone_hit.to_numpy(), g.anat_zone_hit.to_numpy())
        out[m] = rec
    return out


def prior_baseline(lung):
    """Leave-one-out class prior: predict the most frequent box zone among the
    *other* boxes of the same class. What a CAM must beat to show localisation
    beyond 'Effusion is usually basal'."""
    first = lung[lung.method == lung.method.iloc[0]]
    res = {}
    for p in ("grid", "anat"):
        hits = 0
        for _, g in first.groupby("label"):
            zones = g[f"{p}_box_zone"].to_numpy()
            for i in range(len(zones)):
                others = pd.Series(np.delete(zones, i))
                hits += int(not others.empty and others.value_counts().index[0] == zones[i])
        res[p] = hits / len(first)
    return res


def fig_zone_hit(stats, prior):
    methods = [m for m in METHOD_ORDER if m in stats]
    x = np.arange(len(methods))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for off, p, color, label, hatch in ((-w / 2, "grid", GRID_C, "Fixed 3x2 grid", "///"),
                                        (w / 2, "anat", ANAT_C, "Anatomy-aware zones", None)):
        vals = [stats[m][f"{p}_zone_hit"] for m in methods]
        lo = [stats[m][f"{p}_zone_hit"] - stats[m][f"{p}_zone_hit_ci"][0] for m in methods]
        hi = [stats[m][f"{p}_zone_hit_ci"][1] - stats[m][f"{p}_zone_hit"] for m in methods]
        bars = ax.bar(x + off, vals, w - 0.03, color=color, label=label, hatch=hatch,
                      edgecolor="white", linewidth=0.8)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.045, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    for p, ls in (("grid", ":"), ("anat", "--")):
        ax.axhline(prior[p], color=MUTED, ls=ls, lw=1)
    ax.text(len(methods) - 0.45, min(prior.values()) - 0.045,
            f"class prior: grid {prior['grid']:.2f} (dotted), anatomical {prior['anat']:.2f} (dashed)",
            fontsize=7, color=MUTED, ha="right")
    ax.axhline(1 / 6, color=MUTED, lw=0.8)
    ax.text(len(methods) - 0.45, 1 / 6 + 0.012, "chance (1/6)", fontsize=7, color=MUTED, ha="right")
    ax.set_xticks(x, [METHOD_NAME[m] for m in methods])
    ax.set_ylabel("Zone-hit accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_title("Peak CAM zone matches the expert-box zone (lung findings, n boxes = %d)"
                 % stats[methods[0]]["n"], fontsize=9, color=INK, loc="left")
    save(fig, "fig_zone_hit")


def fig_per_class(lung, method):
    g = lung[lung.method == method]
    per = g.groupby("label")[["grid_zone_hit", "anat_zone_hit"]].mean()
    counts = g.groupby("label").box_id.nunique()
    per = per.loc[per.anat_zone_hit.sort_values().index]
    y = np.arange(len(per))
    h = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.grid(axis="x", color=GRIDLINE, lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.barh(y + h / 2, per.grid_zone_hit, h - 0.04, color=GRID_C, hatch="///", edgecolor="white",
            linewidth=0.8, label="Fixed 3x2 grid")
    ax.barh(y - h / 2, per.anat_zone_hit, h - 0.04, color=ANAT_C, edgecolor="white", linewidth=0.8,
            label="Anatomy-aware zones")
    ax.set_yticks(y, [f"{lab} (n={counts[lab]})" for lab in per.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Zone-hit accuracy")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"Per-pathology zone hit, {METHOD_NAME[method]}", fontsize=9, color=INK, loc="left")
    save(fig, "fig_per_class")
    return per


def fig_empty_cam(details):
    bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    labels = ["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", ">=0.5"]
    d = details.copy()
    d["pbin"] = pd.cut(d.probability, bins, labels=labels, include_lowest=True)
    rates = d.pivot_table(index="pbin", columns="method", values="cam_empty", aggfunc="mean", observed=False)
    counts = d[d.method == METHOD_ORDER[0]].groupby("pbin", observed=False).size()
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    gc = rates.get("gradcam")
    if gc is not None:
        ax.bar(range(len(labels)), gc.values, 0.6, color=GRID_C, edgecolor="white")
        for i, v in enumerate(gc.values):
            ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(labels)), [f"{lab}\n(n={counts[lab]})" for lab in labels])
    ax.set_xlabel("Model probability for the annotated class")
    ax.set_ylabel("Share of empty Grad-CAM maps")
    ax.set_ylim(0, 1)
    ax.set_title("Grad-CAM returns an all-zero map when class evidence is weak\n"
                 "(Grad-CAM++, LayerCAM, Score-CAM: %s empty overall)" % ", ".join(
                     f"{details[details.method == m].cam_empty.mean():.0%}" for m in METHOD_ORDER[1:]
                     if m in set(details.method)), fontsize=9, color=INK, loc="left")
    save(fig, "fig_empty_cam")
    return rates


def fig_asfi(lung, method):
    g = lung[(lung.method == method)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, col, title in ((axes[0], "activated_fraction", "Activated fraction (Otsu)"),
                           (axes[1], "asfi", "ASFI")):
        ax.scatter(g.box_area_frac, g[col], s=9, color=GRID_C, alpha=0.55, linewidths=0)
        rho, p = spearmanr(g.box_area_frac, g[col])
        ax.set_xlabel("Expert box area (fraction of image)")
        ax.set_ylabel(title)
        ax.grid(axis="x", color=GRIDLINE, lw=0.6)
        ax.set_title(f"{title}: Spearman rho = {rho:.2f} (p = {p:.1g})", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "fig_asfi_area")


def main():
    details = pd.read_csv(DETAILS)
    lung = details[(details.label != "Cardiomegaly") & (details.seg_ok == 1)]
    held = lung[lung.split.isin(["val", "test"])]

    stats_all, stats_held = zone_hit_stats(lung, "all"), zone_hit_stats(held, "held-out")
    prior_all, prior_held = prior_baseline(lung), prior_baseline(held)
    # Primary method: best anatomical zone hit on all lung boxes.
    primary = max(stats_all, key=lambda m: stats_all[m]["anat_zone_hit"])

    fig_zone_hit(stats_all, prior_all)
    per_class = fig_per_class(lung, primary)
    empty_rates = fig_empty_cam(details)
    fig_asfi(lung, primary)

    g = lung[lung.method == primary]
    asfi = {col: dict(zip(("rho", "p"), map(float, spearmanr(g.box_area_frac, g[col]))))
            for col in ("asfi", "activated_fraction", "energy_in_box")}
    zci_focal, zci_diffuse = g[g.label.isin(FOCAL)].anat_zci, g[g.label.isin(DIFFUSE)].anat_zci
    zci = {
        "median_by_class": g.groupby("label").anat_zci.median().round(3).to_dict(),
        "focal_median": float(zci_focal.median()), "diffuse_median": float(zci_diffuse.median()),
        "mannwhitney_focal_gt_diffuse_p": float(mannwhitneyu(zci_focal, zci_diffuse, alternative="greater").pvalue),
        "kruskal_across_classes_p": float(kruskal(*[x.anat_zci for _, x in g.groupby("label")]).pvalue),
    }
    # Methods compared on anatomical zone hit (paired, same boxes).
    pairwise = {}
    for i, a in enumerate(METHOD_ORDER):
        for b in METHOD_ORDER[i + 1:]:
            ga = lung[lung.method == a].sort_values("box_id").anat_zone_hit.to_numpy()
            gb = lung[lung.method == b].sort_values("box_id").anat_zone_hit.to_numpy()
            if len(ga) and len(ga) == len(gb):
                r = mcnemar_exact(ga, gb)
                pairwise[f"{a} vs {b}"] = {f"{b}_only": r["anat_only"], f"{a}_only": r["grid_only"], "p_value": r["p_value"]}

    seg_fail = int((details[details.method == details.method.iloc[0]].seg_ok == 0).sum())
    summary = {
        "n_boxes_total": int(details.box_id.nunique()), "lung_segmentation_failures": seg_fail,
        "n_lung_boxes": int(lung.box_id.nunique()), "n_lung_boxes_heldout": int(held.box_id.nunique()),
        "primary_method": primary,
        "zone_hit_all": stats_all, "zone_hit_heldout": stats_held,
        "class_prior_baseline_all": prior_all, "class_prior_baseline_heldout": prior_held,
        "per_class_primary": per_class.round(3).to_dict(), "pairwise_methods_anat_mcnemar": pairwise,
        "empty_cam_rate_by_probability": empty_rates.round(3).to_dict(),
        "asfi_vs_box_area_spearman": asfi, "zci": zci,
        "box_inside_zones_mean": {"grid": float(lung.grid_box_in_zones.mean()), "anat": float(lung.anat_box_in_zones.mean())},
    }
    with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else float(o))
    print(json.dumps(summary, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else float(o)))


if __name__ == "__main__":
    main()
