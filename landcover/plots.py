"""Figures. Class colors are the official ESA WorldCover legend colors."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .data import WORLDCOVER  # noqa: E402

INK = "#1f2328"
MUTED = "#57606a"
GRID = "#d0d7de"
# Validation-scheme colors: random CV (optimistic) vs spatial schemes.
SCHEME_COLORS = {"random": "#b4b4b4", "spatial": "#2f5d8a", "test": "#16324f"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def class_cmap(classes):
    """Colormap and norm mapping each class code (0 = no data, white) to its legend color."""
    codes = [0] + list(classes)
    colors = ["#ffffff"] + [WORLDCOVER[c][1] for c in classes]
    bounds = np.arange(len(codes) + 1) - 0.5
    lut = {c: k for k, c in enumerate(codes)}
    return ListedColormap(colors), BoundaryNorm(bounds, len(codes)), lut


def _index_image(img, lut):
    out = np.zeros(img.shape, dtype=np.int16)
    for c, k in lut.items():
        out[img == c] = k
    return out


def _legend_handles(classes):
    return [Patch(facecolor=WORLDCOVER[c][1], edgecolor="#888888", label=WORLDCOVER[c][0]) for c in classes]


def _draw_map(ax, img, classes, title, extent_km, test_outline=None):
    cmap, norm, lut = class_cmap(classes)
    ax.imshow(_index_image(img, lut), cmap=cmap, norm=norm, interpolation="nearest", extent=extent_km)
    if test_outline is not None:
        ax.contour(
            test_outline.astype(float), levels=[0.5], colors="black", linewidths=0.8,
            extent=extent_km, origin="upper",
        )
    ax.set_title(title, loc="left")
    ax.set_xlabel("km east")
    ax.set_ylabel("km north")
    ax.set_aspect("equal")


def map_comparison(pred, ref, classes, extent_km, test_mask, path, subtitle=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4), constrained_layout=True)
    _draw_map(axes[0], pred, classes, "Random Forest classification (Sentinel-2, 20 m)", extent_km, test_mask)
    _draw_map(axes[1], ref, classes, "ESA WorldCover (reference labels)", extent_km, test_mask)
    handles = _legend_handles(classes) + [Patch(facecolor="none", edgecolor="black", label="Held-out test blocks")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.06))
    if subtitle:
        fig.suptitle(subtitle, color=MUTED, fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def confusion_figure(n, classes, ua, pa, path, title):
    """Sample count matrix (rows = map, columns = reference), shaded by row share."""
    n = np.asarray(n)
    share = n / np.maximum(n.sum(1, keepdims=True), 1)
    names = [WORLDCOVER[c][0] for c in classes]
    fig, ax = plt.subplots(figsize=(9, 7.5), constrained_layout=True)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{n[i, j]}", ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(classes)), [f"{nm}\nPA {p:.2f}" for nm, p in zip(names, pa)], rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), [f"{nm}  UA {u:.2f}" for nm, u in zip(names, ua)])
    ax.set_xlabel("Reference (WorldCover)")
    ax.set_ylabel("Map (Random Forest)")
    ax.set_title(title, loc="left")
    ax.spines[:].set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


DESIGN_COLORS = {"scattered": "#2f5d8a", "clustered": "#c4661f"}
SCHEMES = {"random_cv": "Random 5-fold CV", "spatial_cv": "Spatial block CV", "test": "Held-out spatial test"}


def _group_label(key, cluster_px):
    model, design = key.split("|")
    m = {"random_forest": "Random Forest", "hist_gradient_boosting": "Gradient boosting"}[model]
    d = "scattered pixels" if design == "scattered" else f"{cluster_px}x{cluster_px}-pixel patches"
    return f"{m}\n{d}"


SCHEME_STYLE = {
    "random_cv": dict(color=SCHEME_COLORS["random"], marker="o"),
    "spatial_cv": dict(color=SCHEME_COLORS["spatial"], marker="s"),
    "test": dict(color=SCHEME_COLORS["test"], marker="D"),
}


SCHEME_SHORT = {"random_cv": "random CV", "spatial_cv": "spatial CV", "test": "held-out test"}


def _dot_plot(ax, cv, keys, cluster_px, metric, fontsize=9, value_size=9, marker_size=9, compact=False):
    """Dot plot: one row per model/design, one sub-row per validation scheme.

    Dots encode position only, so the axis need not start at zero; the gap between
    random CV and the held-out test is printed under each row label.
    """
    offsets = {"random_cv": -0.25, "spatial_cv": 0.0, "test": 0.25}
    allv = [cv[key][sc][metric] for key in keys for sc in SCHEMES]
    lo, hi = min(allv), max(allv)
    span = hi - lo
    for i, key in enumerate(keys):
        for sc, off in offsets.items():
            v = cv[key][sc][metric]
            y = i + off
            ax.plot([lo - 0.6 * span, v], [y, y], color=GRID, linewidth=1, zorder=1)
            ax.plot(v, y, linestyle="none", markersize=marker_size, zorder=3,
                    label=SCHEMES[sc] if i == 0 else None, **SCHEME_STYLE[sc])
            txt = f"{v:.3f}  {SCHEME_SHORT[sc]}" if compact else f"{v:.3f}"
            ax.text(v + 0.06 * span, y, txt, va="center", ha="left", fontsize=value_size, color=INK)
    labels = []
    for key in keys:
        gap = cv[key]["random_cv"][metric] - cv[key]["test"][metric]
        labels.append(f"{_group_label(key, cluster_px)}\nrandom CV − test: {gap:+.3f}")
    if compact:  # labels inside the axes, above each group (for narrow layouts)
        ax.set_yticks([])
        for i, lab in enumerate(labels):
            ax.text(lo - 0.6 * span, i - 0.5, lab.replace("\n", ", ", 1), fontsize=fontsize, color=INK,
                    weight="bold", va="center", ha="left")
        ax.set_ylim(len(keys) - 0.6, -0.75)
        ax.set_xlim(lo - 0.6 * span, hi + 1.0 * span)
    else:
        ax.set_yticks(range(len(keys)), labels, fontsize=fontsize)
        ax.set_ylim(len(keys) - 0.5, -0.6)
        ax.set_xlim(lo - 0.6 * span, hi + 0.45 * span)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def cv_comparison(cv, sweeps, path, cluster_px, metric="balanced_accuracy"):
    """Left: random vs spatial CV vs held-out test per model/sampling design. Right: RF accuracy vs block size."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.3, 1]})
    _dot_plot(ax, cv, list(cv), cluster_px, metric)
    ax.set_xlabel("Balanced accuracy (mean producer's accuracy); axis zoomed, dots not bars")
    ax.set_title("Random CV vs spatial CV vs the held-out test blocks", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)

    for design, sweep in sweeps.items():
        sizes = [s for s, _ in sweep]
        xs = np.arange(len(sizes))
        vals = [v[metric] for _, v in sweep]
        sds = [v[f"{metric}_sd"] for _, v in sweep]
        label = "scattered pixels" if design == "scattered" else f"{cluster_px}x{cluster_px}-pixel patches"
        ax2.errorbar(xs, vals, yerr=sds, color=DESIGN_COLORS[design], marker="o", markersize=8, linewidth=2,
                     capsize=4, label=label)
    ax2.set_xticks(xs, ["random" if s == 0 else f"{s:g} km" for s in sizes])
    ax2.set_xlabel("Cross-validation fold unit (blocks)")
    ax2.set_ylabel("Balanced accuracy (Random Forest)")
    ax2.set_title("CV accuracy vs block size", loc="left")
    ax2.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="lower left", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def importance_figure(names, mean, sd, path, top=15):
    top = min(top, len(names))
    order = np.argsort(mean)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    ax.barh(np.arange(len(order)), np.asarray(mean)[order], xerr=np.asarray(sd)[order],
            color=SCHEME_COLORS["spatial"], height=0.7, error_kw={"ecolor": MUTED, "capsize": 3})
    ax.set_yticks(np.arange(len(order)), [names[i] for i in order])
    ax.set_xlabel("Drop in balanced accuracy when the feature group is shuffled")
    ax.set_title("Grouped permutation importance (held-out test sample)", loc="left")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def hero_figure(pred, ref, classes, extent_km, cv, test_oa, test_oa_ci, path, place, cluster_px, year=2021):
    """16:9 portfolio card: map pair plus the random-vs-spatial accuracy gap."""
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.95], height_ratios=[0.13, 1], left=0.03, right=0.98,
                          top=0.97, bottom=0.13, wspace=0.12, hspace=0.12)
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.text(0, 0.75, "Sentinel-2 land cover, honestly assessed", fontsize=24, weight="bold", color=INK,
              va="center")
    head.text(0, 0.05, f"{place}  |  {year} Sentinel-2 composites, ESA WorldCover {year} labels",
              fontsize=14, color=MUTED, va="center")
    for k, (img, title) in enumerate([(pred, "Random Forest map"), (ref, f"ESA WorldCover {year}")]):
        ax = fig.add_subplot(gs[1, k])
        cmap, norm, lut = class_cmap(classes)
        ax.imshow(_index_image(img, lut), cmap=cmap, norm=norm, interpolation="nearest", extent=extent_km)
        ax.set_title(title, loc="left", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
    ax = fig.add_subplot(gs[1, 2])
    keys = ["random_forest|scattered", "random_forest|clustered"]
    _dot_plot(ax, cv, keys, cluster_px, "balanced_accuracy", fontsize=11, value_size=11, marker_size=11,
              compact=True)
    ax.set_xlabel("Random Forest balanced accuracy\n(zoomed axis: dots, not bars)", fontsize=11, color=MUTED)
    head.text(1.0, 0.75, f"Agreement with WorldCover: {test_oa:.2f} ± {test_oa_ci:.2f}", fontsize=17, weight="bold",
              color=INK, ha="right", va="center")
    head.text(1.0, 0.05, "held-out blocks, area-weighted overall accuracy, 95% CI", fontsize=12, color=MUTED,
              ha="right", va="center")
    fig.legend(handles=_legend_handles(classes), loc="lower left", ncol=len(classes), frameon=False,
               bbox_to_anchor=(0.03, 0.01), fontsize=12)
    fig.savefig(path, dpi=100)
    plt.close(fig)
