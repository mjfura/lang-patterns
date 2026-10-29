"""Draw the paper's figures from the experiments' saved results.

Run from the repository root:

    uv run python paper/clean/figures/make_figures.py

Reads only results JSON under ``experiments/`` (no model is trained, no experiment is run) and
writes ``fig*.png`` (300 dpi) and ``fig*.pdf`` next to this script.

- fig1_method: schematic of the indicator (no data).
- fig2_recipe: recipe grammar, ``experiments/syntactic_sufficiency/results/{small,medium}/``.
- fig3_dose: dose and response on FineWeb, ``experiments/dose_response/results/summary.json``.
- fig4_attribution: corpus or model, ``experiments/model_sweep/results/summary.json``,
  ``experiments/scale/results/summary.json`` and ``experiments/class_patterns/results/
  {summary,replication_summary}.json``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPERIMENTS = ROOT / "experiments"

CM = 1 / 2.54
FULL_WIDTH = 16 * CM
SINGLE_WIDTH = 8.4 * CM

# Okabe-Ito: colorblind-safe, and the three main hues differ in lightness for grayscale.
BLACK = "#000000"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
GREY = "#7F7F7F"
LIGHT = "#EDEDED"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "hatch.linewidth": 0.4,
        "hatch.color": "white",
    }
)


def load(path: Path) -> Any:  # noqa: ANN401 (JSON of any shape)
    return json.loads(path.read_text())


def save(fig: Figure, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {name}.png / .pdf")


def panel_letter(ax: Axes, letter: str, x: float = -0.16, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


# --------------------------------------------------------------------------------------------
# Figure 1: the method
# --------------------------------------------------------------------------------------------


def box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = "white",
    edge: str = BLACK,
    lw: float = 0.7,
    dashed: bool = False,
    radius: float = 0.12,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            linestyle=(0, (3, 2)) if dashed else "solid",
        )
    )


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    style: str = "-|>",
    connection: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=7,
            linewidth=0.8,
            color=BLACK,
            connectionstyle=connection,
            shrinkA=0,
            shrinkB=0,
        )
    )


def chip(
    ax: Axes, x: float, y: float, w: float, label: str, flag: bool, dashed: bool = False
) -> None:
    h = 0.42
    fill = "#F6D4BF" if flag else LIGHT
    if dashed:
        fill = "white"
    box(
        ax,
        x,
        y,
        w,
        h,
        fill=fill,
        edge=VERMILLION if flag else GREY,
        lw=0.7,
        dashed=dashed,
        radius=0.08,
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.3, style="italic")


def mini_tree(ax: Axes, x0: float, y0: float, w: float, h: float) -> None:
    """A small dendrogram: filler classes at a cut (dashed), head classes at every node."""
    leaves = np.linspace(x0, x0 + w, 8)
    # Merge structure: pairs, then quads, then root; heights in box units.
    h1, h2, h3 = y0 + 0.30 * h, y0 + 0.62 * h, y0 + 0.95 * h
    centers1 = []
    for i in range(0, 8, 2):
        a, b = leaves[i], leaves[i + 1]
        ax.plot([a, a, b, b], [y0, h1, h1, y0], color=BLACK, lw=0.6)
        centers1.append((a + b) / 2)
    centers2 = []
    for i in range(0, 4, 2):
        a, b = centers1[i], centers1[i + 1]
        ax.plot([a, a, b, b], [h1, h2, h2, h1], color=BLACK, lw=0.6)
        centers2.append((a + b) / 2)
    a, b = centers2
    ax.plot([a, a, b, b], [h2, h3, h3, h2], color=BLACK, lw=0.6)
    # Every internal node is a head class.
    for cx in centers1:
        ax.plot(cx, h1, "o", ms=2.6, color=BLUE, zorder=3)
    for cx in centers2:
        ax.plot(cx, h2, "o", ms=2.6, color=BLUE, zorder=3)
    ax.plot((a + b) / 2, h3, "o", ms=2.6, color=BLUE, zorder=3)
    # The filler-class cut.
    cut = (h2 + h3) / 2
    ax.plot([x0 - 0.12, x0 + w + 0.12], [cut, cut], color=VERMILLION, lw=0.8, linestyle=(0, (3, 2)))


def draws_inset(fig: Figure, rect: tuple[float, float, float, float]) -> None:
    """Schematic of one slot read against the draws: their spread, the bar, the corpus."""
    ax = fig.add_axes(rect)
    x = np.linspace(-3.2, 8.2, 400)
    density = np.exp(-0.5 * x**2)
    ax.fill_between(x, density, color=SKY, alpha=0.55, linewidth=0)
    ax.plot(x, density, color=BLUE, lw=0.7)
    bar = 4.2
    corpus = 6.9
    ax.plot([bar, bar], [0, 1.0], color=BLACK, lw=0.8, linestyle=(0, (3, 2)))
    ax.annotate(
        "",
        xy=(corpus, 0.0),
        xytext=(corpus, 0.62),
        arrowprops={"arrowstyle": "-|>", "color": VERMILLION, "lw": 1.0, "mutation_scale": 6},
    )
    ax.text(0, 1.08, "draws", ha="center", va="bottom", fontsize=6.8, color=BLUE)
    ax.text(bar, 1.08, "bar", ha="center", va="bottom", fontsize=6.8)
    ax.text(corpus, 0.68, "corpus", ha="center", va="bottom", fontsize=6.8, color=VERMILLION)
    ax.set_xlim(-3.4, 8.6)
    ax.set_ylim(0, 1.45)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"$t$ of $\it{she}$ $\rightarrow$ verb", fontsize=6.8, labelpad=1)
    ax.patch.set_alpha(0)


def fig1_method() -> None:
    width, height = 16.0, 7.9
    fig = plt.figure(figsize=(width * CM, height * CM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    top, top_h = 4.95, 2.85
    mid = top + top_h / 2

    # 1. Corpus.
    box(ax, 0.1, top, 2.5, top_h)
    ax.text(1.35, top + top_h - 0.32, "Corpus", ha="center", va="top", fontweight="bold")
    ax.text(
        1.35,
        top + top_h - 0.80,
        "the text\nto diagnose",
        ha="center",
        va="top",
        fontsize=7.3,
        linespacing=1.15,
    )
    box(ax, 0.32, top + 0.22, 2.06, 0.95, dashed=True, edge=GREY, radius=0.08)
    ax.text(
        1.35,
        top + 0.695,
        "reference\n(optional)",
        ha="center",
        va="center",
        fontsize=7.3,
        color="#444444",
        linespacing=1.1,
    )

    # 2. Tokenizer.
    box(ax, 3.05, top, 2.75, top_h)
    ax.text(4.425, top + top_h - 0.32, "BPE tokenizer", ha="center", va="top", fontweight="bold")
    ax.text(
        4.425,
        top + top_h - 0.80,
        "learned on the domain\nto the fewest merges\nleaving 1.50 tokens\nper word",
        ha="center",
        va="top",
        fontsize=7.3,
        linespacing=1.15,
    )

    # 3. Class tree.
    box(ax, 6.25, top, 4.0, top_h)
    ax.text(
        8.25, top + top_h - 0.32, "Induced class tree", ha="center", va="top", fontweight="bold"
    )
    ax.text(
        8.25,
        top + top_h - 0.72,
        "Ward on neighbor profiles (Hellinger)",
        ha="center",
        va="top",
        fontsize=7.0,
    )
    mini_tree(ax, 6.65, top + 0.30, 1.6, 1.45)
    ax.plot(8.67, top + 1.42, "o", ms=2.6, color=BLUE)
    ax.text(
        8.80, top + 1.42, "head class:\nevery node", va="center", fontsize=6.8, linespacing=1.05
    )
    ax.plot([8.52, 8.74], [top + 0.72, top + 0.72], color=VERMILLION, lw=0.8, linestyle=(0, (3, 2)))
    ax.text(
        8.80, top + 0.72, "filler classes:\none cut", va="center", fontsize=6.8, linespacing=1.05
    )

    # 4. Patterns.
    box(ax, 10.7, top, 5.2, top_h)
    ax.text(13.3, top + top_h - 0.32, "Patterns", ha="center", va="top", fontweight="bold")
    ax.text(10.95, top + top_h - 0.82, "head pattern", va="top", fontsize=7.3, fontweight="bold")
    ax.text(
        10.95,
        top + top_h - 1.17,
        r"head $\rightarrow$ filler class:  $\it{she}$"
        r" $\rightarrow$ verb",
        va="top",
        fontsize=7.3,
    )
    ax.text(10.95, top + top_h - 1.62, "class pattern", va="top", fontsize=7.3, fontweight="bold")
    ax.text(
        10.95,
        top + top_h - 1.97,
        r"class node $\rightarrow$ pooled fillers:",
        va="top",
        fontsize=7.3,
    )
    ax.text(10.95, top + top_h - 2.32, r"pronoun $\rightarrow$ verb", va="top", fontsize=7.3)

    for x0, x1 in ((2.6, 3.05), (5.8, 6.25), (10.25, 10.7)):
        arrow(ax, (x0, mid), (x1, mid))

    # Two verdict paths.
    low_h = 4.1
    low_y = 0.1
    left_x, left_w = 0.1, 7.75
    right_x, right_w = 8.15, 7.75
    # Arrows from Patterns down to both paths.
    arrow(ax, (12.0, top), (left_x + left_w / 2, low_y + low_h), connection="arc3,rad=0.0")
    arrow(ax, (14.6, top), (right_x + right_w / 2, low_y + low_h))

    for x, w, title in (
        (left_x, left_w, "(a) With a reference"),
        (right_x, right_w, "(b) Without a reference"),
    ):
        box(ax, x, low_y, w, low_h, fill="#FAFAFA")
        ax.text(x + 0.22, low_y + low_h - 0.25, title, va="top", fontweight="bold")

    # (a) text.
    ax.text(
        left_x + 0.22,
        low_y + low_h - 0.72,
        "Read each pattern against 101 random\n"
        "whole-document samples of the reference,\n"
        "matched to the corpus in bytes:\n"
        r"$t$ = distance from the draws' mean,"
        "\n"
        "in the draws' spread.",
        va="top",
        fontsize=7.3,
        linespacing=1.2,
    )
    ax.text(
        left_x + 0.22,
        low_y + 1.42,
        r"Bar: 6th largest of the 101 leave-one-out"
        "\n"
        r"maxima of $t$ ($\alpha$ = 0.06 per family).",
        va="top",
        fontsize=7.3,
        linespacing=1.2,
    )
    chips_y = low_y + 0.28
    cx = left_x + 0.22
    for label, w, flag, dashed in (
        ("under-shown", 1.85, True, False),
        ("absent", 1.25, True, False),
        ("within-noise", 1.95, False, False),
        ("unread", 1.35, False, True),
    ):
        chip(ax, cx, chips_y, w, label, flag, dashed)
        cx += w + 0.2

    # (b) text.
    ax.text(
        right_x + 0.22,
        low_y + low_h - 0.72,
        "Compare each child of a node of the head\n"
        "tree with its parent, per filler class:\n"
        "how many distinct fillers would it show had\n"
        "the parent's occurrences fallen to its two\n"
        "children at random? (101 random splits)",
        va="top",
        fontsize=7.3,
        linespacing=1.2,
    )
    ax.text(
        right_x + 0.22,
        low_y + 1.42,
        r"Bar: built from the splits as in (a),"
        "\n"
        r"at the same $\alpha$.",
        va="top",
        fontsize=7.3,
        linespacing=1.2,
    )
    cx = right_x + 0.22
    for label, w, flag, dashed in (
        ("weak", 1.25, True, False),
        ("rare", 1.25, False, False),
        ("within-noise", 1.95, False, False),
        ("unread", 1.35, False, True),
    ):
        chip(ax, cx, chips_y, w, label, flag, dashed)
        cx += w + 0.2

    # Inset in (a): one slot against the draws.
    inset_x, inset_y, inset_w, inset_h = 5.0, 1.55, 2.6, 1.55
    draws_inset(fig, (inset_x / width, inset_y / height, inset_w / width, inset_h / height))

    save(fig, "fig1_method")


# --------------------------------------------------------------------------------------------
# Figure 2: recipe grammar
# --------------------------------------------------------------------------------------------

RECIPE_VARIANTS = ("1", "2", "4", "8", "16", "20", "24", "all")
RECIPE_SEEDS = (0, 1, 2)
SIZES = {"small": (BLUE, "o", "small model"), "medium": (VERMILLION, "^", "medium model")}


def recipe_runs() -> dict[str, dict[str, list[dict[str, Any]]]]:
    base = EXPERIMENTS / "syntactic_sufficiency" / "results"
    return {
        size: {
            v: [load(base / size / f"{v}_seed{s}.json") for s in RECIPE_SEEDS]
            for v in RECIPE_VARIANTS
        }
        for size in SIZES
    }


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        r = np.empty(len(values))
        arr = np.asarray(values)
        r[order] = np.arange(len(values), dtype=float)
        # Average ranks over ties.
        for value in np.unique(arr):
            mask = arr == value
            r[mask] = r[mask].mean()
        return r

    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def fig2_recipe() -> dict[str, Any]:
    runs = recipe_runs()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, 6.0 * CM), gridspec_kw={"wspace": 0.32}
    )
    check: dict[str, Any] = {"spearman": {}, "mean_ratio": {}}

    offsets = {"small": -0.006, "medium": 0.006}
    for size, (color, marker, label) in SIZES.items():
        xs_all: list[float] = []
        ys_all: list[float] = []
        mean_cov: list[float] = []
        mean_ratio: list[float] = []
        for variant in RECIPE_VARIANTS:
            covs, ratios = [], []
            for run in runs[size][variant]:
                for verb in run["verbs"].values():
                    covs.append(verb["coverage"])
                    ratios.append(verb["ratio"])
            xs_all += covs
            ys_all += ratios
            mean_cov.append(statistics.mean(r["mean_coverage"] for r in runs[size][variant]))
            mean_ratio.append(statistics.mean(r["mean_ratio"] for r in runs[size][variant]))
        ax_a.scatter(
            np.asarray(xs_all) + offsets[size],
            ys_all,
            s=3.5,
            marker=marker,
            facecolor=color,
            edgecolor="none",
            alpha=0.28,
            rasterized=True,
        )
        ax_a.plot(
            mean_cov,
            mean_ratio,
            color=color,
            marker=marker,
            ms=3.8,
            lw=1.0,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=label,
        )
        check["mean_ratio"][size] = dict(zip(RECIPE_VARIANTS, mean_ratio, strict=True))
        check["spearman"][size] = []
        for seed_index in range(len(RECIPE_SEEDS)):
            cov, rat = [], []
            for variant in RECIPE_VARIANTS:
                for verb in runs[size][variant][seed_index]["verbs"].values():
                    cov.append(verb["coverage"])
                    rat.append(verb["ratio"])
            check["spearman"][size].append(spearman(cov, rat))

    ax_a.set_xlabel("Coverage of the verb's slot (fillers seen / possible)")
    ax_a.set_ylabel("Generalization ratio")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(-0.03, 1.12)
    ax_a.axhline(1.0, color=GREY, lw=0.5, linestyle=(0, (2, 2)), zorder=0)
    ax_a.legend(loc="upper left", handlelength=1.6)
    panel_letter(ax_a, "a")

    # (b) Held-out loss against mean ratio across variants.
    check["loss_ratio_spearman"] = {}
    for size, (color, marker, label) in SIZES.items():
        losses, ratios = [], []
        for variant in RECIPE_VARIANTS:
            seed_runs = runs[size][variant]
            ax_b.scatter(
                [r["best_heldout_loss"] for r in seed_runs],
                [r["mean_ratio"] for r in seed_runs],
                s=5,
                marker=marker,
                facecolor=color,
                edgecolor="none",
                alpha=0.35,
            )
            losses.append(statistics.mean(r["best_heldout_loss"] for r in seed_runs))
            ratios.append(statistics.mean(r["mean_ratio"] for r in seed_runs))
        ax_b.plot(
            losses,
            ratios,
            color=color,
            marker=marker,
            ms=3.8,
            lw=1.0,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=label,
        )
        check["loss_ratio_spearman"][size] = [
            spearman(
                [runs[size][v][i]["best_heldout_loss"] for v in RECIPE_VARIANTS],
                [runs[size][v][i]["mean_ratio"] for v in RECIPE_VARIANTS],
            )
            for i in range(len(RECIPE_SEEDS))
        ]
        if size == "small":
            offsets_b = {
                "1": (0.0, 0.035),
                "2": (0.0, 0.035),
                "4": (0.0, 0.035),
                "8": (0.0, 0.035),
                "16": (-0.012, 0.03),
                "20": (-0.012, 0.03),
                "24": (-0.014, -0.02),
                "all": (-0.014, 0.0),
            }
            for variant, x, y in zip(RECIPE_VARIANTS, losses, ratios, strict=True):
                dx, dy = offsets_b[variant]
                ha = "center" if dx == 0 else "right"
                ax_b.text(x + dx, y + dy, variant, fontsize=6.8, ha=ha, va="bottom", color=color)
    ax_b.set_xlabel("Held-out loss (nats per token)")
    ax_b.set_ylabel("Mean generalization ratio")
    ax_b.set_ylim(-0.03, 1.12)
    ax_b.axhline(1.0, color=GREY, lw=0.5, linestyle=(0, (2, 2)), zorder=0)
    ax_b.text(
        0.03,
        0.97,
        "labels: fillers per verb",
        transform=ax_b.transAxes,
        fontsize=6.8,
        va="top",
        color="#444444",
    )
    panel_letter(ax_b, "b")
    save(fig, "fig2_recipe")
    return check


# --------------------------------------------------------------------------------------------
# Figure 3: dose and response
# --------------------------------------------------------------------------------------------


def fig3_dose() -> dict[str, Any]:
    summary = load(EXPERIMENTS / "dose_response" / "results" / "summary.json")
    fig, ax = plt.subplots(figsize=(SINGLE_WIDTH, 6.2 * CM))
    styles = {"A": (BLUE, "o", "assignment A"), "B": (ORANGE, "s", "assignment B")}
    check: dict[str, Any] = {"planted_heads": set(), "comparison": {}}
    ax.axhline(0, color=GREY, lw=0.5, zorder=0)
    for assignment, (color, marker, label) in styles.items():
        xs, ys = [], []
        for head in summary["heads"]:
            if head["assignment"] != assignment or head["target"] is None:
                continue
            check["planted_heads"].add(head["head"])
            for seed in ("0", "1"):
                xs.append(head["diversity"])
                ys.append(head["seeds"][seed]["drop"])
        ax.scatter(
            xs,
            ys,
            s=11,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.9,
            label=label,
            zorder=3,
        )
    comp_x, comp_y = [], []
    for head in summary["heads"]:
        if head["target"] is None:
            for seed in ("0", "1"):
                comp_x.append(head["diversity"])
                comp_y.append(head["seeds"][seed]["drop"])
            check["comparison"].setdefault(head["head"], []).extend(
                v["drop"] for v in head["seeds"].values()
            )
    ax.scatter(
        comp_x,
        comp_y,
        s=11,
        marker="D",
        facecolor="white",
        edgecolor=BLACK,
        linewidth=0.6,
        label="comparison heads (unplanted)",
        zorder=3,
    )
    levels = sorted(summary["level_mean_drops"], key=float)
    ax.plot(
        [float(v) for v in levels],
        [summary["level_mean_drops"][v] for v in levels],
        color=BLACK,
        lw=1.0,
        marker="_",
        ms=6,
        label="mean per target level",
        zorder=2,
    )
    ax.set_xlabel("Diversity of the head's slot (fillers / expected fillers)")
    ax.set_ylabel("Drop at the head (nats, true next word)")
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-3.9, 0.6)
    ax.legend(loc="lower right", handlelength=1.4, borderaxespad=0.2, labelspacing=0.3)
    check["planted_heads"] = sorted(check["planted_heads"])
    check["level_mean_drops"] = summary["level_mean_drops"]
    check["spearman"] = {k: v["spearman"] for k, v in summary["readings"].items()}
    check["strong"] = {k: v["strong_mean_drop"] for k, v in summary["readings"].items()}
    check["mild"] = {k: v["mild_mean_drop"] for k, v in summary["readings"].items()}
    save(fig, "fig3_dose")
    return check


# --------------------------------------------------------------------------------------------
# Figure 4: corpus or model
# --------------------------------------------------------------------------------------------

CONFIGS = ("small", "medium", "long")
CLASS_KINDS = (
    ("pronoun", "pronouns", "control", "class_cut"),
    ("modal", "modals", "control", "class_cut"),
    ("possessive", "possessives", "poss_control", "poss_class_cut"),
)


def fig4_attribution() -> dict[str, Any]:
    sweep = load(EXPERIMENTS / "model_sweep" / "results" / "summary.json")["class_g"]
    scale = load(EXPERIMENTS / "scale" / "results" / "summary.json")["g"]
    patterns = load(EXPERIMENTS / "class_patterns" / "results" / "summary.json")
    replication = load(EXPERIMENTS / "class_patterns" / "results" / "replication_summary.json")

    fig = plt.figure(figsize=(FULL_WIDTH, 11.2 * CM))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.95), hspace=0.62, wspace=0.12)
    check: dict[str, Any] = {"best": {}, "cut_trend": {}, "delta": {}}

    series = (
        ("full", "F (full pool)", BLACK, "o", "-"),
        ("control", "unflagged control", BLUE, "s", "-"),
        ("cut", "flagged class cut", VERMILLION, "^", "-"),
        ("r10", "unflagged 10% cut", GREY, "D", (0, (2, 1.5))),
    )
    x_cfg = np.arange(len(CONFIGS), dtype=float)
    x_scale = 3.75
    first_axes: Axes | None = None
    for col, (kind, label, control, cut) in enumerate(CLASS_KINDS):
        ax = fig.add_subplot(grid[0, col], sharey=first_axes)
        if first_axes is None:
            first_axes = ax
        names = {"full": "full", "control": control, "cut": cut, "r10": "r10"}
        for key, legend, color, marker, style in series:
            values = [sweep[kind][names[key]][c] for c in CONFIGS]
            ax.plot(
                x_cfg,
                values,
                color=color,
                marker=marker,
                linestyle=style,
                ms=3.8,
                markeredgecolor="white",
                markeredgewidth=0.4,
                lw=1.0,
                label=legend,
                zorder=3 if key != "r10" else 2,
            )
        check["best"][label] = {k: max(sweep[kind][names[k]].values()) for k in names}
        check["cut_trend"][label] = [sweep[kind][cut][c] for c in CONFIGS]
        if kind in ("pronoun", "modal"):
            for key, scale_name, dx in (
                ("full", "bigf", -0.13),
                ("control", "big_control", 0.13),
                ("cut", "big_class_cut", 0.0),
            ):
                _, _, color, marker, _ = next(s for s in series if s[0] == key)
                ax.plot(
                    [x_scale + dx],
                    [scale[scale_name][kind]],
                    color=color,
                    marker=marker,
                    ms=3.8,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    linestyle="none",
                    zorder=3,
                )
            check.setdefault("scale", {})[label] = {
                k: scale[k][kind] for k in ("bigf", "big_control", "big_class_cut")
            }
            ticks = [*x_cfg, x_scale]
            ticklabels = [*CONFIGS, "scale"]
        else:
            ticks = list(x_cfg)
            ticklabels = list(CONFIGS)
            ax.text(x_scale, 0.75, "n/a", ha="center", va="center", fontsize=7, color=GREY)
        ax.axvline(3.0 + 0.375, color=GREY, lw=0.5, linestyle=(0, (1, 1.5)))
        ax.set_xticks(ticks, ticklabels)
        ax.set_xlim(-0.35, 4.15)
        ax.set_ylim(0.45, 1.03)
        ax.text(
            0.97,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=8,
            fontstyle="italic",
            va="top",
            ha="right",
        )
        if col == 0:
            ax.set_ylabel("G (share of minimal pairs won)")
            panel_letter(ax, "a", x=-0.30, y=1.08)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if col == 1:
            ax.set_xlabel("Model configuration")
    assert first_axes is not None
    handles, labels = first_axes.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.005),
        handlelength=2.0,
        columnspacing=1.4,
    )

    # (b) Head cut against class cut.
    ax = fig.add_subplot(grid[1, :])
    pron = patterns["delta_g"]
    pron_class = pron["class_cut"]
    pron_head = pron["head_cut"]
    pronouns = ("I", "you", "he", "she", "we", "they")
    removed_pron = ("she", "they")
    pron_sib = tuple(h for h in pronouns if h not in removed_pron)
    groups: dict[str, dict[str, list[float]]] = {
        "pronouns": {
            "class": [pron_class[h] for h in pronouns],
            "removed_class": [pron_class[h] for h in removed_pron],
            "removed_head": [pron_head[h] for h in removed_pron],
            "siblings_head": [pron_head[h] for h in pron_sib],
        }
    }
    for kind, label, removed in (
        ("modal", "modals", ("can", "will")),
        ("possessive", "possessives", ("his", "their")),
    ):
        delta = replication[kind]["delta_g"]
        members = tuple(delta["class"])
        siblings = tuple(h for h in members if h not in removed)
        groups[label] = {
            "class": [delta["class"][h] for h in members],
            "removed_class": [delta["class"][h] for h in removed],
            "removed_head": [delta["head"][h] for h in removed],
            "siblings_head": [delta["head"][h] for h in siblings],
        }
    bars = (
        ("class", "class cut: all members", VERMILLION, None),
        ("removed_class", "class cut: the two heads", "#F2B48C", None),
        ("removed_head", "head cut: the two heads", BLUE, "////"),
        ("siblings_head", "head cut: their siblings", "#A6CDE6", "////"),
    )
    width = 0.19
    for gi, (group, values) in enumerate(groups.items()):
        means = {k: statistics.mean(v) for k, v in values.items()}
        check["delta"][group] = means
        for bi, (key, legend, color, hatch) in enumerate(bars):
            x = gi + (bi - 1.5) * width
            ax.bar(
                x,
                means[key],
                width=width * 0.92,
                color=color,
                edgecolor=BLACK,
                linewidth=0.5,
                hatch=hatch,
                label=legend if gi == 0 else None,
                zorder=2,
            )
            jitter = np.linspace(-width * 0.22, width * 0.22, len(values[key]))
            ax.scatter(x + jitter, values[key], s=5, color=BLACK, zorder=3, linewidth=0)
        share = means["removed_head"] / means["removed_class"]
        ax.text(
            gi,
            -0.50,
            f"head cut / class cut: {share:.0%}",
            ha="center",
            va="center",
            fontsize=7,
            color="#333333",
        )
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(-0.55, 0.12)
    ax.set_ylabel(r"$\Delta$G against the control")
    ax.legend(
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.2
    )
    panel_letter(ax, "b", x=-0.095)
    save(fig, "fig4_attribution")
    return check


def main() -> None:
    fig1_method()
    checks = {
        "fig2_recipe": fig2_recipe(),
        "fig3_dose": fig3_dose(),
        "fig4_attribution": fig4_attribution(),
    }
    print(json.dumps(checks, indent=1, default=float))


if __name__ == "__main__":
    main()
