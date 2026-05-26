from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Hugging Face ──────────────────────────────────────────────────────────────
HF_REPO = "vitaliykinakh/binary-ddpm-tabular"

# ── Datasets ──────────────────────────────────────────────────────────────────
# DATASETS = ["adult", "diabetes", "heloc", "housing", "sick", "travel"]
DATASETS = ["adult"]   # reduce for quick tests

CURRENT_DIR = Path(__file__).parent

# ── Results CSV ───────────────────────────────────────────────────────────────
RESULTS_CSV = CURRENT_DIR / "study_results.csv"

# ── Baseline parameters ───────────────────────────────────────────────────────
DEFAULT_SAMPLE_PARAMS = dict(
    n_timesteps=5,
    batch_size=2048,
    guidance_scale=5.0,
    use_ema=True,
    strategy="target",
    schedule="const",
    renoise_factor=1.0,
    use_t_next=True,
    dropna=True,
    threshold=0.5,
)

# ── Parameter grid ────────────────────────────────────────────────────────────
PARAM_GRID = dict(
    strategy=["target", "mask"],
    use_t_next=[False, True],
    renoise_factor=[1.0, 1.5, 2.0],
    n_timesteps=[2, 5, 10, 20],
    schedule=["const", "linear", "quad"],
)
PARAM_KEYS = list(PARAM_GRID.keys())

PLOT_CONFIG = {
    "sensitivity": {
        "enabled": True,
        "params_to_show": None,
    },
    "timestep_lines": {
        "enabled": True,
        "combos_to_show": None,
        "renoise_factors": None,
    },
    "heatmaps": {
        "enabled": True,
        "pairs_to_show": None,
    },
}

def load_results(csv_path):
    df = pd.read_csv(csv_path)
    for col in ["use_t_next", "is_baseline"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    baseline_df = df[df["is_baseline"]].copy()
    study_df = df[~df["is_baseline"]].copy()

    baseline_mean_per_ds = baseline_df.groupby("dataset")["mean"].mean().to_dict()
    return df, baseline_mean_per_ds, baseline_df, study_df


def plot_sensitivity(param_keys, datasets, plot_config):
    df, baseline_mean_per_ds, _, _ = load_results(RESULTS_CSV)
    cfg = plot_config.get("sensitivity", {})
    if not cfg.get("enabled", True):
        print("sensitivity plots disabled — skipping.")
        return

    params_to_show = cfg.get("params_to_show") or param_keys
    palette = plt.cm.tab10.colors
    ncols = 3
    nrows = -(-len(params_to_show) // ncols)

    _SCOPE = {
        "schedule": (lambda d: d[d["strategy"] == "mask"], "strategy = mask only"),
        "renoise_factor": (lambda d: d[d["use_t_next"] == True], "use_t_next = True only"),
    }

    for ds in datasets:
        df_ds = df[(df["dataset"] == ds) & ~df["is_baseline"]]
        metric_name = df_ds["metric"].iloc[0].capitalize()
        b_mean = baseline_mean_per_ds.get(ds, float("nan"))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            constrained_layout=True,
        )
        axes = np.array(axes).flatten()

        for ax, param in zip(axes, params_to_show):
            if param in _SCOPE:
                filter_fn, scope_note = _SCOPE[param]
                scope_df = filter_fn(df_ds)
            else:
                scope_df, scope_note = df_ds, ""

            grp = scope_df.groupby(param)["mean"].agg(["mean", "std"]).reset_index()
            labels = grp[param].astype(str).tolist()
            colors = [palette[i % len(palette)] for i in range(len(labels))]

            bars = ax.bar(
                labels, grp["mean"], yerr=grp["std"],
                capsize=5, color=colors, alpha=0.85,
                edgecolor="white", linewidth=0.8,
            )
            for bar, (_, row) in zip(bars, grp.iterrows()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + row["std"] + 1e-4,
                    f"{row['mean']:.4f}",
                    ha="center", va="bottom", fontsize=8, color="dimgray",
                )

            title = param + (f"\n({scope_note})" if scope_note else "")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel(param, fontsize=9)
            ax.set_ylabel(metric_name, fontsize=9)
            ax.axhline(b_mean, color="steelblue", linestyle="--",
                       linewidth=1.2, label=f"baseline ({b_mean:.4f})")
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.tick_params(axis="x", labelsize=9)

        for ax in axes[len(params_to_show):]:
            ax.set_visible(False)

        fig.suptitle(
            f"Marginal Parameter Sensitivity — Dataset: {ds}\n"
            f"Mean {metric_name} per parameter value "
            f"(averaged over all other params & downstream models)\n"
            f"Dashed line = baseline ({b_mean:.4f})",
            fontsize=13, fontweight="bold",
        )
        plt.show()


def plot_timestep_lines(datasets, plot_config):
    df, baseline_mean_per_ds, _, _ = load_results(RESULTS_CSV)
    cfg = plot_config.get("timestep_lines", {})
    if not cfg.get("enabled", True):
        print("timestep_lines plots disabled — skipping.")
        return

    filter_combos = cfg.get("combos_to_show")
    filter_renoise = cfg.get("renoise_factors")
    timesteps = sorted(df["n_timesteps"].unique())

    sched_style = {
        "const": {"color": "#1f77b4", "marker": "o"},
        "quad": {"color": "#ff7f0e", "marker": "s"},
        "linear": {"color": "#2ca02c", "marker": "^"},
        "cosine": {"color": "#d62728", "marker": "D"},
    }
    fallback_style = {"color": "gray", "marker": "x"}

    for ds in datasets:
        df_ds = df[(df["dataset"] == ds) & ~df["is_baseline"]]
        metric_name = df_ds["metric"].iloc[0].capitalize()
        b_mean = baseline_mean_per_ds.get(ds, float("nan"))

        for strat in sorted(df_ds["strategy"].unique()):
            for tnxt in [True, False]:
                if filter_combos is not None and (strat, tnxt) not in filter_combos:
                    continue

                df_sub = df_ds[
                    (df_ds["strategy"] == strat) &
                    (df_ds["use_t_next"] == tnxt)
                ]
                if df_sub.empty:
                    continue

                avail_rn = sorted(df_sub["renoise_factor"].unique())
                if filter_renoise is not None and tnxt:
                    avail_rn = [r for r in avail_rn if r in filter_renoise]
                if not avail_rn:
                    continue

                schedules = sorted(df_sub["schedule"].unique())
                nrows_fig = len(avail_rn)

                fig, axes = plt.subplots(
                    nrows_fig, 1,
                    figsize=(8, 4 * nrows_fig),
                    constrained_layout=True,
                    squeeze=False,
                )

                for ri, rn in enumerate(avail_rn):
                    ax = axes[ri, 0]
                    df_cell = df_sub[df_sub["renoise_factor"] == rn]

                    for sched in schedules:
                        df_line = (
                            df_cell[df_cell["schedule"] == sched]
                            .groupby("n_timesteps")["mean"]
                            .agg(["mean", "std"])
                            .reindex(timesteps)
                            .reset_index()
                        )
                        style = sched_style.get(sched, fallback_style)
                        ax.errorbar(
                            df_line["n_timesteps"], df_line["mean"],
                            yerr=df_line["std"],
                            label=f"schedule = {sched}",
                            color=style["color"], marker=style["marker"],
                            capsize=3, linewidth=1.8, markersize=5,
                        )

                    ax.axhline(b_mean, color="steelblue", linestyle="--",
                               linewidth=1.2, label=f"baseline ({b_mean:.4f})")

                    rn_label = (
                        f"renoise_factor = {rn}"
                        if tnxt else
                        "renoise_factor = 1.0  (fixed — use_t_next = False)"
                    )
                    ax.set_title(rn_label, fontsize=9, fontweight="bold")
                    ax.set_xscale("log")
                    ax.set_xticks(timesteps)
                    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
                    ax.set_xlabel("n_timesteps", fontsize=9)
                    ax.set_ylabel(metric_name, fontsize=9)
                    ax.legend(fontsize=7, loc="best")
                    ax.grid(True, linestyle="--", alpha=0.35)

                tnxt_str = "True" if tnxt else "False"
                sched_note = (
                    "Schedule varies (mask strategy applies it)"
                    if strat == "mask" else
                    "Schedule fixed at const (not applied for target strategy)"
                )
                rn_note = (
                    "Rows: renoise_factor" if nrows_fig > 1 else
                    "renoise_factor fixed at 1.0 (use_t_next=False)"
                )
                fig.suptitle(
                    f"Effect of n_timesteps on {metric_name}\n"
                    f"Dataset: {ds}  |  Strategy: {strat}  |  use_t_next: {tnxt_str}\n"
                    f"{sched_note}  |  {rn_note}\n"
                    f"Dashed line = baseline ({b_mean:.4f})",
                    fontsize=12, fontweight="bold",
                )
                plt.show()


def plot_heatmaps_v1(param_keys, datasets, plot_config):
    df, baseline_mean_per_ds, _, _ = load_results(RESULTS_CSV)
    cfg = plot_config.get("heatmaps", {})
    if not cfg.get("enabled", True):
        print("heatmaps disabled — skipping.")
        return

    all_pairs = list(combinations(param_keys, 2))
    pairs_filter = cfg.get("pairs_to_show")
    pairs = [p for p in all_pairs if pairs_filter is None or p in pairs_filter or list(p) in pairs_filter]
    if not pairs:
        print("No heatmap pairs to show.")
        return

    ncols = 3
    nrows = -(-len(pairs) // ncols)

    _SCOPE = {
        "schedule": (lambda d: d[d["strategy"] == "mask"], "mask only"),
        "renoise_factor": (lambda d: d[d["use_t_next"] == True], "use_t_next=True only"),
    }

    for ds in datasets:
        df_ds = df[(df["dataset"] == ds) & ~df["is_baseline"]]
        metric_name = df_ds["metric"].iloc[0].capitalize()
        b_mean = baseline_mean_per_ds.get(ds, float("nan"))
        vmin = df_ds["mean"].quantile(0.05)
        vmax = df_ds["mean"].quantile(0.95)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 5 * nrows),
            constrained_layout=True,
        )
        axes = np.array(axes).flatten()

        for ax, (p1, p2) in zip(axes, pairs):
            scope_df = df_ds.copy()
            scope_notes = []
            for p in (p1, p2):
                if p in _SCOPE:
                    filter_fn, note = _SCOPE[p]
                    scope_df = filter_fn(scope_df)
                    scope_notes.append(note)

            pivot = (
                scope_df.groupby([p1, p2])["mean"]
                .mean()
                .unstack(p2)
            )
            pivot.index = pivot.index.astype(str)
            pivot.columns = pivot.columns.astype(str)

            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.8, label=metric_name)

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_xlabel(p2, fontsize=9)
            ax.set_ylabel(p1, fontsize=9)

            scope_str = f"  [{', '.join(scope_notes)}]" if scope_notes else ""
            ax.set_title(f"{p1}  ×  {p2}{scope_str}", fontsize=10, fontweight="bold")

            for r in range(len(pivot.index)):
                for c in range(len(pivot.columns)):
                    val = pivot.values[r, c]
                    if not np.isnan(val):
                        ax.text(c, r, f"{val:.4f}", ha="center", va="center", fontsize=7)

        for ax in axes[len(pairs):]:
            ax.set_visible(False)

        fig.suptitle(
            f"Pairwise Parameter Interaction Heatmaps — Dataset: {ds}\n"
            f"Mean {metric_name} (averaged over remaining params & downstream models)\n"
            f"Colour scale: 5th–95th percentile of study scores  |  "
            f"Baseline reference: {b_mean:.4f}",
            fontsize=13, fontweight="bold",
        )
        plt.show()


import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dataframe_image as dfi


def build_study_table(df, param_keys):
    """
    Wide-format summary: one row per (dataset, combo).
    Columns:
        param_keys
        + one per downstream model
        + mean_all_models
        + std_all_models

    Sorted best-first by mean_all_models.
    """

    model_names = sorted(df["model"].unique())

    pivot = (
        df.pivot_table(
            index=["dataset", "key", *param_keys],
            columns="model",
            values="mean",
            aggfunc="mean",
        )
        .reset_index()
    )

    pivot.columns.name = None

    pivot["mean_all_models"] = (
        pivot[model_names].mean(axis=1)
    )

    pivot["std_all_models"] = (
        pivot[model_names].std(axis=1)
    )

    # IMPORTANT
    # sort BEFORE ranking
    pivot = pivot.sort_values(
        "mean_all_models",
        ascending=False,
    ).reset_index(drop=True)

    pivot.insert(
        0,
        "rank",
        range(1, len(pivot) + 1),
    )

    return pivot, model_names


def plot_study_table(
    n_rows=10,
    save_dir="study_tables",
    export_format="png",
    preview=True,
):

    os.makedirs(save_dir, exist_ok=True)

    (
        results_df,
        baseline_mean_per_ds,
        baseline_df,
        study_df,
    ) = load_results(RESULTS_CSV)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    for ds in DATASETS:

        df_ds = study_df[
            study_df["dataset"] == ds
        ]

        df_base_ds = baseline_df[
            baseline_df["dataset"] == ds
        ]

        metric_name = df_ds["metric"].iloc[0]

        b_mean = baseline_mean_per_ds.get(
            ds,
            float("nan"),
        )

        # ==================================================
        # BUILD RANKED TABLE
        # ==================================================

        table, model_names = build_study_table(
            df_ds,
            PARAM_KEYS,
        )

        table["delta_vs_baseline"] = (
            table["mean_all_models"] - b_mean
        )

        # ==================================================
        # KEEP ONLY TOP N ROWS
        # ==================================================

        table = table.head(n_rows)

        # ==================================================
        # BASELINE ROW
        # ==================================================

        b_means = (
            df_base_ds.groupby("model")["mean"]
            .mean()
            .to_dict()
        )

        baseline_row = pd.DataFrame([{
            "rank": "BASE",
            "dataset": ds,
            "key": "BASELINE",

            **{
                k: DEFAULT_SAMPLE_PARAMS.get(k, "—")
                for k in PARAM_KEYS
            },

            **{
                m: round(
                    b_means.get(m, float("nan")),
                    4,
                )
                for m in model_names
            },

            "mean_all_models": round(
                b_mean,
                4,
            ),

            "std_all_models": round(
                pd.Series(list(b_means.values())).std(),
                4,
            ),

            "delta_vs_baseline": 0.0,
        }])

        # ==================================================
        # DISPLAY TABLE
        # ==================================================

        display_cols = (
            ["rank", "dataset"]
            + PARAM_KEYS
            + model_names
            + [
                "mean_all_models",
                "std_all_models",
                "delta_vs_baseline",
            ]
        )

        float_cols = (
            model_names
            + [
                "mean_all_models",
                "std_all_models",
                "delta_vs_baseline",
            ]
        )

        study_display = table[
            [c for c in display_cols if c in table.columns]
        ].copy()

        baseline_display = baseline_row[
            [c for c in display_cols if c in baseline_row.columns]
        ].copy()

        full_display = pd.concat(
            [baseline_display, study_display],
            ignore_index=True,
        )

        # ==================================================
        # STYLING
        # ==================================================

        study_means = study_display[
            "mean_all_models"
        ].dropna()

        vmin_style = study_means.quantile(0.1)
        vmax_style = study_means.quantile(0.9)

        def highlight_baseline(row):

            if row["rank"] == "BASE":

                return [
                    "background-color: #cce5ff; font-weight: bold"
                ] * len(row)

            return [""] * len(row)

        def color_delta(val):

            if pd.isna(val) or val == 0:
                return ""

            return (
                "color: green; font-weight: bold"
                if val > 0
                else "color: #cc0000"
            )

        styled = (
            full_display.style

            .apply(
                highlight_baseline,
                axis=1,
            )

            .background_gradient(
                subset=["mean_all_models"],
                cmap="RdYlGn",
                vmin=vmin_style,
                vmax=vmax_style,
            )

            .background_gradient(
                subset=model_names,
                cmap="Blues",
                axis=0,
            )

            .map(
                color_delta,
                subset=["delta_vs_baseline"],
            )

            .format(
                {
                    c: "{:.4f}"
                    for c in float_cols
                },
                na_rep="—",
            )

            .set_caption(
                f"{len(study_display)} configurations — "
                f"Dataset: {ds} | "
                f"Metric: {metric_name} | "
                f"Sorted best → worst | "
                f"Baseline (blue row): {b_mean:.4f}"
            )

            .set_table_styles([{
                "selector": "caption",
                "props": (
                    "font-size:13px; "
                    "font-weight:bold; "
                    "text-align:left; "
                    "margin-bottom:6px; "
                    "caption-side:top;"
                ),
            }])
        )

        # ==================================================
        # EXPORT
        # ==================================================

        save_path = os.path.join(
            save_dir,
            f"{ds}_study_table.{export_format}",
        )

        dfi.export(
            styled,
            save_path,
            table_conversion="chrome",
            dpi=300,
        )

        # =========================
        # PREVIEW SAVED IMAGE
        # =========================

        if preview and export_format == "png":

            img = mpimg.imread(save_path)

            plt.figure(figsize=(16, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{ds} study table")

            plt.show()

"""
plot_heatmaps.py
================
A fully-customisable heatmap-grid function for tabular experiment results.
 
Quick start
-----------
    from plot_heatmaps import plot_heatmaps
    import pandas as pd
 
    df = pd.read_csv("results.csv")
 
    fig = plot_heatmaps(
        df,
        value_col     = "mean_all_models",
        x_col         = "n_timesteps",
        y_col         = "renoise_factor",
        fixed_filters = {"strategy": "mask"},
        subplot_cols  = ["schedule", "use_t_next"],
        ncols         = 3,
        suptitle      = "Mask strategy — accuracy grid",
    )
    plt.show()
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from typing import Any, Callable, Dict, List, Optional, Union
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _subset(df: pd.DataFrame, subplot_cols: List[str], key: tuple) -> pd.DataFrame:
    """Return rows of *df* matching *key* across *subplot_cols*."""
    if not subplot_cols:
        return df
    mask = pd.Series(True, index=df.index)
    for col, val in zip(subplot_cols, key):
        mask &= df[col] == val
    return df[mask]
 
 
def _auto_text_color(rgba, threshold: float = 0.50) -> str:
    """Black or white text depending on perceived luminance of *rgba*."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if lum > threshold else "#f5f5f5"
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Main function
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_heatmaps(
    df: Optional[pd.DataFrame] = None,
    # ── data mapping ──────────────────────────────────────────────────────────
    value_col: str                       = "mean_all_models",
    x_col:     str                       = "n_timesteps",
    y_col:     str                       = "renoise_factor",
    # ── filtering & grouping ──────────────────────────────────────────────────
    fixed_filters: Optional[Dict[str, Any]] = None,
    subplot_cols:  Optional[List[str]]      = None,
    aggfunc:       Union[str, Callable]     = "mean",
    # ── layout ────────────────────────────────────────────────────────────────
    ncols:     int            = 3,
    cell_size: tuple          = (3.8, 3.0),
    figsize:   Optional[tuple] = None,
    # ── colormap ──────────────────────────────────────────────────────────────
    cmap:   str            = "RdYlGn",
    vmin:   Optional[float] = None,
    vmax:   Optional[float] = None,
    center: Optional[float] = None,
    # ── annotations ───────────────────────────────────────────────────────────
    annot:            bool  = True,
    fmt:              str   = ".4f",
    annot_fontsize:   int   = 10,
    annot_fontweight: str   = "bold",
    annot_alpha:      float = 0.95,
    # ── cell borders ──────────────────────────────────────────────────────────
    linewidths: float = 1.2,
    linecolor:  str   = "#00000030",
    # ── colorbar ──────────────────────────────────────────────────────────────
    show_cbar:     bool           = False,
    cbar_label:    Optional[str]  = None,
    cbar_fraction: float          = 0.025,
    cbar_pad:      float          = 0.03,
    cbar_n_ticks:  int            = 6,
    # ── titles & labels ───────────────────────────────────────────────────────
    suptitle:                 str              = "Heatmap Grid",
    suptitle_fontsize:        int              = 16,
    suptitle_fontweight:      str              = "bold",
    suptitle_y:               float            = 1.01,
    subplot_titles:           Optional[List[str]] = None,
    subplot_title_template:   Optional[str]       = None,
    subplot_title_fontsize:   int              = 11,
    subplot_title_fontweight: str              = "semibold",
    subplot_title_pad:        float            = 8.0,
    xlabel:               Optional[str] = None,
    ylabel:               Optional[str] = None,
    axis_label_fontsize:  int           = 10,
    tick_fontsize:        int           = 9,
    tick_rotation_x:      int           = 0,
    tick_rotation_y:      int           = 0,
    # ── colours (direct overrides, no preset system) ──────────────────────────
    facecolor:    str = "white",
    ax_facecolor: str = "white",
    text_color:   str = "#111111",
    spine_color:  str = "#cccccc",
    # ── output ────────────────────────────────────────────────────────────────
    tight_layout_pad: float         = 2.0,
    savefig:          Optional[str] = None,
    dpi:              int           = 150,
    pad_inches:       float         = 0.2,
) -> plt.Figure:
    """
    Plot a grid of heatmaps from a DataFrame of experiment results.
 
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
 
    value_col : str
        Column whose aggregated value fills each heatmap cell.
        e.g. ``"mean_all_models"``, ``"delta_vs_baseline"``.
 
    x_col : str
        Column mapped to the x-axis of every heatmap (e.g. ``"n_timesteps"``).
 
    y_col : str
        Column mapped to the y-axis of every heatmap (e.g. ``"renoise_factor"``).
 
    fixed_filters : dict, optional
        ``{column: value}`` pairs applied before anything else.
        e.g. ``{"strategy": "mask", "dataset": "adult"}``.
 
    subplot_cols : list of str, optional
        Columns whose unique combinations each produce one subplot.
        e.g. ``["schedule"]``            → one subplot per schedule value.
        e.g. ``["schedule", "use_t_next"]`` → one subplot per pair.
        ``None`` → single heatmap.
 
    aggfunc : str or callable
        Aggregation when multiple rows share the same (x, y) cell.
        e.g. ``"mean"``, ``"max"``, ``np.median``.
 
    ncols : int
        Number of subplot columns in the grid.
 
    cell_size : (w, h)
        Width × height in inches per heatmap, used for auto figsize.
 
    figsize : (w, h), optional
        Override the auto-computed figure size.
 
    cmap : str
        Matplotlib colormap. Diverging: ``"RdYlGn"``, ``"coolwarm"``.
        Sequential: ``"viridis"``, ``"YlOrRd"``.
 
    vmin, vmax : float, optional
        Colour-scale limits, shared across all subplots.
        Auto-computed from filtered data if not supplied.
 
    center : float, optional
        Centre the colormap at this value (activates TwoSlopeNorm).
        e.g. ``center=0.0`` for delta columns.
 
    annot : bool
        Overlay cell values as text.
 
    fmt : str
        Python format spec for annotations: ``".4f"``, ``"+.4f"``, ``".2%"``.
 
    show_cbar : bool
        Show a shared colorbar (default False).
 
    subplot_titles : list of str, optional
        Explicit titles for each subplot in order. Overrides auto-generation.
 
    subplot_title_template : str, optional
        f-string template filled with subplot-group column values.
        e.g. ``"schedule={schedule} | use_t_next={use_t_next}"``.
 
    facecolor : str
        Figure background colour (default ``"white"``).
 
    ax_facecolor : str
        Axes background colour (default ``"white"``).
 
    text_color : str
        Colour for all text elements (default ``"#111111"``).
 
    spine_color : str
        Colour for axes spines and colorbar outline (default ``"#cccccc"``).
 
    savefig : str, optional
        File path. Saves the figure before returning if given.
 
    Returns
    -------
    fig : matplotlib.figure.Figure
 
    Examples
    --------
    >>> fig = plot_heatmaps(
    ...     df,
    ...     value_col     = "mean_all_models",
    ...     x_col         = "n_timesteps",
    ...     y_col         = "renoise_factor",
    ...     fixed_filters = {"strategy": "mask"},
    ...     subplot_cols  = ["schedule", "use_t_next"],
    ...     ncols         = 3,
    ...     suptitle      = "Mask strategy — accuracy by timesteps & renoise",
    ... )
    >>> plt.show()
 
    # Delta view with a diverging colormap centred on 0
    >>> fig = plot_heatmaps(
    ...     df,
    ...     value_col    = "delta_vs_baseline",
    ...     x_col        = "n_timesteps",
    ...     y_col        = "renoise_factor",
    ...     fixed_filters= {"strategy": "target"},
    ...     subplot_cols = ["schedule"],
    ...     cmap         = "RdYlGn",
    ...     center       = 0.0,
    ...     fmt          = "+.4f",
    ...     suptitle     = "Target strategy — Δ vs baseline",
    ...     show_cbar    = True,
    ... )
    """

    if df is None:
        df, _, _, _ = load_results(RESULTS_CSV)
 
    # ── 1. Filter ─────────────────────────────────────────────────────────────
    data = df.copy()
    if fixed_filters:
        for col, val in fixed_filters.items():
            if col not in data.columns:
                raise KeyError(f"fixed_filters key '{col}' not in DataFrame columns.")
            data = data[data[col] == val]
    if data.empty:
        raise ValueError("No data remaining after applying fixed_filters.")
 
    # ── 2. Group keys (one key → one subplot) ─────────────────────────────────
    _subplot_cols = subplot_cols or []
    if _subplot_cols:
        if len(_subplot_cols) == 1:
            groups   = data.groupby(_subplot_cols[0], sort=True)
            raw_keys = [(k,) for k in groups.groups.keys()]
        else:
            groups   = data.groupby(_subplot_cols, sort=True)
            raw_keys = list(groups.groups.keys())
        group_keys = [k if isinstance(k, tuple) else (k,) for k in raw_keys]
    else:
        group_keys = [()]
 
    n_plots = len(group_keys)
 
    # ── 3. Build pivots & global colour scale ─────────────────────────────────
    pivots   = {}
    all_vals = []
    for key in group_keys:
        sub = _subset(data, _subplot_cols, key)
        piv = sub.pivot_table(
            index=y_col, columns=x_col, values=value_col, aggfunc=aggfunc
        )
        pivots[key] = piv
        all_vals.append(piv.values.ravel())
 
    finite  = np.concatenate(all_vals)
    finite  = finite[np.isfinite(finite)]
    _vmin   = float(vmin if vmin is not None else np.min(finite))
    _vmax   = float(vmax if vmax is not None else np.max(finite))
 
    if center is not None:
        _norm = mcolors.TwoSlopeNorm(vmin=_vmin, vcenter=float(center), vmax=_vmax)
    else:
        _norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)
 
    _cmap_obj = plt.get_cmap(cmap)
 
    # ── 4. Figure layout ──────────────────────────────────────────────────────
    ncols_  = min(ncols, n_plots)
    nrows_  = int(np.ceil(n_plots / ncols_))
 
    if figsize is None:
        n_x     = data[x_col].nunique()
        n_y     = data[y_col].nunique()
        fw      = ncols_ * (cell_size[0] + n_x * 0.25)
        fh      = nrows_ * (cell_size[1] + n_y * 0.20)
        figsize = (max(fw, 6), max(fh, 3))
 
    fig, axes = plt.subplots(
        nrows_, ncols_,
        figsize   = figsize,
        facecolor = facecolor,
        squeeze   = False,
        layout    = "constrained",
    )
    fig.patch.set_facecolor(facecolor)
 
    # ── 5. Draw each subplot ──────────────────────────────────────────────────
    last_img = None
    for idx, key in enumerate(group_keys):
        row, col = divmod(idx, ncols_)
        ax       = axes[row][col]
        ax.set_facecolor(ax_facecolor)
 
        piv      = pivots[key]
        mat      = piv.values.astype(float)
        x_labels = [str(v) for v in piv.columns]
        y_labels = [str(v) for v in piv.index]
        n_y_ax, n_x_ax = mat.shape
 
        # Draw cells
        img = ax.imshow(
            mat, cmap=_cmap_obj, norm=_norm,
            aspect="auto", interpolation="nearest",
        )
        last_img = img
 
        # Cell borders
        for yi in range(n_y_ax):
            for xi in range(n_x_ax):
                ax.add_patch(Rectangle(
                    (xi - 0.5, yi - 0.5), 1, 1,
                    fill=False, edgecolor=linecolor,
                    linewidth=linewidths, zorder=2,
                ))
 
        # Annotations
        if annot:
            for yi in range(n_y_ax):
                for xi in range(n_x_ax):
                    val = mat[yi, xi]
                    if not np.isfinite(val):
                        continue
                    txt_c = _auto_text_color(_cmap_obj(_norm(val)))
                    ax.text(
                        xi, yi, format(val, fmt),
                        ha="center", va="center",
                        fontsize=annot_fontsize,
                        fontweight=annot_fontweight,
                        color=txt_c, alpha=annot_alpha,
                        zorder=3,
                    )
 
        # Axes cosmetics
        ax.set_xticks(range(n_x_ax))
        ax.set_yticks(range(n_y_ax))
        ax.set_xticklabels(
            x_labels, color=text_color, fontsize=tick_fontsize,
            rotation=tick_rotation_x,
            ha="center" if tick_rotation_x == 0 else "right",
        )
        ax.set_yticklabels(
            y_labels, color=text_color, fontsize=tick_fontsize,
            rotation=tick_rotation_y,
        )
        ax.tick_params(axis="both", which="both", length=0, colors=text_color)
        ax.set_xlabel(xlabel or x_col, color=text_color,
                      fontsize=axis_label_fontsize, labelpad=6)
        ax.set_ylabel(ylabel or y_col, color=text_color,
                      fontsize=axis_label_fontsize, labelpad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)
            spine.set_linewidth(0.8)
 
        # Subplot title
        if subplot_titles and idx < len(subplot_titles):
            title_str = subplot_titles[idx]
        elif subplot_title_template and _subplot_cols:
            title_str = subplot_title_template.format(
                **dict(zip(_subplot_cols, key))
            )
        elif _subplot_cols:
            title_str = "  ·  ".join(
                f"{c} = {v}" for c, v in zip(_subplot_cols, key)
            )
        else:
            title_str = value_col
 
        ax.set_title(
            title_str, color=text_color,
            fontsize=subplot_title_fontsize,
            fontweight=subplot_title_fontweight,
            pad=subplot_title_pad,
        )
 
    # ── 6. Hide empty axes ────────────────────────────────────────────────────
    for idx in range(n_plots, nrows_ * ncols_):
        row, col = divmod(idx, ncols_)
        axes[row][col].set_visible(False)
 
    # ── 7. Shared colorbar (opt-in) ───────────────────────────────────────────
    if show_cbar and last_img is not None:
        visible_axes = [
            axes[r][c]
            for r in range(nrows_) for c in range(ncols_)
            if axes[r][c].get_visible()
        ]
        cb = fig.colorbar(
            last_img, ax=visible_axes,
            fraction=cbar_fraction, pad=cbar_pad, shrink=0.85,
        )
        cb.set_label(cbar_label or value_col, color=text_color,
                     fontsize=axis_label_fontsize, labelpad=8)
        cb.ax.tick_params(labelsize=tick_fontsize, colors=text_color, length=4)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=text_color)
        cb.outline.set_edgecolor(spine_color)
        cb.outline.set_linewidth(0.8)
        cb.locator = mticker.MaxNLocator(nbins=cbar_n_ticks)
        cb.update_ticks()
 
    # ── 8. Supertitle & spacing ───────────────────────────────────────────────
    fig.suptitle(
        suptitle, color=text_color,
        fontsize=suptitle_fontsize, fontweight=suptitle_fontweight,
        y=suptitle_y,
    )
    fig.get_layout_engine().set(
        w_pad=tight_layout_pad / 72,
        h_pad=tight_layout_pad / 72,
        hspace=0.06, wspace=0.06,
    )
 
    # ── 9. Save ───────────────────────────────────────────────────────────────
    if savefig:
        fig.savefig(
            savefig, dpi=dpi, bbox_inches="tight",
            facecolor=facecolor, pad_inches=pad_inches,
        )
 
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Demo  (run this file directly: python plot_heatmaps.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, _, _, _ = load_results(RESULTS_CSV)
    # ensure correct types
    df["use_t_next"]     = df["use_t_next"].astype(str)
    df["renoise_factor"] = df["renoise_factor"].astype(float)
    df["n_timesteps"]    = df["n_timesteps"].astype(int)

    # ── Example 1: mask strategy, one subplot per schedule ─────────────────
    print("Rendering Example 1 …")
    fig1 = plot_heatmaps(
        df,
        value_col     = "mean_all_models",
        x_col         = "n_timesteps",
        y_col         = "renoise_factor",
        fixed_filters = {"strategy": "mask"},
        subplot_cols  = ["schedule", "use_t_next"],
        ncols         = 3,
        cmap          = "RdYlGn",
        suptitle      = "Strategy = mask  ·  mean accuracy across models",
        style         = "dark",
        highlight_best  = True,
        highlight_worst = True,
        fmt           = ".4f",
        annot_fontsize= 9,
        savefig       = "heatmap_mask_dark.png",
    )
    plt.show()

    # ── Example 2: delta view, diverging colormap ──────────────────────────
    print("Rendering Example 2 …")
    fig2 = plot_heatmaps(
        df,
        value_col               = "delta_vs_baseline",
        x_col                   = "n_timesteps",
        y_col                   = "renoise_factor",
        fixed_filters           = {"strategy": "target"},
        subplot_cols            = ["use_t_next"],
        ncols                   = 2,
        cmap                    = "RdYlGn",
        center                  = 0.0,
        fmt                     = "+.4f",
        suptitle                = "Strategy = target  ·  Δ vs baseline",
        subplot_title_template  = "use_t_next = {use_t_next}",
        style                   = "midnight",
        highlight_best          = True,
        highlight_worst         = True,
        savefig                 = "heatmap_target_delta.png",
    )
    plt.show()

    print("Done.")