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

# """
# plot_heatmaps.py  –  Annotated heatmap grid from a tidy DataFrame.
# """
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib.patches import Rectangle


# # use_kid sub-cell definitions: (column, low_is_better, short_label)
# # Layout per cell:  FID  | FID-C
# #                    IS  |  KID
# _KID = [
#     ("fid",             True,  "FID"),
#     ("fid_clip",        True,  "FID-C"),
#     ("inception_score", False, "IS"),
#     ("kid",             True,  "KID"),
# ]
# # (dx, dy) offset from integer cell-centre to the sub-cell's bottom-left corner
# _KID_OFFSETS = [(-0.5, -0.5), (0.0, -0.5), (-0.5, 0.0), (0.0, 0.0)]


# def _text_color(rgba):
#     """Return #111 or #eee depending on background luminance."""
#     lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
#     return "#111" if lum > 0.45 else "#eee"


# def plot_heatmaps(
#     df:             pd.DataFrame,
#     value_col:      str,
#     x_col:          str,
#     y_col:          str,
#     fixed_filters:  dict  = None,
#     subplot_cols:   list  = None,
#     ncols:          int   = 3,
#     cell_size:      tuple = (5, 4),
#     suptitle:       str   = "",
#     low_is_better:  bool  = False,
#     baseline_col:   str   = None,
#     aggfunc:        str   = "mean",
#     fmt:            str   = ".4f",
#     annot_fontsize: int   = 10,
#     use_kid:        bool  = False,
# ) -> plt.Figure:
#     """
#     Plot a grid of annotated heatmaps from a tidy DataFrame.

#     If use_kid=True each cell shows four colour-coded sub-cells:
#         FID  | FID-C      (both: lower = greener)
#          IS  |  KID       (IS: higher = greener; KID: lower = greener)
#     Each metric uses its own global colour scale across all subplots.
#     value_col is still used for the baseline reference when baseline_col is set.

#     All other parameters behave identically to the original.
#     """

#     # ── 1. Fixed filters ────────────────────────────────────────────────────
#     data = df.copy()
#     for col, val in (fixed_filters or {}).items():
#         data = data[data[col] == val]

#     # ── 2. Baseline (value shown in suptitle; no in-axes box) ───────────────
#     baseline_val = None
#     if baseline_col and baseline_col in data.columns:
#         mask = data[baseline_col].isin(["BASELINE", True])
#         if mask.any():
#             baseline_val = data.loc[mask, value_col].mean()
#         data = data[~mask]

#     # ── 3. Subplot groups ────────────────────────────────────────────────────
#     subplot_cols = subplot_cols or []
#     groups = (
#         [(k if isinstance(k, tuple) else (k,), g)
#          for k, g in data.groupby(subplot_cols)]
#         if subplot_cols else [((), data)]
#     )
#     n      = len(groups)
#     ncols_ = min(ncols, n)
#     nrows_ = int(np.ceil(n / ncols_))

#     def make_pivot(grp, col):
#         return grp.pivot_table(index=y_col, columns=x_col,
#                                values=col, aggfunc=aggfunc)

#     # ── 4. Colour scales ─────────────────────────────────────────────────────
#     if use_kid:
#         # One (pivots, cmap, norm) per metric, shared across all subplots
#         kid_info = {}
#         for col, lib, _ in _KID:
#             pivs = [make_pivot(g, col) for _, g in groups]
#             flat = np.concatenate([p.values.ravel() for p in pivs])
#             flat = flat[np.isfinite(flat)]
#             kid_info[col] = (
#                 pivs,
#                 plt.get_cmap("RdYlGn_r" if lib else "RdYlGn"),
#                 mcolors.Normalize(flat.min(), flat.max()),
#             )
#         ref_pivots = kid_info["fid"][0]          # shape / tick labels reference
#     else:
#         ref_pivots = [make_pivot(g, value_col) for _, g in groups]
#         flat = np.concatenate([p.values.ravel() for p in ref_pivots])
#         flat = flat[np.isfinite(flat)]
#         cmap_s = plt.get_cmap("RdYlGn_r" if low_is_better else "RdYlGn")
#         norm_s = (
#             mcolors.TwoSlopeNorm(vmin=flat.min(), vcenter=baseline_val,
#                                  vmax=flat.max())
#             if baseline_val is not None and flat.min() < baseline_val < flat.max()
#             else mcolors.Normalize(flat.min(), flat.max())
#         )

#     # ── 5. Draw ──────────────────────────────────────────────────────────────
#     fig, axes = plt.subplots(
#         nrows_, ncols_,
#         figsize   = (ncols_ * cell_size[0], nrows_ * cell_size[1] + 0.8),
#         squeeze   = False,
#         facecolor = "white",
#     )

#     for idx, ((key, _), ref_piv) in enumerate(zip(groups, ref_pivots)):
#         ax     = axes[idx // ncols_][idx % ncols_]
#         ny, nx = ref_piv.shape

#         if use_kid:
#             # ── 4-quadrant sub-cell rendering ───────────────────────────────
#             ax.set_xlim(-0.5, nx - 0.5)
#             ax.set_ylim(ny - 0.5, -0.5)          # invert y to match imshow

#             for yi in range(ny):
#                 for xi in range(nx):
#                     for (col, _, label), (dx, dy) in zip(_KID, _KID_OFFSETS):
#                         pivs, cmap_, norm_ = kid_info[col]
#                         v = float(pivs[idx].values[yi, xi])
#                         if not np.isfinite(v):
#                             continue
#                         rgba = cmap_(norm_(v))
#                         tc   = _text_color(rgba)
#                         rx, ry = xi + dx, yi + dy
#                         ax.add_patch(Rectangle(
#                             (rx, ry), 0.5, 0.5,
#                             facecolor=rgba, edgecolor="white",
#                             linewidth=0.3, zorder=1,
#                         ))
#                         cx, cy = rx + 0.25, ry + 0.25
#                         # label (small, above centre) + value (below centre)
#                         ax.text(cx, cy - 0.10, label,
#                                 ha="center", va="center", color=tc,
#                                 fontsize=annot_fontsize * 0.55,
#                                 fontweight="bold", zorder=3)
#                         ax.text(cx, cy + 0.10, format(v, fmt),
#                                 ha="center", va="center", color=tc,
#                                 fontsize=annot_fontsize * 0.72, zorder=3)
#         else:
#             # ── standard single-metric imshow ────────────────────────────────
#             mat = ref_piv.values.astype(float)
#             ax.imshow(mat, cmap=cmap_s, norm=norm_s,
#                       aspect="auto", interpolation="nearest")
#             for yi in range(ny):
#                 for xi in range(nx):
#                     v = mat[yi, xi]
#                     if not np.isfinite(v):
#                         continue
#                     rgba = cmap_s(norm_s(v))
#                     ax.text(xi, yi, format(v, fmt),
#                             ha="center", va="center",
#                             fontsize=annot_fontsize, fontweight="bold",
#                             color=_text_color(rgba), zorder=3)

#         # ── shared: cell-border separators + axis labels ─────────────────────
#         for x in np.arange(-0.5, nx, 1):
#             ax.axvline(x, color="white", linewidth=0.8, zorder=2)
#         for y in np.arange(-0.5, ny, 1):
#             ax.axhline(y, color="white", linewidth=0.8, zorder=2)

#         ax.set_xticks(range(nx))
#         ax.set_yticks(range(ny))
#         ax.set_xticklabels([str(v) for v in ref_piv.columns])
#         ax.set_yticklabels([str(v) for v in ref_piv.index])
#         ax.set_xlabel(x_col, fontsize=10, labelpad=6)
#         ax.set_ylabel(y_col, fontsize=10, labelpad=6)

#         title = ("  ·  ".join(f"{c}={v}" for c, v in zip(subplot_cols, key))
#                  if subplot_cols else value_col)
#         ax.set_title(title, fontsize=10, fontweight="semibold", pad=8)

#     # Hide unused axes
#     for idx in range(n, nrows_ * ncols_):
#         axes[idx // ncols_][idx % ncols_].set_visible(False)

#     # ── Suptitle — baseline goes here, not inside the axes ───────────────────
#     sup = suptitle
#     if baseline_val is not None:
#         sup += f"\n[baseline {value_col} = {format(baseline_val, fmt)}]"
#     fig.suptitle(sup, fontsize=14, fontweight="bold")
#     plt.tight_layout()
#     return fig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# use_kid sub-cell definitions: (column, low_is_better, short_label)
# Layout per cell:  FID  | FID-C
#                    IS  |  KID
_KID = [
    ("fid",             True,  "FID"),
    ("fid_clip",        True,  "FID-C"),
    ("inception_score", False, "IS"),
    ("kid",             True,  "KID"),
]
_KID_OFFSETS = [(-0.5, -0.5), (0.0, -0.5), (-0.5, 0.0), (0.0, 0.0)]


def _text_color(rgba):
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "#111" if lum > 0.45 else "#eee"


def plot_heatmaps(
    df:             pd.DataFrame,
    x_col:          str,
    y_col:          str,
    value_col:      str   = None,   # required when use_kid=False
    fixed_filters:  dict  = None,
    subplot_cols:   list  = None,
    ncols:          int   = 3,
    cell_size:      tuple = (5, 4),
    suptitle:       str   = "",
    low_is_better:  bool  = False,
    aggfunc:        str   = "mean",
    fmt:            str   = ".4f",
    annot_fontsize: int   = 10,
    use_kid:        bool  = False,
    std_col:        str   = None,   # ← NEW: pre-computed std column (non-KID mode).
                                    #        KID mode auto-detects "{metric}_std" columns
                                    #        (e.g. "fid_std", "kid_std", …).
) -> plt.Figure:
    """
    Plot a grid of annotated heatmaps from a tidy DataFrame.

    value_col is optional when use_kid=True (metrics are fid, fid_clip,
    inception_score, kid). Each cell shows four colour-coded sub-cells:
        FID  | FID-C      lower = greener
         IS  |  KID       IS: higher = greener; others: lower = greener

    Colour scale is a plain min→max linear ramp per metric,
    computed independently per subplot (so each subplot's own min/max
    always maps to the full red→green range).

    std_col (non-KID): name of a column holding pre-computed std values;
        shown as "mean\\n±std" inside each cell.
    KID mode: std is shown automatically whenever a column named
        "{metric}_std" exists in df (e.g. "fid_std", "inception_score_std").
    """
    if not use_kid and value_col is None:
        raise ValueError("value_col is required when use_kid=False")

    # ── 1. Fixed filters ────────────────────────────────────────────────────
    data = df.copy()
    for col, val in (fixed_filters or {}).items():
        data = data[data[col] == val]

    # ── 2. Subplot groups ────────────────────────────────────────────────────
    subplot_cols = subplot_cols or []
    groups = (
        [(k if isinstance(k, tuple) else (k,), g)
         for k, g in data.groupby(subplot_cols)]
        if subplot_cols else [((), data)]
    )
    n      = len(groups)
    ncols_ = min(ncols, n)
    nrows_ = int(np.ceil(n / ncols_))

    def make_pivot(grp, col):
        return grp.pivot_table(index=y_col, columns=x_col,
                               values=col, aggfunc=aggfunc)

    # ── 3. Precompute pivots & colormaps (norms are per-subplot, see § 4) ────
    std_pivots   = None  # ← NEW: populated below for non-KID mode
    kid_std_info = {}    # ← NEW: populated below for KID mode

    if use_kid:
        kid_info = {}
        for col, lib, _ in _KID:
            pivs = [make_pivot(g, col) for _, g in groups]
            kid_info[col] = (
                pivs,
                plt.get_cmap("RdYlGn_r" if lib else "RdYlGn"),
            )
        ref_pivots = kid_info["fid"][0]

        # ← NEW: auto-detect "{metric}_std" columns ──────────────────────────
        for col, _, _ in _KID:
            std_c = f"{col}_std"
            if std_c in data.columns:
                kid_std_info[col] = [make_pivot(g, std_c) for _, g in groups]
        # ─────────────────────────────────────────────────────────────────────
    else:
        ref_pivots = [make_pivot(g, value_col) for _, g in groups]
        cmap_s = plt.get_cmap("RdYlGn_r" if low_is_better else "RdYlGn")

        # ← NEW: build std pivots when std_col is provided ────────────────────
        if std_col is not None and std_col in data.columns:
            std_pivots = [make_pivot(g, std_col) for _, g in groups]
        # ─────────────────────────────────────────────────────────────────────

    # ── 4. Draw ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        nrows_, ncols_,
        figsize   = (ncols_ * cell_size[0], nrows_ * cell_size[1] + 0.8),
        squeeze   = False,
        facecolor = "white",
    )

    for idx, ((key, _), ref_piv) in enumerate(zip(groups, ref_pivots)):
        ax     = axes[idx // ncols_][idx % ncols_]
        ny, nx = ref_piv.shape

        if use_kid:
            # Per-subplot, per-metric norms (each subplot's own min/max)
            subplot_norms = {}
            for col, _, _ in _KID:
                pivs, _ = kid_info[col]
                local_vals = pivs[idx].values.ravel()
                local_vals = local_vals[np.isfinite(local_vals)]
                subplot_norms[col] = mcolors.Normalize(
                    local_vals.min(), local_vals.max()
                )

            ax.set_xlim(-0.5, nx - 0.5)
            ax.set_ylim(ny - 0.5, -0.5)          # invert y like imshow

            for yi in range(ny):
                for xi in range(nx):
                    for (col, _, label), (dx, dy) in zip(_KID, _KID_OFFSETS):
                        pivs, cmap_ = kid_info[col]
                        norm_ = subplot_norms[col]
                        v = float(pivs[idx].values[yi, xi])
                        if not np.isfinite(v):
                            continue
                        rgba = cmap_(norm_(v))
                        tc   = _text_color(rgba)
                        rx, ry = xi + dx, yi + dy
                        ax.add_patch(Rectangle(
                            (rx, ry), 0.5, 0.5,
                            facecolor=rgba, edgecolor="white",
                            linewidth=0.3, zorder=1,
                        ))
                        cx, cy_c = rx + 0.25, ry + 0.25

                        # ← NEW: build value string, shift positions when std present
                        val_str  = format(v, fmt)
                        label_dy = -0.10                   # default label offset
                        val_dy   = +0.10                   # default value offset
                        val_fs   = annot_fontsize * 0.72   # default value font size
                        if col in kid_std_info:
                            sv = float(kid_std_info[col][idx].values[yi, xi])
                            if np.isfinite(sv):
                                val_str  = f"{val_str}\n±{format(sv, fmt)}"
                                label_dy = -0.14           # push label up
                                val_dy   = +0.06           # center the 2-liner
                                val_fs   = annot_fontsize * 0.60
                        # ─────────────────────────────────────────────────────

                        ax.text(cx, cy_c + label_dy, label,
                                ha="center", va="center", color=tc,
                                fontsize=annot_fontsize * 0.55,
                                fontweight="bold", zorder=3)
                        ax.text(cx, cy_c + val_dy, val_str,
                                ha="center", va="center", color=tc,
                                fontsize=val_fs, zorder=3,
                                linespacing=1.15)          # ← NEW: tighten 2-liner
        else:
            mat = ref_piv.values.astype(float)
            # Per-subplot norm (this subplot's own min/max)
            flat_local = mat.ravel()
            flat_local = flat_local[np.isfinite(flat_local)]
            norm_s = mcolors.Normalize(flat_local.min(), flat_local.max())

            ax.imshow(mat, cmap=cmap_s, norm=norm_s,
                      aspect="auto", interpolation="nearest")
            for yi in range(ny):
                for xi in range(nx):
                    v = mat[yi, xi]
                    if not np.isfinite(v):
                        continue
                    rgba = cmap_s(norm_s(v))

                    # ← NEW: append "±std" line when std_pivots available
                    txt    = format(v, fmt)
                    ann_fs = annot_fontsize
                    if std_pivots is not None:
                        sv = std_pivots[idx].values[yi, xi]
                        if np.isfinite(sv):
                            txt    = f"{txt}\n±{format(sv, fmt)}"
                            ann_fs = annot_fontsize * 0.82
                    # ─────────────────────────────────────────────────────────

                    ax.text(xi, yi, txt,
                            ha="center", va="center",
                            fontsize=ann_fs, fontweight="bold",
                            color=_text_color(rgba), zorder=3,
                            linespacing=1.2)               # ← NEW: tighten 2-liner

        for x in np.arange(-0.5, nx, 1):
            ax.axvline(x, color="white", linewidth=0.8, zorder=2)
        for y in np.arange(-0.5, ny, 1):
            ax.axhline(y, color="white", linewidth=0.8, zorder=2)

        ax.set_xticks(range(nx))
        ax.set_yticks(range(ny))
        ax.set_xticklabels([str(v) for v in ref_piv.columns])
        ax.set_yticklabels([str(v) for v in ref_piv.index])
        ax.set_xlabel(x_col, fontsize=10, labelpad=6)
        ax.set_ylabel(y_col, fontsize=10, labelpad=6)

        title = ("  ·  ".join(f"{c}={v}" for c, v in zip(subplot_cols, key))
                 if subplot_cols else (value_col or "metrics"))
        ax.set_title(title, fontsize=10, fontweight="semibold", pad=8)

    for idx in range(n, nrows_ * ncols_):
        axes[idx // ncols_][idx % ncols_].set_visible(False)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig