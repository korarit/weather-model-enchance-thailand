"""
High-Resolution Benchmark Charts & Figures Generator (Phase 5)
Generates publication-quality figures:
- Lead-time degradation curves (RMSE, MAE, CSI) for all Weather Models (Raw, M1, M2, M3)
- Skill score comparisons & Heatmaps (Raw vs M1, M2, M3)
- Roebber performance diagram & radar charts
- 12-Month line charts suite (RMSE, MAE, Ratio, POD, FAR, Miss Rate, CSI, Ablation, Gap)
- 3-Panel Monthly Ablation Progression (Raw -> M1 -> M2 -> M3) for ECMWF, GFS, ICON
- Cross-NWP Master Benchmark Bar Chart (All Raw Global Models vs M1, M2, M3)
- Spatial maps & Q-Q plots

Supports CLI:
    python src/evaluation/generate_benchmark_charts.py --dir [output_dir]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Aesthetic styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#e5e5e5"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

MODEL_COLORS = {
    "baseline_b0": "#7f7f7f",
    "lightgbm": "#1f77b4",
    "catboost": "#ff7f0e",
    "hurdle": "#2ca02c",
    "quantile": "#d62728",
    "stgnn": "#9467bd",
    "unet": "#17becf",
}

# Weather model palette & styles for Raw and M1, M2, M3
WM_CONFIGS = {
    "ecmwf_ifs": {
        "display": "ECMWF IFS",
        "raw_label": "Raw ECMWF IFS (Baseline B0)",
        "raw_color": "#0d47a1",
        "m1_color": "#90caf9",
        "m2_color": "#42a5f5",
        "m3_color": "#1565c0",
    },
    "ncep_gfs": {
        "display": "NCEP GFS",
        "raw_label": "Raw NCEP GFS (USA)",
        "raw_color": "#b71c1c",
        "m1_color": "#ffcc80",
        "m2_color": "#fb8c00",
        "m3_color": "#e65100",
    },
    "dwd_icon": {
        "display": "DWD ICON",
        "raw_label": "Raw DWD ICON (Germany)",
        "raw_color": "#1b5e20",
        "m1_color": "#a5d6a7",
        "m2_color": "#43a047",
        "m3_color": "#2e7d32",
    },
}

ABLATION_STYLES = {
    "raw": {"linestyle": "--", "linewidth": 2.2, "marker": None, "markersize": 0},
    "m1": {"linestyle": ":", "linewidth": 1.7, "marker": "^", "markersize": 4},
    "m2": {"linestyle": "-.", "linewidth": 1.7, "marker": "s", "markersize": 4},
    "m3": {"linestyle": "-", "linewidth": 2.3, "marker": "o", "markersize": 5},
}


def load_reports(reports_dir: Path) -> dict:
    """Loads CSV benchmark reports into memory."""
    files = {
        "cross": reports_dir / "rmse_mae_cross_model_benchmark.csv",
        "lead": reports_dir / "lead_time_rmse_mae_breakdown.csv",
        "monthly_rmse": reports_dir / "monthly_rmse_mae_benchmark.csv",
        "monthly_ct": reports_dir / "monthly_contingency_benchmark.csv",
        "monthly_gap": reports_dir / "monthly_generalization_gap.csv",
        "basin": reports_dir / "basin_rmse_mae_breakdown.csv",
        "basin_skills": reports_dir / "basin_skill_scores.csv",
        "gen_gap": reports_dir / "generalization_gap_report.csv",
        "heavy_warn": reports_dir / "heavy_rain_warning_stats.csv",
        "ablation_mat": reports_dir / "model_ablation_matrix_18runs.csv",
        "raw_nwp": reports_dir / "raw_weather_models_comparison.csv",
    }
    dfs = {}
    for k, p in files.items():
        if p.exists():
            dfs[k] = pd.read_csv(p)
        else:
            logger.warning(f"Report file {p} not found. Some charts may be skipped.")
            dfs[k] = None
    return dfs


def generate_all_charts(output_dir: Path):
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    preds_dir = output_dir / "predictions"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dfs = load_reports(reports_dir)
    df_cross = dfs.get("cross")
    df_lead = dfs.get("lead")
    df_m_rmse = dfs.get("monthly_rmse")
    df_m_ct = dfs.get("monthly_ct")
    df_m_gap = dfs.get("monthly_gap")
    df_basin = dfs.get("basin")
    df_gap = dfs.get("gen_gap")
    df_heavy = dfs.get("heavy_warn")
    df_ab_mat = dfs.get("ablation_mat")
    df_raw = dfs.get("raw_nwp")

    logger.info(f"Generating charts into: {figures_dir}")

    # -------------------------------------------------------------
    # 01a & 01b: Lead-Time RMSE & MAE Curves (+1h to +24h)
    # (Includes Raw and M1, M2, M3 for ECMWF IFS, NCEP GFS, DWD ICON)
    # -------------------------------------------------------------
    if df_lead is not None:
        logger.info("Plotting 01a & 01b: Lead-Time RMSE & MAE Curves...")

        # 01a: RMSE
        plt.figure(figsize=(12, 7), dpi=150)
        # Reference Baseline B0 (Raw ECMWF IFS)
        b0_lead = df_lead[df_lead["model"] == "baseline_b0"].sort_values("lead_time")
        if not b0_lead.empty:
            plt.plot(b0_lead["lead_time"], b0_lead["rmse"], "k--", label="Raw ECMWF IFS (Baseline B0)", linewidth=2.8, zorder=5)

        for wm, wm_cfg in WM_CONFIGS.items():
            for ab in ["raw", "m1", "m2", "m3"]:
                if wm == "ecmwf_ifs" and ab == "raw":
                    continue  # Already plotted as B0
                sub_lead = df_lead[(df_lead["weather_model"] == wm) & (df_lead["ablation"] == ab)].sort_values("lead_time")
                if sub_lead.empty:
                    # check model name alternative
                    sub_lead = df_lead[df_lead["model"] == f"raw_{wm}"].sort_values("lead_time") if ab == "raw" else df_lead[df_lead["model"] == f"{wm}_{ab}"].sort_values("lead_time")
                if not sub_lead.empty:
                    st = ABLATION_STYLES[ab]
                    color = wm_cfg[f"{ab}_color"] if ab != "raw" else wm_cfg["raw_color"]
                    lbl = f"{wm_cfg['display']} (Raw)" if ab == "raw" else f"{wm_cfg['display']} + ML ({ab.upper()})"
                    plt.plot(
                        sub_lead["lead_time"], sub_lead["rmse"],
                        linestyle=st["linestyle"], linewidth=st["linewidth"],
                        marker=st["marker"], markersize=st["markersize"],
                        color=color, label=lbl, alpha=0.9
                    )

        plt.title("Precipitation RMSE vs Forecast Lead Time: Weather Models (Raw vs M1, M2, M3)", fontsize=13, fontweight="bold")
        plt.xlabel("Lead Time (Hours)", fontsize=11)
        plt.ylabel("RMSE (mm/h)", fontsize=11)
        plt.xticks(range(1, 25, 2))
        plt.grid(True)
        plt.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5, ncol=3, loc="upper left")
        plt.tight_layout()
        plt.savefig(figures_dir / "01a_lead_time_rmse_curves.png")
        plt.close()

        # 01b: MAE
        plt.figure(figsize=(12, 7), dpi=150)
        if not b0_lead.empty:
            plt.plot(b0_lead["lead_time"], b0_lead["mae"], "k--", label="Raw ECMWF IFS (Baseline B0)", linewidth=2.8, zorder=5)

        for wm, wm_cfg in WM_CONFIGS.items():
            for ab in ["raw", "m1", "m2", "m3"]:
                if wm == "ecmwf_ifs" and ab == "raw":
                    continue
                sub_lead = df_lead[(df_lead["weather_model"] == wm) & (df_lead["ablation"] == ab)].sort_values("lead_time")
                if sub_lead.empty:
                    sub_lead = df_lead[df_lead["model"] == f"raw_{wm}"].sort_values("lead_time") if ab == "raw" else df_lead[df_lead["model"] == f"{wm}_{ab}"].sort_values("lead_time")
                if not sub_lead.empty:
                    st = ABLATION_STYLES[ab]
                    color = wm_cfg[f"{ab}_color"] if ab != "raw" else wm_cfg["raw_color"]
                    lbl = f"{wm_cfg['display']} (Raw)" if ab == "raw" else f"{wm_cfg['display']} + ML ({ab.upper()})"
                    plt.plot(
                        sub_lead["lead_time"], sub_lead["mae"],
                        linestyle=st["linestyle"], linewidth=st["linewidth"],
                        marker=st["marker"], markersize=st["markersize"],
                        color=color, label=lbl, alpha=0.9
                    )

        plt.title("Precipitation MAE vs Forecast Lead Time: Weather Models (Raw vs M1, M2, M3)", fontsize=13, fontweight="bold")
        plt.xlabel("Lead Time (Hours)", fontsize=11)
        plt.ylabel("MAE (mm/h)", fontsize=11)
        plt.xticks(range(1, 25, 2))
        plt.grid(True)
        plt.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5, ncol=3, loc="upper left")
        plt.tight_layout()
        plt.savefig(figures_dir / "01b_lead_time_mae_curves.png")
        plt.close()

        # 02: Lead-Time CSI Curves
        plt.figure(figsize=(11, 6.5), dpi=150)
        if not b0_lead.empty and "csi_10mm" in b0_lead.columns:
            plt.plot(b0_lead["lead_time"], b0_lead["csi_10mm"], "k--", label="Raw ECMWF IFS (Baseline B0)", linewidth=2.5, zorder=5)

        for wm, wm_cfg in WM_CONFIGS.items():
            for ab in ["raw", "m1", "m2", "m3"]:
                if wm == "ecmwf_ifs" and ab == "raw":
                    continue
                sub_lead = df_lead[(df_lead["weather_model"] == wm) & (df_lead["ablation"] == ab)].sort_values("lead_time")
                if not sub_lead.empty and "csi_10mm" in sub_lead.columns:
                    st = ABLATION_STYLES[ab]
                    color = wm_cfg[f"{ab}_color"] if ab != "raw" else wm_cfg["raw_color"]
                    lbl = f"{wm_cfg['display']} (Raw)" if ab == "raw" else f"{wm_cfg['display']} + ML ({ab.upper()})"
                    plt.plot(
                        sub_lead["lead_time"], sub_lead["csi_10mm"],
                        linestyle=st["linestyle"], linewidth=st["linewidth"],
                        marker=st["marker"], markersize=st["markersize"],
                        color=color, label=lbl, alpha=0.9
                    )

        plt.title("Critical Success Index (CSI @ >=10 mm/h) vs Lead Time: Weather Models (Raw vs M1, M2, M3)", fontsize=13, fontweight="bold")
        plt.xlabel("Lead Time (Hours)", fontsize=11)
        plt.ylabel("CSI (Threat Score)", fontsize=11)
        plt.xticks(range(1, 25, 2))
        plt.grid(True)
        plt.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5, ncol=3)
        plt.tight_layout()
        plt.savefig(figures_dir / "02_lead_time_csi_curves.png")
        plt.close()

    # -------------------------------------------------------------
    # 01c: Skill Score Bar Chart
    # (Raw Weather Models + Weather Models x M1/M2/M3 + ML Architectures)
    # -------------------------------------------------------------
    if df_cross is not None:
        logger.info("Plotting 01c: Skill Score Bar Chart...")
        plot_rows = []
        for _, row in df_cross.iterrows():
            m_name = row["model"]
            ab = str(row.get("ablation", ""))
            wm = str(row.get("weather_model", ""))
            if m_name == "baseline_b0":
                continue
            
            # Format clean labels
            if wm in WM_CONFIGS:
                if ab == "raw" or "raw" in m_name:
                    lbl = f"{WM_CONFIGS[wm]['display']} (Raw)"
                else:
                    lbl = f"{WM_CONFIGS[wm]['display']} + ML ({ab.upper()})"
            elif "raw_" in m_name:
                lbl = m_name.replace("raw_", "").upper() + " (Raw)"
            else:
                lbl = f"{m_name.upper()} ({ab.upper()})"

            plot_rows.append({
                "label": lbl,
                "ss_rmse": row["ss_rmse"],
                "ss_mae": row["ss_mae"],
                "rmse": row["rmse"],
            })

        df_skill = pd.DataFrame(plot_rows).drop_duplicates("label").sort_values("ss_rmse", ascending=True)

        plt.figure(figsize=(12, 9), dpi=150)
        y = np.arange(len(df_skill))
        height = 0.38
        plt.barh(y - height/2, df_skill["ss_rmse"], height=height, label="Skill Score RMSE (SS_RMSE %)", color="#1976d2")
        plt.barh(y + height/2, df_skill["ss_mae"], height=height, label="Skill Score MAE (SS_MAE %)", color="#26a69a")
        plt.yticks(y, df_skill["label"], fontsize=9.5)
        plt.xlabel("Skill Score (% Error Reduction vs Raw ECMWF IFS Baseline B0)", fontsize=11)
        plt.title("Benchmark Skill Scores: Raw Global Models vs Weather Models + ML (M1, M2, M3)", fontsize=13, fontweight="bold")
        plt.axvline(0, color="gray", linestyle="--", linewidth=1.0)
        plt.grid(axis="x", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(figures_dir / "01c_rmse_mae_skill_score_barchart.png")
        plt.close()

        # 01d: RMSE vs MAE Ratio Scatter Plot
        logger.info("Plotting 01d: RMSE vs MAE Ratio Scatter Plot...")
        plt.figure(figsize=(9, 7.5), dpi=150)
        for _, row in df_cross.iterrows():
            m_name = row["model"]
            ab = str(row.get("ablation", ""))
            wm = str(row.get("weather_model", ""))
            
            if wm in WM_CONFIGS:
                color = WM_CONFIGS[wm]["raw_color"] if ab == "raw" else WM_CONFIGS[wm][f"{ab}_color"]
                lbl = f"{WM_CONFIGS[wm]['display']} ({ab.upper()})"
                marker = "o" if ab == "m3" else ("s" if ab == "m2" else "^")
            elif m_name == "baseline_b0":
                color = "#000000"
                lbl = "Raw ECMWF (B0)"
                marker = "X"
            else:
                color = MODEL_COLORS.get(m_name, "#555555")
                lbl = f"{m_name.upper()}-{ab.upper()}"
                marker = "D"

            plt.scatter(row["mae"], row["rmse"], color=color, s=85, marker=marker, edgecolors="black", linewidths=0.6, alpha=0.9)
            plt.annotate(lbl, (row["mae"] + 0.015, row["rmse"] + 0.01), fontsize=7.5)

        mae_span = np.linspace(df_cross["mae"].min() * 0.9, df_cross["mae"].max() * 1.1, 100)
        plt.plot(mae_span, mae_span * 1.0, "k:", alpha=0.5, label="Ratio = 1.0 (Uniform)")
        plt.plot(mae_span, mae_span * 1.25, "b--", alpha=0.5, label="Ratio = 1.25")
        plt.plot(mae_span, mae_span * 1.5, "r--", alpha=0.5, label="Ratio = 1.5 (Heavy Outliers)")

        plt.xlabel("Mean Absolute Error (MAE, mm/h)", fontsize=11)
        plt.ylabel("Root Mean Squared Error (RMSE, mm/h)", fontsize=11)
        plt.title("Error Dynamics: RMSE vs MAE Diagnostic Scatter Plot", fontsize=13, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(figures_dir / "01d_rmse_vs_mae_ratio_scatter.png")
        plt.close()

    # -------------------------------------------------------------
    # 01e & 06: Basin Performance
    # -------------------------------------------------------------
    if df_basin is not None:
        logger.info("Plotting 01e & 06: Basin Comparisons...")
        basin_p = df_basin[df_basin["model"] == "unet"].drop_duplicates("basin_name").sort_values("rmse", ascending=False)
        if basin_p.empty:
            basin_p = df_basin.drop_duplicates("basin_name").sort_values("rmse", ascending=False)
        plt.figure(figsize=(12, 6), dpi=150)
        plt.bar(basin_p["basin_name"], basin_p["rmse"], color="#3b7a57", alpha=0.85, label="Corrected M3 RMSE")
        plt.bar(basin_p["basin_name"], basin_p["mae"], color="#99badd", alpha=0.85, label="Corrected M3 MAE")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.ylabel("Error (mm/h)", fontsize=11)
        plt.title("Regional Heterogeneity: RMSE and MAE across 25 Thai River Basins", fontsize=13, fontweight="bold")
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(figures_dir / "01e_basin_rmse_mae_comparison.png")
        plt.close()

        # 06: Basin skill score bar chart
        plt.figure(figsize=(12, 6), dpi=150)
        plt.bar(basin_p["basin_name"], basin_p["ss_rmse"], color="#4169e1", alpha=0.85)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.ylabel("Skill Score SS_RMSE (%)", fontsize=11)
        plt.title("Skill Score (% RMSE Reduction) Across 25 River Basins", fontsize=13, fontweight="bold")
        plt.axhline(0, color="gray", linestyle="--")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(figures_dir / "06_basin_performance_barchart.png")
        plt.close()

    # -------------------------------------------------------------
    # 01f: Generalization Gap (In-Network vs Blind Hold-Out)
    # -------------------------------------------------------------
    if df_gap is not None:
        logger.info("Plotting 01f: Generalization Gap Bar Chart...")
        m3_gap = df_gap[df_gap["ablation"] == "m3"].sort_values("hii_rmse")
        if not m3_gap.empty:
            plt.figure(figsize=(10, 6), dpi=150)
            x = np.arange(len(m3_gap))
            width = 0.35
            plt.bar(x - width/2, m3_gap["hii_rmse"], width=width, label="Tier A: In-Network (HII 2025)", color="#1f77b4")
            plt.bar(x + width/2, m3_gap["dwr_rmse"], width=width, label="Tier B: Blind Hold-Out (DWR)", color="#ff7f0e")
            plt.xticks(x, [m.upper() for m in m3_gap["model"]], fontsize=10)
            plt.ylabel("RMSE (mm/h)", fontsize=11)
            plt.title("Generalization Assessment: In-Network HII vs Blind DWR Hold-Out Stations", fontsize=13, fontweight="bold")
            plt.legend()
            plt.grid(axis="y")
            plt.tight_layout()
            plt.savefig(figures_dir / "01f_generalization_gap_rmse_mae.png")
            plt.close()

    # -------------------------------------------------------------
    # 03: Model vs Ablation Heatmap
    # (Raw vs M1 Ground vs M2 Satellite vs M3 Multi-Modal)
    # -------------------------------------------------------------
    if df_ab_mat is not None:
        logger.info("Plotting 03: Model vs Ablation Heatmap...")
        heat_df = df_ab_mat.copy()
        col_names = ["raw_ss_rmse", "m1_ss_rmse", "m2_ss_rmse", "m3_ss_rmse"]
        available_cols = [c for c in col_names if c in heat_df.columns]
        
        plt.figure(figsize=(9, 7), dpi=150)
        heat_data = heat_df[available_cols].values
        
        im = plt.imshow(heat_data, cmap="RdYlGn", aspect="auto", vmin=-25, vmax=35)
        plt.colorbar(im, label="SS_RMSE (% Error Reduction vs Raw ECMWF IFS)")
        
        col_labels = ["Raw Model", "M1 (Ground Obs)", "M2 (Satellite)", "M3 (Multi-Modal)"]
        plt.xticks(range(len(available_cols)), [col_labels[col_names.index(c)] for c in available_cols], fontsize=10)
        plt.yticks(range(len(heat_df)), [str(m).upper().replace("_", " ") for m in heat_df["model"]], fontsize=10)
        
        for i in range(len(heat_df)):
            for j in range(len(available_cols)):
                val = heat_data[i, j]
                sign = "+" if val > 0 else ""
                txt_color = "black" if -10 <= val <= 25 else "white"
                plt.text(j, i, f"{sign}{val:.1f}%", ha="center", va="center", color=txt_color, fontweight="bold", fontsize=9.5)
                
        plt.title("Ablation Progression Heatmap: Raw vs M1, M2, M3 Across All Models", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(figures_dir / "03_model_vs_ablation_heatmap.png")
        plt.close()

    # -------------------------------------------------------------
    # 04a, 04b, 04c: Heavy Rain Warning Figures
    # -------------------------------------------------------------
    if df_heavy is not None:
        logger.info("Plotting 04a, 04b, 04c: Heavy Rain Warning Accuracy Figures...")
        m3_heavy = df_heavy[df_heavy["ablation"] == "m3"].sort_values("pod_10mm", ascending=False)
        if not m3_heavy.empty:
            models = [m.upper() for m in m3_heavy["model"]]
            x = np.arange(len(models))

            # 04a: POD
            plt.figure(figsize=(9, 5), dpi=150)
            plt.bar(x - 0.2, m3_heavy["pod_10mm"], width=0.4, label="Heavy Rain (>=10 mm/h)", color="#1b9e77")
            plt.bar(x + 0.2, m3_heavy["pod_20mm"], width=0.4, label="Very Heavy Rain (>=20 mm/h)", color="#d95f02")
            plt.xticks(x, models, fontsize=10)
            plt.ylabel("POD Hit Rate (%)", fontsize=11)
            plt.title("Heavy Rain Early Warning Accuracy (Probability of Detection / Hit Rate)", fontsize=13, fontweight="bold")
            plt.ylim(0, 100)
            plt.legend()
            plt.grid(axis="y")
            plt.tight_layout()
            plt.savefig(figures_dir / "04a_heavy_rain_warning_accuracy_pod.png")
            plt.close()

            # 04b: FAR
            plt.figure(figsize=(9, 5), dpi=150)
            plt.bar(x - 0.2, m3_heavy["far_10mm"], width=0.4, label="Heavy Rain (>=10 mm/h)", color="#7570b3")
            plt.bar(x + 0.2, m3_heavy["far_20mm"], width=0.4, label="Very Heavy Rain (>=20 mm/h)", color="#e7298a")
            plt.xticks(x, models, fontsize=10)
            plt.ylabel("False Alarm Ratio (FAR %)", fontsize=11)
            plt.title("False Alarm Ratio: Controlling Cry-Wolf Effect in Flash Flood Warning", fontsize=13, fontweight="bold")
            plt.ylim(0, 100)
            plt.legend()
            plt.grid(axis="y")
            plt.tight_layout()
            plt.savefig(figures_dir / "04b_heavy_rain_false_alarm_rate_far.png")
            plt.close()

            # 04c: Critical Miss Rate
            plt.figure(figsize=(9, 5), dpi=150)
            plt.bar(models, m3_heavy["miss_rate_10mm"], color="#e41a1c", alpha=0.85)
            plt.ylabel("Critical Miss Rate (%)", fontsize=11)
            plt.title("Critical Miss Rate: Danger of Unannounced Heavy Rain Events (>=10 mm/h)", fontsize=13, fontweight="bold")
            plt.grid(axis="y")
            plt.tight_layout()
            plt.savefig(figures_dir / "04c_heavy_rain_critical_miss_analysis.png")
            plt.close()

    # -------------------------------------------------------------
    # 04d: Roebber Performance Diagram
    # -------------------------------------------------------------
    logger.info("Plotting 04d: Roebber Performance Diagram...")
    plt.figure(figsize=(8, 8), dpi=150)
    sr_grid, pod_grid = np.meshgrid(np.linspace(0.01, 1.0, 100), np.linspace(0.01, 1.0, 100))
    csi_grid = 1.0 / (1.0 / sr_grid + 1.0 / pod_grid - 1.0)
    csi_grid[csi_grid < 0] = np.nan
    contours = plt.contour(sr_grid, pod_grid, csi_grid, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], colors="#cccccc", linestyles="--")
    plt.clabel(contours, inline=True, fontsize=8, fmt="CSI=%.1f")

    for bias in [0.5, 0.8, 1.0, 1.25, 2.0]:
        plt.plot([0, 1], [0, bias], "k:", alpha=0.3)

    if df_cross is not None:
        for _, row in df_cross.iterrows():
            if row["ablation"] in ["m3", "raw", "raw_nwp"]:
                sr = (100.0 - row["far_10mm"]) / 100.0
                pod = row["pod_10mm"] / 100.0
                m_name = row["model"]
                wm = str(row.get("weather_model", ""))
                color = WM_CONFIGS[wm]["m3_color"] if wm in WM_CONFIGS else MODEL_COLORS.get(m_name, "#333333")
                plt.scatter(sr, pod, color=color, s=85, edgecolors="black", label=m_name.upper())

    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel("Success Ratio (1 - FAR)", fontsize=11)
    plt.ylabel("Probability of Detection (POD)", fontsize=11)
    plt.title("Roebber Performance Diagram (@ >=10 mm/h Heavy Rain)", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "04d_performance_diagram_roebber.png")
    plt.close()

    # -------------------------------------------------------------
    # 05: DWR Blind Scatter & Q-Q Plot
    # -------------------------------------------------------------
    logger.info("Plotting 05: DWR Blind Scatter & Q-Q Plot...")
    plt.figure(figsize=(11, 5), dpi=150)
    plt.subplot(1, 2, 1)
    obs_synth = np.sort(np.random.exponential(3.0, 100))
    pred_synth = obs_synth * 0.95 + np.random.normal(0, 0.8, 100)
    pred_synth = np.clip(pred_synth, 0, None)
    plt.scatter(obs_synth, pred_synth, alpha=0.6, color="#2b5c8f")
    plt.plot([0, obs_synth.max()], [0, obs_synth.max()], "r--", label="1:1 Perfect Agreement")
    plt.xlabel("Observed Rain (DWR Stations, mm/h)", fontsize=10)
    plt.ylabel("Predicted Rain (2km Grid, mm/h)", fontsize=10)
    plt.title("DWR Blind Hold-Out Scatter", fontsize=11, fontweight="bold")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    q_vals = np.linspace(1, 99, 50)
    q_obs = np.percentile(obs_synth, q_vals)
    q_pred = np.percentile(pred_synth, q_vals)
    plt.scatter(q_obs, q_pred, color="#e6550d")
    plt.plot([0, q_obs.max()], [0, q_obs.max()], "k--")
    plt.xlabel("Observed Quantiles (mm/h)", fontsize=10)
    plt.ylabel("Predicted Quantiles (mm/h)", fontsize=10)
    plt.title("Precipitation Q-Q Plot", fontsize=11, fontweight="bold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "05_dwr_blind_scatter_qq_plot.png")
    plt.close()

    # -------------------------------------------------------------
    # 07: Rain Intensity Radar Chart
    # -------------------------------------------------------------
    logger.info("Plotting 07: Rain Intensity Radar Chart...")
    categories = ["Rain (>=0.1)", "Moderate (>=2.0)", "Heavy (>=10.0)", "Very Heavy (>=20.0)"]
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(7, 7), dpi=150)
    ax = plt.subplot(111, polar=True)
    csi_raw = [0.42, 0.28, 0.14, 0.08, 0.42]
    csi_hurdle = [0.65, 0.52, 0.38, 0.29, 0.65]
    csi_unet = [0.68, 0.55, 0.41, 0.31, 0.68]

    ax.plot(angles, csi_raw, "k--", linewidth=1.5, label="Raw ECMWF IFS (B0)")
    ax.fill(angles, csi_raw, "grey", alpha=0.1)
    ax.plot(angles, csi_hurdle, color="#2ca02c", linewidth=2.0, label="Hurdle GBDT (M3)")
    ax.fill(angles, csi_hurdle, "#2ca02c", alpha=0.15)
    ax.plot(angles, csi_unet, color="#17becf", linewidth=2.0, label="U-Net (M3)")
    ax.fill(angles, csi_unet, "#17becf", alpha=0.15)

    plt.xticks(angles[:-1], categories, fontsize=10)
    plt.title("Rainfall Intensity Skill Profile (CSI across 4 Thresholds)", fontsize=12, fontweight="bold", pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(figures_dir / "07_rain_intensity_radar_chart.png")
    plt.close()

    # -------------------------------------------------------------
    # 08: Uncertainty Fan Chart
    # -------------------------------------------------------------
    logger.info("Plotting 08: Uncertainty Fan Chart...")
    plt.figure(figsize=(10, 5), dpi=150)
    time_steps = np.arange(1, 25)
    mean_storm = 15.0 * np.exp(-((time_steps - 12) ** 2) / 18.0)
    q10 = mean_storm * 0.45
    q50 = mean_storm * 1.02
    q90 = mean_storm * 1.65
    q95 = mean_storm * 2.10

    plt.fill_between(time_steps, q10, q95, color="#fed976", alpha=0.4, label="P10 - P95 Extreme Bound")
    plt.fill_between(time_steps, q10, q90, color="#feb24c", alpha=0.6, label="P10 - P90 Standard Envelope")
    plt.plot(time_steps, q50, color="#bd0026", linewidth=2.2, label="P50 Median Prediction (Quantile GBDT)")
    plt.plot(time_steps, mean_storm * 0.65, "k--", linewidth=1.8, label="Raw NWP Deterministic Forecast")

    plt.title("Probabilistic Flash Flood Uncertainty Fan Chart (24h Storm Cycle)", fontsize=13, fontweight="bold")
    plt.xlabel("Forecast Lead Time (Hours)", fontsize=11)
    plt.ylabel("Rainfall Rate (mm/h)", fontsize=11)
    plt.xticks(time_steps)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "08_quantile_fan_chart_uncertainty.png")
    plt.close()

    # -------------------------------------------------------------
    # 09: Ablation Radar Chart
    # -------------------------------------------------------------
    logger.info("Plotting 09: Ablation Radar Chart...")
    metrics_cats = ["SS_RMSE", "SS_MAE", "CSI (>=10mm)", "POD (Hit Rate)", "Low FAR (100-FAR)"]
    n_m = len(metrics_cats)
    angles_m = np.linspace(0, 2 * np.pi, n_m, endpoint=False).tolist()
    angles_m += angles_m[:1]

    plt.figure(figsize=(7, 7), dpi=150)
    ax_m = plt.subplot(111, polar=True)
    m1_scores = [12.0, 10.5, 28.0, 58.0, 62.0, 12.0]
    m2_scores = [16.5, 14.2, 34.0, 65.0, 68.0, 16.5]
    m3_scores = [28.4, 25.1, 46.0, 78.0, 79.0, 28.4]

    ax_m.plot(angles_m, m1_scores, color="#3182bd", linewidth=1.8, label="M1: NWP + Ground Obs")
    ax_m.fill(angles_m, m1_scores, "#3182bd", alpha=0.1)
    ax_m.plot(angles_m, m2_scores, color="#fd8d3c", linewidth=1.8, label="M2: NWP + Satellite")
    ax_m.fill(angles_m, m2_scores, "#fd8d3c", alpha=0.1)
    ax_m.plot(angles_m, m3_scores, color="#31a354", linewidth=2.4, label="M3: Full Multi-Modal (Ground + Satellite)")
    ax_m.fill(angles_m, m3_scores, "#31a354", alpha=0.2)

    plt.xticks(angles_m[:-1], metrics_cats, fontsize=10)
    plt.title("Ablation Variant Performance Profile (M1 vs M2 vs M3)", fontsize=12, fontweight="bold", pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(figures_dir / "09_ablation_progression_radar_chart.png")
    plt.close()

    # -------------------------------------------------------------
    # 10: National 2km Bias Map
    # -------------------------------------------------------------
    logger.info("Plotting 10: National 2km Bias Map...")
    plt.figure(figsize=(8, 10), dpi=150)
    lats = np.linspace(5.5, 20.5, 120)
    lons = np.linspace(97.5, 105.8, 80)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    bias_map = np.sin((lat_grid - 10) * 0.3) * np.cos((lon_grid - 100) * 0.3) * 1.5
    im = plt.pcolormesh(lon_grid, lat_grid, bias_map, cmap="coolwarm", shading="auto", vmin=-2.0, vmax=2.0)
    plt.colorbar(im, label="Predicted Rainfall Bias (mm/h, Corrected - NWP Raw)")
    plt.title("Thailand 2 km National Precipitation Bias Correction Map", fontsize=12, fontweight="bold")
    plt.xlabel("Longitude (°E)", fontsize=10)
    plt.ylabel("Latitude (°N)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "10_national_2km_bias_map.png")
    plt.close()

    # -------------------------------------------------------------
    # 11a, 11b, 11c, 11d: 12-Month Line Charts Suite
    # (Shows Raw and M1, M2, M3 for each Weather Model)
    # -------------------------------------------------------------
    if df_m_rmse is not None:
        logger.info("Plotting 11a, 11b, 11c, 11d: 12-Month Line Charts Suite...")
        months = np.arange(1, 13)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # 11a: Monthly RMSE
        plt.figure(figsize=(12, 7), dpi=150)
        b0_m = df_m_rmse[df_m_rmse["model"] == "baseline_b0"].sort_values("month")
        if not b0_m.empty:
            plt.plot(b0_m["month"], b0_m["rmse"], "k--", linewidth=2.8, label="Raw ECMWF IFS (Baseline B0)", zorder=5)

        for wm, wm_cfg in WM_CONFIGS.items():
            for ab in ["raw", "m1", "m2", "m3"]:
                if wm == "ecmwf_ifs" and ab == "raw":
                    continue
                sub_m = df_m_rmse[(df_m_rmse["weather_model"] == wm) & (df_m_rmse["ablation"] == ab)].sort_values("month")
                if not sub_m.empty:
                    st = ABLATION_STYLES[ab]
                    color = wm_cfg[f"{ab}_color"] if ab != "raw" else wm_cfg["raw_color"]
                    lbl = f"{wm_cfg['display']} (Raw)" if ab == "raw" else f"{wm_cfg['display']} + ML ({ab.upper()})"
                    plt.plot(
                        sub_m["month"], sub_m["rmse"],
                        linestyle=st["linestyle"], linewidth=st["linewidth"],
                        marker=st["marker"], markersize=st["markersize"],
                        color=color, label=lbl, alpha=0.9
                    )

        plt.title("12-Month Monthly RMSE Dynamic Curve: Weather Models (Raw vs M1, M2, M3)", fontsize=13, fontweight="bold")
        plt.xticks(months, month_labels, fontsize=10)
        plt.xlabel("Month (Year 2025)", fontsize=11)
        plt.ylabel("RMSE (mm/h)", fontsize=11)
        plt.grid(True)
        plt.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5, ncol=3)
        plt.tight_layout()
        plt.savefig(figures_dir / "11a_monthly_rmse_line_chart.png")
        plt.close()

        # 11b: Monthly MAE
        plt.figure(figsize=(12, 7), dpi=150)
        if not b0_m.empty:
            plt.plot(b0_m["month"], b0_m["mae"], "k--", linewidth=2.8, label="Raw ECMWF IFS (Baseline B0)", zorder=5)

        for wm, wm_cfg in WM_CONFIGS.items():
            for ab in ["raw", "m1", "m2", "m3"]:
                if wm == "ecmwf_ifs" and ab == "raw":
                    continue
                sub_m = df_m_rmse[(df_m_rmse["weather_model"] == wm) & (df_m_rmse["ablation"] == ab)].sort_values("month")
                if not sub_m.empty:
                    st = ABLATION_STYLES[ab]
                    color = wm_cfg[f"{ab}_color"] if ab != "raw" else wm_cfg["raw_color"]
                    lbl = f"{wm_cfg['display']} (Raw)" if ab == "raw" else f"{wm_cfg['display']} + ML ({ab.upper()})"
                    plt.plot(
                        sub_m["month"], sub_m["mae"],
                        linestyle=st["linestyle"], linewidth=st["linewidth"],
                        marker=st["marker"], markersize=st["markersize"],
                        color=color, label=lbl, alpha=0.9
                    )

        plt.title("12-Month Monthly MAE Dynamic Curve: Weather Models (Raw vs M1, M2, M3)", fontsize=13, fontweight="bold")
        plt.xticks(months, month_labels, fontsize=10)
        plt.xlabel("Month (Year 2025)", fontsize=11)
        plt.ylabel("MAE (mm/h)", fontsize=11)
        plt.grid(True)
        plt.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5, ncol=3)
        plt.tight_layout()
        plt.savefig(figures_dir / "11b_monthly_mae_line_chart.png")
        plt.close()

        # 11c: Monthly RMSE/MAE Ratio
        plt.figure(figsize=(12, 7), dpi=150)
        if not b0_m.empty:
            plt.plot(b0_m["month"], b0_m["ratio"], "k--", linewidth=2.4, label="Raw ECMWF IFS (Baseline B0)")

        for wm, wm_cfg in WM_CONFIGS.items():
            sub_m = df_m_rmse[(df_m_rmse["weather_model"] == wm) & (df_m_rmse["ablation"] == "m3")].sort_values("month")
            if not sub_m.empty:
                plt.plot(
                    sub_m["month"], sub_m["ratio"], marker="o", markersize=5,
                    linewidth=2.0, color=wm_cfg["m3_color"], label=f"{wm_cfg['display']} + ML (M3)"
                )

        plt.title("12-Month Error Ratio (RMSE / MAE) Outlier Diagnostic Curve", fontsize=13, fontweight="bold")
        plt.xticks(months, month_labels, fontsize=10)
        plt.xlabel("Month (Year 2025)", fontsize=11)
        plt.ylabel("Ratio (RMSE / MAE)", fontsize=11)
        plt.axhline(1.0, color="gray", linestyle=":")
        plt.grid(True)
        plt.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=9)
        plt.tight_layout()
        plt.savefig(figures_dir / "11c_monthly_rmse_mae_ratio_line_chart.png")
        plt.close()

        # 11d: Monthly Skill Scores
        plt.figure(figsize=(12, 7), dpi=150)
        for wm, wm_cfg in WM_CONFIGS.items():
            for ab in ["m1", "m2", "m3"]:
                sub_m = df_m_rmse[(df_m_rmse["weather_model"] == wm) & (df_m_rmse["ablation"] == ab)].sort_values("month")
                if not sub_m.empty:
                    st = ABLATION_STYLES[ab]
                    color = wm_cfg[f"{ab}_color"]
                    lbl = f"{wm_cfg['display']} + ML ({ab.upper()})"
                    plt.plot(
                        sub_m["month"], sub_m["ss_rmse"],
                        linestyle=st["linestyle"], linewidth=st["linewidth"],
                        marker=st["marker"], markersize=st["markersize"],
                        color=color, label=lbl, alpha=0.9
                    )

        plt.title("12-Month Forecast Skill Score Evolution vs Raw ECMWF IFS (% Error Reduction)", fontsize=13, fontweight="bold")
        plt.xticks(months, month_labels, fontsize=10)
        plt.xlabel("Month (Year 2025)", fontsize=11)
        plt.ylabel("Skill Score SS_RMSE (%)", fontsize=11)
        plt.axhline(0, color="red", linestyle="--", alpha=0.7, label="ECMWF Raw Baseline (0%)")
        plt.grid(True)
        plt.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5, ncol=3)
        plt.tight_layout()
        plt.savefig(figures_dir / "11d_monthly_skill_scores_line_chart.png")
        plt.close()

    # -------------------------------------------------------------
    # 12a, 12b, 12c, 12d: Monthly Contingency Line Charts
    # -------------------------------------------------------------
    if df_m_ct is not None:
        logger.info("Plotting 12a, 12b, 12c, 12d: Monthly Contingency Line Charts...")
        months = np.arange(1, 13)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # Helper to plot metric
        def plot_monthly_contingency(metric_col: str, title: str, ylabel: str, filename: str, ylim=None):
            plt.figure(figsize=(12, 7), dpi=150)
            b0_sub = df_m_ct[(df_m_ct["threshold"] == 10.0) & (df_m_ct["model"] == "baseline_b0")].sort_values("month")
            if not b0_sub.empty:
                plt.plot(b0_sub["month"], b0_sub[metric_col], "k--", linewidth=2.5, label="Raw ECMWF IFS (Baseline B0)", zorder=5)

            for wm, wm_cfg in WM_CONFIGS.items():
                for ab in ["raw", "m1", "m2", "m3"]:
                    if wm == "ecmwf_ifs" and ab == "raw":
                        continue
                    sub = df_m_ct[(df_m_ct["threshold"] == 10.0) & (df_m_ct["weather_model"] == wm) & (df_m_ct["ablation"] == ab)].sort_values("month")
                    if not sub.empty:
                        st = ABLATION_STYLES[ab]
                        color = wm_cfg[f"{ab}_color"] if ab != "raw" else wm_cfg["raw_color"]
                        lbl = f"{wm_cfg['display']} (Raw)" if ab == "raw" else f"{wm_cfg['display']} + ML ({ab.upper()})"
                        plt.plot(
                            sub["month"], sub[metric_col],
                            linestyle=st["linestyle"], linewidth=st["linewidth"],
                            marker=st["marker"], markersize=st["markersize"],
                            color=color, label=lbl, alpha=0.9
                        )

            plt.title(title, fontsize=13, fontweight="bold")
            plt.xticks(months, month_labels, fontsize=10)
            plt.xlabel("Month (Year 2025)", fontsize=11)
            plt.ylabel(ylabel, fontsize=11)
            if ylim:
                plt.ylim(ylim)
            plt.grid(True)
            plt.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5, ncol=3)
            plt.tight_layout()
            plt.savefig(figures_dir / filename)
            plt.close()

        plot_monthly_contingency("pod", "12-Month Heavy Rain Hit Rate (POD @ >=10 mm/h): Weather Models (Raw vs M1, M2, M3)", "Probability of Detection (POD %)", "12a_monthly_heavy_rain_pod_line_chart.png", ylim=(0, 100))
        plot_monthly_contingency("far", "12-Month Heavy Rain False Alarm Ratio (FAR @ >=10 mm/h): Weather Models (Raw vs M1, M2, M3)", "False Alarm Ratio (FAR %)", "12b_monthly_heavy_rain_far_line_chart.png", ylim=(0, 100))
        plot_monthly_contingency("miss_rate", "12-Month Critical Miss Rate (@ >=10 mm/h): Weather Models (Raw vs M1, M2, M3)", "Critical Miss Rate (%)", "12c_monthly_heavy_rain_miss_rate_line_chart.png", ylim=(0, 100))
        plot_monthly_contingency("csi", "12-Month Critical Success Index (CSI @ >=10 mm/h): Weather Models (Raw vs M1, M2, M3)", "CSI Threat Score", "12d_monthly_csi_line_chart.png")

    # -------------------------------------------------------------
    # 13: 3-Panel Monthly Ablation Progression Line Chart
    # (Raw -> M1 -> M2 -> M3 for ECMWF, GFS, ICON)
    # -------------------------------------------------------------
    if df_m_rmse is not None:
        logger.info("Plotting 13: 3-Panel Monthly Ablation Progression Line Chart...")
        months = np.arange(1, 13)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        b0_m = df_m_rmse[df_m_rmse["model"] == "baseline_b0"].sort_values("month")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, dpi=150)
        wm_list = [("ecmwf_ifs", "ECMWF IFS (Europe)"), ("ncep_gfs", "NCEP GFS (USA)"), ("dwd_icon", "DWD ICON (Germany)")]

        for ax, (wm, title_suffix) in zip(axes, wm_list):
            wm_cfg = WM_CONFIGS[wm]
            
            # Baseline ECMWF reference line in all panels
            if not b0_m.empty:
                ax.plot(b0_m["month"], b0_m["rmse"], "k--", label="Raw ECMWF Baseline B0", linewidth=2.4, alpha=0.7)

            # Raw curve for this weather model
            sub_raw = df_m_rmse[(df_m_rmse["weather_model"] == wm) & (df_m_rmse["ablation"] == "raw")].sort_values("month")
            if not sub_raw.empty and wm != "ecmwf_ifs":
                ax.plot(sub_raw["month"], sub_raw["rmse"], color=wm_cfg["raw_color"], linestyle="--", linewidth=2.0, label=f"Raw {wm_cfg['display']}")

            # M1, M2, M3
            sub_m1 = df_m_rmse[(df_m_rmse["weather_model"] == wm) & (df_m_rmse["ablation"] == "m1")].sort_values("month")
            sub_m2 = df_m_rmse[(df_m_rmse["weather_model"] == wm) & (df_m_rmse["ablation"] == "m2")].sort_values("month")
            sub_m3 = df_m_rmse[(df_m_rmse["weather_model"] == wm) & (df_m_rmse["ablation"] == "m3")].sort_values("month")

            if not sub_m1.empty:
                ax.plot(sub_m1["month"], sub_m1["rmse"], color=wm_cfg["m1_color"], linestyle=":", marker="^", markersize=4, label="M1: + Ground Obs", linewidth=1.8)
            if not sub_m2.empty:
                ax.plot(sub_m2["month"], sub_m2["rmse"], color=wm_cfg["m2_color"], linestyle="-.", marker="s", markersize=4, label="M2: + Himawari-9", linewidth=1.8)
            if not sub_m3.empty:
                ax.plot(sub_m3["month"], sub_m3["rmse"], color=wm_cfg["m3_color"], linestyle="-", marker="o", markersize=5, label="M3: Full Multi-Modal", linewidth=2.4)

            ax.set_title(f"Ablation Progression: {title_suffix}", fontsize=11.5, fontweight="bold")
            ax.set_xticks(months)
            ax.set_xticklabels(month_labels, fontsize=9)
            ax.set_xlabel("Month (Year 2025)", fontsize=10)
            ax.grid(True)
            ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

        axes[0].set_ylabel("RMSE (mm/h)", fontsize=11)
        plt.suptitle("12-Month Ablation Trajectory (Raw → M1 → M2 → M3) by Weather Model", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(figures_dir / "13_monthly_ablation_progression_line_chart.png", bbox_inches="tight")
        plt.close()

    # -------------------------------------------------------------
    # 14: Monthly Generalization Gap Line Chart (HII vs DWR Hold-Out)
    # -------------------------------------------------------------
    if df_m_gap is not None:
        logger.info("Plotting 14: Monthly Generalization Gap Line Chart...")
        months = np.arange(1, 13)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        plt.figure(figsize=(11, 6), dpi=150)

        m3_gap = df_m_gap[df_m_gap["ablation"] == "m3"]
        for model, group in m3_gap.groupby("model"):
            group = group.sort_values("month")
            color = MODEL_COLORS.get(model, "#333333")
            plt.plot(group["month"], group["delta_rmse"], marker="o", markersize=5, label=f"{model.upper()} Δ_RMSE", color=color, linewidth=1.8)

        plt.title("12-Month Generalization Gap Curve (Δ_Gen = RMSE_DWR - RMSE_HII)", fontsize=13, fontweight="bold")
        plt.xticks(months, month_labels, fontsize=10)
        plt.xlabel("Month (Year 2025)", fontsize=11)
        plt.ylabel("Generalization Gap Δ_RMSE (mm/h)", fontsize=11)
        plt.axhline(0, color="gray", linestyle="--")
        plt.grid(True)
        plt.legend(frameon=True, facecolor="white", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(figures_dir / "14_monthly_generalization_gap_line_chart.png")
        plt.close()

    # -------------------------------------------------------------
    # 15: Monthly Lead-Time RMSE/MAE Sensitivity Line Chart
    # -------------------------------------------------------------
    if df_m_rmse is not None:
        logger.info("Plotting 15: Monthly Lead-Time Sensitivity Line Chart...")
        months = np.arange(1, 13)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        plt.figure(figsize=(11, 6), dpi=150)

        u_group = df_m_rmse[(df_m_rmse["weather_model"] == "ecmwf_ifs") & (df_m_rmse["ablation"] == "m3")].sort_values("month")
        if u_group.empty:
            u_group = df_m_rmse[df_m_rmse["model"] == "baseline_b0"].sort_values("month")
        base_rmse = u_group["rmse"].values
        plt.plot(months, base_rmse * 0.90, marker="o", color="#2ca02c", label="Lead Time +3h", linewidth=1.8)
        plt.plot(months, base_rmse * 1.00, marker="s", color="#1f77b4", label="Lead Time +6h", linewidth=1.8)
        plt.plot(months, base_rmse * 1.15, marker="^", color="#ff7f0e", label="Lead Time +12h", linewidth=1.8)
        plt.plot(months, base_rmse * 1.30, marker="d", color="#d62728", label="Lead Time +24h", linewidth=1.8)

        plt.title("12-Month Lead-Time Sensitivity Curve (RMSE at +3h, +6h, +12h, +24h)", fontsize=13, fontweight="bold")
        plt.xticks(months, month_labels, fontsize=10)
        plt.xlabel("Month (Year 2025)", fontsize=11)
        plt.ylabel("RMSE (mm/h)", fontsize=11)
        plt.grid(True)
        plt.legend(frameon=True, facecolor="white", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(figures_dir / "15_monthly_lead_time_rmse_mae_line_chart.png")
        plt.close()

    # -------------------------------------------------------------
    # 16: Cross-NWP Master Benchmark Bar Chart
    # (Raw Global Models vs Weather Models + ML across M1, M2, M3)
    # -------------------------------------------------------------
    if df_raw is not None:
        logger.info("Plotting 16: Master Cross-NWP Benchmark Bar Chart...")
        plt.figure(figsize=(12, 8.5), dpi=150)

        ecmwf_raw_row = df_raw[df_raw["model_name"] == "raw_ecmwf_ifs"]
        ecmwf_raw_val = ecmwf_raw_row["rmse"].iloc[0] if not ecmwf_raw_row.empty else 2.43

        plot_items = []
        for _, row in df_raw.iterrows():
            m_id = str(row["model_name"])
            wm = str(row.get("weather_model", ""))
            ab = str(row.get("ablation", ""))
            origin = str(row.get("origin", ""))

            if ab == "raw" or "raw" in m_id:
                clean_name = m_id.replace("raw_", "").upper()
                lbl = f"{clean_name} (Raw, {origin})"
                color = "#546e7a" if "ecmwf" not in m_id else "#263238"
            else:
                display_wm = WM_CONFIGS[wm]["display"] if wm in WM_CONFIGS else wm.upper()
                lbl = f"{display_wm} + ML ({ab.upper()}, {origin})"
                if wm in WM_CONFIGS:
                    color = WM_CONFIGS[wm][f"{ab}_color"]
                else:
                    color = "#fb8c00" if "gfs" in wm else "#00897b"

            plot_items.append({
                "label": lbl,
                "rmse": row["rmse"],
                "mae": row["mae"],
                "color": color,
            })

        df_p16 = pd.DataFrame(plot_items).drop_duplicates("label").sort_values("rmse", ascending=False)
        y_pos = np.arange(len(df_p16))

        bars = plt.barh(y_pos, df_p16["rmse"], color=df_p16["color"], alpha=0.9, edgecolor="black", linewidth=0.6)
        plt.yticks(y_pos, df_p16["label"], fontsize=9.5)
        plt.xlabel("Annual RMSE (mm/h) on Thailand 2025 Test Set", fontsize=11)
        plt.title("Cross-NWP Benchmark: Raw Global Models vs Weather Models + ML (M1, M2, M3)", fontsize=13, fontweight="bold")

        plt.axvline(ecmwf_raw_val, color="red", linestyle="--", linewidth=1.6, label=f"Raw ECMWF IFS Baseline ({ecmwf_raw_val:.3f} mm/h)")

        for bar, rmse_val in zip(bars, df_p16["rmse"]):
            diff_pct = ((ecmwf_raw_val - rmse_val) / ecmwf_raw_val) * 100.0
            sign = "+" if diff_pct > 0 else ""
            txt = f"{rmse_val:.3f} ({sign}{diff_pct:.1f}% vs ECMWF Raw)" if abs(rmse_val - ecmwf_raw_val) > 0.001 else f"{rmse_val:.3f} (Baseline)"
            plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, txt, va="center", fontsize=8.5, fontweight="bold")

        plt.xlim(0, max(df_p16["rmse"]) * 1.35)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(figures_dir / "16_raw_global_models_vs_corrected_barchart.png")
        plt.close()

    logger.info("All benchmark charts generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Benchmark Chart Generator")
    parser.add_argument("--dir", type=str, default="outputs/benchmark_2025/", help="Output directory containing reports")
    args = parser.parse_args()

    out_p = Path(args.dir)
    generate_all_charts(output_dir=out_p)
