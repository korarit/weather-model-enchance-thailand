"""
Full End-to-End Pipeline Orchestrator for Thailand 2 km Precipitation Bias Correction
Executes all 6 stages with real-time step and progress logging:
  [STAGE 1/6] Ingesting HII Station Metadata
  [STAGE 2/6] Acquiring Ground Telemetry (Clean Parquet)
  [STAGE 3/6] Extracting NWP Global Forecasts & Himawari-9 Satellite Data
  [STAGE 4/6] Assembling Multi-Modal 2 km Feature Matrix
  [STAGE 5/6] Training Precipitation Bias Correction Model(s)
  [STAGE 6/6] Executing Hydrological Evaluation Benchmark Suite
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import (
    resolve_hii_paths,
    resolve_forecast_paths,
    DEFAULT_GEO_DIR,
    DEFAULT_FEATURES_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PipelineOrchestrator")


def print_stage_banner(stage_num: int, total_stages: int, stage_name: str):
    logger.info("=" * 80)
    logger.info(">>> [STAGE %d/%d] %s", stage_num, total_stages, stage_name.upper())
    logger.info("=" * 80)


def run_full_pipeline(
    hii_dir: str = None,
    forecast_dir: str = None,
    output_dir: str = "outputs/pipeline_run",
    model_name: str = "catboost",
    weather_model: str = "ecmwf_ifs",
    ablation: str = "m1",
    downloader: str = "tigge",
    smoke_test: bool = True,
    skip_download: bool = False,
):
    total_stages = 6
    out_base = PROJECT_ROOT / output_dir
    out_base.mkdir(parents=True, exist_ok=True)

    hii_paths = resolve_hii_paths(hii_dir=hii_dir)
    forecast_paths = resolve_forecast_paths(forecast_dir=forecast_dir)

    logger.info("Pipeline Storage Layout:")
    logger.info("  * HII Base Dir       : %s", hii_paths["hii_dir"])
    logger.info("  * HII Clean Parquet  : %s", hii_paths["clean_dir"])
    logger.info("  * HII Metadata       : %s", hii_paths["meta_dir"])
    logger.info("  * Forecast Base Dir  : %s", forecast_paths["forecast_dir"])
    logger.info("  * NWP Raw Runs       : %s", forecast_paths["raw_nwp_dir"])
    logger.info("  * NWP Forecasts      : %s", forecast_paths["out_nwp_dir"])
    logger.info("  * Himawari Satellite : %s", forecast_paths["himawari_dir"])
    logger.info("  * Downloader Backend : %s", downloader.upper())
    logger.info("  * Pipeline Output    : %s", out_base)

    # -------------------------------------------------------------------------
    # STAGE 1: HII Metadata Acquisition
    # -------------------------------------------------------------------------
    print_stage_banner(1, total_stages, "Ingesting HII Station Metadata & Cross-Mapping")
    from src.data.hii_metadata import build_master_metadata
    meta_summary = build_master_metadata(output_dir=hii_paths["meta_dir"])
    logger.info("Metadata ready: %d total stations mapped.", meta_summary.get("total_stations", 0))

    # -------------------------------------------------------------------------
    # STAGE 2: HII Clean Parquet Downloader
    # -------------------------------------------------------------------------
    print_stage_banner(2, total_stages, "Acquiring Ground Telemetry (HII Clean Parquet)")
    if skip_download:
        logger.info("Skip-download active: Bypassing HII download stage.")
    else:
        from src.data.hii_downloader import download_clean_data, get_candidate_stations
        sample_count = 5 if smoke_test else 50
        stns = get_candidate_stations(sample_count)
        years = [2021, 2025] if smoke_test else [2021, 2022, 2023, 2024, 2025]
        download_clean_data(
            stations=stns,
            years=years,
            catalogs=["hourly_rain", "pressure", "humidity"],
            output_dir=hii_paths["clean_dir"],
            max_workers=4
        )

    # -------------------------------------------------------------------------
    # STAGE 3: NWP Forecasts & Himawari-9 Satellite Data
    # -------------------------------------------------------------------------
    print_stage_banner(3, total_stages, f"Extracting NWP Global Forecasts ({downloader.upper()}) & Himawari-9 Satellite")
    if skip_download:
        logger.info("Skip-download active: Bypassing NWP/Satellite acquisition.")
    else:
        if downloader == "hybrid":
            from src.data.hybrid_downloader import run_hybrid_batch
            if weather_model == "all":
                target_models = ["ecmwf_ifs", "gfs", "jma_gsm"]
            elif weather_model in ("jma_gsm", "jma", "rjtd"):
                target_models = ["jma_gsm"]
            elif weather_model in ("ecmwf_ifs", "ecmf", "ecmwf"):
                target_models = ["ecmwf_ifs"]
            else:
                target_models = ["gfs"]
            run_hybrid_batch(
                models=target_models,
                start_date="2021-01-01",
                end_date="2021-01-02" if smoke_test else "2021-01-05",
                output_dir=forecast_paths["out_nwp_dir"],
                meta_dir=hii_paths["meta_dir"],
            )
        else:
            # Standard TIGGE flow
            from src.data.tigge_downloader import run_batch_acquisition
            origins = [weather_model] if weather_model != "all" else ["ecmf", "kwbc"]
            run_batch_acquisition(
                dates=["2021-01-01", "2021-01-02"] if smoke_test else ["2021-01-01", "2021-01-05"],
                origins=origins,
                output_dir=forecast_paths["raw_nwp_dir"]
            )

            # Extract NWP to analysis-ready
            from src.data.tigge_extractor import run_extraction_pipeline
            run_extraction_pipeline(
                raw_dir=forecast_paths["raw_nwp_dir"],
                out_dir=forecast_paths["out_nwp_dir"]
            )

        # Extract satellite
        from src.data.himawari_extractor import run_himawari_acquisition
        sat_end = pd.Timestamp("2021-01-02 23:00:00") if smoke_test else pd.Timestamp("2021-01-05 23:00:00")
        run_himawari_acquisition(
            start_dt=pd.Timestamp("2021-01-01 00:00:00"),
            end_dt=sat_end,
            step_hours=12 if smoke_test else 6,
            out_dir=forecast_paths["himawari_dir"]
        )

    # -------------------------------------------------------------------------
    # STAGE 4: Multi-Modal Feature Assembly (2 km Grid)
    # -------------------------------------------------------------------------
    print_stage_banner(4, total_stages, "Assembling Multi-Modal 2 km Feature Matrix")
    from src.features.feature_builder import run_sample_builder
    features_out = out_base / "features"
    feature_parquet = run_sample_builder(
        output_dir=features_out,
        geo_dir=DEFAULT_GEO_DIR,
        hii_meta_dir=hii_paths["meta_dir"],
        raw_nwp_dir=forecast_paths["raw_nwp_dir"],
        out_nwp_dir=forecast_paths["out_nwp_dir"],
        himawari_dir=forecast_paths["himawari_dir"],
        downloader=downloader,
    )
    logger.info("Feature matrix ready at: %s", feature_parquet)

    # -------------------------------------------------------------------------
    # STAGE 5: Bias Correction Model Training
    # -------------------------------------------------------------------------
    print_stage_banner(5, total_stages, f"Training Bias Correction Model ({model_name.upper()})")
    model_output_dir = out_base / "models" / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df_features = pd.read_parquet(feature_parquet)

    target_models = ["ecmwf_ifs"] if weather_model == "all" else [weather_model]
    target_ablations = ["m1"] if ablation == "all" else [ablation]

    if model_name == "catboost":
        from src.models.runners.run_catboost import train_and_eval_catboost
        for wm in target_models:
            for ab in target_ablations:
                train_and_eval_catboost(df_features, ab, output_dir=model_output_dir, weather_model=wm, smoke_test=smoke_test)
    elif model_name == "lightgbm":
        from src.models.runners.run_lightgbm import train_and_eval_lgbm
        for wm in target_models:
            for ab in target_ablations:
                train_and_eval_lgbm(df_features, ab, output_dir=model_output_dir, weather_model=wm, smoke_test=smoke_test)
    elif model_name == "hurdle":
        from src.models.runners.run_hurdle import train_and_eval_hurdle
        for wm in target_models:
            for ab in target_ablations:
                train_and_eval_hurdle(df_features, ab, output_dir=model_output_dir, weather_model=wm, smoke_test=smoke_test)
    elif model_name == "quantile":
        from src.models.runners.run_quantile import train_and_eval_quantile
        for wm in target_models:
            for ab in target_ablations:
                train_and_eval_quantile(df_features, ab, output_dir=model_output_dir, weather_model=wm, smoke_test=smoke_test)
    elif model_name == "stgnn":
        from src.models.runners.run_stgnn import train_and_eval_stgnn
        for wm in target_models:
            for ab in target_ablations:
                train_and_eval_stgnn(df_features, ab, output_dir=model_output_dir, weather_model=wm, smoke_test=smoke_test)
    elif model_name == "unet":
        from src.models.runners.run_unet import train_and_eval_unet
        for wm in target_models:
            for ab in target_ablations:
                train_and_eval_unet(df_features, ab, output_dir=model_output_dir, weather_model=wm, smoke_test=smoke_test)
    elif model_name == "linknet":
        from src.models.runners.run_linknet import train_and_eval_linknet
        for wm in target_models:
            for ab in target_ablations:
                train_and_eval_linknet(df_features, ab, output_dir=model_output_dir, weather_model=wm, smoke_test=smoke_test)
    elif model_name == "fpn":
        from src.models.runners.run_fpn import train_and_eval_fpn
        for wm in target_models:
            for ab in target_ablations:
                train_and_eval_fpn(df_features, ab, output_dir=model_output_dir, weather_model=wm, smoke_test=smoke_test)

    # -------------------------------------------------------------------------
    # STAGE 6: Hydrological Evaluation Benchmark Suite
    # -------------------------------------------------------------------------
    print_stage_banner(6, total_stages, "Executing Hydrological Evaluation Benchmark Suite")
    from src.evaluation.run_evaluation import run_evaluation_suite
    eval_out = out_base / "benchmark"
    eval_metrics = run_evaluation_suite(output_dir=eval_out, smoke_test=smoke_test)
    logger.info("Evaluation complete! 2025 Hold-out results:")
    for k, v in eval_metrics.items():
        logger.info("  * %s: %s", k, v)

    logger.info("=" * 80)
    logger.info("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY! Output at: %s", out_base)
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Full End-to-End Bias Correction Pipeline")
    parser.add_argument("--hii-dir", type=str, default=None, help="Base directory for HII data (e.g. D:/data/hii)")
    parser.add_argument("--forecast-dir", "--nwp-dir", dest="forecast_dir", type=str, default=None, help="Base directory for NWP forecast data (e.g. E:/data/weather_nwp)")
    parser.add_argument("--output-dir", type=str, default="outputs/pipeline_run", help="Output directory")
    parser.add_argument("--model", choices=["catboost", "lightgbm", "hurdle", "quantile", "stgnn", "unet", "linknet", "fpn"], default="catboost", help="ML Architecture")
    parser.add_argument("--weather-model", choices=["ecmwf_ifs", "ncep_gfs", "jma_gsm", "dwd_icon", "cmc_gem", "bom_access", "all"], default="ecmwf_ifs", help="Target Weather Model")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default="m1", help="Feature ablation variant")
    parser.add_argument("--downloader", choices=["tigge", "hybrid"], default="tigge", help="NWP downloader backend ('tigge' or 'hybrid')")
    parser.add_argument("--smoke-test", action="store_true", default=True, help="Fast smoke test mode")
    parser.add_argument("--full", action="store_true", help="Production full mode (disables smoke-test)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download stages if files already exist")
    args = parser.parse_args()

    smoke = not args.full if args.full else args.smoke_test

    run_full_pipeline(
        hii_dir=args.hii_dir,
        forecast_dir=args.forecast_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        weather_model=args.weather_model,
        ablation=args.ablation,
        downloader=args.downloader,
        smoke_test=smoke,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
