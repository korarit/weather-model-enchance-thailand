"""
Generates all 18 Model Experiment Configuration YAMLs for Phase 4
6 Models x 3 Variants (M1, M2, M3)
"""

from pathlib import Path
import yaml

CONFIG_BASE = Path(__file__).resolve().parent.parent.parent / "configs" / "models"

MODELS = {
    "lightgbm": {
        "model_type": "gbdt_leaf_wise",
        "objective": "regression_l1",
        "hyperparameters": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
        }
    },
    "catboost": {
        "model_type": "oblivious_trees",
        "objective": "RMSE",
        "hyperparameters": {
            "iterations": 600,
            "learning_rate": 0.06,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
        }
    },
    "hurdle": {
        "model_type": "two_stage_hurdle",
        "stage1_objective": "binary_logloss",
        "stage2_objective": "regression_l1",
        "hyperparameters": {
            "clf_n_estimators": 400,
            "reg_n_estimators": 400,
            "prob_threshold": 0.35,
            "rain_threshold": 0.1,
        }
    },
    "quantile": {
        "model_type": "multi_quantile_gbdt",
        "quantiles": [0.10, 0.50, 0.90, 0.95],
        "hyperparameters": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
        }
    },
    "stgnn": {
        "model_type": "spatio_temporal_gnn",
        "objective": "huber_loss",
        "hyperparameters": {
            "epochs": 50,
            "batch_size": 32,
            "hidden_dim": 32,
            "lr": 0.005,
            "weight_decay": 1e-4,
        }
    },
    "unet": {
        "model_type": "conv_unet_2d",
        "objective": "huber_loss",
        "hyperparameters": {
            "epochs": 40,
            "batch_size": 32,
            "hidden_dim": 32,
            "lr": 0.005,
            "weight_decay": 1e-4,
        }
    },
    "linknet": {
        "model_type": "spatial_linknet",
        "objective": "huber_loss",
        "hyperparameters": {
            "epochs": 40,
            "batch_size": 32,
            "hidden_dim": 32,
            "lr": 0.005,
            "weight_decay": 1e-4,
        }
    },
    "fpn": {
        "model_type": "spatial_fpn",
        "objective": "huber_loss",
        "hyperparameters": {
            "epochs": 40,
            "batch_size": 32,
            "hidden_dim": 32,
            "pyramid_channels": 24,
            "lr": 0.005,
            "weight_decay": 1e-4,
        }
    }
}

VARIANTS = {
    "m1": {"name": "M1_Ground_Only", "desc": "NWP + Topo + Ground Telemetry"},
    "m2": {"name": "M2_Satellite_Only", "desc": "NWP + Topo + Himawari-9 Satellite"},
    "m3": {"name": "M3_Joint_Ground_Satellite", "desc": "NWP + Topo + Ground + Himawari-9 Satellite"},
}

TARGET_WEATHER_MODELS = ["ecmwf_ifs", "ncep_gfs", "dwd_icon"]

def generate_configs():
    for wm in TARGET_WEATHER_MODELS:
        for model_name, m_info in MODELS.items():
            m_dir = CONFIG_BASE / model_name
            m_dir.mkdir(parents=True, exist_ok=True)
            
            for v_code, v_info in VARIANTS.items():
                cfg = {
                    "experiment_id": f"{wm.upper()}_{model_name.upper()}_{v_code.upper()}",
                    "weather_model": wm,
                    "model_name": model_name,
                    "ablation_variant": v_code,
                    "variant_description": f"[{wm.upper()}] " + v_info["desc"],
                    "data": {
                        "train_years": [2021, 2022, 2023, 2024],
                        "val_years": [2024],
                        "frozen_test_year": 2025,
                        "target_column": "target_bias",
                        "nwp_rain_column": f"nwp_rain_{wm}" if wm != "ecmwf_ifs" else "nwp_rain_raw",
                    },
                    **m_info
                }
                
                out_file = m_dir / f"{wm}_{model_name}_{v_code}.yaml"
                with open(out_file, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                
                # Default alias for ecmwf_ifs
                if wm == "ecmwf_ifs":
                    compat_file = m_dir / f"{model_name}_{v_code}.yaml"
                    with open(compat_file, "w", encoding="utf-8") as f:
                        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                print(f"Generated config: {out_file.relative_to(CONFIG_BASE)}")

if __name__ == "__main__":
    generate_configs()
