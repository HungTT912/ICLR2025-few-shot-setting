import yaml
import os

# Base configuration template
base_config = {
    "base_GP": {
        "delta_lengthscale": 2.5,
        "delta_variance": 4.5,
        "initial_lengthscale": 7.5,
        "initial_outputscale": 5.5,
        "noise": 0.01
    },
    "GP": {
        "delta_lengthscale": 0.25,
        "delta_variance": 0.25,
        "initial_lengthscale": None,  # Placeholder for varying values
        "initial_outputscale": None,  # Placeholder for matching initial_lengthscale
        "noise": 0.01,
        "num_fit_samples": 10004,
        "num_functions": 8,
        "num_gradient_steps": 100,
        "num_points": 1024,
        "sampling_from_GP_lr": 0.001,
        "threshold_diff": 0.001
    },
    "model": {
        "BB": {
            "lr_scheduler": {
                "cooldown": 200,
                "factor": 0.5,
                "min_lr": 5.0e-07,
                "patience": 200,
                "threshold": 0.0001
            },
            "optimizer": {
                "beta1": 0.9,
                "lr": 0.001,
                "optimizer": "Adam",
                "weight_decay": 0.0
            },
            "params": {
                "MLPParams": {
                    "condition_key": "SpatialRescaler",
                    "hidden_size": 1024,
                    "image_size": 60
                },
                "loss_type": "l1",
                "max_var": 1.0,
                "mt_type": "linear",
                "num_timesteps": 1000,
                "objective": "grad",
                "sample_step": 200,
                "sample_type": "linear",
                "skip_sample": True
            }
        },
        "CondStageParams": {
            "in_channels": 3,
            "n_stages": 2,
            "out_channels": 3
        },
        "EMA": {
            "ema_decay": 0.995,
            "start_ema_step": 4000,
            "update_ema_interval": 8,
            "use_ema": True
        },
        "latent_before_quant_conv": False,
        "model_name": "BrownianBridge",
        "model_type": "BBDM",
        "normalize_latent": False,
        "only_load_latent_mean_std": False
    },
    "runner": "BBDMRunner",
    "task": {
        "name": "AntMorphology-Exact-v0",
        "normalize_x": True,
        "normalize_y": True
    },
    "testing": {
        "clip_denoised": False,
        "num_candidates": 128,
        "percentile_sampling": 0.2,
        "type_sampling": "random",
        "eta": 0,
        "classifier_free_guidance_weight": -1,
        "alpha": 0.95
    },
    "training": {
        "accumulate_grad_batches": 2,
        "batch_size": 64,
        "classifier_free_guidance_prob": 0.15,
        "n_epochs": 100,
        "save_interval": 20,
        "val_frac": 0.1,
        "validation_interval": 20
    },
    "data_ratio": 0.01,
    "tune": "few_shot/tune_4",
    "wandb_name": None  # Placeholder for varying names
}

# Directory to save configurations
output_dir = "./configs/few-shot-setting/tune_4"

if os.path.exists(output_dir)==False:
    os.makedirs(output_dir, exist_ok=True)

# Values for initial_lengthscale and initial_outputscale
lengthscales = [7.5]
delta_lengthscales = [2.5, 1.5,0.5]
variances  = [5.5] 
delta_variances = [4.5, 3.5, 2.5 ] 

# Generate a YAML file for each initial_lengthscale and initial_outputscale pair
for delta_lengthscale in delta_lengthscales: 
    for lengthscale in lengthscales:
        for variance in variances: 
            for delta_variance in delta_variances: 
                # Update config with current lengthscale and outputscale values
                config = base_config.copy()
                config["GP"]["initial_lengthscale"] = lengthscale
                config["GP"]["initial_outputscale"] = variance
                config["GP"]["delta_lengthscale"] = delta_lengthscale
                config["GP"]["delta_variance"] = delta_variance 
                config["wandb_name"] = f"few-shot-setting-ant-rand-l{lengthscale}-dl{delta_lengthscale}-v{variance}-dv{delta_variance}"

                # Save to a YAML file
                filename = f"Template-BBDM-ant-l{lengthscale}-dl{delta_lengthscale}-v{variance}-dv{delta_variance}.yaml"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w') as file:
                    yaml.dump(config, file)

                print(f"Generated {filepath}")
