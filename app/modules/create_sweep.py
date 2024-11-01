from wandb_integration import get_sweep_config

import wandb

if __name__ == "__main__":
    sweep_name = input("Name of Sweep: ")

    sweep_config = get_sweep_config(sweep_name)

    sweep_id = wandb.sweep(sweep_config, project="VisionTransformer")
