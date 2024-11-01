from pathlib import Path
import wandb


def get_sweep_config(sweep_name: str) -> dict:
    return {
        "name": sweep_name,
        "method": "grid",
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"values": [0.0001, 0.001, 0.01]},
            "batch_size": {"values": [16, 32, 64]},
            "dropout": {"values": [0.2, 0.3, 0.4, 0.5]},
        },
    }


def get_sweep_run_name(learning_rate: float, batch_size: int, dropout: float) -> str:
    return f"lr_{learning_rate}_bs_{batch_size}_do_{dropout:.2f}"


def log_evaluation(loss: float, accuracy: float) -> None:
    wandb.log({"test_loss": loss, "test_acc": accuracy})


def log_image(image_name: str, image):
    wandb.log({image_name: wandb.Image(image)})


def log_model_artifact(model_name: str, model_file_path: Path):
    model_artifact = wandb.Artifact(model_name, type="model")
    model_artifact.add_file(model_file_path)
    wandb.log_artifact(model_artifact)
