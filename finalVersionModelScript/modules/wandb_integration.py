import wandb
from pathlib import Path

class SweepOptimizer:
    def __init__(self, project_name, sweep_name, metric):
        self.project_name = project_name
        self.sweep_name = sweep_name
        self.metric = metric
        self.sweep_id = self._get_sweep_id()
        self.api = wandb.Api()

    def _get_sweep_id(self):
        SWEEPS = {
            "CNN-SWEEP": "silvan-wiedmer-fhgr/VisionTransformer/k1h9h3lh",
            "INCEPTION-SWEEP": "silvan-wiedmer-fhgr/VisionTransformer/5ghtealo",
            "VISION-TRANSFORMER-SWEEP": "silvan-wiedmer-fhgr/VisionTransformer/fpspjbrh"
        }
        return SWEEPS.get(self.sweep_name)

    def get_best_parameters(self):
        if not self.sweep_id:
            raise ValueError(f"Sweep-ID für {self.sweep_name} nicht gefunden.")
        
        sweep = self.api.sweep(self.sweep_id)
        runs = sorted(sweep.runs, key=lambda run: run.summary.get(self.metric, 0), reverse=True)
        
        if not runs:
            raise ValueError("Keine Läufe im Sweep gefunden.")
        
        best_run = runs[0]
        best_params = best_run.config
        
        print(f"Best run: {best_run.name} with {best_run.summary.get(self.metric, 0)} for metric '{self.metric}'")
        
        return best_params

def log_evaluation(loss: float, accuracy: float) -> None:
    wandb.log({"test_loss": loss, "test_acc": accuracy})


def log_image(image_name: str, image):
    wandb.log({image_name: wandb.Image(image)})

def log_model_artifact(model_name: str, model_file_path: Path):
    model_artifact = wandb.Artifact(model_name, type="model")
    model_artifact.add_file(model_file_path)
    wandb.log_artifact(model_artifact)

