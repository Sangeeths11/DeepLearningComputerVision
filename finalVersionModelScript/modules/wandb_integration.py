import wandb

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
    
if __name__ == "__main__":
    optimizer = SweepOptimizer("VisionTransformer", "CNN-SWEEP", "test_acc")
    best_params = optimizer.get_best_parameters()
    print("Beste Hyperparameter:", best_params)

