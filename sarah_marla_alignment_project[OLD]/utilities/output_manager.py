# Standard library imports
from typing import Optional
from datetime import datetime
import os
import yaml
# Third-party imports
import torch
# Data processing
import pandas as pd
from pathlib import Path

class OutputManager:

    def __init__(
        self,
        experiment_name: str,
        root_dir: str = "experiments",
        config: Optional[dict] = None
    ):
        
        parent_dir = Path(__file__).resolve().parent.parent

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        self.output_dir = os.path.join(
            parent_dir,
            root_dir,
            experiment_name,
            self.timestamp
        )

        # Standard subfolders
        self.dirs = {
            "checkpoints": os.path.join(self.output_dir, "checkpoints"),
            "plots": os.path.join(self.output_dir, "plots"),
            "metrics": os.path.join(self.output_dir, "metrics"),
            "activations": os.path.join(self.output_dir, "activations"),
            "analysis": os.path.join(self.output_dir, "analysis"),
            "saliency": os.path.join(self.output_dir, "saliency"),
        }

        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        if config is not None:
            self.save_yaml(config, "config.yaml")

        print(f"Experiment outputs: {self.output_dir}")

    def save_yaml(self, obj: dict, filename: str, subdir: str = ""):
        path = os.path.join(self.output_dir, subdir, filename)
        with open(path, "w") as f:
            yaml.safe_dump(obj, f, sort_keys=False)
        return path

    def save_tensor(self, tensor: torch.Tensor, filename: str, subdir: str):
        path = os.path.join(self.dirs[subdir], filename)
        torch.save(tensor.cpu(), path)
        return path

    def save_dataframe(self, df: pd.DataFrame, filename: str, subdir: str):
        path = os.path.join(self.dirs[subdir], filename)
        df.to_csv(path, index=False)
        return path

    def save_figure(self, fig, filename: str, subdir: str):
        path = os.path.join(self.dirs[subdir], filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        return path

    def save_text(self, text: str, filename: str, subdir: str):
        path = os.path.join(self.dirs[subdir], filename)
        with open(path, "w") as f:
            f.write(text)
        return path
    
    def get_checkpoint_dir(self):
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

