import torch 
import numpy as np
import os 
from architectures.vit import Transformer
from architectures.cnn import CNNModel
import yaml 
from pathlib import Path 
from utilities.load_data import load_petface_data
import torch as t 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 


def plot_cka_heatmap(df: pd.DataFrame) -> plt.Figure:
    pivot = df.pivot(index="cnn_layer", columns="vit_layer", values="cka")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, cmap="viridis", ax=ax)
    ax.set_title("CKA: CNN vs ViT")
    return fig


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    #No flattening. No hooks. No batch accumulation.
    """
    X, Y: (B, D)
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Sample count mismatch")

    B = X.shape[0]
    print(f"CKA computed over {B} samples")

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.T
    L = Y @ Y.T

    hsic = torch.trace(K @ L)
    norm = torch.norm(K, p="fro") * torch.norm(L, p="fro")

    return (hsic / norm).item()
     

def run_cka(cnn_act_dir: Path, vit_act_dir: Path):
    cnn_files = sorted(cnn_act_dir.glob("*.pt"))
    vit_files = sorted(vit_act_dir.glob("*.pt"))

    results = []

    for cnn_file in cnn_files:
        cnn_layer = cnn_file.stem
        X = torch.load(cnn_file)  # (B, D)

        for vit_file in vit_files:
            vit_layer = vit_file.stem
            Y = torch.load(vit_file)  # (B, D)

            if X.shape[0] != Y.shape[0]:
                raise ValueError("Batch size mismatch for CKA")

            score = linear_cka(X, Y)

            results.append({
                "cnn_layer": cnn_layer,
                "vit_layer": vit_layer,
                "cka": score
            })

    df = pd.DataFrame(results)
    fig = plot_cka_heatmap(df)

    return df, fig