import torch
from pathlib import Path
import yaml
import os
from utilities.load_data import load_petface_data, plot_breed_distribution
from pathlib import Path

from extraction.activations import extract_and_save_activations
from analysis.cka import run_cka
from analysis.umap import run_umap
from analysis.saliency import run_saliency
from extraction.postprocess import cnn_postprocess, vit_postprocess

from utilities.output_manager import OutputManager
from training.checkpoints import load_model_checkpoint
from architectures.factory import build_model


def main():

    # 0. Global setup
    Current_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(Current_dir, "config", "cnn.yaml")) as f:
        cnn_config = yaml.safe_load(f)

    with open(os.path.join(Current_dir, "config", "vit_train.yaml")) as f:
        vit_config = yaml.safe_load(f)

    torch.manual_seed(42)  ## TODO config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO config
    print("Using {} device".format(device))

    experiment_cfg = {
        "experiment": "cnn_vs_vit_petface",
        "dataset": "petface",
        "num_samples": 5,
        "cnn_layers": ["pool3", "pool4", "gap"],
        "vit_layers": ["block0", "block1", "block2"],
        "seed": 42,
    }

    out = OutputManager(experiment_name="cnn_vs_vit_petface", config=experiment_cfg)

    # 1. Load dataset (fixed subset)

    # TODO one train config - so same hyperparameters for models are used
    # TODO how do you want you did you use dataset
    train_ldr, val_ldr, test_ldr, n_breeds = load_petface_data(
        batch_size=cnn_config["training"]["batch_size"],
        label_path="./data/cat/cat.csv",
        strip_percent=cnn_config["data"]["strip_percent"],
        visualize_breed_distribution=False,
    )

    images, labels, _ = next(iter(test_ldr))
    images = images[: experiment_cfg["num_samples"]].to(device)
    labels = labels[: experiment_cfg["num_samples"]]

    # 2. Load models

    cnn_run_dir = cnn_config["analysis"]["cnn_run"]
    vit_run_dir = vit_config["analysis"]["vit_run"]

    cnn_ckpt = os.path.join(cnn_run_dir, "checkpoints", "best.pt")
    vit_ckpt = os.path.join(vit_run_dir, "checkpoints", "best.pt")

    # TODO enforce same breed number overall, run main, write run_saliency
    #infer_num_classes_from_checkpoint(ckpt_path=cnn_ckpt)
    #nfer_num_classes_from_checkpoint(ckpt_path=vit_ckpt)

    cnn = build_model(cnn_config, num_classes=n_breeds).to(device)
    cnn, cnn_meta = load_model_checkpoint(
        model=cnn,
        checkpoint_path=cnn_ckpt,
        device=device,
    )
    print("Loaded CNN from epoch:", cnn_meta.get("epoch"))

    vit = build_model(vit_config, num_classes=n_breeds).to(device)
    vit, vit_meta = load_model_checkpoint(
        model=vit,
        checkpoint_path=vit_ckpt,
        device=device,
    )
    print("Loaded ViT from epoch:", vit_meta.get("epoch"))

    # 3. Define layers

    cnn_layers = {
        "pool3": cnn.pool3,
        "pool4": cnn.pool4,
        "gap": cnn.global_avg_pool,
    }

    vit_layers = {
        "block2": vit.encoder_blocks[2],
        "block0": vit.encoder_blocks[0],
        "block1": vit.encoder_blocks[1],
    }

    # 4. Activation extraction (ONCE)

    cnn_act_dir = Path(out.dirs["activations"]) / "cnn"
    vit_act_dir = Path(out.dirs["activations"]) / "vit"

    if not cnn_act_dir.exists():
        extract_and_save_activations(
            model=cnn,
            layers=cnn_layers,
            images=images,
            save_dir=cnn_act_dir,
            postprocess_fn=cnn_postprocess,
            model_type="cnn",
        )

    if not vit_act_dir.exists():
        extract_and_save_activations(
            model=vit,
            layers=vit_layers,
            images=images,
            save_dir=vit_act_dir,
            postprocess_fn=vit_postprocess,
            model_type="vit",
        )

    # 5. CKA

    cka_df, cka_fig = run_cka(cnn_act_dir=cnn_act_dir, vit_act_dir=vit_act_dir)

    out.save_dataframe(cka_df, "cka.csv", subdir="analysis")
    out.save_figure(cka_fig, "cka_heatmap.png", subdir="analysis")

    # 6. UMAP    
    umap_figs = run_umap(
        act_dir=Path(out.dirs["activations"]),
        labels=labels
    )

    for name, fig in umap_figs.items():
        out.save_figure(fig, f"umap_{name}.png", subdir="analysis")

    # 7. Saliency (NO caching)

    cnn_saliency_figs = run_saliency(
        model=cnn,
        images=images[:4],
        device=device
    )

    for i, fig in enumerate(cnn_saliency_figs):
        out.save_figure(fig, f"cnn_saliency_{i}.png", subdir="saliency")

    vit_saliency_figs = run_saliency(
        model=vit,
        images=images[:4],
        device=device
    )

    for i, fig in enumerate(vit_saliency_figs):
        out.save_figure(fig, f"vit_saliency_{i}.png", subdir="saliency")
    
    print("All experiments completed successfully.")

if __name__ == "__main__":
    main()
