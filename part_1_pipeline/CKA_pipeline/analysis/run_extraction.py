"""
Extrahiert Layer-Aktivierungen aus dem Vision- ODER Language-Modell und
speichert sie als .h5 im Format, das das Dashboard erwartet (scripts/io.py).

Aufruf aus dem Ordner pipeline_sara/:
    python -m probing.run_extraction --modality vision
    python -m probing.run_extraction --modality language
    python -m probing.run_extraction --modality vision --max-batches 8 --out ../data/vision/dinov2.h5

--max-batches begrenzt die Anzahl Batches (None = ganzer test_loader).
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

# pipeline_sara/ auf den Pfad legen, damit `data`, `models`, `probing` importierbar sind
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataloader import load_data
from models.load_model import load_model
from probing.get_activations import (
    extract_activations_to_h5,
    postprocess_vision,
    postprocess_language,
)


def run(modality: str, out_path: str | None, max_batches: int | None):
    assert modality in ("vision", "language")

    with open("configs/run_config.yaml", "r") as f:
        run_cfg = yaml.safe_load(f)
    cfg_file = "configs/vision.yaml" if modality == "vision" else "configs/language.yaml"
    with open(cfg_file, "r") as f:
        model_cfg = yaml.safe_load(f)

    device = torch.device(
        run_cfg["running_params"]["device"] if torch.cuda.is_available() else "cpu"
    )

    # Modelltyp auf die gewünschte Modalität setzen (steuert load_model + Dataset)
    run_cfg["model"]["m_type"] = modality

    # ── Modell laden ──────────────────────────────────────────────────────────
    model, processor = load_model(run_cfg, model_cfg, run_cfg["data"]["n_classes"])

    # Optional: trainierten Checkpoint laden (nur wenn in der Config gesetzt)
    ckpt_path = model_cfg.get("path")
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[OK] Checkpoint geladen: {ckpt_path}")
        if missing:
            print(f"  -> fehlende Keys ({len(missing)})")
        if unexpected:
            print(f"  -> unerwartete Keys ({len(unexpected)})")
    else:
        print("[INFO] Kein Checkpoint-Pfad in der Config -> vortrainiertes Modell.")

    model = model.to(device)
    model.eval()

    # ── Daten laden ───────────────────────────────────────────────────────────
    _, test_ldr = load_data(run_cfg=run_cfg, processor=processor)

    # ── Extrahieren + als .h5 speichern ───────────────────────────────────────
    if modality == "vision":
        postprocess_fn = postprocess_vision
        default_out = "../data/vision/dinov2.h5"
    else:
        postprocess_fn = postprocess_language
        default_out = "../data/embeddings/llama_3b/layers.h5"

    out = Path(out_path or default_out)
    extract_activations_to_h5(
        model=model,
        out_path=out,
        postprocess_fn=postprocess_fn,
        model_type=modality,
        test_loader=test_ldr,
        device=device,
        max_batches=max_batches,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=["vision", "language"], required=True)
    p.add_argument("--out", default=None, help="Ziel-.h5-Pfad (sonst Default pro Modalität)")
    p.add_argument("--max-batches", type=int, default=None, help="Batches begrenzen (Default: alle)")
    args = p.parse_args()
    run(args.modality, args.out, args.max_batches)
