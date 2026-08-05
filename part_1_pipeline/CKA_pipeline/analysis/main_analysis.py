import torch 
from models.load_model import load_model
import yaml 
import random 
from pathlib import Path 
from probing.get_activations import (
    extract_activations_to_h5,
    postprocess_language,
    postprocess_vision,
)
from data.dataloader import load_data
### MODELLE LADEN
MODEL_KEYS = {"model_id", "m_type", "probe_layer", "hidden_dim", "lora", "token"}

MODEL_DIRS =["/work2/lt83cico-mathAi/lt83cico-mathAi-1783300802/pipeline_sara/pipeline_sara/training/models/alignment_phase2/llama_3B_coco_multilabel_cka_phase_2/20260715_040425"
,]
SEED = 42        # gleiches Zufallsbatch reproduzierbar; None für rein zufällig

# Wohin die .h5 fürs Dashboard geschrieben werden. Struktur entspricht dem, was
# scripts/io.py im Dashboard-Repo liest: vision/<slug>.h5 bzw. embeddings/<slug>/layers.h5.
DASHBOARD_ROOT = Path("dashboard_data")
# Ganzer Test-Loader (None) oder begrenzen, z. B. 20 Batches à 64 = 1280 Samples.
MAX_BATCHES = None

def load_split_config(config_path):
    cfg = yaml.safe_load(open(config_path))
    if "run_cfg" in cfg and "model_cfg" in cfg:        # bereits getrenntes Format
        return cfg["run_cfg"], cfg["model_cfg"]
    # gemergtes Format -> nach Schlüsseln aufteilen
    model_cfg = {k: cfg[k] for k in MODEL_KEYS if k in cfg}
    run_cfg   = {k: v for k, v in cfg.items() if k not in MODEL_KEYS}
    return run_cfg, model_cfg
    
def process_model(exp_dir):
    exp_dir    = Path(exp_dir)
    config_p   = exp_dir / "config.yaml"
    ckpt_p     = exp_dir / "checkpoints" / "best_model.pt"

    print(f"\n=== {exp_dir} ===")

    # 1) Config laden + trennen
    run_cfg, model_cfg = load_split_config(config_p)
    m_type = "language"

    # 2) Device
    requested = run_cfg["running_params"]["device"]
    if isinstance(requested, str) and requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA angefragt, aber nicht verfügbar -> CPU.")
        requested = "cpu"
    device = torch.device(requested)

    # 3) Modell + Processor bauen
    model, processor = load_model(run_cfg=run_cfg, model_cfg=model_cfg,
                                  n_classes=run_cfg["data"]["n_classes"])

    # 4) Trainierte Gewichte laden
    ckpt = torch.load(ckpt_p, map_location="cpu")
    model.load_state_dict(ckpt["model"])               # ggf. strict=False
    print(f"Checkpoint geladen (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')})")

    # 5) Auf device + passendes Postprocessing
    if m_type == "vision":
        model = model.to(device)
        postprocess_fn = postprocess_vision
    else:
        model.classifier.to(device)
        postprocess_fn = postprocess_language

    # 6) Test-Loader (gleicher processor -> gleiches collate/padding)
    _, test_loader = load_data(run_cfg=run_cfg, processor=processor)

    # 7) Zielpfad im Dashboard-Format ableiten.
    #    exp_dir = .../<slug>/<timestamp>  ->  slug = exp_dir.parent.name
    slug = exp_dir.parent.name
    if m_type == "vision":
        out_path = DASHBOARD_ROOT / "vision" / f"{slug}.h5"
    else:
        out_path = DASHBOARD_ROOT / "embeddings" / slug / "layers.h5"

    # 8) Aktivierungen ziehen und als .h5 speichern
    extract_activations_to_h5(
        model=model, out_path=out_path, postprocess_fn=postprocess_fn,
        model_type=m_type, test_loader=test_loader, device=device,
        max_batches=MAX_BATCHES,
    )

    # Speicher freigeben, bevor das nächste Modell kommt
    del model, processor, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        
        
def main():
    if SEED is not None:
        random.seed(SEED)
        torch.manual_seed(SEED)

    for exp_dir in MODEL_DIRS:
        process_model(exp_dir)


if __name__ == "__main__":
    main()

