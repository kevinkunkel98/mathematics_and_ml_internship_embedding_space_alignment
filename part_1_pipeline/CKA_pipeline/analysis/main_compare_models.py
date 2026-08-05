import torch
import torch.nn as nn
import yaml
import os
import glob
import numpy as np
from sklearn.metrics import average_precision_score

from data.dataloader import load_data
from models.load_model import load_model

# ==========================================================================
# NUR DAS HIER ANPASSEN: zwei Experiment-Ordner mit der Struktur
#   activations  analysis  checkpoints  config.yaml  metrics  plots
# In <ordner>/config.yaml steht unter model.m_type, ob es ein
# vision- oder language-Modell ist (alignment wird als language geladen).
# ==========================================================================
P1_MODEL_PATH = "/work2/lt83cico-mathAi/lt83cico-mathAi-1783300802/training_pipeline_coco/training/models/language/llama_3B_coco_multilabel_full_train_set/20260714_011103"
#P2_MODEL_PATH = "/work2/lt83cico-mathAi/lt83cico-mathAi-1783300802/pipeline_sara/pipeline_sara/training/models/vision/20260714_010757"
P2_MODEL_PATH = "/work2/lt83cico-mathAi/lt83cico-mathAi-1783300802/pipeline_sara/pipeline_sara/training/models/alignment_phase2/llama_3B_coco_multilabel_cka_phase_2/20260715_040425"

CKA_MAX_SAMPLES = 2000     # CKA ueber hoechstens so viele Beispiele (Speicher/Tempo); None = alle

# Architektur-Config je Modalitaet (liefert model_id, hidden_dim, lora, ...)
ARCH_CONFIG = {
    "vision":   "configs/vision.yaml",
    "language": "configs/language.yaml",
}


class CKALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Sample count mismatch für CKA Loss")
        X = X.to(torch.float32)
        Y = Y.to(torch.float32)
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        K = X @ X.T
        L = Y @ Y.T
        hsic = torch.sum(K * L)
        norm = torch.norm(K, p="fro") * torch.norm(L, p="fro")
        cka = hsic / (norm + 1e-8)
        return 1.0 - cka


# --------------------------------------------------------------------------
# Modell aus einem Experiment-Ordner laden
# --------------------------------------------------------------------------
def read_mtype(model_dir):
    """Liest model.m_type aus <model_dir>/config.yaml. 'alignment' -> 'language'."""
    with open(os.path.join(model_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    m_type = cfg["model"]["m_type"]
    if m_type == "alignment":     # der trainierte Student im Alignment ist das Sprachmodell
        m_type = "language"
    if m_type not in ARCH_CONFIG:
        raise ValueError(f"Unbekannter m_type '{m_type}' in {model_dir}/config.yaml")
    return m_type


def find_checkpoint(model_dir):
    """Sucht die Checkpoint-Datei in <model_dir>/checkpoints/."""
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    for name in ("best_model.pt", "final_language_model.pt"):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not pts:
        raise FileNotFoundError(f"Keine .pt-Datei in {ckpt_dir}")
    return pts[0]


def load_from_dir(model_dir, run_cfg, device):
    """Baut das richtige Modell (vision/language) und laedt seinen Checkpoint."""
    m_type = read_mtype(model_dir)

    with open(ARCH_CONFIG[m_type], "r") as f:
        model_cfg = yaml.safe_load(f)

    run_cfg["model"]["m_type"] = m_type                     # steuert load_model
    model, processor = load_model(run_cfg, model_cfg, run_cfg["data"]["n_classes"])

    ckpt_path = find_checkpoint(model_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[OK] {m_type}-Modell geladen aus: {model_dir}")
    print(f"     Checkpoint: {ckpt_path}")
    print(f"     missing={len(missing)}  unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("     WARNUNG: fehlende Keys -> Checkpoint evtl. unvollstaendig/inkompatibel.")

    model = model.to(device).eval()
    return model, processor, m_type


def model_inputs(batch, modality, device):
    """Zieht aus dem Batch nur die zur Modalitaet passenden Inputs."""
    keys = ["pixel_values"] if modality == "vision" else ["input_ids", "attention_mask"]
    return {k: batch[k].to(device) for k in keys if k in batch}


def pick_loader(run_cfg, modalities, vision_proc, lang_proc):
    """Waehlt den Dataloader-Modus abhaengig von den beiden Modalitaeten."""
    if modalities == {"language"}:
        run_cfg["model"]["m_type"] = "language"
        return load_data(run_cfg, processor=lang_proc)
    if modalities == {"vision"}:
        run_cfg["model"]["m_type"] = "vision"
        return load_data(run_cfg, processor=vision_proc)
    # gemischt (vision + language) -> gepaarte Batches
    run_cfg["model"]["m_type"] = "alignment"
    return load_data(run_cfg, vision_processor=vision_proc, language_processor=lang_proc)


# --------------------------------------------------------------------------
# Auswertung: Task-Metriken pro Modell + CKA zwischen beiden
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(models, test_loader, device, cka_max=None):
    """models: Liste [(model, modality, name), (model, modality, name)]."""
    n = len(models)
    losses  = [0.0] * n
    preds   = [[] for _ in range(n)]
    feats   = [[] for _ in range(n)]
    targets = []

    for batch in test_loader:
        labels = batch["labels"].to(device)
        targets.append(labels.cpu().numpy())

        for i, (model, modality, _) in enumerate(models):
            inputs = model_inputs(batch, modality, device)
            loss, logits, hidden = model(inputs, labels=labels)
            losses[i] += loss.item()
            preds[i].append(torch.sigmoid(logits).float().cpu().numpy())
            feats[i].append(hidden.float().cpu())

    nb = len(test_loader)
    targets = np.vstack(targets)

    results = []
    for i in range(n):
        p = np.vstack(preds[i])
        try:
            mAP = average_precision_score(targets, p, average="macro")
        except ValueError:
            mAP = 0.0
        results.append({"loss": losses[i] / nb, "mAP": float(mAP)})

    # CKA EINMAL ueber den ganzen (optional gesubsampleten) Datensatz.
    # Reihenfolge ist ueber beide Modelle identisch (shuffle=False) -> Paarung stimmt.
    X = torch.cat(feats[0])
    Y = torch.cat(feats[1])
    if cka_max and X.shape[0] > cka_max:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(X.shape[0], generator=g)[:cka_max]
        X, Y = X[idx], Y[idx]
    cka = float(1.0 - CKALoss()(X, Y).item())

    return results, cka


def main():
    with open("configs/run_config.yaml", "r") as f:
        run_cfg = yaml.safe_load(f)
    device = torch.device(run_cfg["running_params"]["device"] if torch.cuda.is_available() else "cpu")

    # --- beide Modelle aus ihren Ordnern laden (Typ kommt aus config.yaml) ---
    model_a, proc_a, mod_a = load_from_dir(P1_MODEL_PATH, run_cfg, device)
    model_b, proc_b, mod_b = load_from_dir(P2_MODEL_PATH, run_cfg, device)

    # Processoren nach Modalitaet einsortieren (fuer den passenden Loader)
    vision_proc = proc_a if mod_a == "vision" else (proc_b if mod_b == "vision" else None)
    lang_proc   = proc_a if mod_a == "language" else (proc_b if mod_b == "language" else None)

    # --- Testdaten passend zum Modalitaets-Paar laden ---
    modalities = {mod_a, mod_b}
    _, test_ldr = pick_loader(run_cfg, modalities, vision_proc, lang_proc)

    # --- auswerten ---
    models = [(model_a, mod_a, "Modell A"), (model_b, mod_b, "Modell B")]
    results, cka = evaluate(models, test_ldr, device, CKA_MAX_SAMPLES)

    # --- Ausgabe ---
    print(f"\n{'Metrik':<15} | {'Modell A':<24} | {'Modell B':<24}")
    print(f"{'Ordner':<15} | {os.path.basename(P1_MODEL_PATH.rstrip('/')):<24} | {os.path.basename(P2_MODEL_PATH.rstrip('/')):<24}")
    print(f"{'Modalitaet':<15} | {mod_a:<24} | {mod_b:<24}")
    print(f"{'Task Loss':<15} | {results[0]['loss']:<24.4f} | {results[1]['loss']:<24.4f}")
    print(f"{'Task mAP':<15} | {results[0]['mAP']:<24.4f} | {results[1]['mAP']:<24.4f}")
    print(f"{'CKA (A<->B)':<15} | {cka:<24.4f} |")


if __name__ == "__main__":
    main()
