# Setup Guide

---

## Part 1 — Vision Models (ResNet-18 + ViT-B/16 on CIFAR-10)

### Option A — Mock data (no GPU)

```bash
python scripts/generate_mock_vision.py
```

Generates synthetic activations and CAM heatmaps for both models in `data/vision/`. Sufficient to test the full dashboard.

---

### Option B — Real extraction (GPU required)

```bash
python scripts/extract_vision_embeddings.py --epochs 10 --n-train 10000 --n-test 1000
```

- Downloads CIFAR-10 automatically to `data/cifar10/`
- Fine-tunes pretrained ResNet-18 and ViT-B/16 for the given number of epochs
- Extracts layer-wise activations from the test set
- Computes GradCAM heatmaps for 40 sample images
- Saves to `data/vision/resnet18.h5` and `data/vision/vit_b16.h5`

**Note:** ViT-B/16 CAM is currently a placeholder (uniform maps) — GradCAM for ViT will be added once GPU cluster access is available.

Reduce `--batch-size` (default 64) if you hit OOM.

---

## Part 2 — Language Models (Llama-3-8B)

### 1. Request model access on Hugging Face

Both models are gated. Log into [huggingface.co](https://huggingface.co) and request access:

- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

Approval is usually automatic within a few minutes.

---

### 2. Get a HuggingFace token

Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token → **Read** scope is sufficient.

```bash
export HF_TOKEN=hf_...
```

---

### 3. 4-bit quantization (RTX 2080 Super — 8 GB VRAM)

Llama-3-8B in float16 requires ~16 GB VRAM. 4-bit quantization (already configured in `extract_embeddings.py`) brings it to ~5 GB. `bitsandbytes` is included in `requirements.txt`.

---

### 4. Extract embeddings

Run one model at a time (each requires ~5 GB VRAM peak):

```bash
python scripts/extract_embeddings.py \
  --model meta-llama/Meta-Llama-3-8B \
  --n-rows 500 \
  --batch-size 4

python scripts/extract_embeddings.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --n-rows 500 \
  --batch-size 4
```

> Use `--batch-size 2` if you hit OOM. Expect ~1–2 hours per model on a 2080 Super.

Verify output:

```bash
python -c "
from scripts.io import load_embeddings
l, lab = load_embeddings('data/embeddings/meta-llama--Meta-Llama-3-8B/layers.h5')
print(len(l), 'layers, shape:', l[0].shape, 'labels:', lab.shape)
"
# Expected: 33 layers, shape: (1000, 4096) labels: (1000,)
```

---

## Launch the Dashboard

```bash
python app/app.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050).

First run fits UMAP, t-SNE, LinearSVC, and CKA for all layer combinations. With mock data this takes ~5–10 minutes on CPU; with real LLM embeddings ~15–30 minutes. Results are cached to `data/cache/` — subsequent launches start in seconds.

If vision data is missing, the Part 1 tab degrades gracefully with an error message rather than crashing the app.

---

## Export Slide Figures

To regenerate the PNG figures used in the slides from the current mock (or real) data:

```bash
python3 scripts/export_slide_figures.py
```

Output goes to `assets/slides/`. Requires `kaleido` and `plotly` in the system Python (not the venv).

```bash
pip3 install kaleido plotly --break-system-packages
```
