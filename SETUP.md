# Real Data Setup — RLHF Embedding Extraction

### 1. Request model access on Hugging Face

Both models are gated. Log into [huggingface.co](https://huggingface.co) and request access on each model page:

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

### 3. Add 4-bit quantization support (RTX 2080 Super — 8 GB VRAM)

Llama-3-8B in float16 requires ~16 GB VRAM. 4-bit quantization brings it to ~5 GB.

Install `bitsandbytes`:

```bash
.venv/bin/pip install bitsandbytes>=0.43.0
```

Edit `scripts/extract_embeddings.py` — replace the model loading block:

```python
# before
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    token=token,
)

# after
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=token,
)
```

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

### 5. Launch the dashboard

```bash
python app/app.py
```

First run fits UMAP, t-SNE, and LinearSVC for all 66 layer × model combinations. This takes **~15–30 minutes** on CPU and is cached to `data/cache/` — subsequent launches start in seconds.

Open [http://127.0.0.1:8050](http://127.0.0.1:8050).
