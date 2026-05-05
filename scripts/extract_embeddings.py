import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.data import sample_hh_rlhf
from scripts.io import save_embeddings


def extract(model_id: str, n_rows: int, batch_size: int) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Error: HF_TOKEN environment variable not set. Export it before running.")

    print(f"Loading {n_rows} rows from Anthropic/hh-rlhf...")
    texts, labels = sample_hh_rlhf(n_rows)

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers + 1  # embedding layer + transformer layers
    all_layers: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_num, i in enumerate(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_idx, hidden in enumerate(outputs.hidden_states):
            # hidden: [batch, seq_len, hidden_dim] — take last token
            all_layers[layer_idx].append(hidden[:, -1, :].cpu().float().numpy())

        if (batch_num + 1) % 10 == 0:
            print(f"  Batch {batch_num + 1}/{total_batches}")

    layers = {i: np.concatenate(arrs, axis=0) for i, arrs in all_layers.items()}

    slug = model_id.replace("/", "--")
    save_path = Path("data/embeddings") / slug / "layers.h5"
    save_embeddings(save_path, layers, labels)
    print(f"Saved {n_layers} layers × {len(texts)} samples → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract layer-wise embeddings from an LLM.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--n-rows", type=int, default=500, help="Number of hh-rlhf rows to sample")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    args = parser.parse_args()
    extract(args.model, args.n_rows, args.batch_size)
