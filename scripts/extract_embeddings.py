import argparse
import gc
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts.data import sample_hh_rlhf
from scripts.io import save_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# 3 public AllenAI checkpoints — no HF approval needed
TULU3_UNGATED = [
    ("allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai--Llama-3.1-Tulu-3-8B-SFT"),
    ("allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai--Llama-3.1-Tulu-3-8B-DPO"),
    ("allenai/Llama-3.1-Tulu-3-8B",      "allenai--Llama-3.1-Tulu-3-8B"),
]


def _build_device_map(model_id: str, token: str | None) -> dict | str:
    """Route embed_tokens/lm_head to CPU, transformer layers to GPU.

    Prevents VRAM overflow on 8 GB cards when loading 8B models in 4-bit:
    quantized blocks ~4.25 GB + fp16 embeddings ~2 GB would exceed the limit.
    """
    try:
        cfg = AutoConfig.from_pretrained(model_id, token=token)
        n_layers = cfg.num_hidden_layers
        device_map: dict = {
            "model.embed_tokens": "cpu",
            "model.norm": "cuda:0",
            "lm_head": "cpu",
        }
        for i in range(n_layers):
            device_map[f"model.layers.{i}"] = "cuda:0"
        logger.info(
            "Explicit device_map: %d transformer layers on GPU, embed_tokens+lm_head on CPU",
            n_layers,
        )
        return device_map
    except Exception as exc:
        logger.warning("Could not build explicit device_map (%s); falling back to 'auto'", exc)
        return "auto"


def _load_model_and_tokenizer(model_id: str, token: str | None):
    logger.info("Loading tokenizer for %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    on_gpu = torch.cuda.is_available()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    ) if on_gpu else None

    device_map = _build_device_map(model_id, token) if on_gpu else None
    torch_dtype = torch.float16 if not on_gpu else None

    logger.info("Loading model %s (4-bit=%s) ...", model_id, on_gpu)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        token=token,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def _extract_one(
    model_id: str,
    slug: str,
    texts: list[str],
    labels: np.ndarray,
    batch_size: int,
    token: str | None,
) -> None:
    model, tokenizer = _load_model_and_tokenizer(model_id, token)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_layers = model.config.num_hidden_layers + 1
    layer_buffers: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}

    backbone = getattr(model, "model", model)
    device_map = getattr(model, "hf_device_map", {})
    embed_tokens = getattr(backbone, "embed_tokens", None)
    embed_on_cpu = (
        embed_tokens is not None
        and device_map.get("model.embed_tokens", device) == "cpu"
    )

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for b_idx, i in enumerate(range(0, len(texts), batch_size)):
        inputs = tokenizer(
            texts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        if embed_on_cpu:
            # Run embedding lookup on CPU to avoid copying 1 GB weight matrix to GPU
            ids_cpu = inputs["input_ids"].cpu()
            hook = getattr(embed_tokens, "_hf_hook", None)
            if hook is not None:
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(embed_tokens)
            inputs_embeds = embed_tokens(ids_cpu).to(device)
            if hook is not None:
                from accelerate.hooks import add_hook_to_module
                add_hook_to_module(embed_tokens, hook)
            fwd = {k: v.to(device) for k, v in inputs.items() if k != "input_ids"}
            fwd["inputs_embeds"] = inputs_embeds
        else:
            fwd = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = backbone(**fwd, output_hidden_states=True)

        for li, hidden in enumerate(out.hidden_states):
            layer_buffers[li].append(hidden[:, -1, :].cpu().float().numpy())

        if (b_idx + 1) % 100 == 0 or (b_idx + 1) == total_batches:
            logger.info("  %s — batch %d/%d", slug, b_idx + 1, total_batches)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    layers = {i: np.concatenate(arrs, axis=0) for i, arrs in layer_buffers.items()}
    save_path = Path("data/embeddings") / slug / "layers.h5"
    save_embeddings(save_path, layers, labels)
    logger.info("Saved %d layers × %d samples → %s", n_layers, len(texts), save_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract layer-wise embeddings from an LLM.",
        epilog="""
Examples
--------
# Tulu-3 alignment trajectory (3 public AllenAI checkpoints):
python scripts/extract_embeddings.py --trajectory tulu3

# Single model:
python scripts/extract_embeddings.py --model allenai/Llama-3.1-Tulu-3-8B
""",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", help="HuggingFace model ID (single-model mode)")
    mode.add_argument(
        "--trajectory",
        choices=["tulu3"],
        help="Extract all 3 Tulu-3-8B checkpoints (SFT → DPO → RLHF).",
    )
    parser.add_argument("--n-rows", type=int, default=2000, help="hh-rlhf rows to sample")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    logger.info("Loading %d rows from Anthropic/hh-rlhf ...", args.n_rows)
    texts, labels = sample_hh_rlhf(args.n_rows)

    if args.trajectory == "tulu3":
        for model_id, slug in TULU3_UNGATED:
            logger.info("=== %s (%s) ===", slug, model_id)
            _extract_one(model_id, slug, texts, labels, args.batch_size, token)
    else:
        slug = args.model.replace("/", "--")
        _extract_one(args.model, slug, texts, labels, args.batch_size, token)


if __name__ == "__main__":
    main()
