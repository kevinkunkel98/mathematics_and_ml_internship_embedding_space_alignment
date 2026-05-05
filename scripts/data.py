import numpy as np
from datasets import load_dataset


def sample_hh_rlhf(n_rows: int = 500) -> tuple[list[str], np.ndarray]:
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    rows = ds.select(range(n_rows))

    texts: list[str] = []
    labels: list[int] = []
    for row in rows:
        texts.append(row["chosen"])
        labels.append(1)
        texts.append(row["rejected"])
        labels.append(0)

    return texts, np.array(labels, dtype=np.int8)
