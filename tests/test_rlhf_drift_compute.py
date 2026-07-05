import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.io import save_embeddings


def test_fit_rlhf_drift_returns_expected_keys():
    from app.rlhf_drift_compute import fit_rlhf_drift

    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 10, dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        sft_path = tmp / "sft.h5"
        dpo_path = tmp / "dpo.h5"
        rlhf_path = tmp / "rlhf.h5"

        for path in (sft_path, dpo_path, rlhf_path):
            layers = {i: rng.standard_normal((20, 8)).astype(np.float32) for i in range(3)}
            save_embeddings(path, layers, labels)

        with patch("app.rlhf_drift_compute.CACHE_PATH", tmp / "cache" / "cka.pkl"):
            result = fit_rlhf_drift(sft_path, dpo_path, rlhf_path)

    assert set(result.keys()) == {"layers", "sft_dpo", "dpo_rlhf", "sft_rlhf"}
    assert result["layers"] == [0, 1, 2]
    assert len(result["sft_dpo"]) == 3
    assert len(result["dpo_rlhf"]) == 3
    assert len(result["sft_rlhf"]) == 3
    for key in ("sft_dpo", "dpo_rlhf", "sft_rlhf"):
        for value in result[key]:
            assert 0.0 <= value <= 1.0 + 1e-9


def test_fit_rlhf_drift_uses_cache_on_second_call():
    from app.rlhf_drift_compute import fit_rlhf_drift

    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 10, dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        sft_path = tmp / "sft.h5"
        dpo_path = tmp / "dpo.h5"
        rlhf_path = tmp / "rlhf.h5"

        for path in (sft_path, dpo_path, rlhf_path):
            layers = {0: rng.standard_normal((20, 8)).astype(np.float32)}
            save_embeddings(path, layers, labels)

        cache_path = tmp / "cache" / "cka.pkl"
        with patch("app.rlhf_drift_compute.CACHE_PATH", cache_path):
            result1 = fit_rlhf_drift(sft_path, dpo_path, rlhf_path)
            # Overwrite one file with different data to prove second call reads cache, not recomputes
            save_embeddings(sft_path, {0: np.zeros((20, 8), dtype=np.float32)}, labels)
            result2 = fit_rlhf_drift(sft_path, dpo_path, rlhf_path)

    assert result1 == result2
