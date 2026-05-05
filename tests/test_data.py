import numpy as np
from unittest.mock import patch, MagicMock

from scripts.data import sample_hh_rlhf


def _make_mock_dataset(n: int):
    rows = [{"chosen": f"chosen_{i}", "rejected": f"rejected_{i}"} for i in range(n)]
    mock_ds = MagicMock()
    mock_ds.select.return_value = rows
    return mock_ds


def test_returns_paired_texts_and_labels():
    with patch("scripts.data.load_dataset", return_value=_make_mock_dataset(5)):
        texts, labels = sample_hh_rlhf(n_rows=5)

    assert len(texts) == 10
    assert len(labels) == 10
    # chosen then rejected, interleaved
    assert texts[0] == "chosen_0"
    assert texts[1] == "rejected_0"
    assert texts[2] == "chosen_1"
    assert texts[3] == "rejected_1"


def test_label_values():
    with patch("scripts.data.load_dataset", return_value=_make_mock_dataset(4)):
        _, labels = sample_hh_rlhf(n_rows=4)

    assert set(labels.tolist()) == {0, 1}
    # chosen=1, rejected=0, alternating
    np.testing.assert_array_equal(labels[::2], 1)
    np.testing.assert_array_equal(labels[1::2], 0)
