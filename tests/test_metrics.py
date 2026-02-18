"""
Tests for evaluation metrics.
"""

import pytest
from htr_essays.evaluation.metrics import (
    compute_cer,
    compute_wer,
    compute_all_metrics,
    compute_sequence_accuracy,
)


def test_compute_cer():
    """Test CER computation."""
    # Identical strings
    assert compute_cer("hello", "hello") == 0.0

    # Single character error
    cer = compute_cer("hello", "helo")
    assert 0.0 < cer < 1.0

    # Empty reference
    assert compute_cer("hello", "") == 1.0


def test_compute_wer():
    """Test WER computation."""
    # Identical strings
    assert compute_wer("hello world", "hello world") == 0.0

    # Single word error
    wer = compute_wer("hello world", "hello wrld")
    assert 0.0 < wer <= 1.0


def test_compute_all_metrics():
    """Test computing all metrics."""
    predictions = ["hello world", "test"]
    references = ["hello world", "tset"]

    metrics = compute_all_metrics(predictions, references)

    assert 'cer' in metrics
    assert 'wer' in metrics
    assert 'sequence_accuracy' in metrics
    assert 'total_samples' in metrics

    assert metrics['total_samples'] == 2
    assert 0.0 <= metrics['cer'] <= 1.0
    assert 0.0 <= metrics['wer'] <= 1.0


def test_sequence_accuracy():
    """Test sequence accuracy computation."""
    predictions = ["abc", "def", "ghi"]
    references = ["abc", "def", "xyz"]

    acc = compute_sequence_accuracy(predictions, references)
    assert acc == 2/3  # 2 out of 3 match


if __name__ == '__main__':
    test_compute_cer()
    test_compute_wer()
    test_compute_all_metrics()
    test_sequence_accuracy()
    print("All metrics tests passed!")
