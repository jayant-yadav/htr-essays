"""
Basic tests for dataset module.
"""

import pytest
import json
import tempfile
from pathlib import Path


def test_create_data_splits():
    """Test data split creation."""
    from htr_essays.data.dataset import create_data_splits

    # Create a dummy annotations file
    dummy_annotations = [{"id": i} for i in range(100)]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dummy_annotations, f)
        annotations_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_file = f.name

    # Create splits
    splits = create_data_splits(
        annotations_file=annotations_file,
        output_file=output_file,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    # Verify splits
    assert len(splits['train']) == 70
    assert len(splits['val']) == 15
    assert len(splits['test']) == 15

    # Verify no overlap
    train_set = set(splits['train'])
    val_set = set(splits['val'])
    test_set = set(splits['test'])

    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0

    # Cleanup
    Path(annotations_file).unlink()
    Path(output_file).unlink()


def test_dataset_initialization():
    """Test dataset can be initialized (with mock data)."""
    # This is a placeholder - would need actual test data
    pass


if __name__ == '__main__':
    test_create_data_splits()
    print("Dataset tests passed!")
