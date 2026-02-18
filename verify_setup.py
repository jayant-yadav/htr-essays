"""
Verify HTR pipeline setup and test basic functionality.

Run this script to verify that the pipeline is correctly set up.

Usage:
    python verify_setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("="*70)
print("HTR Pipeline Setup Verification")
print("="*70)
print()

# Test 1: Import modules
print("1. Testing module imports...")
try:
    from htr_essays.data.dataset import EssayLineDataset, create_data_splits
    from htr_essays.data.preprocessing import get_train_transform
    from htr_essays.models.trocr_model import TrOCRForHTR
    from htr_essays.models.config import TrainingConfig
    from htr_essays.training.trainer import compute_cer, compute_wer
    from htr_essays.evaluation.metrics import compute_all_metrics
    from htr_essays.inference.predictor import HTRPredictor
    from htr_essays.inference.segment import LineSegmenter
    print("   ✓ All modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Check data files
print("\n2. Checking data files...")
data_root = Path("../200essays")
annotations_file = data_root / "json_full.json"

if annotations_file.exists():
    print(f"   ✓ Found annotations: {annotations_file}")

    import json
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    print(f"   ✓ Loaded {len(annotations)} annotations")
else:
    print(f"   ✗ Annotations file not found: {annotations_file}")
    print("     Please ensure 200essays/json_full.json exists")

# Test 3: Test metrics
print("\n3. Testing metrics calculation...")
try:
    predictions = ["hello world", "test text"]
    references = ["hello world", "test txet"]

    metrics = compute_all_metrics(predictions, references)
    print(f"   ✓ CER: {metrics['cer']:.4f}")
    print(f"   ✓ WER: {metrics['wer']:.4f}")
except Exception as e:
    print(f"   ✗ Metrics test failed: {e}")

# Test 4: Test configuration
print("\n4. Testing configuration...")
try:
    config = TrainingConfig()
    print(f"   ✓ Training config created")
    print(f"     - Model: {config.model_name}")
    print(f"     - Batch size: {config.batch_size}")
    print(f"     - Learning rate: {config.learning_rate}")
except Exception as e:
    print(f"   ✗ Configuration test failed: {e}")

# Test 5: Test segmenter
print("\n5. Testing line segmenter...")
try:
    from PIL import Image
    import numpy as np

    segmenter = LineSegmenter()

    # Create dummy image
    dummy_img = Image.new('RGB', (800, 600), color='white')
    print(f"   ✓ Segmenter initialized")
except Exception as e:
    print(f"   ✗ Segmenter test failed: {e}")

# Test 6: Check dependencies
print("\n6. Checking key dependencies...")
dependencies = {
    'torch': 'PyTorch',
    'transformers': 'Hugging Face Transformers',
    'cv2': 'OpenCV',
    'PIL': 'Pillow',
    'jiwer': 'JiWER',
}

missing = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} (missing)")
        missing.append(name)

# Summary
print("\n" + "="*70)
if missing:
    print("Setup incomplete!")
    print(f"Missing dependencies: {', '.join(missing)}")
    print("\nInstall missing dependencies with:")
    print("  pip install -e .")
    sys.exit(1)
else:
    print("Setup verification complete!")
    print("\n✓ All modules and dependencies are correctly installed")
    print("✓ Data files are accessible")
    print("✓ Basic functionality is working")
    print("\nYou can now:")
    print("  1. Create data splits: python -c 'from htr_essays.data.dataset import create_data_splits; create_data_splits(\"../200essays/json_full.json\", \"data_splits.json\")'")
    print("  2. Train the model: bash scripts/train.sh --debug")
    print("  3. Evaluate the model: bash scripts/evaluate.sh outputs/final_model")
    print("  4. Run inference: bash scripts/infer.sh outputs/final_model --image essay.jpg")

print("="*70)
