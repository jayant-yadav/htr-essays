"""
Evaluate HTR model on test set and generate comprehensive reports.

Usage:
    python -m htr_essays.evaluation.evaluate --checkpoint outputs/final_model --split test
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from htr_essays.data.dataset import EssayLineDataset
from htr_essays.models.trocr_model import TrOCRForHTR, create_processor
from htr_essays.evaluation.metrics import (
    compute_all_metrics,
    compute_per_year_metrics,
    analyze_errors,
    format_metrics_report,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate HTR model")

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='../../../200essays', help='Data directory')
    parser.add_argument('--annotations_file', type=str, default='../../../200essays/json_full.json')
    parser.add_argument('--split_file', type=str, default='data_splits.json')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_beams', type=int, default=4, help='Number of beams for beam search')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions to JSON')

    return parser.parse_args()


def evaluate_model(
    model,
    processor,
    dataset: EssayLineDataset,
    batch_size: int = 16,
    num_beams: int = 4,
    device: str = 'cuda',
) -> tuple:
    """
    Evaluate model on a dataset.

    Args:
        model: TrOCR model
        processor: TrOCR processor
        dataset: Dataset to evaluate on
        batch_size: Batch size for inference
        num_beams: Number of beams for beam search
        device: Device to run on

    Returns:
        Tuple of (predictions, references, image_paths)
    """
    model.eval()
    predictions = []
    references = []
    image_paths = []

    # Create dataloader
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return pixel_values, labels

    # Process in batches
    print(f"Evaluating on {len(dataset)} samples...")

    with torch.no_grad():
        for idx in tqdm(range(0, len(dataset), batch_size)):
            batch_samples = [dataset[i] for i in range(idx, min(idx + batch_size, len(dataset)))]

            # Get pixel values and labels
            pixel_values = torch.stack([s['pixel_values'] for s in batch_samples]).to(device)
            labels = torch.stack([s['labels'] for s in batch_samples])

            # Generate predictions
            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=num_beams,
                early_stopping=True,
            )

            # Decode predictions
            pred_strs = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Decode references
            labels[labels == -100] = processor.tokenizer.pad_token_id
            ref_strs = processor.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(pred_strs)
            references.extend(ref_strs)

            # Track image paths for per-year analysis
            for i in range(idx, min(idx + batch_size, len(dataset))):
                sample = dataset.samples[i]
                image_paths.append(sample['image_path'])

    return predictions, references, image_paths


def extract_year_from_path(path: str) -> str:
    """Extract year from image path."""
    if '2000' in path:
        return '2000'
    elif '2006' in path:
        return '2006'
    elif '2012' in path:
        return '2012'
    elif '2018' in path:
        return '2018'
    return 'unknown'


def main():
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("HTR Model Evaluation")
    print("="*70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load processor
    print("Loading processor...")
    processor = create_processor()

    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    from transformers import VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint)
    model = model.to(device)

    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = EssayLineDataset(
        annotations_file=args.annotations_file,
        images_root=args.data_dir,
        processor=processor,
        split_file=args.split_file,
        split_type=args.split,
        transform=None,
    )

    print(f"Dataset size: {len(dataset)} line samples")
    print()

    # Evaluate
    predictions, references, image_paths = evaluate_model(
        model=model,
        processor=processor,
        dataset=dataset,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        device=device,
    )

    # Compute overall metrics
    print("\n" + "="*70)
    print("Computing metrics...")
    print("="*70 + "\n")

    metrics = compute_all_metrics(predictions, references)
    print(format_metrics_report(metrics))

    # Compute per-year metrics
    years = [extract_year_from_path(path) for path in image_paths]
    per_year_metrics = compute_per_year_metrics(predictions, references, years)

    print("\nPer-Year Metrics:")
    print("="*70)
    for year, year_metrics in sorted(per_year_metrics.items()):
        print(f"\n{year}:")
        print(f"  Samples: {year_metrics['total_samples']}")
        print(f"  CER: {year_metrics['cer']*100:.2f}%")
        print(f"  WER: {year_metrics['wer']*100:.2f}%")
        print(f"  Sequence Accuracy: {year_metrics['sequence_accuracy']*100:.2f}%")

    # Analyze errors
    print("\n" + "="*70)
    print("Error Analysis")
    print("="*70)
    error_analysis = analyze_errors(predictions, references)

    print(f"\nTotal errors: {error_analysis['total_errors']}")
    print(f"Perfect predictions: {error_analysis['perfect_predictions']}")

    if 'swedish_char_analysis' in error_analysis:
        print("\nSwedish Character Analysis:")
        for char, stats in error_analysis['swedish_char_analysis'].items():
            if stats['count_in_reference'] > 0:
                print(f"  '{char}': {stats['count_in_reference']} in reference, "
                      f"{stats['count_in_prediction']} in prediction "
                      f"(diff: {stats['difference']})")

    # Save results
    results = {
        'overall_metrics': metrics,
        'per_year_metrics': {k: v for k, v in per_year_metrics.items()},
        'error_analysis': error_analysis,
        'args': vars(args),
    }

    results_file = os.path.join(args.output_dir, f'{args.split}_metrics.json')
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            return obj

        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {results_file}")

    # Save predictions if requested
    if args.save_predictions:
        predictions_file = os.path.join(args.output_dir, f'{args.split}_predictions.json')

        predictions_data = [
            {
                'image_path': path,
                'prediction': pred,
                'reference': ref,
                'year': year,
            }
            for path, pred, ref, year in zip(image_paths, predictions, references, years)
        ]

        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f, indent=2, ensure_ascii=False)

        print(f"Predictions saved to: {predictions_file}")

    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70)


if __name__ == '__main__':
    main()
