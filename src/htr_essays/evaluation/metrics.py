"""
Evaluation metrics for HTR model.
"""

from typing import List, Dict
import numpy as np
import jiwer
from collections import defaultdict


def compute_cer(prediction: str, reference: str) -> float:
    """
    Compute Character Error Rate (CER).

    CER = (Substitutions + Deletions + Insertions) / Total characters in reference

    Args:
        prediction: Predicted string
        reference: Ground truth string

    Returns:
        CER as float (0-1, lower is better)
    """
    if len(reference) == 0:
        return 1.0 if len(prediction) > 0 else 0.0

    return jiwer.cer(reference, prediction)


def compute_wer(prediction: str, reference: str) -> float:
    """
    Compute Word Error Rate (WER).

    WER = (Substitutions + Deletions + Insertions) / Total words in reference

    Args:
        prediction: Predicted string
        reference: Ground truth string

    Returns:
        WER as float (0-1, lower is better)
    """
    if len(reference.split()) == 0:
        return 1.0 if len(prediction.split()) > 0 else 0.0

    return jiwer.wer(reference, prediction)


def compute_normalized_cer(prediction: str, reference: str) -> float:
    """
    Compute normalized CER (case-insensitive, no punctuation).

    Args:
        prediction: Predicted string
        reference: Ground truth string

    Returns:
        Normalized CER as float
    """
    # Normalize
    pred_norm = prediction.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    ref_norm = reference.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')

    return compute_cer(pred_norm, ref_norm)


def compute_normalized_wer(prediction: str, reference: str) -> float:
    """
    Compute normalized WER (case-insensitive, no punctuation).

    Args:
        prediction: Predicted string
        reference: Ground truth string

    Returns:
        Normalized WER as float
    """
    # Normalize
    pred_norm = prediction.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    ref_norm = reference.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')

    return compute_wer(pred_norm, ref_norm)


def compute_sequence_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Compute percentage of perfectly transcribed sequences.

    Args:
        predictions: List of predicted strings
        references: List of ground truth strings

    Returns:
        Accuracy as float (0-1)
    """
    if len(predictions) == 0 or len(references) == 0:
        return 0.0

    correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    return correct / len(predictions)


def compute_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute all metrics for a set of predictions.

    Args:
        predictions: List of predicted strings
        references: List of ground truth strings

    Returns:
        Dict with all metrics
    """
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references")

    # Compute individual metrics
    cers = [compute_cer(pred, ref) for pred, ref in zip(predictions, references)]
    wers = [compute_wer(pred, ref) for pred, ref in zip(predictions, references)]
    norm_cers = [compute_normalized_cer(pred, ref) for pred, ref in zip(predictions, references)]
    norm_wers = [compute_normalized_wer(pred, ref) for pred, ref in zip(predictions, references)]

    # Compute sequence accuracy
    seq_acc = compute_sequence_accuracy(predictions, references)

    return {
        'cer': np.mean(cers),
        'cer_std': np.std(cers),
        'cer_median': np.median(cers),
        'wer': np.mean(wers),
        'wer_std': np.std(wers),
        'wer_median': np.median(wers),
        'normalized_cer': np.mean(norm_cers),
        'normalized_wer': np.mean(norm_wers),
        'sequence_accuracy': seq_acc,
        'total_samples': len(predictions),
    }


def compute_per_year_metrics(
    predictions: List[str],
    references: List[str],
    years: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by year.

    Args:
        predictions: List of predicted strings
        references: List of ground truth strings
        years: List of year labels (e.g., "2000", "2006")

    Returns:
        Dict mapping year to metrics dict
    """
    # Group by year
    year_groups = defaultdict(lambda: {'predictions': [], 'references': []})

    for pred, ref, year in zip(predictions, references, years):
        year_groups[year]['predictions'].append(pred)
        year_groups[year]['references'].append(ref)

    # Compute metrics for each year
    results = {}
    for year, data in year_groups.items():
        results[year] = compute_all_metrics(data['predictions'], data['references'])

    return results


def analyze_errors(predictions: List[str], references: List[str]) -> Dict:
    """
    Analyze common errors in predictions.

    Args:
        predictions: List of predicted strings
        references: List of ground truth strings

    Returns:
        Dict with error analysis
    """
    # Aggregate error counts using current jiwer API (compute_measures was removed)
    character_errors = {
        'substitutions': 0,
        'deletions': 0,
        'insertions': 0,
        'hits': 0,
    }
    word_errors = {
        'substitutions': 0,
        'deletions': 0,
        'insertions': 0,
        'hits': 0,
    }

    for pred, ref in zip(predictions, references):
        char_alignment = jiwer.process_characters(ref, pred)
        word_alignment = jiwer.process_words(ref, pred)

        character_errors['substitutions'] += int(getattr(char_alignment, 'substitutions', 0))
        character_errors['deletions'] += int(getattr(char_alignment, 'deletions', 0))
        character_errors['insertions'] += int(getattr(char_alignment, 'insertions', 0))
        character_errors['hits'] += int(getattr(char_alignment, 'hits', 0))

        word_errors['substitutions'] += int(getattr(word_alignment, 'substitutions', 0))
        word_errors['deletions'] += int(getattr(word_alignment, 'deletions', 0))
        word_errors['insertions'] += int(getattr(word_alignment, 'insertions', 0))
        word_errors['hits'] += int(getattr(word_alignment, 'hits', 0))

    # Find most difficult characters (Swedish specific)
    swedish_chars = ['å', 'ä', 'ö', 'Å', 'Ä', 'Ö']
    swedish_char_errors = {}

    for char in swedish_chars:
        char_count_ref = sum(ref.count(char) for ref in references)
        char_count_pred = sum(pred.count(char) for pred in predictions)

        if char_count_ref > 0:
            swedish_char_errors[char] = {
                'count_in_reference': char_count_ref,
                'count_in_prediction': char_count_pred,
                'difference': abs(char_count_ref - char_count_pred),
            }

    return {
        'swedish_char_analysis': swedish_char_errors,
        'total_errors': len([p for p, r in zip(predictions, references) if p != r]),
        'perfect_predictions': len([p for p, r in zip(predictions, references) if p == r]),
        'character_error_counts': character_errors,
        'word_error_counts': word_errors,
    }


def format_metrics_report(metrics: Dict) -> str:
    """
    Format metrics as a readable report.

    Args:
        metrics: Metrics dictionary

    Returns:
        Formatted string
    """
    report = []
    report.append("="*70)
    report.append("EVALUATION METRICS")
    report.append("="*70)
    report.append("")

    # Main metrics
    report.append("Character Error Rate (CER):")
    report.append(f"  Mean: {metrics['cer']*100:.2f}%")
    report.append(f"  Std:  {metrics['cer_std']*100:.2f}%")
    report.append(f"  Median: {metrics['cer_median']*100:.2f}%")
    report.append("")

    report.append("Word Error Rate (WER):")
    report.append(f"  Mean: {metrics['wer']*100:.2f}%")
    report.append(f"  Std:  {metrics['wer_std']*100:.2f}%")
    report.append(f"  Median: {metrics['wer_median']*100:.2f}%")
    report.append("")

    report.append("Normalized Metrics (case-insensitive, no punctuation):")
    report.append(f"  CER: {metrics['normalized_cer']*100:.2f}%")
    report.append(f"  WER: {metrics['normalized_wer']*100:.2f}%")
    report.append("")

    report.append(f"Sequence Accuracy: {metrics['sequence_accuracy']*100:.2f}%")
    report.append(f"Total Samples: {metrics['total_samples']}")
    report.append("")
    report.append("="*70)

    return "\n".join(report)


if __name__ == '__main__':
    # Test metrics
    predictions = [
        "Hello world",
        "This is a test",
        "Swedish text",
    ]

    references = [
        "Hello world",
        "This is a tset",
        "Swedish txet",
    ]

    metrics = compute_all_metrics(predictions, references)
    print(format_metrics_report(metrics))

    # Test Swedish character analysis
    pred_swedish = ["Hej världen", "Det är söndng"]
    ref_swedish = ["Hej världen", "Det är söndag"]

    errors = analyze_errors(pred_swedish, ref_swedish)
    print("\nError Analysis:")
    print(errors)
