"""
Training utilities and metrics computation for HTR.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import TrainerCallback

import jiwer


def compute_cer(pred_str: str, label_str: str) -> float:
    """
    Compute Character Error Rate (CER).

    Args:
        pred_str: Predicted string
        label_str: Ground truth string

    Returns:
        CER as a float (0-1)
    """
    return jiwer.cer(label_str, pred_str)


def compute_wer(pred_str: str, label_str: str) -> float:
    """
    Compute Word Error Rate (WER).

    Args:
        pred_str: Predicted string
        label_str: Ground truth string

    Returns:
        WER as a float (0-1)
    """
    return jiwer.wer(label_str, pred_str)


def compute_metrics_batch(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute metrics for a batch of predictions.

    Args:
        predictions: List of predicted strings
        references: List of ground truth strings

    Returns:
        Dict with 'cer' and 'wer' keys
    """
    # Compute CER and WER for each pair
    cers = [compute_cer(pred, ref) for pred, ref in zip(predictions, references)]
    wers = [compute_wer(pred, ref) for pred, ref in zip(predictions, references)]

    return {
        'cer': np.mean(cers),
        'wer': np.mean(wers),
        'cer_std': np.std(cers),
        'wer_std': np.std(wers),
    }


def create_compute_metrics_fn(processor):
    """
    Create a compute_metrics function for the Trainer.

    Args:
        processor: TrOCR processor for decoding

    Returns:
        Function that computes CER/WER metrics
    """
    def compute_metrics_htr(eval_pred) -> Dict[str, float]:
        """
        Compute HTR metrics from predictions and labels.

        Args:
            eval_pred: EvalPrediction with predictions and label_ids

        Returns:
            Dict with eval_cer and eval_wer metrics
        """
        predictions, label_ids = eval_pred.predictions, eval_pred.label_ids

        # Decode predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Get predicted token IDs
        pred_ids = np.argmax(predictions, axis=-1)

        # Decode to text
        pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # Decode labels (replace -100 with pad token)
        label_ids = label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_strs = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Compute metrics
        metrics = compute_metrics_batch(pred_strs, label_strs)

        # Return with 'eval_' prefix
        return {f'eval_{k}': v for k, v in metrics.items()}

    return compute_metrics_htr


class HTRTrainer(Trainer):
    """
    Custom Trainer for HTR.

    Note: We rely on the compute_metrics function for metric computation
    and use the default training/evaluation loops for distributed training compatibility.
    """

    def __init__(self, *args, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor

    def save_model(self, output_dir=None, _internal_call=False):
        """
        Save model in Hugging Face format for checkpoint compatibility.

        If the training model is wrapped (e.g., TrOCRForHTR), save the wrapped
        VisionEncoderDecoderModel so checkpoints can be loaded directly with
        VisionEncoderDecoderModel.from_pretrained(...).
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Unwrap DDP/FSDP wrappers first
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        # Save underlying HF model when using TrOCRForHTR wrapper
        hf_model = model_to_save.model if hasattr(model_to_save, 'model') else model_to_save

        # Keep PyTorch .bin format for compatibility with current workflows
        hf_model.save_pretrained(output_dir, safe_serialization=False)

        # Save the processor
        if hasattr(self, 'processor') and self.processor is not None:
            self.processor.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")


class CERMetricCallback(TrainerCallback):
    """
    Callback to compute and log CER/WER during training.
    """

    def __init__(self, processor):
        self.processor = processor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log CER/WER after evaluation."""
        if metrics is not None:
            if 'eval_cer' in metrics:
                print(f"\n{'='*50}")
                print(f"Evaluation Results (Step {state.global_step}):")
                print(f"  CER: {metrics['eval_cer']:.4f}")
                if 'eval_wer' in metrics:
                    print(f"  WER: {metrics['eval_wer']:.4f}")
                if 'eval_loss' in metrics:
                    print(f"  Loss: {metrics['eval_loss']:.4f}")
                print(f"{'='*50}\n")


def setup_trainer(
    model,
    processor,
    train_dataset,
    eval_dataset,
    training_args: TrainingArguments,
    compute_metrics_fn=None,
) -> HTRTrainer:
    """
    Setup HTRTrainer with all components.

    Args:
        model: TrOCR model
        processor: TrOCR processor
        train_dataset: Training dataset
        eval_dataset: Validation dataset
        training_args: Training arguments
        compute_metrics_fn: Optional custom metrics function

    Returns:
        Configured HTRTrainer
    """
    # Data collator
    def collate_fn(batch):
        """Collate function for DataLoader."""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'pixel_values': pixel_values,
            'labels': labels,
        }

    # Create compute_metrics function if not provided
    if compute_metrics_fn is None:
        compute_metrics_fn = create_compute_metrics_fn(processor)

    # Create trainer
    trainer = HTRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processor=processor,
        compute_metrics=compute_metrics_fn,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            CERMetricCallback(processor),
        ],
    )

    return trainer


def create_training_arguments(config) -> TrainingArguments:
    """
    Create TrainingArguments from config.

    Args:
        config: TrainingConfig instance

    Returns:
        TrainingArguments
    """
    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,  # Smaller batch size for evaluation
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        eval_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        fp16=config.fp16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to=config.report_to,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        ddp_find_unused_parameters=True,  # Allow unused parameters in distributed training
    )

    return args


if __name__ == '__main__':
    # Test CER/WER computation
    pred = "hello world"
    ref = "helo wrld"

    cer = compute_cer(pred, ref)
    wer = compute_wer(pred, ref)

    print(f"Prediction: '{pred}'")
    print(f"Reference: '{ref}'")
    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")
