"""
Main training script for HTR model.

Usage:
    python -m htr_essays.training.train --config config.json
    python -m htr_essays.training.train --debug
"""

import argparse
import os
import sys
from numbers import Real
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import set_seed
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from htr_essays.data.dataset import EssayLineDataset, create_data_splits
from htr_essays.data.preprocessing import get_train_transform, get_val_transform
from htr_essays.models.trocr_model import TrOCRForHTR, create_processor
from htr_essays.models.config import TrainingConfig, get_debug_config, get_multi_gpu_config
from htr_essays.training.trainer import setup_trainer, create_training_arguments
from htr_essays.evaluation.metrics import compute_all_metrics


def format_error_rate(value: float) -> str:
    return f"{value:.4f} ({value * 100:.2f}%)"


def save_training_loss_curve(trainer, output_path: str) -> str | None:
    if not trainer.is_world_process_zero():
        return None

    log_history = trainer.state.log_history
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for item in log_history:
        step = item.get('step')
        if step is None:
            continue

        if 'loss' in item and 'eval_loss' not in item:
            train_steps.append(step)
            train_losses.append(float(item['loss']))

        if 'eval_loss' in item:
            eval_steps.append(step)
            eval_losses.append(float(item['eval_loss']))

    if not train_losses and not eval_losses:
        return None

    import matplotlib.pyplot as plt

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(train_steps, train_losses, label='train_loss', linewidth=2)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label='eval_loss', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path_obj), dpi=200)
    plt.close()

    return str(output_path_obj)


def evaluate_with_generation(model, processor, dataset, device, batch_size: int = 16, num_beams: int = 4):
    """Compute evaluation metrics from generated sequences (not argmax over logits)."""
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Final eval (generate)"):
            batch_samples = [dataset[i] for i in range(start_idx, min(start_idx + batch_size, len(dataset)))]

            pixel_values = torch.stack([sample['pixel_values'] for sample in batch_samples]).to(device)
            labels = torch.stack([sample['labels'] for sample in batch_samples])

            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=num_beams,
                early_stopping=True,
            )

            pred_strs = processor.batch_decode(generated_ids, skip_special_tokens=True)

            labels = labels.clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id
            ref_strs = processor.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(pred_strs)
            references.extend(ref_strs)

    return compute_all_metrics(predictions, references)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train HTR model on Swedish student essays")

    parser.add_argument('--debug', action='store_true', help='Run in debug mode with small dataset')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--data_dir', type=str, default='../../../200essays', help='Data directory')
    parser.add_argument('--annotations_file', type=str, default='../../../200essays/json_full.json',
                       help='Path to annotations JSON')
    parser.add_argument('--split_file', type=str, default='data_splits.json', help='Path to data splits JSON')
    parser.add_argument('--create_splits', action='store_true', help='Create new data splits')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs')
    parser.add_argument('--model_name', type=str, default='microsoft/trocr-large-handwritten',
                       help='Pretrained model name')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--loss_curve_path', type=str, default='logs/train/training_loss_curve.png',
                       help='Path to save training loss curve image')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup configuration
    if args.debug:
        print("Running in DEBUG mode")
        config = get_debug_config()
    elif args.num_gpus is not None:
        print(f"Configured for {args.num_gpus} GPUs")
        config = get_multi_gpu_config(args.num_gpus)
    else:
        config = TrainingConfig()

    # Override config with command line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.annotations_file:
        config.annotations_file = args.annotations_file
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.model_name:
        config.model_name = args.model_name

    loss_curve_path = args.loss_curve_path

    # Set random seed for reproducibility
    set_seed(config.seed)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    print("="*70)
    print("HTR Training Pipeline for Swedish Student Essays")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  FP16: {config.fp16}")
    print(f"  Debug mode: {config.debug}")
    print()

    # Create data splits if needed
    split_file_path = args.split_file

    # Verify annotations file exists
    if not os.path.exists(config.annotations_file):
        print(f"\nERROR: Annotations file not found: {config.annotations_file}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Please ensure json_full.json is in the correct location.")
        sys.exit(1)

    if args.create_splits or not os.path.exists(split_file_path):
        print("Creating data splits...")
        create_data_splits(
            annotations_file=config.annotations_file,
            output_file=split_file_path,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=config.seed,
        )
        print()

    # Setup processor
    print("Loading TrOCR processor...")
    processor = create_processor(config.model_name)
    print(f"  Vocabulary size: {len(processor.tokenizer)}")
    print()

    # Create datasets
    print("Loading datasets...")
    train_dataset = EssayLineDataset(
        annotations_file=config.annotations_file,
        images_root=config.data_dir,
        processor=processor,
        split_file=split_file_path,
        split_type='train',
        transform=get_train_transform() if not config.debug else None,
    )

    val_dataset = EssayLineDataset(
        annotations_file=config.annotations_file,
        images_root=config.data_dir,
        processor=processor,
        split_file=split_file_path,
        split_type='val',
        transform=get_val_transform(),
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print()

    # Limit dataset size for debug mode
    if config.debug:
        train_dataset.samples = train_dataset.samples[:config.debug_samples]
        val_dataset.samples = val_dataset.samples[:config.debug_samples // 5]
        print(f"  Debug mode: Limited to {len(train_dataset)} train, {len(val_dataset)} val samples")
        print()

    # Setup model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = TrOCRForHTR.create(
        model_name=config.model_name,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    # Create training arguments
    print("Setting up training...")
    training_args = create_training_arguments(config)
    if hasattr(training_args, 'predict_with_generate'):
        training_args.predict_with_generate = True

    # Setup trainer
    trainer = setup_trainer(
        model=model,  # Pass the TrOCRForHTR wrapper which properly handles all arguments
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_args=training_args,
    )

    print(f"  Trainer configured")
    print(f"  Logging to: {config.logging_dir}")
    print()

    # Start training
    print("="*70)
    print("Starting training...")
    print("="*70)
    print()

    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        curve_path = save_training_loss_curve(trainer, loss_curve_path)
        if curve_path:
            print(f"Training loss curve saved to: {curve_path}")
        print("Saving model...")
        trainer.save_model(os.path.join(config.output_dir, "interrupted_checkpoint"))
        return

    curve_path = save_training_loss_curve(trainer, loss_curve_path)
    if curve_path:
        print(f"Training loss curve saved to: {curve_path}")

    # Save final model
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    print(f"\nSaving model to {config.output_dir}/final_model")
    trainer.save_model(os.path.join(config.output_dir, "final_model"))

    # Evaluate on validation set using generation-based metrics
    print("\nRunning final evaluation (generation-based CER/WER)...")
    eval_batch_size = getattr(config, 'eval_batch_size', 16)
    metrics = evaluate_with_generation(
        model=trainer.model.module if hasattr(trainer.model, 'module') else trainer.model,
        processor=processor,
        dataset=val_dataset,
        device=device,
        batch_size=eval_batch_size,
        num_beams=4,
    )

    print("\nFinal Metrics:")
    for key, value in metrics.items():
        if isinstance(value, Real) and ('cer' in key.lower() or 'wer' in key.lower()):
            print(f"  {key}: {format_error_rate(float(value))}")
        elif isinstance(value, Real):
            print(f"  {key}: {float(value):.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nModel saved to: {config.output_dir}")
    print("Training pipeline complete!")


if __name__ == '__main__':
    main()
