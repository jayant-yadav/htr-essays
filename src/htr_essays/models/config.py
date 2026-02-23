"""
Training configuration for HTR pipeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model settings
    model_name: str = "microsoft/trocr-base-handwritten"
    pretrained: bool = True

    # Data settings
    data_dir: str = "../../../200essays"
    annotations_file: str = "../../../200essays/json_full.json"
    split_file: str = "data_splits.json"

    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 8  # Per GPU batch size for training
    eval_batch_size: int = 2  # Per GPU batch size for evaluation (smaller to reduce memory)
    num_epochs: int = 50
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4

    # Optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler_type: str = "cosine"  # "linear", "cosine", "polynomial"
    num_warmup_steps: int = 500

    # Mixed precision training
    fp16: bool = True  # Enable for A100 GPUs
    fp16_opt_level: str = "O1"

    # Distributed training
    local_rank: int = -1
    world_size: int = 1

    # Checkpointing
    output_dir: str = "outputs"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True

    # Evaluation
    eval_steps: int = 500
    evaluation_strategy: str = "steps"  # "steps" or "epoch"
    metric_for_best_model: str = "cer"
    greater_is_better: bool = False  # Lower CER is better

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001

    # Logging
    logging_dir: str = "logs"
    logging_steps: int = 100
    report_to: str = "tensorboard"  # "tensorboard", "wandb", or "none"

    # Data processing
    max_length: int = 128  # Maximum sequence length
    num_workers: int = 4

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Reproducibility
    seed: int = 42

    # Debug mode (use small subset)
    debug: bool = False
    debug_samples: int = 100


@dataclass
class ModelConfig:
    """Configuration for TrOCR model."""

    # Architecture
    encoder_model: str = "microsoft/trocr-base-handwritten"
    decoder_model: Optional[str] = None  # Use same as encoder if None

    # Image processing
    image_size: tuple = (384, 384)
    patch_size: int = 16

    # Text processing
    vocab_size: int = 50265
    max_position_embeddings: int = 512
    bos_token_id: int = 0
    eos_token_id: int = 2
    pad_token_id: int = 1

    # Model dimensions
    encoder_hidden_size: int = 768
    decoder_hidden_size: int = 768
    encoder_layers: int = 12
    decoder_layers: int = 6
    encoder_attention_heads: int = 12
    decoder_attention_heads: int = 12

    # Dropout
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Training
    label_smoothing: float = 0.0  # Label smoothing factor (0 = disabled)


def get_default_config() -> TrainingConfig:
    """Get default training configuration for A100 GPUs."""
    return TrainingConfig(
        batch_size=8,
        num_epochs=50,
        fp16=True,
        gradient_accumulation_steps=4,
    )


def get_debug_config() -> TrainingConfig:
    """Get debug configuration for quick testing."""
    config = TrainingConfig()
    config.debug = True
    config.debug_samples = 100
    config.num_epochs = 2
    config.batch_size = 4
    config.save_steps = 50
    config.eval_steps = 50
    config.logging_steps = 10
    return config


def get_multi_gpu_config(num_gpus: int = 4) -> TrainingConfig:
    """
    Get configuration optimized for multi-GPU training.

    Args:
        num_gpus: Number of GPUs available

    Returns:
        TrainingConfig optimized for multi-GPU
    """
    config = TrainingConfig()
    config.batch_size = 8  # Per GPU
    config.gradient_accumulation_steps = 4
    config.world_size = num_gpus

    # Effective batch size = batch_size * gradient_accumulation_steps * num_gpus
    # = 8 * 4 * 4 = 128

    return config


if __name__ == '__main__':
    # Print default configuration
    config = get_default_config()
    print("Default Training Configuration:")
    for field, value in config.__dict__.items():
        print(f"  {field}: {value}")

    print(f"\nEffective batch size (1 GPU): {config.batch_size * config.gradient_accumulation_steps}")

    multi_config = get_multi_gpu_config(4)
    print(f"\nEffective batch size (4 GPUs): {multi_config.batch_size * multi_config.gradient_accumulation_steps * 4}")
