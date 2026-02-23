"""
TrOCR model utilities for Swedish handwriting recognition.
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    TrOCRProcessor,
)

from .config import ModelConfig


class TrOCRForHTR(VisionEncoderDecoderModel):
    """
    TrOCR model wrapper for handwritten text recognition.

    Hugging Face VisionEncoderDecoderModel with HTR-specific defaults.
    """

    def __init__(
        self,
        config: VisionEncoderDecoderConfig,
        htr_config: Optional[ModelConfig] = None,
    ):
        """
        Initialize TrOCR model.

        Args:
            config: HuggingFace VisionEncoderDecoderConfig
            htr_config: HTR-specific model configuration
        """
        super().__init__(config)
        self.htr_config = htr_config or ModelConfig()

        # Set model config parameters
        self.config.decoder_start_token_id = self.htr_config.bos_token_id
        self.config.pad_token_id = self.htr_config.pad_token_id
        self.config.eos_token_id = self.htr_config.eos_token_id

        # Enable gradient checkpointing for memory efficiency
        self.gradient_checkpointing_enable()

    @classmethod
    def create(
        cls,
        model_name: str = "microsoft/trocr-base-handwritten",
        htr_config: Optional[ModelConfig] = None,
    ) -> "TrOCRForHTR":
        """
        Create model from a pretrained checkpoint.

        Args:
            model_name: HuggingFace model name or path
            htr_config: HTR-specific model configuration

        Returns:
            TrOCRForHTR instance loaded from pretrained weights
        """
        return cls.from_pretrained(model_name, htr_config=htr_config)

    @classmethod
    def create_from_scratch(
        cls,
        encoder_config,
        decoder_config,
        htr_config: Optional[ModelConfig] = None,
    ) -> "TrOCRForHTR":
        """
        Create model from encoder/decoder configs with random weights.
        """
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )
        return cls(config, htr_config=htr_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pixel_values: Batch of images [batch_size, 3, height, width]
            labels: Batch of target sequences [batch_size, seq_len]
            **kwargs: Additional arguments to pass to the model

        Returns:
            Dict with 'loss' and 'logits'
        """
        outputs = super().forward(
            pixel_values=pixel_values,
            labels=labels,
            **kwargs
        )

        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits,
        }

    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        early_stopping: bool = True,
    ) -> torch.Tensor:
        """
        Generate text predictions.

        Args:
            pixel_values: Batch of images
            max_length: Maximum sequence length
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop when all beams finish

        Returns:
            Generated token IDs [batch_size, seq_len]
        """
        outputs = super().generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
        )

        return outputs


def create_processor(model_name: str = "microsoft/trocr-base-handwritten") -> TrOCRProcessor:
    """
    Create TrOCR processor for image and text processing.

    Args:
        model_name: HuggingFace model name

    Returns:
        TrOCRProcessor instance
    """
    processor = TrOCRProcessor.from_pretrained(model_name)
    return processor


def setup_model_and_processor(
    model_name: str = "microsoft/trocr-base-handwritten",
    config: Optional[ModelConfig] = None,
    device: str = "cuda",
) -> tuple:
    """
    Setup model and processor.

    Args:
        model_name: HuggingFace model name
        config: Model configuration
        device: Device to load model on

    Returns:
        Tuple of (model, processor)
    """
    # Create processor
    processor = create_processor(model_name)

    # Create model
    model = TrOCRForHTR.create(
        model_name=model_name,
        htr_config=config,
    )

    # Move to device
    model = model.to(device)

    return model, processor


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_encoder(model: TrOCRForHTR):
    """
    Freeze encoder parameters for faster fine-tuning.

    Args:
        model: TrOCR model
    """
    for param in model.encoder.parameters():
        param.requires_grad = False

    print("Encoder frozen. Only decoder will be trained.")


def unfreeze_encoder(model: TrOCRForHTR):
    """
    Unfreeze encoder parameters.

    Args:
        model: TrOCR model
    """
    for param in model.encoder.parameters():
        param.requires_grad = True

    print("Encoder unfrozen. Full model will be trained.")


if __name__ == '__main__':
    # Test model creation
    print("Creating TrOCR model...")
    model = TrOCRForHTR.create()

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 384, 384)
    labels = torch.randint(0, 50265, (batch_size, 32))

    print("\nTesting forward pass...")
    outputs = model(pixel_values=pixel_values, labels=labels)
    print(f"Loss: {outputs['loss']}")
    print(f"Logits shape: {outputs['logits'].shape}")

    print("\nModel setup successful!")
