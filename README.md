# HTR Pipeline for Swedish Student Essays

A complete Handwritten Text Recognition (HTR) pipeline for transcribing Swedish student essays using Microsoft's TrOCR transformer model.

## Features

- **Pre-trained TrOCR Model**: Fine-tuned `microsoft/trocr-base-handwritten` for Swedish handwriting
- **Automated Line Segmentation**: Open CV-based line detection for unseen essays
- **Multi-GPU Training**: Optimized for 4xA100 GPUs with distributed training
- **Comprehensive Evaluation**: CER, WER, and per-year metrics with error analysis
- **End-to-End Inference**: Segment and transcribe new essays automatically

## Dataset

- **200 annotated essays** from years 2000, 2006, 2012, and 2018
- **4,041 line-level annotations** with Swedish transcriptions
- **Bounding box coordinates** for each text line
- **Data split**: 140 train / 30 validation / 30 test essays (70/15/15)

## Installation

Install dependencies:

```bash
cd htr-essays
pip install -e .
```

## Quick Start

### 1. Create Data Splits

```bash
cd /mimer/NOBACKUP/groups/studentessays
python -c "from htr_essays.data.dataset import create_data_splits; create_data_splits('200essays/json_full.json', 'htr-essays/data_splits.json')"
```

### 2. Train the Model

```bash
cd htr-essays
bash scripts/train.sh
```

### 3. Evaluate

```bash
bash scripts/evaluate.sh outputs/final_model test
```

### 4. Inference

```bash
bash scripts/infer.sh outputs/final_model --image path/to/essay.jpg --visualize
```

## Performance Targets

- **With ground truth bounding boxes**: CER < 15%, WER < 25%
- **With automated segmentation**: CER < 20%, WER < 30%

See full documentation in the source code modules.
