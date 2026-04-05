# Model_Training.md — Model Training Guide
## Smart-Stock: TrOCR + DistilBERT Fine-Tuning

**Version:** 1.0  
**Training Environment:** Google Colab Pro / Kaggle (T4/P100 GPU)

---

## 1. Overview

Two models are fine-tuned for Smart-Stock:

| Model | Base | Task | Dataset |
|---|---|---|---|
| TrOCR | `microsoft/trocr-base-printed` | Receipt OCR | SROIE + CORD |
| DistilBERT NER | `distilbert-base-uncased` | Food entity extraction | Annotated receipt NER corpus |

Both are fine-tuned on Colab, serialized to ONNX for efficient CPU inference in production.

---

## 2. TrOCR Fine-Tuning

### 2.1 Dataset Sources

**Primary: SROIE (Scanned Receipts OCR and Information Extraction)**  
- 1,000 scanned receipt images with full OCR annotations
- Download: [ICDAR 2019 SROIE Challenge](https://rrc.cvc.uab.es/?ch=13)
- Format: Image + bounding boxes + transcribed text per box

**Secondary: CORD (Consolidated Receipt Dataset)**  
- 1,000 Indonesian/English receipts with structured annotations
- Download: HuggingFace `naver-clova-ix/cord-v2`
- Used to improve generalization across receipt formats

**Augmentation (see Section 2.3):** Synthetic dirty/noisy receipt generation expands dataset to ~5,000 training examples.

### 2.2 Data Preparation

```python
from datasets import load_dataset
from transformers import TrOCRProcessor
from PIL import Image
import os

# Load CORD from HuggingFace
dataset = load_dataset("naver-clova-ix/cord-v2")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

def preprocess_trocr(example):
    image = Image.open(example["image_path"]).convert("RGB")
    
    # Resize to 384x384 (ViT input)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    # Tokenize ground truth text
    labels = processor.tokenizer(
        example["text"],
        padding="max_length",
        max_length=128,
        truncation=True
    ).input_ids
    
    # Replace padding token id with -100 (ignored in loss)
    labels = [l if l != processor.tokenizer.pad_token_id else -100 for l in labels]
    
    return {"pixel_values": pixel_values.squeeze(), "labels": labels}
```

### 2.3 Augmentation Strategy

Thermal receipt images degrade in specific ways. The augmentation pipeline simulates real-world receipt degradation:

```python
import albumentations as A
import cv2
import numpy as np

receipt_augmentation = A.Compose([
    # Simulate thermal fade
    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.1), p=0.5),
    
    # Simulate scanner noise
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    
    # Simulate slight rotation (pocket crumple)
    A.Rotate(limit=5, border_mode=cv2.BORDER_REPLICATE, p=0.5),
    
    # Simulate perspective distortion (photo from angle)
    A.Perspective(scale=(0.02, 0.05), p=0.3),
    
    # Simulate motion blur (shaky photo)
    A.MotionBlur(blur_limit=3, p=0.2),
    
    # Simulate JPEG compression artifacts
    A.ImageCompression(quality_lower=60, quality_upper=90, p=0.4),
])
```

**Augmentation ratio:** 3 augmented copies per original image → ~4,000 training images total from SROIE + CORD.

### 2.4 Fine-Tuning Configuration

```python
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Configure decoder for generation
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-smart-stock",
    num_train_epochs=10,
    per_device_train_batch_size=8,       # T4 GPU: fits at batch 8
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    predict_with_generate=True,
    fp16=True,                           # Mixed precision for T4
    generation_max_length=128,
    dataloader_num_workers=2,
)
```

### 2.5 Evaluation Metric: CER

```python
from jiwer import cer, wer

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
    return {
        "cer": cer(label_str, pred_str),
        "wer": wer(label_str, pred_str),
    }
```

### 2.6 Training Schedule (Colab Estimate)

| Phase | Duration | Notes |
|---|---|---|
| Data prep + augmentation | ~1 hour | Run once, save to Drive |
| Fine-tuning (10 epochs) | ~4–6 hours | T4 GPU, batch 8 |
| Evaluation | ~30 min | On held-out SROIE test set |
| ONNX export | ~15 min | — |

---

## 3. DistilBERT NER Fine-Tuning

### 3.1 Dataset Sources

**Option A: CORD NER Annotations**  
CORD includes structured key-value annotations (menu items, prices, quantities). These can be re-mapped to BIO tags.

**Option B: Self-Annotated Receipt NER Corpus (Recommended)**  
Collect 200–300 real receipt text outputs (from TrOCR stage) and annotate manually using [Label Studio](https://labelstud.io/) or [Doccano](https://github.com/doccano/doccano).

Label schema:
```
ORG    STRWBRY   1     LB     2.99
B-FOOD I-FOOD  B-QTY B-UNIT B-PRICE
```

Target dataset size: **500–800 annotated receipt text lines** is sufficient for DistilBERT fine-tuning given the constrained vocabulary.

**Augmentation:** Token-level synonym replacement for `O`-tagged tokens. Food token substitution from the abbreviation map (swap `STRWBRY` ↔ `BLUBRY` with matching labels) to increase entity diversity.

### 3.2 Data Format (HuggingFace NER Format)

```python
# Each example
{
    "tokens": ["ORG", "STRWBRY", "1", "LB", "2.99"],
    "ner_tags": [0, 1, 4, 5, 7]
    # 0=B-FOOD, 1=I-FOOD, 2=B-QTY, 3=B-UNIT, 4=B-BRAND, 5=B-PRICE, 6=O
}
```

### 3.3 Fine-Tuning Configuration

```python
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

label_list = ["B-FOOD", "I-FOOD", "B-QTY", "B-UNIT", "B-BRAND", "B-PRICE", "O"]
id2label = {i: l for i, l in enumerate(label_list)}
label2id = {l: i for i, l in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./distilbert-ner-smart-stock",
    num_train_epochs=15,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
```

### 3.4 Tokenization Alignment

DistilBERT's WordPiece tokenizer may split one token into multiple subword tokens. Labels must be aligned:

```python
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)          # Special tokens
            elif word_id != prev_word_id:
                aligned.append(label[word_id])  # First subtoken: use real label
            else:
                aligned.append(-100)          # Subsequent subtokens: ignore
            prev_word_id = word_id
        labels.append(aligned)
    
    tokenized["labels"] = labels
    return tokenized
```

### 3.5 Evaluation: seqeval F1

```python
import evaluate
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
```

### 3.6 Training Schedule (Colab Estimate)

| Phase | Duration |
|---|---|
| Data annotation (manual) | 4–8 hours (one-time) |
| Data formatting + augmentation | ~30 min |
| Fine-tuning (15 epochs) | ~45 min (T4 GPU, batch 32) |
| Evaluation | ~10 min |
| ONNX export | ~5 min |

---

## 4. ONNX Export (Both Models)

ONNX serialization enables fast CPU inference in production without requiring PyTorch or GPU.

### TrOCR → ONNX
```python
# Note: TrOCR is a seq2seq model. Use optimum for export.
from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="./trocr-smart-stock/best-model",
    output="./ml_service/models/trocr_onnx/",
    task="image-to-text",
    opset=14
)
```

### DistilBERT NER → ONNX
```python
from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="./distilbert-ner-smart-stock/best-model",
    output="./ml_service/models/distilbert_ner_onnx/",
    task="token-classification",
    opset=14
)
```

### ONNX Runtime Inference
```python
from optimum.onnxruntime import ORTModelForTokenClassification

ner_model = ORTModelForTokenClassification.from_pretrained(
    "./ml_service/models/distilbert_ner_onnx"
)
```

---

## 5. Model Versioning & Storage

- Trained model checkpoints saved to Google Drive during Colab training.
- Best checkpoint (by eval metric) exported to ONNX.
- ONNX models committed to repo under `ml_service/models/` (tracked via Git LFS).
- Model version recorded in `config.yaml`:

```yaml
models:
  trocr:
    version: "1.0.0"
    path: "ml_service/models/trocr_onnx"
    trained_on: "SROIE+CORD+augmented"
    eval_cer: 0.041
  distilbert_ner:
    version: "1.0.0"
    path: "ml_service/models/distilbert_ner_onnx"
    trained_on: "receipt_ner_v1"
    eval_f1: 0.891
```

---

## 6. Retraining Trigger Criteria

Retrain when any of the following are true:
- End-to-end item accuracy on test receipts drops below **82%**
- New retail receipt format identified with < 70% extraction accuracy
- NER F1 drops below **0.85** on evaluation set
- Normalization LLM fallback rate exceeds **30%** (indicates NER is missing common items)

---

## 7. Colab Notebook Structure

Recommended notebook organization for reproducibility:

```
smart_stock_training/
├── 01_trocr_data_prep.ipynb       # Download SROIE/CORD, augment, save to Drive
├── 02_trocr_finetune.ipynb        # Fine-tune TrOCR, eval, export ONNX
├── 03_ner_annotation_guide.md     # Instructions for manual annotation
├── 04_ner_data_prep.ipynb         # Format annotations, augment, split
├── 05_ner_finetune.ipynb          # Fine-tune DistilBERT NER, eval, export ONNX
└── 06_pipeline_integration_test.ipynb  # End-to-end test with real receipts
```

All notebooks mount Google Drive at `/content/drive/MyDrive/smart-stock/` for persistent checkpoint storage across Colab sessions.
