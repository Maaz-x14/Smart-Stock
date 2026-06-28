# OCR_Training.md — Model Training Guide
## SmartStock: TrOCR Fine-Tuning

**Version:** 13.0  
**Training Environment:** Kaggle (2× T4 GPU, 30 hr/week quota — single GPU enforced via `CUDA_VISIBLE_DEVICES`)  
**Last Updated:** Current as of v3 dataset integration + WildReceipt addition

---

## Pipeline Overview

```
Receipt Image → TrOCR (OCR) → NER → Normalization → Expiry Prediction
```

This document covers **Stage 1: TrOCR fine-tuning only.**

**Model:** `microsoft/trocr-base-printed` + LoRA (decoder only)  
**Target metrics:** CER ≤ 0.05 (5%) | WER ≤ 0.10 (10%)  
**Current best:** CER ~0.0856 (Colab LoRA run, stored as `trocr-smart-stock-best`)  
**Baseline to beat:** 0.0856 — do not overwrite this Kaggle dataset until a better run is confirmed

---

## Part 1: Architecture

### 1.1 Why LoRA, Why Frozen Encoder

TrOCR is a Vision Encoder-Decoder: ViT encoder reads the image, RoBERTa decoder generates text. Full fine-tuning of 335M parameters caused plateau and optimizer instability. Current approach:

- **Encoder (ViT): fully frozen** — preserves pretrained visual feature extraction
- **Decoder: LoRA on `q_proj` + `v_proj`, all 12 layers** — adapts text generation with minimal parameters
- **Trainable params: 1,523,712 (0.45% of 335M total)**

LoRA config:
```python
LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
```

**Known ceiling:** Frozen encoder is the current performance ceiling. Once a stable single-GPU baseline is re-established, unfreezing the top 2–4 ViT encoder blocks alongside LoRA is the highest-impact next step.

### 1.2 Why Line Crops

TrOCR was pretrained on single-line text images. Feeding full receipts (20–50 lines) is a domain mismatch. Line crops are the single biggest architectural fix — CER dropped from 0.757 (full images) → 0.133 after switching.

**Input:** Single cropped text line from a receipt (PIL RGB)  
**Output:** Text of that one line (e.g., `"MULTIGRAIN CHEERIO 1.50"`)

---

## Part 2: Dataset — v3 (Current)

### 2.1 Composition

| Source | Train | Val | Test | Notes |
|--------|-------|-----|------|-------|
| CORD | ~2,105 | 221 | ~251 | Indonesian restaurant receipts |
| SROIE | ~14,476 | — | ~8,050 | English retail receipts, 2× weighted in train |
| WildReceipt | ~11,400 | ~1,267 | 1,000 | Diverse English receipts, test capped at 1,000 |
| WildReceipt excess test | ~4,103 | — | — | Moved to train |
| **Total v3** | **~46,500** | **~1,488** | **~9,301** | |

**Dataset name on Kaggle:** `smart-stock-dataset-v3`  
**Input path:** `/kaggle/input/datasets/maazahmad69/smart-stock-dataset-v3/smart_stock_dataset_v3`

### 2.2 Key Dataset Decisions

- **SROIE 2× weighted** in train — more English receipt signal vs CORD's Indonesian menus
- **Validation = CORD + WildReceipt val only** — SROIE has no val split; retraining from scratch so new baseline will be established on this mixed val set
- **WildReceipt test capped at 1,000 crops** — prevents test eval from hanging (previous session lost 6+ hours to beam search on 8,301 SROIE test samples alone); excess ~4,103 crops folded into train
- **WildReceipt labels excluded:** 0 (empty/illegible) and 25 (catch-all noise like terminal IDs, legal text). All other labels included.
- **WildReceipt line grouping:** relative Y-tolerance (2% of image height, min 10px) — handles WildReceipt's large image sizes (1,000–1,800px) better than SROIE's fixed 15px

### 2.3 CORD Extractor

```python
import json
from collections import defaultdict
from PIL import Image

def extract_cord_crops(image: Image.Image, ground_truth_str: str) -> list:
    try:
        data = json.loads(ground_truth_str)
    except (json.JSONDecodeError, AttributeError):
        return []

    valid_lines = data.get("valid_line", [])
    if not valid_lines:
        return []

    groups = defaultdict(list)
    for line in valid_lines:
        groups[line["group_id"]].append(line)

    w, h = image.size
    crops = []

    for gid, lines in groups.items():
        if not any(l.get("category", "").startswith("menu.") for l in lines):
            continue

        all_words, all_xs, all_ys = [], [], []
        for line in lines:
            for word in line.get("words", []):
                text = word.get("text", "").strip()
                if text:
                    all_words.append(text)
                q = word.get("quad", {})
                if q:
                    all_xs += [q.get("x1",0), q.get("x2",0), q.get("x3",0), q.get("x4",0)]
                    all_ys += [q.get("y1",0), q.get("y2",0), q.get("y3",0), q.get("y4",0)]

        if not all_words or not all_xs:
            continue

        text = " ".join(all_words)
        x1, y1 = max(0, min(all_xs)), max(0, min(all_ys))
        x2, y2 = min(w, max(all_xs)), min(h, max(all_ys))
        if x2 <= x1 or y2 <= y1:
            continue

        crops.append((image.crop((x1, y1, x2, y2)), text))

    return crops
```

### 2.4 SROIE Extractor

```python
def extract_sroie_crops(image: Image.Image, words: list, bboxes: list) -> list:
    if not words or not bboxes:
        return []

    items = sorted(zip(words, bboxes), key=lambda x: x[1][1])

    lines = []
    current_words, current_boxes = [items[0][0]], [items[0][1]]
    for word, box in items[1:]:
        if abs(box[1] - current_boxes[-1][1]) <= 15:
            current_words.append(word)
            current_boxes.append(box)
        else:
            lines.append((current_words, current_boxes))
            current_words, current_boxes = [word], [box]
    lines.append((current_words, current_boxes))

    w, h = image.size
    crops = []
    for line_words, line_boxes in lines:
        text = " ".join(line_words).strip()
        if not text:
            continue
        x1 = max(0, min(b[0] for b in line_boxes))
        y1 = max(0, min(b[1] for b in line_boxes))
        x2 = min(w, max(b[2] for b in line_boxes))
        y2 = min(h, max(b[3] for b in line_boxes))
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append((image.crop((x1, y1, x2, y2)), text))
    return crops
```

### 2.5 WildReceipt Extractor

```python
import json
WILDRECEIPT_EXCLUDE = {0, 25}

def extract_wildreceipt_crops(annotation_file: Path) -> list:
    crops = []

    with open(annotation_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            img_path = WILDRECEIPT_DIR / record["file_name"]
            if not img_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            img_w, img_h = image.size
            y_tolerance = max(10, int(img_h * 0.02))  # 2% of height, min 10px

            valid_annotations = [
                ann for ann in record["annotations"]
                if ann["label"] not in WILDRECEIPT_EXCLUDE
                and ann.get("text", "").strip()
            ]
            if not valid_annotations:
                continue

            def ann_coords(ann):
                box = ann["box"]
                xs = [box[0], box[2], box[4], box[6]]
                ys = [box[1], box[3], box[5], box[7]]
                center_y = (min(ys) + max(ys)) / 2
                return center_y, min(xs), min(ys), max(xs), max(ys)

            parsed = []
            for ann in valid_annotations:
                cy, x1, y1, x2, y2 = ann_coords(ann)
                parsed.append((cy, x1, y1, x2, y2, ann["text"].strip()))
            parsed.sort(key=lambda r: (r[0], r[1]))

            lines = []
            current_line = [parsed[0]]
            for item in parsed[1:]:
                if abs(item[0] - current_line[-1][0]) <= y_tolerance:
                    current_line.append(item)
                else:
                    lines.append(current_line)
                    current_line = [item]
            lines.append(current_line)

            for line_items in lines:
                text = " ".join(item[5] for item in line_items)
                if not text:
                    continue
                x1 = max(0, min(item[1] for item in line_items))
                y1 = max(0, min(item[2] for item in line_items))
                x2 = min(img_w, max(item[3] for item in line_items))
                y2 = min(img_h, max(item[4] for item in line_items))
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                crops.append((image.crop((x1, y1, x2, y2)), text))

    return crops
```

### 2.6 Dataset Builder (v3)

```python
WR_TEST_KEEP = 1000  # cap WildReceipt test crops, remainder moves to train

def build_and_save_dataset():
    if DATASET_SAVE.exists():
        print(f"Dataset found at {DATASET_SAVE} — loading from disk...")
        return DatasetDict.load_from_disk(str(DATASET_SAVE))

    # ... load CORD, SROIE, WildReceipt (see notebook Cell 4 + WildReceipt extractor cell) ...

    # WildReceipt: 90/10 train/val split from train.txt crops
    wr_train_split = wr_train_raw.train_test_split(test_size=0.1, seed=42)
    wr_train_final = wr_train_split["train"]
    wr_val         = wr_train_split["test"]

    # Cap test.txt crops at 1000, move remainder to train
    wr_test_split    = wr_test_raw.train_test_split(test_size=WR_TEST_KEEP, seed=42)
    wr_test_final    = wr_test_split["test"]
    wr_test_to_train = wr_test_split["train"]

    train_parts = [cord_train, sroie_train, sroie_train, wr_train_final, wr_test_to_train]

    dataset_dict = DatasetDict({
        "train":      concatenate_datasets(train_parts),
        "validation": concatenate_datasets([cord_validation, wr_val]),
        "test":       concatenate_datasets([cord_test, sroie_test, wr_test_final]),
    })

    dataset_dict.save_to_disk(str(WORKING_DIR / "smart_stock_dataset_v3"))
```

---

## Part 3: Augmentation

Applied **only to training images**, inline during `preprocess_trocr`. Simulates real-world receipt degradation.

```python
import albumentations as A
import cv2

receipt_augmentation = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.15), p=0.6),  # thermal fade
    A.GaussNoise(p=0.5),                                                 # scanner noise
    A.Rotate(limit=8, border_mode=cv2.BORDER_REPLICATE, p=0.5),         # crumple tilt
    A.Perspective(scale=(0.02, 0.08), p=0.4),                           # photo angle
    A.MotionBlur(blur_limit=5, p=0.3),                                  # shaky photo
    A.ImageCompression(p=0.5),                                           # JPEG artifact
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),                           # focus blur
    A.RandomShadow(p=0.2),                                               # shadows
])

def apply_augmentation(pil_image: Image.Image) -> Image.Image:
    pil_image = pil_image.convert("RGB")
    if pil_image.height < 32:
        pad = 32 - pil_image.height
        pil_image = ImageOps.expand(pil_image, border=(0, pad//2, 0, pad - pad//2), fill=255)
    img_np = np.array(pil_image)
    augmented = receipt_augmentation(image=img_np)["image"]
    return Image.fromarray(augmented)
```

---

## Part 4: Preprocessing & Dataset Class

```python
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

def preprocess_trocr(example, augment: bool = False):
    image = Image.open(io.BytesIO(example["image_bytes"])).convert("RGB")
    if augment:
        image = apply_augmentation(image)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    labels = processor.tokenizer(
        example["text"], padding="max_length", max_length=128, truncation=True,
    ).input_ids
    labels = [t if t != processor.tokenizer.pad_token_id else -100 for t in labels]
    return {
        "pixel_values": pixel_values.squeeze(),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

class TrOCRDataset(TorchDataset):
    # On-the-fly preprocessing — zero RAM accumulation, compatible with Seq2SeqTrainer
    def __init__(self, hf_dataset, augment: bool = False):
        self.data = hf_dataset
        self.augment = augment
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return preprocess_trocr(self.data[idx], augment=self.augment)
```

---

## Part 5: Model Setup

```python
base_model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_INPUT))

# Freeze encoder entirely
for param in base_model.encoder.parameters():
    param.requires_grad = False

# Apply LoRA to decoder (or resume from checkpoint — see priority logic below)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(base_model, lora_config)

# Checkpoint resume priority:
# 1. /kaggle/working/trocr-smart-stock (current session)
# 2. /kaggle/input/.../trocr-smart-stock (uploaded from prior session)
# 3. Fresh LoRA (no prior checkpoint)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.vocab_size             = model.config.decoder.vocab_size
model.generation_config.eos_token_id         = processor.tokenizer.sep_token_id
model.generation_config.early_stopping       = True
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty       = 2.0
model.generation_config.num_beams            = 4
```

**LoRASaveCallback** — critical, must not be removed:
```python
class LoRASaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        adapter_dir = checkpoint_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        kwargs["model"].save_pretrained(str(adapter_dir))
        return control
```
Without this, PEFT silently resets adapter weights on checkpoint reload.

---

## Part 6: Training Arguments (Current Active Config)

```python
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" is set in Cell 1 (Setup & Paths)
# — must be the very first line before any torch import anywhere in the notebook

training_args = Seq2SeqTrainingArguments(
    output_dir=str(CHECKPOINT_DIR),

    num_train_epochs=8,
    per_device_train_batch_size=8,   # effective batch = 8 (single GPU, no DataParallel)
    per_device_eval_batch_size=8,

    learning_rate=1.4824e-4,         # Optuna best (Trial 0, subset run)
    warmup_ratio=0.02672,
    weight_decay=0.01,
    lr_scheduler_type="cosine_with_restarts",
    lr_scheduler_kwargs={"num_cycles": 2},
    max_grad_norm=1.0,

    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=False,
    save_total_limit=5,

    predict_with_generate=True,
    generation_max_length=128,

    fp16=True,
    dataloader_num_workers=2,
    logging_steps=50,
    log_level="info",
    report_to="none",

    # gradient_accumulation_steps=2,  # commented — use only if batch 8 causes OOM
)
```

**Note:** There is also an older commented-out block (`lr=1.695e-4`, `cosine`) above this one in the notebook — leave it commented, it is not used.

---

## Part 7: Kaggle Paths (Current)

```python
# Cell 1 — Setup & Paths
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # MUST be first line, before all imports

INPUT_DIR   = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")

DATASET_DIR     = Path("/kaggle/input/datasets/maazahmad69/smart-stock-dataset-v3/smart_stock_dataset_v3")
WILDRECEIPT_DIR = Path("/kaggle/input/datasets/maazahmad69/wild-receipt/wildreceipt")
MODEL_INPUT     = Path("/kaggle/input/datasets/maazahmad69/trocr-smart-stock-best/content/drive/MyDrive/SmartStock/trocr-smart-stock-best")
INPUT_CHECKPOINT_DIR = Path("/kaggle/input/smart-stock-dataset/trocr-smart-stock")

CHECKPOINT_DIR = WORKING_DIR / "trocr-smart-stock"
BEST_MODEL_DIR = WORKING_DIR / "trocr-smart-stock-best"
```

**Kaggle dataset inputs for current session:**
| Dataset slug | Contains |
|---|---|
| `smart-stock-dataset-v3` | Combined CORD + SROIE + WildReceipt v3 dataset (build once, reuse) |
| `wild-receipt` | Raw WildReceipt images + annotation files (needed only during dataset build) |
| `trocr-smart-stock-best` | Best model weights (CER 0.0856) — loaded as base for fine-tuning |

---

## Part 8: Notebook Cell Run Order

| Cell | Name | Run? |
|------|------|------|
| 1 | Setup & Paths | ✅ Always |
| 2 | CORD crop extractor | ✅ (defines function, fast) |
| 3 | SROIE crop extractor | ✅ (defines function, fast) |
| 4 | WildReceipt extractor | ✅ (defines function, fast) |
| 5 | Dataset Builder | ✅ (loads from disk if v3 exists, else builds ~30 min) |
| 6 | Augmentation | ✅ |
| 7 | Preprocessing + TrOCRDataset | ✅ |
| 8 | collate_fn | ✅ |
| 9 | Model Setup | ✅ |
| 10 | Training Arguments (Technique 1 block) | ✅ |
| 11 | Metrics | ✅ |
| 12 | Optuna | ⛔ Keep commented — run after epoch 1 baseline confirmed |
| 13 | Trainer and Training | ✅ |
| 14 | Training Curves | ✅ |
| 15 | Save & Export | ✅ |
| 16 | Evaluate on Test Set | ✅ (manual loop — not trainer.evaluate) |
| 17 | Qualitative Evaluation | ✅ |

---

## Part 9: Save & Export

```python
from peft import PeftModel

merged_model = model.merge_and_unload()

merged_model.save_pretrained(str(BEST_MODEL_DIR))
processor.save_pretrained(str(BEST_MODEL_DIR))
merged_model.generation_config.save_pretrained(str(BEST_MODEL_DIR))
```

Expected files in `trocr-smart-stock-best/`:
`model.safetensors` (~1.3 GB), `config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`, `processor_config.json`

**Rule:** Only overwrite the `trocr-smart-stock-best` Kaggle dataset if the new run beats CER 0.0856. Otherwise keep it as-is.

---

## Part 10: Test Set Evaluation

```python
# Manual loop — DO NOT use trainer.evaluate(test_dataset)
# trainer.evaluate() hangs on degenerate 1-pixel-wide crops in the test set

model.eval()
all_preds, all_labels = [], []
skipped = 0

for sample in combined_dataset["test"]:
    try:
        image = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")
        w, h = image.size
        if w < 4 or h < 4:
            skipped += 1
            continue
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_preds.append(pred)
        all_labels.append(sample["text"])
    except Exception:
        skipped += 1

test_cer = cer(all_labels, all_preds)
test_wer = wer(all_labels, all_preds)
print(f"Test CER : {test_cer:.4f}")
print(f"Test WER : {test_wer:.4f}")
print(f"Skipped  : {skipped} / {len(combined_dataset['test'])}")
```

---

## Part 11: Training History

| Run | Setup | Best Val CER | Notes |
|-----|-------|-------------|-------|
| Full finetune | All 333M params | 0.1758 | Plateau, optimizer mismatch |
| LoRA first (Colab) | Frozen enc + LoRA dec | 0.0894 | 1 epoch only |
| Optuna search | 1 epoch, 1/8 data, batch 4 | 0.0803–0.0886 | Best: lr=1.4824e-4, warmup=0.02672 |
| Kaggle run 1 | 5 epochs, DataParallel batch 16 | 0.1325→0.1372 | DataParallel doubled batch silently |
| Kaggle run 2 | 8 epochs, attempted single GPU | 0.1332 | DataParallel STILL fired — env var placed after torch imports |
| **v3 run (next)** | Single GPU confirmed, v3 dataset | TBD | GPU bug fixed, WildReceipt added |

**Current stored best:** `trocr-smart-stock-best` — CER ~0.0856 (Colab LoRA run). This is the benchmark.

---

## Part 12: Bugs Fixed

### Bug 1 — CRITICAL: DataParallel not disabled (FIXED)
`os.environ["CUDA_VISIBLE_DEVICES"] = "0"` was placed after PyTorch was already imported, so Kaggle's 2×T4 silently ran DataParallel, doubling effective batch from 8→16. Optuna tuned at batch 4, so LR was always mismatched.

**Fix:** `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` is now the **very first line of Cell 1**, before any import including `from pathlib import Path`.

### Bug 2 — Test eval hangs on degenerate crops (FIXED)
Test set contains 1-pixel-wide images (`torch.Size([3, 42, 1])`). Beam search hangs on these — previous session spent 6+ hours on test eval and never completed.

**Fix:** Manual loop in Cell 16 skips any crop with `w < 4 or h < 4`.

### Bug 3 — PEFT checkpoint save (FIXED)
Without `LoRASaveCallback`, PEFT silently resets adapter weights on checkpoint reload. Callback saves `lora_adapter/` subdir alongside each checkpoint.

---

## Part 13: Improvement Techniques — Roadmap

### Tier 1 — High impact, do next
| Technique | Status | Notes |
|-----------|--------|-------|
| Fix GPU isolation bug | ✅ Done | `CUDA_VISIBLE_DEVICES` moved to Cell 1 top |
| WildReceipt integration | ✅ Done | v3 dataset, ~46,500 train examples |
| Encoder partial unfreeze | ⏳ After stable baseline | Unfreeze top 2–4 ViT blocks alongside LoRA — current performance ceiling |
| Re-run Optuna | ⏳ After epoch 1 confirmed | Previous LR was tuned against corrupted DataParallel val — may need retuning on v3 |

### Tier 2 — Medium impact
| Technique | Status | Notes |
|-----------|--------|-------|
| Label smoothing | Not yet | Helps with noisy WildReceipt labels. Value: 0.1 |
| LoRA rank increase | Not yet | r=16 → r=32, adds ~3M params, may help with WildReceipt diversity |
| Gradient accumulation | Commented out | `gradient_accumulation_steps=2` for effective batch 16 if OOM after encoder unfreeze |

### Tier 3 — Lower priority
| Technique | Status | Notes |
|-----------|--------|-------|
| Beam search tuning | Not yet | `num_beams=4` → 6–8 at inference, no retraining needed |
| WildReceipt weighting | Not yet | Currently 1× — monitor if it dominates training signal |

---

## Part 14: Runtime Estimate (v3, Single GPU T4)

| Phase | Duration |
|-------|----------|
| Dataset build (first time only) | ~30–40 min |
| Dataset load from disk (subsequent) | ~2 min |
| 8 epochs training, ~46,500 examples, batch 8 | ~10–11 hours |
| Test evaluation (manual loop, ~9,300 samples) | ~45–60 min |
| **Total first session** | **~12 hours** |

Use **Save & Run All** — not draft mode. Draft mode does not commit outputs to Kaggle.

---

## Part 15: Troubleshooting

| Issue | Fix |
|-------|-----|
| DataParallel fires despite `CUDA_VISIBLE_DEVICES` | Env var was set after torch import. Must be first line of first cell. |
| Test eval hangs for hours | Use manual loop with `w < 4 or h < 4` skip — never `trainer.evaluate()` on test set |
| LoRA adapter resets on resume | `LoRASaveCallback` missing or not attached to trainer |
| Stale checkpoints without `lora_adapter/` subdir | Trainer cell removes them: `shutil.rmtree(ckpt)` for any checkpoint missing `lora_adapter/` |
| Dataset rebuilds every session | `DATASET_SAVE.exists()` check skips rebuild — only triggers if v3 path is not mounted |
| OOM on batch 8 | Uncomment `gradient_accumulation_steps=2`, keep batch 8 |
| WildReceipt images not found | Check `WILDRECEIPT_DIR` path; only needed during dataset build, not inference |
| `AttributeError: 'TrOCRDataset' has no .select()` | Use `combined_dataset["train"].select(...)` (HF dataset) then wrap in `TrOCRDataset()` |
| `GaussNoise` var_limit warning | Use `A.GaussNoise(p=0.5)` without named args — API changed in albumentations |
| CORD train: 0 crops extracted | `quad` is in `valid_line[].words[].quad`, not in `gt_parse.menu` |
| fp16 scaler error on resume | Ensure `fp16=True` in training_args when resuming — scaler state in checkpoint requires it |

---

## Part 16: DistilBERT NER (Stage 2 — Not Yet Started)

CORD's structured `ground_truth` JSON maps directly to food NER schema: `nm` → `FOOD_ITEM`, `cnt` → `QUANTITY`, `price` → `PRICE`. No manual annotation needed. SROIE not used for NER (tags have zero overlap with food entities).

Full NER fine-tuning documented here once TrOCR CER ≤ 0.05 is achieved and validated.
