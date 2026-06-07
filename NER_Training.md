# NER_Training.md — DistilBERT NER Fine-Tuning Guide
## Smart-Stock: Stage 2 — Named Entity Recognition

**Version:** 1.2 (BIO logic fix for I-QTY, expanded synthetic generator v2)  
**Training Environment:** Kaggle (T4 GPU)  
**Model:** `distilbert-base-uncased` → token classification

---

## 1. Overview

Stage 2 takes raw OCR text lines (TrOCR output from Stage 1) and extracts structured food entities using a fine-tuned DistilBERT token classification model.

```
Input:  ["ORG", "STRWBRY", "1", "LB", "2.99"]
Output: [B-FOOD, I-FOOD, B-QTY, B-UNIT, B-PRICE]

Grouped: {food: "ORG STRWBRY", quantity: "1", unit: "LB", price: "2.99"}
```

Uses BIO (Beginning-Inside-Outside) tagging. Consecutive B/I tags of the same type are grouped into entity spans during post-processing.

---

## 2. Entity Labels

| Label | Meaning | Example token |
|---|---|---|
| `B-FOOD` | Beginning of food item | `STRWBRY` |
| `I-FOOD` | Inside food item (continuation) | `BRST` in `CHKN BRST` |
| `B-QTY` | Quantity | `1`, `2`, `0.5` |
| `B-UNIT` | Unit of measure | `LB`, `OZ`, `GAL`, `CT` |
| `B-PRICE` | Price | `2.99` |
| `O` | Outside / not an entity | store name, date, tax code |

**Note:** Only `B-` prefixes are used for single-token entities (QTY, UNIT, PRICE). Multi-token spans use `B-` for first token and `I-` for continuation. No `I-QTY`, `I-UNIT`, `I-PRICE` in schema.

Label mapping:
```python
LABEL2ID = {
    "O":       0,
    "B-FOOD":  1,
    "I-FOOD":  2,
    "B-QTY":   3,
    "B-UNIT":  4,
    "B-PRICE": 5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
```

---

## 3. Dataset Strategy

Three sources combined. Each contributes different signal:

| Dataset | HF Path | Size | Entities | Role |
|---|---|---|---|---|
| CORD v2 | `naver-clova-ix/cord-v2` | ~11k lines | FOOD, QTY, PRICE | Primary — receipt domain, structured annotations |
| TASTEset | `dmargutierrez/TASTESet` | 700 recipes | FOOD, QTY, UNIT | Auxiliary — English food names + UNIT labels |
| Synthetic | Generated in-notebook | 5,000 lines | FOOD, QTY, UNIT, PRICE | Gap filler — English grocery abbreviations |

**Why each source:**
- **CORD** has the only structured receipt annotations with FOOD/QTY/PRICE — primary source
- **TASTEset** is the only dataset with clean UNIT labels in English food context — fills CORD's gap
- **Synthetic** bridges the gap between recipe language ("two cups diced tomatoes") and receipt abbreviations ("DICED TOMATOE 2CT")

---

## 4. Dataset 1 — CORD v2

### 4.1 Schema

CORD's `ground_truth` is a JSON string. The `valid_line` field contains per-word annotations:

```python
# valid_line structure (confirmed from actual dataset inspection)
# category is on the LINE object, NOT on individual word objects
# all words in a line share the same category
{
  "valid_line": [
    {
      "category": "menu.cnt",   # <- entity category is HERE (line level)
      "group_id": 5,
      "sub_group_id": 0,
      "words": [
        {
          "text": "1",          # word objects have NO category field
          "quad": {"x1": 176, "y1": 624, "x2": 194, ...},
          "is_key": 0,
          "row_id": 2068524
        }
      ]
    },
    {
      "category": "menu.nm",
      "group_id": 3,
      "words": [
        {"text": "REAL",    "quad": {...}, "is_key": 0, "row_id": 2068522},
        {"text": "GANACHE", "quad": {...}, "is_key": 0, "row_id": 2068522}
      ]
    },
    {
      "category": "menu.price",
      "group_id": 3,
      "words": [
        {"text": "16,500", "quad": {...}, "is_key": 0, "row_id": 2068522}
      ]
    }
  ]
}
```

**Critical:** `ground_truth` is a raw JSON string — must `json.loads()` it before accessing.

### 4.2 CORD Loader

> `CORD_TO_ENTITY` is defined at the top of the loader cell — module-level so all functions can access it.

```python
import json
from datasets import load_dataset

# Module-level — must be defined before calling any loader or build function
CORD_TO_ENTITY = {
    "menu.nm":        "FOOD",
    "menu.sub_nm":    "FOOD",
    "menu.cnt":       "QTY",
    "menu.unitprice": "PRICE",
    "menu.price":     "PRICE",
}

def load_cord_ner() -> list[dict]:
    """
    Extract word-level NER annotations from CORD valid_line.
    Groups words by group_id into logical receipt lines.
    Returns list of {"tokens": [...], "ner_tags": [...]} dicts.
    """
    cord = load_dataset("naver-clova-ix/cord-v2")
    examples = []

    for split in ["train", "validation", "test"]:
        for sample in cord[split]:
            try:
                data = json.loads(sample["ground_truth"])
            except (json.JSONDecodeError, KeyError):
                continue

            valid_lines = data.get("valid_line", [])

            # Group by group_id — same group = same receipt line
            from collections import defaultdict
            groups = defaultdict(list)
            for line in valid_lines:
                groups[line["group_id"]].append(line)

            for gid, lines in groups.items():
                tokens, tags = [], []
                prev_entity = None

                for line in lines:
                    # category is on the line object — shared by all words in the line
                    category = line.get("category", "")
                    entity = CORD_TO_ENTITY.get(category, None)

                    for word in line.get("words", []):
                        text = word.get("text", "").strip()
                        if not text:
                            continue

                        if entity is None:
                            tags.append(LABEL2ID["O"])
                            prev_entity = None
                        elif entity == "FOOD" and entity == prev_entity:
                            # Only FOOD spans get I- continuation prefix
                            # QTY/UNIT/PRICE are always single-token → always B-
                            tags.append(LABEL2ID["I-FOOD"])
                        else:
                            tags.append(LABEL2ID[f"B-{entity}"])
                            prev_entity = entity

                        tokens.append(text)

                if tokens and any(t != LABEL2ID["O"] for t in tags):
                    examples.append({"tokens": tokens, "ner_tags": tags})

    print(f"CORD: {len(examples)} annotated lines")
    return examples
```

**Expected yield:** ~11,000 lines with at least one non-O entity.

---

## 5. Dataset 2 — TASTEset

### 5.1 Schema

TASTEset on HuggingFace (`dmargutierrez/TASTESet`) is **pre-tokenized** — it already has `input_ids`, `attention_mask`, `labels`, and `ner_tags` fields. The `recipes` field contains the original word list. The `ner_tags` field contains string labels like `"B-FOOD"`, `"B-UNIT"`, `"B-QUANTITY"`.

```python
# TASTEset example
{
  "recipes":  ["2", "cups", "all", "purpose", "flour"],
  "ner_tags": ["O", "B-QUANTITY", "B-UNIT", "B-FOOD", "I-FOOD", "I-FOOD"],
}
```

**Important:** TASTEset uses `B-QUANTITY` not `B-QTY`. Remap on load.

### 5.2 TASTEset Entity Mapping

| TASTEset tag | Smart-Stock tag | Include? |
|---|---|---|
| `B-FOOD` / `I-FOOD` | `B-FOOD` / `I-FOOD` | ✅ Yes |
| `B-QUANTITY` / `I-QUANTITY` | `B-QTY` | ✅ Yes (first token only) |
| `B-UNIT` / `I-UNIT` | `B-UNIT` | ✅ Yes (first token only) |
| `B-PHYSICAL_QUALITY` | `O` | ❌ Drop |
| `B-PROCESS` | `O` | ❌ Drop |
| `B-COLOR`, `B-TASTE`, etc. | `O` | ❌ Drop |

### 5.3 TASTEset Loader

```python
def load_tasteset_ner() -> list[dict]:
    """
    Load TASTEset and remap entity tags to Smart-Stock schema.
    TASTEset is pre-tokenized — use 'recipes' field as tokens,
    'ner_tags' field for labels.
    """
    TASTESET_MAP = {
        "B-FOOD":     "B-FOOD",
        "I-FOOD":     "I-FOOD",
        "B-QUANTITY": "B-QTY",
        "I-QUANTITY": "O",    # collapse multi-token quantities to first only
        "B-UNIT":     "B-UNIT",
        "I-UNIT":     "O",    # collapse multi-token units to first only
        # everything else → O
    }

    tasteset = load_dataset("dmargutierrez/TASTESet")
    examples = []

    for split in ["train", "test"]:
        for sample in tasteset[split]:
            tokens = sample["recipes"]
            raw_tags = sample["ner_tags"]

            if not tokens or not raw_tags:
                continue

            tags = [
                LABEL2ID.get(TASTESET_MAP.get(t, "O"), LABEL2ID["O"])
                for t in raw_tags[:len(tokens)]
            ]

            if any(t != LABEL2ID["O"] for t in tags):
                examples.append({"tokens": tokens, "ner_tags": tags})

    print(f"TASTEset: {len(examples)} annotated lines")
    return examples
```

**Expected yield:** ~680 examples (490 train + 210 test after filtering empty).

---

## 6. Dataset 3 — Synthetic Grocery Lines

Generates English grocery receipt abbreviations with ground-truth annotations. Fills the gap between recipe language and receipt abbreviations.

```python
# Full vocabulary — 200+ items across 10 categories + Pakistani and global brands
# See generate_synthetic_dataset() below — paste the full expanded generator here
# Key changes from v1:
# - ALL_FOOD_ITEMS replaces FOOD_ITEMS (200+ items vs 20)
# - 20 receipt layout styles vs 4
# - get_food_variant() adds modifier + structural corruption
# - add_ocr_noise() adds character + token-level noise
# - ASSERT token/tag length match at end of each style
# Full code: use the expanded generate_synthetic_dataset provided separately

UNITS = ["LB", "OZ", "GAL", "GL", "CT", "PK", "EA", "BAG", "BTL", "BX", "CAN", "DOZ", "PT", "QT"]
PRICE_RANGE = (0.49, 24.99)
```

---

## 7. Combined Dataset Builder

```python
from sklearn.model_selection import train_test_split

def build_ner_dataset() -> dict:
    """
    Combine all sources, split into train/val/test.
    Weights: CORD 1x, TASTEset 1x, Synthetic 0.8x
    """
    cord_data      = load_cord_ner()
    tasteset_data  = load_tasteset_ner()
    synthetic_data = generate_synthetic_dataset(5000)

    # Apply synthetic weight (0.8x)
    synthetic_sample = random.sample(synthetic_data, int(len(synthetic_data) * 0.8))

    all_data = cord_data + tasteset_data + synthetic_sample
    random.shuffle(all_data)

    # 80/10/10 split
    train_val, test = train_test_split(all_data, test_size=0.10, random_state=42)
    train, val      = train_test_split(train_val, test_size=0.111, random_state=42)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return {"train": train, "validation": val, "test": test}

ner_splits = build_ner_dataset()
```

---

## 8. Tokenization & Label Alignment

DistilBERT uses WordPiece tokenization which splits words into subword tokens. Labels must be aligned: first subword gets the real label, continuation subwords get `-100` (ignored in loss).

```python
from transformers import AutoTokenizer

MODEL_CHECKPOINT = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align(example: dict) -> dict:
    """
    Tokenize pre-split word list and align NER labels to subword tokens.

    Example:
      Input tokens:  ["STRWBRY",  "1",  "LB"]
      WordPiece:     ["str", "##wb", "##ry", "1", "lb"]
      Aligned tags:  [B-FOOD,     -100,  -100,  B-QTY, B-UNIT]
    """
    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        max_length=128,
        is_split_into_words=True,   # CRITICAL: input is pre-tokenized word list
    )

    word_ids = tokenized.word_ids()
    labels = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)              # [CLS] / [SEP] tokens
        elif word_id != prev_word_id:
            labels.append(example["ner_tags"][word_id])  # first subword → real label
        else:
            labels.append(-100)              # continuation subwords → ignore
        prev_word_id = word_id

    tokenized["labels"] = labels
    return tokenized


def build_hf_dataset(split_data: list[dict]):
    """Convert list of examples to HuggingFace Dataset and tokenize."""
    from datasets import Dataset
    ds = Dataset.from_list(split_data)
    ds = ds.map(
        tokenize_and_align,
        remove_columns=["tokens", "ner_tags"],
    )
    return ds

train_ds = build_hf_dataset(ner_splits["train"])
val_ds   = build_hf_dataset(ner_splits["validation"])
test_ds  = build_hf_dataset(ner_splits["test"])

print(f"Tokenized — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
```

---

## 9. Model Setup

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
```

---

## 10. Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./distilbert-ner-smart-stock",

    # Training schedule
    num_train_epochs=15,
    per_device_train_batch_size=32,    # NER is much lighter than TrOCR — batch 32 fits fine
    per_device_eval_batch_size=32,

    # Optimizer
    learning_rate=3e-5,
    warmup_ratio=0.06,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,

    # Eval & saving
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=3,

    # Performance
    fp16=True,

    # Logging
    logging_steps=50,
    log_level="info",
    report_to="none",
)
```

---

## 11. Evaluation — seqeval F1

Token accuracy is misleading for NER — `O` tokens dominate and inflate it. Always use entity-level F1 via seqeval.

```python
import evaluate
import numpy as np

seqeval = evaluate.load("seqeval")
label_names = list(LABEL2ID.keys())

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_preds = [
        [label_names[pred] for pred, lab in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[lab] for lab in label if lab != -100]
        for label in labels
    ]

    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": round(results["overall_precision"], 4),
        "recall":    round(results["overall_recall"],    4),
        "f1":        round(results["overall_f1"],        4),
        "accuracy":  round(results["overall_accuracy"],  4),
    }
```

---

## 12. Data Collator

```python
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    label_pad_token_id=-100,    # pads labels with -100 (ignored in loss)
)
```

---

## 13. Trainer

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

## 14. Save & Export

```python
from pathlib import Path

save_path = "/kaggle/working/distilbert-ner-smart-stock-best"

trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

# Verify files
expected = [
    "model.safetensors", "config.json",
    "tokenizer_config.json", "tokenizer.json",
    "vocab.txt", "special_tokens_map.json",
]
print(f"\nSaved to: {save_path}")
for fname in expected:
    fpath = Path(save_path) / fname
    exists = fpath.exists()
    size = fpath.stat().st_size / 1e6 if exists else 0
    print(f"  {'✅' if exists else '❌'} {fname} ({size:.1f} MB)")
```

---

## 15. Evaluate on Test Set

```python
results = trainer.evaluate(test_ds)
print(f"Test F1:        {results['eval_f1']:.4f}")
print(f"Test Precision: {results['eval_precision']:.4f}")
print(f"Test Recall:    {results['eval_recall']:.4f}")

# Target benchmarks:
# F1        ≥ 0.88
# Precision ≥ 0.90
# Recall    ≥ 0.86
```

---

## 16. Entity Post-Processing (Inference)

```python
def extract_entities(tokens: list[str], predictions: list[str]) -> dict:
    """
    Group BIO predictions into entity spans.

    Input:
      tokens:      ["ORG", "STRWBRY", "1", "LB", "2.99"]
      predictions: ["B-FOOD", "I-FOOD", "B-QTY", "B-UNIT", "B-PRICE"]

    Output:
      {"food_tokens": ["ORG", "STRWBRY"], "quantity": "1", "unit": "LB", "price": "2.99"}
    """
    entities = {"food_tokens": [], "quantity": None, "unit": None, "price": None}

    for token, pred in zip(tokens, predictions):
        if pred in ("B-FOOD", "I-FOOD"):
            entities["food_tokens"].append(token)
        elif pred == "B-QTY":
            entities["quantity"] = token
        elif pred == "B-UNIT":
            entities["unit"] = token
        elif pred == "B-PRICE":
            entities["price"] = token

    return entities
```

---

## 17. ONNX Export

```python
from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path=save_path,
    output="./models/distilbert_ner_onnx/",
    task="token-classification",
    opset=14,
)
# Rename output: distilbert_ner_onnx/model.onnx → distilbert_ner.onnx
# Matches ml_service/models/ structure from ML_Pipeline.md
```

**ONNX inference:**
```python
import onnxruntime as ort
import numpy as np

# Local path (during training session)
session = ort.InferenceSession("./models/distilbert_ner_onnx/model.onnx")
# From Kaggle dataset (future sessions)
# session = ort.InferenceSession("/kaggle/input/datasets/maazahmad69/distilbert-ner-onnx/model.onnx")

def run_ner_onnx(tokens: list[str]) -> list[str]:
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="np",
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    logits = session.run(
        None,
        {
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
    )[0]
    pred_ids = np.argmax(logits, axis=-1)[0]
    word_ids = tokenizer(tokens, is_split_into_words=True).word_ids()

    # Align back to word level — keep only first subword prediction per word
    word_preds = {}
    for idx, wid in enumerate(word_ids):
        if wid is not None and wid not in word_preds:
            word_preds[wid] = ID2LABEL[pred_ids[idx]]

    return [word_preds[i] for i in range(len(tokens))]
```

---

## 18. Optuna Hyperparameter Search

Run after confirming baseline F1 ≥ 0.75. Search on 20% of train data for speed.

```python
# !pip install optuna -q

def optuna_hp_space(trial):
    return {
        "learning_rate":               trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "warmup_ratio":                trial.suggest_float("warmup_ratio", 0.03, 0.15),
        "weight_decay":                trial.suggest_float("weight_decay", 0.0, 0.05),
    }

def model_init():
    return AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

search_train = train_ds.select(range(len(train_ds) // 5))

search_trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=search_train,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

best_run = search_trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=10,
    compute_objective=lambda metrics: metrics["eval_f1"],
)

print("Best hyperparameters:", best_run.hyperparameters)
# Apply best params and retrain full dataset if Δf1 > 0.02
for k, v in best_run.hyperparameters.items():
    setattr(training_args, k, v)
```

---

## 19. Kaggle Setup

### Required Packages
```bash
pip install transformers datasets evaluate seqeval scikit-learn optimum[onnxruntime] optuna
```

### Disk Usage
NER datasets are text-only — negligible disk vs TrOCR image tensors. Full pre-tokenized dataset fits in RAM (~100MB). No need for on-the-fly preprocessing or cache workarounds used in Stage 1.

### Save Strategy
- **Quick Save** → code only
- **Save & Run All** → commits `/kaggle/working/` as output
- **Yes, save models as Kaggle datasets** — same as OCR stage. Anything you want to reuse.

**What to save and dataset names:**

| Output | How to save |
|---|---|
| Everything in output (`distilbert-ner-smart-stock-best/`, `distilbert-ner-smart-stock/`, `models/`) | One combined Kaggle dataset named `distilbert-ner-smart-stock` — Kaggle bundles all output folders together |

**Loading in future sessions:**
```python
# All NER outputs are saved in one combined Kaggle dataset: distilbert-ner-smart-stock
# Subfolders inside it match the output folder names exactly

# Load best model
from transformers import AutoModelForTokenClassification, AutoTokenizer

NER_DATASET = "/kaggle/input/datasets/maazahmad69/distilbert-ner-smart-stock"

model = AutoModelForTokenClassification.from_pretrained(
    f"{NER_DATASET}/distilbert-ner-smart-stock-best"
)
tokenizer = AutoTokenizer.from_pretrained(
    f"{NER_DATASET}/distilbert-ner-smart-stock-best"
)

# Load ONNX model
import onnxruntime as ort
session = ort.InferenceSession(
    f"{NER_DATASET}/models/distilbert_ner_onnx/model.onnx"
)

# Resume training from checkpoint
resume_from = f"{NER_DATASET}/distilbert-ner-smart-stock/checkpoint-1365"
trainer.train(resume_from_checkpoint=resume_from)
```

> **Verify paths after attaching:** `os.listdir("/kaggle/input/datasets/maazahmad69/distilbert-ner-smart-stock/")`

> **Path format reminder:** Always verify with `os.listdir("/kaggle/input/datasets/maazahmad69/")` after attaching a dataset.

### Runtime Estimate (T4 GPU)
| Phase | Duration |
|---|---|
| Dataset loading + building | ~5 min |
| Tokenization | ~2 min |
| Training (15 epochs, ~13k examples) | ~30–45 min |
| Evaluation | ~5 min |
| **Total** | **< 1 hour** |

Fits easily in one Kaggle session.

---

## 20. Actual Training Results (Run 1)

| Epoch | Train Loss | Val Loss | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|
| 1 | 1.5827 | 0.5521 | 0.9596 | 0.6951 | 0.8062 | 0.7972 |
| 5 | 0.3682 | 0.3864 | 0.9441 | 0.7806 | 0.8546 | 0.8467 |
| 9 | 0.2908 | 0.3745 | 0.9026 | 0.8009 | 0.8487 | 0.8511 |
| 15 | 0.2347 | 0.3862 | 0.8794 | 0.8142 | 0.8455 | 0.8502 |

**Best checkpoint:** epoch 9 (F1: 0.8487, lowest val loss: 0.3745)

**Test set:** F1: 0.835 | Precision: 0.929 | Recall: 0.758

**Recall gap analysis:** Precision is strong (0.929) but recall lags target (0.758 vs 0.86). Model misses ~24% of entities. Likely causes: limited CORD receipt coverage (2,577 lines), synthetic data doesn't cover enough abbreviation variants. Fix: expand `FOOD_ITEMS` list to 200+ items with more abbreviation styles before next run.

---

## 21. Target Metrics

| Metric | Target |
|---|---|
| Entity-level F1 | ≥ 0.88 |
| Precision | ≥ 0.90 |
| Recall | ≥ 0.86 |
| ONNX inference (CPU, 128 tokens) | < 150ms |

---

## 21. Known Lessons (from OCR stage — avoid repeating)

- **`is_split_into_words=True`** is mandatory when passing pre-tokenized word lists to the tokenizer. Missing this causes wrong word boundary detection and broken label alignment.
- **`-100` masking** on continuation subwords is mandatory. Failing to do this inflates loss on unimportant positions and degrades F1.
- **seqeval not token accuracy** — `O` tokens dominate token-level accuracy and make it useless. Always report entity-level F1.
- **`eval_strategy` not `evaluation_strategy`** — transformers 4.46+ renamed this.
- **`ground_truth` is a JSON string** — `json.loads()` before accessing any keys.
- **TASTEset uses `B-QUANTITY` not `B-QTY`** — remap on load or labels won't match your schema.
- **ONNX load path:** `main_export` saves to `./models/distilbert_ner_onnx/model.onnx` — load with that exact path, not `models/distilbert_ner.onnx`.
- **`KeyError: 'I-QTY'`** — BIO logic in `load_cord_ner` was generating `I-QTY`/`I-UNIT`/`I-PRICE` continuation tags. Only `FOOD` entities span multiple tokens — QTY/UNIT/PRICE are always single-token, always use `B-` prefix. Fixed in loader.
- **SROIE NER tags** (COMPANY/DATE/ADDRESS/TOTAL) have no food entity overlap — don't use them for NER training.
- **Kaggle disk:** NER datasets are tiny compared to image datasets. No disk issues expected here.

---

## 22. Error Analysis Checklist

After training, inspect failure cases:

| Check | What to look for |
|---|---|
| FOOD boundary errors | Multi-word items split wrong (e.g., "CHKN" B-FOOD but "BRST" predicted O) |
| QTY/UNIT confusion | "1LB" as one token predicted B-QTY instead of being split |
| Price false positives | Dates ("01/25") or product codes predicted B-PRICE |
| Domain gap | Model failing on abbreviations not seen in CORD/TASTEset |
| OOV tokens | New abbreviations split into garbage subwords — check with tokenizer first |
