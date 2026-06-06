# NER_Training.md — DistilBERT NER Fine-Tuning Guide
## Smart-Stock: Stage 2 — Named Entity Recognition

**Version:** 1.0  
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
| SROIE | `sizhkhy/SROIE` | 626+347 receipts | (domain only) | Domain adapter — all tokens labeled O |
| Synthetic | Generated in-notebook | 5,000 lines | FOOD, QTY, UNIT, PRICE | Gap filler — English grocery abbreviations |

**Why each source:**
- **CORD** has the only structured receipt annotations with FOOD/QTY/PRICE — primary source
- **TASTEset** is the only dataset with clean UNIT labels in English food context — fills CORD's gap
- **SROIE** has no food entity labels but exposes the model to real receipt formatting (abbreviations, thermal font artifacts) — domain adaptation only, all tokens O
- **Synthetic** bridges the gap between recipe language ("two cups diced tomatoes") and receipt abbreviations ("DICED TOMATOE 2CT")

---

## 4. Dataset 1 — CORD v2

### 4.1 Schema

CORD's `ground_truth` is a JSON string. The `valid_line` field contains per-word annotations:

```python
# valid_line structure (confirmed from actual dataset inspection)
{
  "valid_line": [
    {
      "words": [
        {
          "text": "REAL",
          "category": "menu.nm",   # <- entity category
          "quad": {...},
          "row_id": 2068522
        },
        {
          "text": "GANACHE",
          "category": "menu.nm"
        },
        {
          "text": "1",
          "category": "menu.cnt"
        },
        {
          "text": "16,500",
          "category": "menu.price"
        }
      ],
      "category": "menu",
      "group_id": 3
    }
  ]
}
```

**Critical:** `ground_truth` is a raw JSON string — must `json.loads()` it before accessing.

### 4.2 Category → BIO Mapping

```python
CORD_TO_ENTITY = {
    "menu.nm":        "FOOD",
    "menu.sub_nm":    "FOOD",   # sub-item / modifier
    "menu.cnt":       "QTY",
    "menu.unitprice": "PRICE",
    "menu.price":     "PRICE",
    # everything else → O (total.*, sub_total.*, void.*, etc.)
}
```

### 4.3 CORD Loader

```python
import json
from datasets import load_dataset

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
                    for word in line.get("words", []):
                        text = word.get("text", "").strip()
                        if not text:
                            continue

                        category = word.get("category", "")
                        entity = CORD_TO_ENTITY.get(category, None)

                        if entity is None:
                            tags.append(LABEL2ID["O"])
                            prev_entity = None
                        elif entity == prev_entity:
                            tags.append(LABEL2ID[f"I-{entity}"])
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

## 6. Dataset 3 — SROIE (Domain Adaptation)

SROIE's NER tags are COMPANY, ADDRESS, DATE, TOTAL — not food entities. But real receipt text patterns (abbreviations, price formatting, receipt structure) are valuable for domain adaptation.

**Approach:** Use all SROIE words with all labels set to `O`. This teaches the model what receipt text looks like without introducing wrong entity associations.

```python
def load_sroie_domain() -> list[dict]:
    """
    Load SROIE as domain adaptation data.
    All tokens labeled O — teaches receipt text patterns, not entity types.
    """
    sroie = load_dataset("sizhkhy/SROIE")
    examples = []

    for split in ["train", "test"]:
        for sample in sroie[split]:
            words = sample.get("words", [])
            if not words:
                continue
            tokens = [w.strip() for w in words if w.strip()]
            tags = [LABEL2ID["O"]] * len(tokens)
            if tokens:
                examples.append({"tokens": tokens, "ner_tags": tags})

    print(f"SROIE domain: {len(examples)} lines")
    return examples
```

---

## 7. Dataset 4 — Synthetic Grocery Lines

Generates English grocery receipt abbreviations with ground-truth annotations. Fills the gap between recipe language and receipt abbreviations.

```python
import random

FOOD_ITEMS = [
    ("Strawberries", "STRWBRY"), ("Whole Milk", "WHL MLK"),
    ("Chicken Breast", "CHKN BRST"), ("Greek Yogurt", "GRK YGRT"),
    ("Organic Eggs", "ORG EGGS"), ("Cheddar Cheese", "CHDR CH"),
    ("Salmon Fillet", "SLMN FLT"), ("Baby Spinach", "BBY SPNCH"),
    ("Orange Juice", "OJ"), ("Pasta Sauce", "PST SCE"),
    ("Ground Beef", "GRD BEEF"), ("Butter", "BTTR"),
    ("Mozzarella", "MOZZ"), ("Avocado", "AVOCDO"),
    ("Blueberries", "BLUBRY"), ("Almond Milk", "ALMD MLK"),
    ("Chicken Thighs", "CHKN THGH"), ("Pork Chops", "PRK CHPS"),
    ("Sweet Potatoes", "SWT POTATO"), ("Brown Rice", "BRN RICE"),
    # Expand to 100+ items before training
]

UNITS = ["LB", "OZ", "GAL", "GL", "CT", "PK", "EA", "BAG", "BTL", "BX", "CAN"]
PRICE_RANGE = (0.99, 14.99)

def generate_synthetic_line() -> dict:
    """Generate one synthetic receipt line with BIO annotations."""
    food_full, food_abbr = random.choice(FOOD_ITEMS)
    qty = random.randint(1, 5)
    unit = random.choice(UNITS)
    price = round(random.uniform(*PRICE_RANGE), 2)

    style = random.randint(0, 3)

    if style == 0:
        # "STRWBRY 1 LB 2.99"
        tokens = food_abbr.split() + [str(qty), unit, str(price)]
        n_food = len(food_abbr.split())
        tags = (
            [LABEL2ID["B-FOOD"]] +
            [LABEL2ID["I-FOOD"]] * (n_food - 1) +
            [LABEL2ID["B-QTY"], LABEL2ID["B-UNIT"], LABEL2ID["B-PRICE"]]
        )
    elif style == 1:
        # "ORG STRWBRY 1LB 2.99"
        food_tokens = ("ORG " + food_abbr).split()
        combined_qty_unit = f"{qty}{unit}"
        tokens = food_tokens + [combined_qty_unit, str(price)]
        n_food = len(food_tokens)
        tags = (
            [LABEL2ID["B-FOOD"]] +
            [LABEL2ID["I-FOOD"]] * (n_food - 1) +
            [LABEL2ID["B-QTY"], LABEL2ID["B-PRICE"]]
        )
    elif style == 2:
        # "STRAWBERRIES 1 LB @2.99/LB"
        tokens = food_full.upper().split() + [str(qty), unit]
        n_food = len(food_full.split())
        tags = (
            [LABEL2ID["B-FOOD"]] +
            [LABEL2ID["I-FOOD"]] * (n_food - 1) +
            [LABEL2ID["B-QTY"], LABEL2ID["B-UNIT"]]
        )
    else:
        # "2 CHKN BRST 5.99"
        food_tokens = food_abbr.split()
        tokens = [str(qty)] + food_tokens + [str(price)]
        n_food = len(food_tokens)
        tags = (
            [LABEL2ID["B-QTY"]] +
            [LABEL2ID["B-FOOD"]] +
            [LABEL2ID["I-FOOD"]] * (n_food - 1) +
            [LABEL2ID["B-PRICE"]]
        )

    return {"tokens": tokens, "ner_tags": tags}


def add_ocr_noise(tokens: list[str], prob: float = 0.05) -> list[str]:
    """Simulate common TrOCR output errors."""
    noise_map = {"0": "O", "O": "0", "1": "I", "l": "1", "S": "5"}
    noisy = []
    for token in tokens:
        chars = list(token)
        for i, c in enumerate(chars):
            if random.random() < prob and c in noise_map:
                chars[i] = noise_map[c]
        noisy.append("".join(chars))
    return noisy


def generate_synthetic_dataset(n: int = 5000) -> list[dict]:
    examples = []
    for _ in range(n):
        ex = generate_synthetic_line()
        ex["tokens"] = add_ocr_noise(ex["tokens"])
        examples.append(ex)
    print(f"Synthetic: {len(examples)} lines generated")
    return examples
```

---

## 8. Combined Dataset Builder

```python
from sklearn.model_selection import train_test_split

def build_ner_dataset() -> dict:
    """
    Combine all sources, split into train/val/test.
    Weights: CORD 1x, TASTEset 1x, SROIE 0.3x, Synthetic 0.8x
    """
    cord_data     = load_cord_ner()
    tasteset_data = load_tasteset_ner()
    sroie_data    = load_sroie_domain()
    synthetic_data = generate_synthetic_dataset(5000)

    # Apply SROIE weight (0.3x — domain signal only)
    sroie_sample = random.sample(sroie_data, int(len(sroie_data) * 0.3))

    # Apply synthetic weight (0.8x)
    synthetic_sample = random.sample(synthetic_data, int(len(synthetic_data) * 0.8))

    all_data = cord_data + tasteset_data + sroie_sample + synthetic_sample
    random.shuffle(all_data)

    # 80/10/10 split
    train_val, test = train_test_split(all_data, test_size=0.10, random_state=42)
    train, val      = train_test_split(train_val, test_size=0.111, random_state=42)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return {"train": train, "validation": val, "test": test}

ner_splits = build_ner_dataset()
```

---

## 9. Tokenization & Label Alignment

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

## 10. Model Setup

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

## 11. Training Arguments

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

## 12. Evaluation — seqeval F1

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

## 13. Data Collator

```python
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    label_pad_token_id=-100,    # pads labels with -100 (ignored in loss)
)
```

---

## 14. Trainer

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

## 15. Save & Export

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

## 16. Evaluate on Test Set

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

## 17. Entity Post-Processing (Inference)

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

## 18. ONNX Export

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

session = ort.InferenceSession("models/distilbert_ner.onnx")

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

## 19. Optuna Hyperparameter Search

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

## 20. Kaggle Setup

### Required Packages
```bash
pip install transformers datasets evaluate seqeval scikit-learn optimum[onnxruntime] optuna
```

### Disk Usage
NER datasets are text-only — negligible disk vs TrOCR image tensors. Full pre-tokenized dataset fits in RAM (~100MB). No need for on-the-fly preprocessing or cache workarounds used in Stage 1.

### Save Strategy
- **Quick Save** → code only
- **Save & Run All** → commits `/kaggle/working/` as output
- Save best model as Kaggle dataset: `distilbert-ner-smart-stock-best`

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

## 21. Target Metrics

| Metric | Target |
|---|---|
| Entity-level F1 | ≥ 0.88 |
| Precision | ≥ 0.90 |
| Recall | ≥ 0.86 |
| ONNX inference (CPU, 128 tokens) | < 150ms |

---

## 22. Known Lessons (from OCR stage — avoid repeating)

- **`is_split_into_words=True`** is mandatory when passing pre-tokenized word lists to the tokenizer. Missing this causes wrong word boundary detection and broken label alignment.
- **`-100` masking** on continuation subwords is mandatory. Failing to do this inflates loss on unimportant positions and degrades F1.
- **seqeval not token accuracy** — `O` tokens dominate token-level accuracy and make it useless. Always report entity-level F1.
- **`eval_strategy` not `evaluation_strategy`** — transformers 4.46+ renamed this.
- **`ground_truth` is a JSON string** — `json.loads()` before accessing any keys.
- **TASTEset uses `B-QUANTITY` not `B-QTY`** — remap on load or labels won't match your schema.
- **Kaggle disk:** NER datasets are tiny compared to image datasets. No disk issues expected here.

---

## 23. Error Analysis Checklist

After training, inspect failure cases:

| Check | What to look for |
|---|---|
| FOOD boundary errors | Multi-word items split wrong (e.g., "CHKN" B-FOOD but "BRST" predicted O) |
| QTY/UNIT confusion | "1LB" as one token predicted B-QTY instead of being split |
| Price false positives | Dates ("01/25") or product codes predicted B-PRICE |
| Domain gap | Model failing on abbreviations not seen in CORD/TASTEset |
| OOV tokens | New abbreviations split into garbage subwords — check with tokenizer first |
