# OCR_Training_Colab.md — TrOCR Fine-Tuning on Google Colab
## Smart-Stock: Colab Migration Guide

**Version:** 2.0  
**Base document:** `Model_Training.md` (Kaggle version) — this doc only shows what changes. All code not listed here stays identical.  
**Dataset:** `maazahmad69/smart-stock-dataset` on Kaggle → imported into Colab via Kaggle API

---

## What Changes vs Kaggle

| Kaggle | Colab |
|---|---|
| `/kaggle/input/datasets/...` | `/content/smart-stock-dataset/` |
| `/kaggle/working/` | `/content/drive/MyDrive/SmartStock/` (persistent) or `/content/` (session only) |
| Outputs saved via "Save & Run All" | Manually copy to Google Drive |
| GPU T4 free tier (12hr limit) | GPU T4 free tier (session limit varies) |
| Dataset attached via UI | Downloaded via Kaggle API |

---

## Cell 0 — Run First: Environment Setup

Add this as the very first cell in your Colab notebook:

```python
# ── Install dependencies ────────────────────────────────────────────────────
!pip install -q transformers datasets evaluate jiwer albumentations kaggle

# ── Mount Google Drive (persistent storage) ─────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# Create persistent output dir on Drive
import os
DRIVE_DIR = "/content/drive/MyDrive/SmartStock"
os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/trocr-smart-stock-best", exist_ok=True)
print(f"Drive mounted. Output dir: {DRIVE_DIR}")

# ── Import Kaggle dataset ────────────────────────────────────────────────────
from google.colab import files

# Upload your kaggle.json API token (one-time per session)
# Get it from: kaggle.com → Account → API → Create New Token
files.upload()  # select kaggle.json when prompted

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download smart-stock-dataset from Kaggle
!kaggle datasets download -d maazahmad69/smart-stock-dataset -p /content/
!unzip -q /content/smart-stock-dataset.zip -d /content/smart-stock-dataset/
!rm /content/smart-stock-dataset.zip

# Verify contents
import os
for item in sorted(os.listdir("/content/smart-stock-dataset")):
    print(item)
```

Expected output after unzip:
```
smart_stock_dataset_v2/
trocr-smart-stock/
```

---

## Cell 1 — Replace: SAVE_PATH definition

**Find this in your notebook (section 1.2 Combined Dataset Builder):**
```python
SAVE_PATH = Path("/kaggle/working/smart_stock_dataset_v2")
```

**Replace with:**
```python
# Colab: dataset is already extracted from Kaggle download
SAVE_PATH = Path("/content/smart-stock-dataset/smart_stock_dataset_v2")
if not SAVE_PATH.exists():
    SAVE_PATH = Path("/content/smart_stock_dataset_v2")  # fallback if rebuilding fresh
```

---

## Cell 2 — Replace: build_and_save_dataset save path

**Inside `build_and_save_dataset()`, find:**
```python
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(SAVE_PATH))
```

**Replace with:**
```python
    # Save to Drive for persistence across sessions
    drive_save_path = Path("/content/drive/MyDrive/SmartStock/smart_stock_dataset_v2")
    drive_save_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(drive_save_path))
    SAVE_PATH = drive_save_path  # update SAVE_PATH to Drive location
```

---

## Cell 3 — Replace: training_args output_dir

**Find:**
```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-smart-stock",
```

**Replace with:**
```python
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/SmartStock/trocr-smart-stock",
```

Saving checkpoints directly to Drive means they survive session disconnects.

---

## Cell 4 — Replace: Checkpoint resume block

**Find:**
```python
# Resume from checkpoint — checks Kaggle input dataset first, then local working dir
CHECKPOINT_INPUT = Path("/kaggle/input/datasets/maazahmad69/smart-stock-dataset/trocr-smart-stock")
checkpoint_dir   = CHECKPOINT_INPUT if CHECKPOINT_INPUT.exists() else Path("./trocr-smart-stock")

checkpoints = sorted(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
resume_from = str(checkpoints[-1]) if checkpoints else None

if resume_from:
    print(f"Resuming from: {resume_from}")
else:
    print("No checkpoint found — training from scratch")

trainer.train(resume_from_checkpoint=resume_from)
```

**Replace with:**
```python
# Resume from checkpoint — check Drive first (persistent), then Kaggle download, then local
from pathlib import Path

CHECKPOINT_DRIVE  = Path("/content/drive/MyDrive/SmartStock/trocr-smart-stock")
CHECKPOINT_KAGGLE = Path("/content/smart-stock-dataset/trocr-smart-stock")

if CHECKPOINT_DRIVE.exists() and any(CHECKPOINT_DRIVE.glob("checkpoint-*")):
    checkpoint_dir = CHECKPOINT_DRIVE
    print("Using Drive checkpoints")
elif CHECKPOINT_KAGGLE.exists() and any(CHECKPOINT_KAGGLE.glob("checkpoint-*")):
    checkpoint_dir = CHECKPOINT_KAGGLE
    print("Using Kaggle-downloaded checkpoints")
else:
    checkpoint_dir = None

if checkpoint_dir:
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
    resume_from = str(checkpoints[-1])
    print(f"Resuming from: {resume_from}")
else:
    resume_from = None
    print("No checkpoint found — training from scratch")

trainer.train(resume_from_checkpoint=resume_from)
```

---

## Cell 5 — Replace: Save & Export block

**Find:**
```python
save_path = "/kaggle/working/trocr-smart-stock-best"
```

**Replace entire save block with:**
```python
from pathlib import Path

# Save to both Drive (persistent) and local /content/ (for this session)
DRIVE_SAVE = "/content/drive/MyDrive/SmartStock/trocr-smart-stock-best"
LOCAL_SAVE = "/content/trocr-smart-stock-best"

for save_path in [DRIVE_SAVE, LOCAL_SAVE]:
    trainer.save_model(save_path)
    processor.save_pretrained(save_path)
    model.generation_config.save_pretrained(save_path)
    print(f"Saved to: {save_path}")

# Verify Drive copy
expected_files = [
    "model.safetensors", "config.json", "generation_config.json",
    "tokenizer_config.json", "tokenizer.json", "processor_config.json",
]
print(f"\nDrive save verification ({DRIVE_SAVE}):")
for fname in expected_files:
    fpath = Path(DRIVE_SAVE) / fname
    exists = fpath.exists()
    size = fpath.stat().st_size / 1e6 if exists else 0
    print(f"  {'✅' if exists else '❌'} {fname} ({size:.1f} MB)")
```

---

## Cell 6 — Replace: "Loading in future sessions" paths

**Find in section 1.9:**
```python
# Line-crop dataset (built once, reuse forever)
SAVE_PATH = Path("/kaggle/input/datasets/maazahmad69/smart-stock-dataset-v2/smart_stock_dataset_v2")
if not SAVE_PATH.exists():
    SAVE_PATH = Path("/kaggle/working/smart_stock_dataset_v2")  # fallback if rebuilding

# Resume training from saved checkpoint
resume_from = "/kaggle/input/datasets/maazahmad69/smart-stock-dataset/trocr-smart-stock/checkpoint-15536"

# Load final trained model (after training completes)
model = VisionEncoderDecoderModel.from_pretrained(
    "/kaggle/input/datasets/maazahmad69/trocr-smart-stock-best/trocr-smart-stock-best"
)
processor = TrOCRProcessor.from_pretrained(
    "/kaggle/input/datasets/maazahmad69/trocr-smart-stock-best/trocr-smart-stock-best"
)
```

**Replace with:**
```python
# In future Colab sessions — mount Drive first, then load from there

# Dataset
SAVE_PATH = Path("/content/drive/MyDrive/SmartStock/smart_stock_dataset_v2")
if not SAVE_PATH.exists():
    # Dataset not on Drive yet — re-download from Kaggle (see Cell 0)
    SAVE_PATH = Path("/content/smart-stock-dataset/smart_stock_dataset_v2")

# Resume from latest checkpoint on Drive
checkpoint_dir = Path("/content/drive/MyDrive/SmartStock/trocr-smart-stock")
checkpoints = sorted(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
resume_from = str(checkpoints[-1]) if checkpoints else None

# Load final trained model
model = VisionEncoderDecoderModel.from_pretrained(
    "/content/drive/MyDrive/SmartStock/trocr-smart-stock-best"
)
processor = TrOCRProcessor.from_pretrained(
    "/content/drive/MyDrive/SmartStock/trocr-smart-stock-best"
)
```

---

## Cell 7 — Replace: Optuna search paths

**Find inside Optuna section:**
```python
search_hf    = combined_dataset["train"].select(range(len(combined_dataset["train"]) // 5))
```

No path change needed here — `combined_dataset` is already loaded in memory. No cell change required.

---

## Persistence Strategy on Colab

Colab's `/content/` is wiped on session disconnect. Google Drive is permanent.

| Data | Where to save | Survives disconnect? |
|---|---|---|
| `smart_stock_dataset_v2` | Drive: `SmartStock/smart_stock_dataset_v2/` | ✅ Yes |
| Checkpoints | Drive: `SmartStock/trocr-smart-stock/` | ✅ Yes |
| Best model | Drive: `SmartStock/trocr-smart-stock-best/` | ✅ Yes |
| HuggingFace cache | `/content/` (re-downloads each session) | ❌ No |

**Rule:** anything you want to keep → save to `/content/drive/MyDrive/SmartStock/`.

---

## Colab-Specific Gotchas

| Issue | Fix |
|---|---|
| Colab disconnects mid-training | Checkpoints save to Drive every epoch (`save_strategy="epoch"`) — resume from Drive on reconnect |
| `kaggle.json` upload required every session | One-time per session — keep the file locally and re-upload when Colab restarts |
| Drive mount slow on large reads | The `smart_stock_dataset_v2` Arrow files are read via Drive — can be slow. Copy to `/content/` at session start for faster access: `!cp -r /content/drive/MyDrive/SmartStock/smart_stock_dataset_v2 /content/` |
| HuggingFace models re-downloaded each session | Normal — they're cached in `/content/` which doesn't persist. ~7s download on Colab. |
| GPU not available | Runtime → Change runtime type → T4 GPU |
| Colab free tier session limit | ~12 hours max. Same as Kaggle. Resume from Drive checkpoints on next session. |
| `fp16=True` fails on some Colab GPUs | If you get a CUDA error on fp16, add `fp16_backend="auto"` to training_args |
