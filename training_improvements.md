# TrOCR Training Improvements

## Root cause recap

Last run: Optuna tuned at batch size 4 → real training ran at effective batch size 16 (DataParallel across 2 GPUs) with the same LR. Result: model never learned, CER went from 0.132 → 0.137 across 5 epochs.

All fixes below address this directly or build on top of a corrected baseline.

---

## Blocks to skip (already done, output in input)

Comment out or do not run:
- **CORD/SROIE crop extractors** — dataset already built
- **Dataset builder** — `smart_stock_dataset_v2` already in input
- **Augmentation** — still needed (runs inline per batch, not pre-built)
- **Optuna** — skip this run, LR is hardcoded below from prior results

---

## Run order for this session

1. Setup & Paths
2. Augmentation
3. Preprocessing Function
4. Model Setup
5. Training Arguments *(replace with updated block below)*
6. Metrics
7. Trainer and Training *(replace with updated block below)*
8. Training Curves
9. Save & Export *(use fixed block below)*
10. Evaluate on Test Set
11. Qualitative Evaluation

---

## Technique 1 — Force single GPU + fix LR (most important, run first)

DataParallel doubled effective batch size silently. Forcing single GPU makes real training match Optuna's conditions exactly.

Replace the existing **Training Arguments** block:

```python
import os
from transformers import Seq2SeqTrainingArguments

# Force single GPU — Optuna tuned at batch size 4 on single GPU.
# DataParallel was silently running batch size 16 (4x mismatch).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

training_args = Seq2SeqTrainingArguments(
    output_dir=str(CHECKPOINT_DIR),

    num_train_epochs=8,                # more epochs — single GPU converges slower but cleaner
    per_device_train_batch_size=8,     # effective batch = 8 now (no DataParallel)
    per_device_eval_batch_size=8,

    # Optuna best from Kaggle run (Trial 0 — CER 0.088 on subset)
    learning_rate=1.4824e-4,
    warmup_ratio=0.02672,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
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

    logging_dir=str(WORKING_DIR / "logs"),
    logging_steps=50,
    log_level="info",
    report_to="none",
)

print(f"Effective batch size: {training_args.per_device_train_batch_size}")
print(f"GPU count visible: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
```

---

## Technique 2 — Gradient accumulation (combine with Technique 1)

Instead of relying on DataParallel for larger effective batch, use gradient accumulation to get a larger effective batch on single GPU without changing LR tuning conditions. Add `gradient_accumulation_steps=2` to Training Arguments above:

```python
# Add this line inside Seq2SeqTrainingArguments (after max_grad_norm):
gradient_accumulation_steps=2,        # effective batch = 8 × 2 = 16, but LR stays valid
                                       # because accumulation doesn't change the optimizer step rate
                                       # the way DataParallel does
```

Note: only add this if you want faster convergence from larger effective batch. If Technique 1 alone already produces improving CER, leave it out.

---

## Technique 3 — Cosine with restarts scheduler

Your current cosine schedule decays to near-zero LR by epoch 3-4. The model stagnates because the LR is too small to escape local minima in later epochs. Restarts periodically reset LR to kick the optimizer out.

Change one line in Training Arguments:

```python
lr_scheduler_type="cosine_with_restarts",
# Also add:
lr_scheduler_kwargs={"num_cycles": 2},   # 2 restarts over total training
```

This is best combined with Technique 1 (single GPU + correct LR). Don't use this alone without fixing the batch size issue first.

---

## Technique 4 — Resume from best LoRA checkpoint in input

Your input directory already has `trocr-smart-stock/checkpoint-13594`, `checkpoint-15536`, `checkpoint-17478`. These are LoRA checkpoints from the last Colab run (CER ~0.138). They're not great, but they're a better starting point than fresh LoRA on the base model — the adapter has already learned receipt-domain patterns.

Replace the existing **Model Setup** block:

```python
from pathlib import Path
from transformers import VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Input checkpoint dir from your uploaded dataset
INPUT_CHECKPOINT_DIR = Path("/kaggle/input/smart-stock-dataset/trocr-smart-stock")

base_model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_INPUT))

for param in base_model.encoder.parameters():
    param.requires_grad = False

# Priority order:
# 1. Working dir checkpoints (current session, most recent)
# 2. Input dir checkpoints (uploaded from Colab — fallback)
# 3. Fresh LoRA (no prior training)

def find_lora_checkpoints(directory: Path):
    """Find checkpoints with LoRA adapter weights in either format."""
    return sorted([
        ckpt for ckpt in directory.glob("checkpoint-*")
        if (ckpt / "lora_adapter").exists() or (ckpt / "adapter_config.json").exists()
    ])

working_checkpoints = find_lora_checkpoints(CHECKPOINT_DIR)
input_checkpoints   = find_lora_checkpoints(INPUT_CHECKPOINT_DIR) if INPUT_CHECKPOINT_DIR.exists() else []

if working_checkpoints:
    latest = working_checkpoints[-1]
    adapter_path = latest / "lora_adapter" if (latest / "lora_adapter").exists() else latest
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)
    print(f"Resumed from working dir: {adapter_path}")
elif input_checkpoints:
    latest = input_checkpoints[-1]
    adapter_path = latest / "lora_adapter" if (latest / "lora_adapter").exists() else latest
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)
    print(f"Resumed from input dir: {adapter_path}")
else:
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    print("No adapter checkpoint found — fresh LoRA applied")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.vocab_size             = model.config.decoder.vocab_size
model.generation_config.eos_token_id         = processor.tokenizer.sep_token_id
model.generation_config.early_stopping       = True
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty       = 2.0
model.generation_config.num_beams            = 4
```

---

## Technique 5 — Trainer with early stopping

If CER stops improving for 2 consecutive epochs, training stops automatically instead of wasting compute. Replace the existing **Trainer and Training** block:

```python
from transformers import Seq2SeqTrainer, TrainerCallback, EarlyStoppingCallback
import shutil

class LoRASaveCallback(TrainerCallback):
    """Save LoRA adapter alongside each Trainer checkpoint."""
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        adapter_dir = checkpoint_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        kwargs["model"].save_pretrained(str(adapter_dir))
        print(f"LoRA adapter saved to {adapter_dir}")
        return control

print(f"LR:           {training_args.learning_rate}")
print(f"Warmup ratio: {training_args.warmup_ratio}")
print(f"Epochs:       {training_args.num_train_epochs}")
print(f"Batch size:   {training_args.per_device_train_batch_size}")

# Remove stale checkpoints from prior sessions that have no LoRA adapter
if CHECKPOINT_DIR.exists():
    for ckpt in CHECKPOINT_DIR.glob("checkpoint-*"):
        has_adapter = (ckpt / "lora_adapter").exists() or (ckpt / "adapter_config.json").exists()
        if not has_adapter:
            shutil.rmtree(ckpt)
            print(f"Removed stale checkpoint: {ckpt}")

# Enable early stopping — needs load_best_model_at_end=True and metric_for_best_model set
# Update training_args for this:
training_args.load_best_model_at_end = True
training_args.metric_for_best_model  = "cer"
training_args.greater_is_better      = False
# Early stopping requires eval_strategy == save_strategy, align them:
training_args.save_strategy          = "epoch"
training_args.save_steps             = None

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[
        LoRASaveCallback(),
        EarlyStoppingCallback(early_stopping_patience=2),  # stop if CER doesn't improve for 2 epochs
    ],
)

# Resume from latest valid checkpoint if available in working dir
valid_checkpoints = sorted([
    ckpt for ckpt in CHECKPOINT_DIR.glob("checkpoint-*")
    if (ckpt / "lora_adapter").exists() or (ckpt / "adapter_config.json").exists()
])
resume_from = str(valid_checkpoints[-1]) if valid_checkpoints else None

if resume_from:
    print(f"Resuming from: {resume_from}")
else:
    print("Starting fresh training")

trainer.train(resume_from_checkpoint=resume_from)
```

---

## Fixed Save & Export block

```python
from peft import PeftModel

merged_model = model.merge_and_unload()

for save_path in [str(BEST_MODEL_DIR)]:
    merged_model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    merged_model.generation_config.save_pretrained(save_path)
    print(f"Saved to: {save_path}")

expected_files = [
    "model.safetensors", "config.json", "generation_config.json",
    "tokenizer_config.json", "tokenizer.json", "processor_config.json",
]
print(f"\nVerification ({BEST_MODEL_DIR}):")
for fname in expected_files:
    fpath = BEST_MODEL_DIR / fname
    exists = fpath.exists()
    size = fpath.stat().st_size / 1e6 if exists else 0
    print(f"  {'✅' if exists else '❌'} {fname} ({size:.1f} MB)")
```

---

## What to download after training

From `/kaggle/working/`:
- `trocr-smart-stock-best/` — the merged final model (all 6 files). Upload this as a new Kaggle dataset version to replace the current `trocr-smart-stock-best` input dataset.
- `trocr-smart-stock/checkpoint-{latest}/lora_adapter/` — the raw LoRA adapter in case you want to continue training in a future session without re-merging.

Only create a new Kaggle dataset (`trocr-smart-stock-best-v2`) if the final CER is **better than 0.0856** (your current best from Colab LoRA run). Otherwise keep the existing input dataset unchanged.
