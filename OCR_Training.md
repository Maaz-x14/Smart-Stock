I am building SmartStock, an ML pipeline for receipt intelligence. The full pipeline is:
Receipt Image → TrOCR OCR (Stage 1) → DistilBERT NER (Stage 2) → Normalization → Expiry Prediction
Stage 2 (NER) is already trained at F1 0.907 across 6 classes. I am currently working on Stage 1 — fine-tuning TrOCR for receipt OCR.

## Stage 1 — TrOCR Overview

Base model: microsoft/trocr-base-printed
Adapter: LoRA on decoder only (q_proj, v_proj, all 12 layers, r=16, lora_alpha=32)
Encoder: fully frozen (current performance ceiling — see Roadmap)
Trainable params: 1,523,712 of 335M (0.45%)
Infrastructure: Kaggle T4 GPU, 30hr/week quota, single GPU enforced via `CUDA_VISIBLE_DEVICES=0`
Target: CER ≤ 0.05, WER ≤ 0.10

**Current best CONFIRMED val CER: 0.0687** (from the prior session's checkpoint-51348).

**v3 run 3 (most recent session) produced val CER 0.0695 at epoch 6 (checkpoint-59906) — essentially flat vs. 0.0687, not an improvement.** This run's headline "Test CER 1.4052" was **invalid** — caused by a test-eval bug (see Bug #6 below), not by real model degradation. True test CER has not yet been measured; re-run the corrected test-eval cell before trusting any test-set number older than this document.

---

## Dataset v3 — smart-stock-dataset-v3

Combined CORD + SROIE + WildReceipt, pre-built and saved to Kaggle:

| Split | Size |
|---|---|
| Train | 68,463 |
| Val | 3,193 |
| Test | 9,301 |

- **CORD**: line-level crops by `group_id`
- **SROIE**: line-level crops by Y-proximity (15px tolerance), 2× weighted in train
- **WildReceipt**: per-annotation crops (not line grouping — two-column layout caused column merging bug that regressed CER from 0.088 → 0.339 when Y-grouping was used)

---

## Kaggle Dataset Structure

Three separate datasets as inputs:

| Slug | Purpose | Path |
|---|---|---|
| `smart-stock-dataset-v3` | Combined dataset, never changes | `/kaggle/input/datasets/maazahmad69/smart-stock-dataset-v3/smart_stock_dataset_v3` |
| `trocr-smart-stock-model` | Best model + resume checkpoint | `/kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock-best/trocr-smart-stock-best` |
| `wild-receipt` | Raw WildReceipt, only for dataset rebuild | `/kaggle/input/datasets/maazahmad69/wild-receipt/wildreceipt` |

Checkpoint resume path (double-nested due to Kaggle upload structure):
```
/kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock/trocr-smart-stock/checkpoint-51348
```

**Action item — verify before next run:** confirm `checkpoint-51348` contains `optimizer.pt` and `scheduler.pt`, not just `model.safetensors` / `lora_adapter/`. The corrected resume logic (Bug #5 fix below) depends on these files being present in the uploaded checkpoint. If they were stripped out during a prior upload to save space, the "restore optimizer state" fix will silently fall back to fresh state.

---

## Current Training Config

```python
num_train_epochs=8
per_device_train_batch_size=8
learning_rate=1.4824e-4        # Optuna best (from pre-encoder-unfreeze search)
warmup_ratio=0.02672
weight_decay=0.01
lr_scheduler_type="cosine_with_restarts"
lr_scheduler_kwargs={"num_cycles": 1}
max_grad_norm=1.0
eval_strategy="epoch"
save_strategy="epoch"
load_best_model_at_end=True
metric_for_best_model="eval_cer"
greater_is_better=False
save_total_limit=3
predict_with_generate=True
generation_max_length=128
fp16=True                      # T4, must stay True
report_to="none"
```

Generation config (on `model.generation_config`):
```python
eos_token_id   = processor.tokenizer.sep_token_id
early_stopping = True
length_penalty = 1.0           # was 2.0, caused hallucination
num_beams      = 4
# no_repeat_ngram_size deliberately removed — caused CER > 1.0 on receipt text
# (receipt text has legitimate repetition, e.g. "60.000 60.000")
```

**Note on LR retuning:** this LR was tuned via Optuna before any encoder unfreezing. Once the encoder is partially unfrozen (Roadmap step 1), gradient flow changes and this LR should be treated as stale — re-run Optuna after that change, not before.

---

## Known Bugs (all fixed as of this version)

1. **DataParallel not disabled** — `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` must be the absolute first Python line before any import.

2. **Test eval hangs** — degenerate crops (w < 4 or h < 4) hang beam search; use manual loop, never `trainer.evaluate()` on test set.

3. **PEFT checkpoint reset** — `LoRASaveCallback` saves `lora_adapter/` subdir alongside each checkpoint; without it PEFT silently resets adapter weights on resume.

4. **Test eval skips all samples** — `model.generate()` must use keyword arg: `model.generate(pixel_values=pixel_values, max_new_tokens=128)` — positional arg raises `TypeError` on `PeftModel`.

5. **Scaler bug on resume — FIX CORRECTED THIS VERSION.**
   Previous "fix" (as of v3 run 3) discarded ALL optimizer/scheduler state on every resume from the uploaded input checkpoint (`resume_from_checkpoint=None`), citing scaler incompatibility. This was overly destructive: it meant the cosine LR schedule restarted from warmup every session, which is why val CER and training loss oscillated in a narrow band (0.0695–0.0739) across epochs instead of trending down — each session was re-doing a partial warmup/anneal cycle on top of already-converged weights rather than continuing to anneal.
   **Corrected fix:** copy the checkpoint to a writable directory, delete only `scaler.pt` from the copy, and pass the **copy's path** to `trainer.train(resume_from_checkpoint=...)`. This restores real `optimizer.pt` + `scheduler.pt` state (Adam momentum, correct position in the LR curve) while avoiding the `AttributeError: NoneType.load_state_dict` crash that the missing/incompatible scaler causes.
   ```python
   resume_copy = WORKING_DIR / "resume_checkpoint"
   if resume_copy.exists():
       shutil.rmtree(resume_copy)
   shutil.copytree(INPUT_CHECKPOINT_DIR, resume_copy)
   scaler_path = resume_copy / "scaler.pt"
   if scaler_path.exists():
       scaler_path.unlink()
   resume_from = str(resume_copy)
   ```

6. **Test CER >> val CER (1.4052 vs 0.0695) — NEW BUG, FOUND AND FIXED THIS VERSION.**
   Root cause: `model.merge_and_unload()` (called to produce `merged_model` for saving) is a **destructive, in-place** operation — it folds LoRA deltas into the base layers and removes the LoRA modules from the underlying module tree that the original `model` (PeftModel wrapper) variable still points to. After this call, `model` is a stale/inconsistent object; only `merged_model` is safe to use for inference.
   The test-eval cell and the single-sample sanity-check cell were both still calling `model.generate(...)` after the merge step had already run, producing corrupted output (token reordering, digit corruption) — this is a structural inference bug, not a real capability regression. **Both cells now use `merged_model.generate(...)`.**
   **Action:** re-run the corrected test-eval cell on the existing checkpoint before drawing any conclusion about real test-set performance — this costs a few minutes of inference, no training/GPU-quota cost, and will very likely land close to val CER (~0.07) rather than 1.4.

7. **Degenerate image crop hang in test eval** — skip images with width/height < 4px (folded into Bug #2/#4's code, same guard).

---

## Corrected Code — Test Eval Cell

```python
from jiwer import cer, wer

# Use merged_model (Bug #6) — NOT model, which is stale after merge_and_unload().
merged_model.eval()

all_preds, all_labels = [], []
skipped = 0
first_error_printed = False

for idx, sample in enumerate(combined_dataset["test"]):
    try:
        image = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")
        w, h = image.size
        if w < 4 or h < 4:
            skipped += 1
            continue

        pixel_values = processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(merged_model.device)

        with torch.no_grad():
            generated_ids = merged_model.generate(
                pixel_values=pixel_values,
                max_new_tokens=128,
            )

        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_preds.append(pred)
        all_labels.append(sample["text"])

    except Exception as e:
        if not first_error_printed:
            print(f"First exception at idx={idx}: {type(e).__name__}: {e}")
            first_error_printed = True
        skipped += 1

if all_preds:
    test_cer = cer(all_labels, all_preds)
    test_wer = wer(all_labels, all_preds)
    print(f"Test CER : {test_cer:.4f}")
    print(f"Test WER : {test_wer:.4f}")
else:
    print("No predictions collected — all samples skipped or errored")

print(f"Skipped  : {skipped} / {len(combined_dataset['test'])}")
print(f"Evaluated: {len(all_preds)} / {len(combined_dataset['test'])}")

# Targets: CER ≤ 0.05, WER ≤ 0.10
```

## Corrected Code — Single-Sample Sanity Check Cell

```python
import io
from PIL import Image

sample = combined_dataset["test"][0]
image = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")

pixel_values = processor(
    image,
    return_tensors="pt"
).pixel_values.to(merged_model.device)   # merged_model, not model

generated_ids = merged_model.generate(pixel_values=pixel_values, max_new_tokens=128)

prediction = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("GROUND TRUTH:")
print(sample["text"])
print("")
print("="*80)
print("PREDICTION:")
print(prediction)

image
```

## Corrected Code — Trainer + Resume Cell

```python
from transformers import Seq2SeqTrainer, TrainerCallback
import shutil

class LoRASaveCallback(TrainerCallback):
    """
    Saves LoRA adapter weights alongside each Trainer checkpoint.
    Without this callback, PEFT silently resets adapter weights on checkpoint
    reload — the base model config is restored instead of the adapter state.
    Saves to: checkpoint-{step}/lora_adapter/
    """
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        adapter_dir = checkpoint_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        kwargs["model"].save_pretrained(str(adapter_dir))
        print(f"LoRA adapter saved to {adapter_dir}")
        return control

print(f"LR: {training_args.learning_rate}")
print(f"Warmup ratio: {training_args.warmup_ratio}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")

# Remove stale checkpoints from prior bad runs (no lora_adapter subdir)
if CHECKPOINT_DIR.exists():
    for ckpt in CHECKPOINT_DIR.glob("checkpoint-*"):
        has_adapter = (ckpt / "lora_adapter").exists()
        if not has_adapter:
            shutil.rmtree(ckpt)
            print(f"Removed stale checkpoint (no LoRA adapter): {ckpt}")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[LoRASaveCallback()],
)

# ── Resume logic (Bug #5, corrected) ─────────────────────────────────────────
# Priority: 1) working-dir checkpoint (same session, full state)
#           2) uploaded input checkpoint (prior session — restore optimizer+
#              scheduler state via scaler-stripped copy, NOT resume_from=None)
#           3) fresh start.
valid_checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint-*"))

if valid_checkpoints:
    resume_from = str(valid_checkpoints[-1])
    print(f"Resuming from working dir (optimizer state intact): {resume_from}")

elif INPUT_CHECKPOINT_DIR.exists():
    resume_copy = WORKING_DIR / "resume_checkpoint"
    if resume_copy.exists():
        shutil.rmtree(resume_copy)
    shutil.copytree(INPUT_CHECKPOINT_DIR, resume_copy)

    scaler_path = resume_copy / "scaler.pt"
    if scaler_path.exists():
        scaler_path.unlink()
        print(f"Removed incompatible scaler.pt from {resume_copy}")
    else:
        print(f"No scaler.pt found in {INPUT_CHECKPOINT_DIR} — "
              f"optimizer.pt/scheduler.pt presence not guaranteed, verify before relying on this resume.")

    resume_from = str(resume_copy)
    print(f"Resuming from input checkpoint copy (optimizer+scheduler state intact): {resume_from}")

else:
    resume_from = None
    print("Starting fresh training")

trainer.train(resume_from_checkpoint=resume_from)
```

---

## Current Situation

The v3-run-3 session completed 8 epochs, best val CER 0.0695 at checkpoint-59906 — flat vs. the prior 0.0687, not a real improvement, and consistent with the LR-schedule-restart issue (Bug #5) suppressing genuine cross-session progress. The reported Test CER of 1.4052 from that run is invalid (Bug #6) and should be disregarded; re-running the corrected test-eval cell against the existing checkpoint is the immediate next step, and requires no new training.

## Next Steps (in order)

1. **Re-run corrected test-eval cell** on the current best checkpoint (`checkpoint-59906` / whatever is currently in `trocr-smart-stock-best`). Pure inference, no GPU-quota cost. This gives the first trustworthy test CER/WER for this model.
2. **Verify `checkpoint-51348` contents** — confirm `optimizer.pt` and `scheduler.pt` exist alongside `scaler.pt`, so the corrected resume logic actually has state to restore.
3. **Run one more training session with the corrected resume logic** (Bug #5 fix) to confirm the LR schedule now continues annealing instead of restarting, and see whether val CER actually drops below 0.0687 when optimizer momentum is preserved across sessions.
4. **Upload the new best model** to `trocr-smart-stock-model` (overwrite `trocr-smart-stock-best/`, keep only the latest checkpoint with its optimizer/scheduler state for resume).
5. **Partial ViT encoder unfreeze** (top 2 blocks) — highest-impact remaining lever; frozen encoder is the current performance ceiling.
6. **Re-run Optuna** after encoder unfreeze — LR will likely need retuning since gradient flow changes with more trainable parameters.
7. Once CER < 0.05 on a **verified** test-eval run, move to Stage 2 NER integration.
