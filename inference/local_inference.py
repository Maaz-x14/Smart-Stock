# ============================================================
# local_inference.py
# Test the saved TrOCR model on a real receipt photo locally.
# Usage: python local_inference.py
# ============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # harmless if no GPU present

import time
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ── Paths ────────────────────────────────────────────────────────────────
MODEL_DIR = Path("/home/maazahmad/Desktop/Smart-Stock/ml_service/models/trocr-smart-stock")
IMAGE_PATH = Path("/home/maazahmad/Desktop/Smart-Stock/inference/1.jpg")

MIN_WIDTH = 50
MIN_HEIGHT = 10
MIN_DARK_PIXEL_RATIO = 0.02


# ── Load model ───────────────────────────────────────────────────────────
def load_model():
    processor = TrOCRProcessor.from_pretrained(str(MODEL_DIR))
    model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_DIR))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}, max_length={model.generation_config.max_length}, "
          f"use_cache={model.generation_config.use_cache}")
    return processor, model, device


# ── Line detection (CRAFT via easyocr) ──────────────────────────────────
def detect_lines_craft(reader, image_path, pad=4):
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)

    horizontal_list, free_list = reader.detect(img_np)
    boxes = horizontal_list[0]
    boxes = sorted(boxes, key=lambda b: (b[2], b[0]))

    w_img, h_img = pil_img.size
    crops = []
    for (x_min, x_max, y_min, y_max) in boxes:
        x1 = max(0, int(x_min) - pad)
        y1 = max(0, int(y_min) - pad)
        x2 = min(w_img, int(x_max) + pad)
        y2 = min(h_img, int(y_max) + pad)
        crop = pil_img.crop((x1, y1, x2, y2))
        crops.append((crop, (x1, y1, x2, y2)))

    return crops, pil_img


def is_valid_crop(crop_img: Image.Image) -> bool:
    w, h = crop_img.size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False
    gray = cv2.cvtColor(np.array(crop_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_ratio = np.count_nonzero(binary) / binary.size
    return dark_ratio >= MIN_DARK_PIXEL_RATIO


# ── Batched TrOCR inference ──────────────────────────────────────────────
def predict_batch(processor, model, device, crop_imgs, batch_size=16):
    texts = []
    for i in range(0, len(crop_imgs), batch_size):
        batch = crop_imgs[i:i + batch_size]
        pixel_values = processor(
            images=[c.convert("RGB") for c in batch],
            return_tensors="pt", padding=True
        ).pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        texts.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
    return texts


def main():
    import easyocr

    processor, model, device = load_model()
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    t0 = time.time()
    crops, orig = detect_lines_craft(reader, IMAGE_PATH)
    t1 = time.time()

    kept = [(img, bbox) for (img, bbox) in crops if is_valid_crop(img)]
    imgs = [c for c, _ in kept]

    preds = predict_batch(processor, model, device, imgs)
    t2 = time.time()

    print(f"\nDetection: {t1 - t0:.2f}s | Recognition: {t2 - t1:.2f}s")
    print(f"{IMAGE_PATH.name}: {len(crops)} detected, {len(kept)} kept\n")

    for i, text in enumerate(preds):
        print(f"  line {i:2d}: {text}")


if __name__ == "__main__":
    main()