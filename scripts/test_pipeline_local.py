# test_pipeline_local.py
#
# Usage: run as a module from the repo root, not `python
# scripts/test_pipeline_local.py` directly - it imports `app.*`, which
# needs the repo root on sys.path: python -m scripts.test_pipeline_local
#
# Also see HANDOFF.md / eval.yml's header comment: IMAGE_PATHS below
# is hardcoded to this project's original author's own local machine
# and the images it points to are not in this repo (removed for
# privacy). Point it at your own receipt images to use this locally.
import asyncio
import time
from datetime import date
from app.db import get_session_factory
from ml_service.ocr.model import _get_ocr  # warm-up only, see main()
from ml_service.pipeline import process_receipt
import logging

logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
logging.getLogger("paddle").setLevel(logging.WARNING)  # silence OCR noise
logging.getLogger("paddlex").setLevel(logging.WARNING)

IMAGE_PATHS = [
    # "/home/maazahmad/Desktop/Smart-Stock/inference/1.jpg",
    "/home/maazahmad/Desktop/Smart-Stock/inference/2.jpg",
    "/home/maazahmad/Desktop/Smart-Stock/inference/3.jpg",
    # "/home/maazahmad/Desktop/Smart-Stock/inference/4.jpg",
]

async def main():
    # Issue #33: warm up the PaddleOCR singleton BEFORE timing starts.
    # model.py already lazy-loads it as a module-level singleton, so
    # in prod it's loaded once at service startup, not per-request -
    # loading it here (outside the timed loop) matches that, instead
    # of letting the first receipt's timer absorb one-time model-load
    # cost that no real request would ever pay.
    print("Warming up OCR model...")
    _get_ocr()

    db = get_session_factory()()
    try:
        for path in IMAGE_PATHS:
            print(f"\n=== {path} ===")
            with open(path, "rb") as f:
                image_bytes = f.read()
            start = time.perf_counter()
            try:
                items = await process_receipt(image_bytes, db, purchase_date=date.today())
            except Exception as e:
                print(f"  PIPELINE FAILED: {type(e).__name__}: {e}")
                continue
            elapsed = time.perf_counter() - start
            print(f"  process_receipt() wall-clock: {elapsed:.2f}s")
            for item in items:
                print(f"  {item.raw_token!r:30} -> is_food={item.is_food!r:6} "
                      f"canonical={item.canonical_name!r:20} pass={item.normalization_pass!r} "
                      f"expiry={item.predicted_expiry_date}")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())
