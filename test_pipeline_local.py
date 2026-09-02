# test_pipeline_local.py
import asyncio
from datetime import date
from app.db import SessionLocal  # adjust to your actual session factory
from ml_service.pipeline import process_receipt
import logging

logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
logging.getLogger("paddle").setLevel(logging.WARNING)  # silence OCR noise
logging.getLogger("paddlex").setLevel(logging.WARNING)

IMAGE_PATHS = [
    # "/home/maazahmad/Desktop/Smart-Stock/inference/1.jpg",
    # "/home/maazahmad/Desktop/Smart-Stock/inference/2.jpg",
    "/home/maazahmad/Desktop/Smart-Stock/inference/3.jpg",
    # "/home/maazahmad/Desktop/Smart-Stock/inference/4.jpg",
]

async def main():
    db = SessionLocal()
    try:
        for path in IMAGE_PATHS:
            print(f"\n=== {path} ===")
            with open(path, "rb") as f:
                image_bytes = f.read()
            try:
                items = await process_receipt(image_bytes, db, purchase_date=date.today())
            except Exception as e:
                print(f"  PIPELINE FAILED: {type(e).__name__}: {e}")
                continue
            for item in items:
                print(f"  {item.raw_token!r:30} -> is_food={item.is_food!r:6} "
                      f"canonical={item.canonical_name!r:20} expiry={item.predicted_expiry_date}")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())