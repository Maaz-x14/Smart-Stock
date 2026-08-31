"""
Stage 1: OCR.

PaddleOCR (PP-OCRv6 small, pretrained, CPU) - chosen over fine-tuned
TrOCR after real-receipt benchmarking (see OCR_Training.md): more
accurate on tested item/quantity/price lines, ~5-6s/receipt on CPU vs
~480s for TrOCR.

Ported from nb_paddleocr.ipynb (Cell 2) - logic unchanged, wrapped as
a lazy-loaded singleton for pipeline use.
"""

from paddleocr import PaddleOCR

_ocr = None


def _get_ocr() -> PaddleOCR:
    """Lazy-loaded singleton - avoid reloading model weights per call."""
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device='cpu',
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv6_small_rec",
        )
    return _ocr


def run_ocr(image_path: str) -> dict:
    """
    Run OCR on a receipt image.

    Args:
        image_path: path to the receipt image on disk.

    Returns:
        dict with keys: rec_texts, rec_scores, rec_boxes, rec_polys
        (raw PaddleOCR output shape - passed directly into
        row_reconstruction.reconstruct_rows()).
    """
    ocr = _get_ocr()
    result = ocr.predict(str(image_path))
    res = result[0]
    return {
        "rec_texts": res["rec_texts"],
        "rec_scores": res["rec_scores"],
        "rec_boxes": res["rec_boxes"],
        "rec_polys": res["rec_polys"],
    }