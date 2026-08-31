"""
Stage 1.5: Row Reconstruction.

Groups PaddleOCR's individual text boxes into logical receipt rows,
correcting for image skew before y-position clustering. Ported from
nb_paddleocr.ipynb - logic unchanged.
"""

import numpy as np


def estimate_skew_angle(polys):
    angles = []
    for poly in polys:
        poly = np.array(poly)
        top_left, top_right = poly[0], poly[1]
        dx = top_right[0] - top_left[0]
        dy = top_right[1] - top_left[1]
        if dx != 0:
            angles.append(np.arctan2(dy, dx))
    return np.median(angles)


def cluster_rows_deskewed(texts, scores, boxes, polys, y_tol_ratio=0.3):
    angle = estimate_skew_angle(polys)
    slope = np.tan(angle)  # dy per dx

    items = []
    for text, score, box in zip(texts, scores, boxes):
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        y_corrected = y_center - slope * x_center  # remove skew-induced drift
        height = y2 - y1
        items.append({"text": text, "score": score, "x": x1, "y": y_corrected, "h": height})

    items.sort(key=lambda i: i["y"])

    rows = [[items[0]]]
    for item in items[1:]:
        avg_h = sum(i["h"] for i in rows[-1]) / len(rows[-1])
        if abs(item["y"] - rows[-1][-1]["y"]) <= avg_h * y_tol_ratio:
            rows[-1].append(item)
        else:
            rows.append([item])

    for row in rows:
        row.sort(key=lambda i: i["x"])
    return rows


def cluster_rows(texts, scores, boxes, y_tol_ratio=0.5):
    """Non-deskewed baseline clustering - kept for reference/comparison
    (nb_paddleocr.ipynb Cell 3), not used by the pipeline.
    cluster_rows_deskewed handles skewed receipt photos and is the
    production path."""
    items = []
    for text, score, box in zip(texts, scores, boxes):
        x1, y1, x2, y2 = box
        y_center = (y1 + y2) / 2
        height = y2 - y1
        items.append({"text": text, "score": score, "x": x1, "y": y_center, "h": height})

    items.sort(key=lambda i: i["y"])

    rows = []
    current_row = [items[0]]
    for item in items[1:]:
        avg_h = sum(i["h"] for i in current_row) / len(current_row)
        if abs(item["y"] - current_row[-1]["y"]) <= avg_h * y_tol_ratio:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
    rows.append(current_row)

    for row in rows:
        row.sort(key=lambda i: i["x"])

    return rows


def reconstruct_rows(ocr_result: dict) -> list[list[dict]]:
    """
    Public entry point for Stage 1.5, called by pipeline.py.

    Args:
        ocr_result: dict from model.run_ocr() - rec_texts, rec_scores,
                    rec_boxes, rec_polys.

    Returns:
        rows - list of rows, each a list of {text, score, x, y, h}
        dicts, left-to-right sorted. This is the shape
        ml_service/parsing (Stage 1.6/1.7) expects.
    """
    return cluster_rows_deskewed(
        ocr_result["rec_texts"],
        ocr_result["rec_scores"],
        ocr_result["rec_boxes"],
        ocr_result["rec_polys"],
    )