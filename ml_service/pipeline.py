"""
Stage 0: Pipeline orchestrator (#25).

Wires OCR -> Row Reconstruction -> Prefilter/Row Parser -> Item Field
Extraction -> (is_food gate) -> Normalization -> Expiry Prediction.

Design decisions (locked, see chat record / HANDOFF.md):
  - Plain async function, not a class. DB session comes from FastAPI's
    per-request dependency injection; Stage 2's LLM provider is owned
    by extractor.py, not this module - pipeline.py doesn't know or
    care whether it's Groq, OpenRouter, or Gemini.
  - Stage 2 (is_food classification) is the ~658ms/item bottleneck
    (measured in #28/#29). Originally ran concurrently via
    asyncio.to_thread + Semaphore (STAGE2_CONCURRENCY). That bounded
    parallel requests, not tokens/min - even at concurrency=5, back-
    to-back Stage 2 + Stage 3 calls with no token-budget awareness
    still exhausted Groq's free-tier TPM (8000/min), causing 429s
    regardless of the concurrency setting (Issue #46).
  - FIX (Issue #46): Stage 2 now calls extract_item_fields_batch(),
    which batches is_food classification across all items in chunks
    of 5 (one Groq call per chunk instead of one per item). This cuts
    call count ~5x, which fixes both the TPM exhaustion and the 10s
    upload budget concern the old concurrency approach was trying to
    solve. asyncio.Semaphore/asyncio.gather-per-item is removed - no
    longer needed, since there's no longer a fan-out of N per-item
    calls to bound.
  - Stage 3 (Normalization) Pass 3 LLM fallback stays per-item, NOT
    batched (Issue #46 scope decision) - it's sparse (cache/fuzzy-match
    miss only) and accuracy-critical, unlike Stage 2 which hits every
    item unconditionally.
  - Stage 3/4 stay sequential per item - not the bottleneck, no
    evidence justifying the complexity of concurrent DB-session use.
  - is_food gating: None (API failure/malformed response) is treated
    identically to False. It must never be silently promoted to True.
  - Error philosophy: stages that make the pipeline structurally
    unable to continue (OCR, Row Reconstruction, Row Parser) raise
    typed exceptions - the FastAPI route maps these to API_Spec.md's
    error codes (OCR_FAILURE, etc.), not this module. Stages that
    produce per-item uncertainty (unit/brand extraction, is_food,
    normalization, expiry) never raise for expected failure modes -
    they return None/missing fields, preserving the item so it's
    still surfaced to the user, never silently dropped.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from datetime import date

from sqlalchemy.orm import Session

from ml_service.ocr.model import run_ocr
from ml_service.ocr.row_reconstruction import reconstruct_rows
from ml_service.parsing.row_parser import parse_receipt_rows
from ml_service.item_extraction.extractor import extract_item_fields_batch, ItemFields
from ml_service.normalization.normalizer import normalize_entity
from ml_service.expiry.predictor import predict_expiry

logger = logging.getLogger(__name__)


# -- Typed exceptions: stages where failure means the pipeline cannot continue --

class PipelineError(Exception):
    """Base class for pipeline-fatal stage failures. The FastAPI route
    is responsible for mapping these to API_Spec.md error codes - not
    this module."""


class OCRError(PipelineError):
    pass


class RowReconstructionError(PipelineError):
    pass


class RowParserError(PipelineError):
    pass


# -- Final per-item output shape --

@dataclass
class ExtractedItem:
    raw_token: str
    canonical_name: str | None
    brand: str | None
    is_food: bool | None
    quantity: float | None
    unit: str | None
    category: str | None
    predicted_expiry_date: date | None
    shelf_life_days: int | None
    confidence: float | None
    storage_context: str | None


def _excluded_item(raw_token: str, parsed: dict, fields: ItemFields) -> ExtractedItem:
    """Shared shape for items that skip Stage 3/4 - either because
    is_food is not True, or because Stage 3/4 couldn't resolve them.
    Matches API_Spec.md §2's 'detected but excluded' null-field shape."""
    return ExtractedItem(
        raw_token=raw_token,
        canonical_name=None,
        brand=fields.brand,
        is_food=fields.is_food,
        quantity=parsed["quantity"],
        unit=fields.unit,
        category=None,
        predicted_expiry_date=None,
        shelf_life_days=None,
        confidence=None,
        storage_context=None,
    )


async def process_receipt(
    image_bytes: bytes,
    db: Session,
    storage_context: str | None = None,
    purchase_date: date | None = None,
) -> list[ExtractedItem]:
    """
    Run the full ML pipeline on a single receipt image.

    Args:
        image_bytes: raw receipt image bytes (from the upload).
        db:          active SQLAlchemy session (FastAPI-injected).
        storage_context: user-selected "fridge"/"freezer"/"pantry",
                         or None to default by category (Stage 4).
        purchase_date:   defaults to today if not provided.

    Returns:
        list[ExtractedItem] - one per parsed receipt row. Non-food /
        unresolved items are included with null downstream fields,
        never dropped.

    Raises:
        OCRError, RowReconstructionError, RowParserError - pipeline-
        fatal failures. Per-item failures (Stage 2/3/4) do not raise;
        see module docstring.
    """
    purchase_date = purchase_date or date.today()

    # -- Stage 1: OCR --
    # run_ocr() takes a file path (ported as-is from the notebook);
    # image_bytes is written to a temp file to bridge that gap.
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            ocr_result = await asyncio.to_thread(run_ocr, tmp.name)
    except Exception as e:
        logger.error(f"Stage 1 OCR failed: {e}")
        raise OCRError(str(e)) from e

    logger.debug(f"Stage 1 OCR: {len(ocr_result['rec_texts'])} text boxes")

    # -- Stage 1.5: Row Reconstruction --
    try:
        rows = await asyncio.to_thread(reconstruct_rows, ocr_result)
    except Exception as e:
        logger.error(f"Stage 1.5 Row Reconstruction failed: {e}")
        raise RowReconstructionError(str(e)) from e

    logger.debug(f"Stage 1.5 Row Reconstruction: {len(rows)} rows")

    # -- Stage 1.6 (Prefilter, runs internally) + Stage 1.7 (Row Parser) --
    try:
        parsed_items = await asyncio.to_thread(parse_receipt_rows, rows)
    except Exception as e:
        logger.error(f"Stage 1.7 Row Parser failed: {e}")
        raise RowParserError(str(e)) from e

    logger.debug(f"Stage 1.7 Row Parser: {len(parsed_items)} items parsed")

    # -- Stage 2: Item Field Extraction (batched, Issue #46) --
    # extract_item_fields_batch() is sync/blocking (sync OpenAI client
    # under the hood) - run in a thread so we don't block the event
    # loop. Unlike the old per-item semaphore/gather approach, this is
    # ONE call covering the whole receipt's unit/brand extraction plus
    # chunked (size 5) is_food classification - no per-item fan-out to
    # bound here anymore.
    item_names = [item["item_name"] or "" for item in parsed_items]
    fields_list = await asyncio.to_thread(extract_item_fields_batch, item_names)

    logger.debug(f"Stage 2 Item Field Extraction: {len(fields_list)} items classified")

    # -- Stage 3 + 4: Normalization + Expiry, gated on is_food --
    results: list[ExtractedItem] = []

    for parsed, fields in zip(parsed_items, fields_list):
        raw_token = parsed["item_name"] or ""

        # is_food gating: None (unknown) is treated identically to
        # False. Never promoted to True under any circumstance.
        if fields.is_food is not True:
            logger.debug(f"'{raw_token}': is_food={fields.is_food!r} -> excluded, skipping Stage 3/4")
            results.append(_excluded_item(raw_token, parsed, fields))
            continue

        try:
            normalized = await asyncio.to_thread(
                normalize_entity, raw_token, parsed["quantity"] or 1.0, fields.unit, db
            )
        except Exception as e:
            logger.error(f"Stage 3 Normalization raised for '{raw_token}': {e}")
            normalized = None

        if normalized is None:
            logger.debug(f"'{raw_token}': Stage 3 could not resolve a canonical name")
            results.append(_excluded_item(raw_token, parsed, fields))
            continue

        logger.debug(
            f"'{raw_token}' -> '{normalized.canonical_name}' "
            f"(pass {normalized.normalization_pass}, category={normalized.category})"
        )

        try:
            expiry = await asyncio.to_thread(
                predict_expiry, normalized, purchase_date, db, storage_context
            )
        except Exception as e:
            logger.error(f"Stage 4 Expiry Prediction raised for '{normalized.canonical_name}': {e}")
            expiry = None

        if expiry is None:
            # Normalized but no expiry - item is still food, still
            # keep everything Stage 3 resolved.
            results.append(ExtractedItem(
                raw_token=raw_token,
                canonical_name=normalized.canonical_name,
                brand=fields.brand,
                is_food=True,
                quantity=normalized.quantity,
                unit=normalized.unit,
                category=normalized.category,
                predicted_expiry_date=None,
                shelf_life_days=None,
                confidence=None,
                storage_context=None,
            ))
            continue

        logger.debug(
            f"'{normalized.canonical_name}': expiry={expiry.predicted_expiry} "
            f"confidence={expiry.confidence} source={expiry.source}"
        )

        results.append(ExtractedItem(
            raw_token=raw_token,
            canonical_name=normalized.canonical_name,
            brand=fields.brand,
            is_food=True,
            quantity=normalized.quantity,
            unit=normalized.unit,
            category=normalized.category,
            predicted_expiry_date=expiry.predicted_expiry,
            shelf_life_days=expiry.shelf_life_days,
            confidence=expiry.confidence,
            storage_context=expiry.storage_context,
        ))

    return results
