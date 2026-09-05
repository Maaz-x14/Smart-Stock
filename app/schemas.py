# Pydantic schemas for POST /receipts/upload response.
# app/schemas.py
#
# Scope (Issue #31): the extraction-response shape only, matching
# API_Spec.md v1.1's already-documented /receipts/upload contract -
# specifically the brand/is_food fields Stage 2 (extractor.py) now
# produces that the API layer never had a schema for, since no
# FastAPI routes/schemas existed in this repo before this issue.
# Other endpoints (auth, inventory CRUD, alerts, recipes, waste-log)
# are out of scope here - separate issues, not touched.

from datetime import date
from pydantic import BaseModel, Field, ConfigDict, model_validator


class ExtractedItem(BaseModel):
    """One item from the ML pipeline's output for a single receipt.

    Field nullability mirrors the real pipeline gate, not arbitrary
    optionality: `is_food=False` (or None/unknown) means Stage 3/4
    never ran for this item, so every downstream field is genuinely
    absent, not just unfilled - matches API_Spec.md's documented
    example for a non-food item (Supravit-M Tablet).
    """

    model_config = ConfigDict(from_attributes=True)

    raw_token: str = Field(..., description="Original OCR'd item_name from Row Parser, pre-Stage-2.")

    # Stage 2 output (extractor.py's ItemFields) - always present,
    # regardless of is_food outcome.
    brand: str | None = Field(None, description="Fuzzy-matched brand, or None if no lexicon match (a valid outcome, not a failure).")
    is_food: bool | None = Field(
        ...,
        description="Stage 2's food gate. None means UNKNOWN (API failure or malformed LLM response) - "
                    "callers must treat this identically to False, per API_Spec.md §2.",
    )
    unit: str | None = Field(None, description="Regex-extracted unit from item_name (Stage 2), independent of is_food.")

    # Stage 3/4 output - only populated when is_food resolved True.
    # None across the board otherwise, per API_Spec.md's documented
    # non-food example - this is not "missing data", it's "did not run".
    canonical_name: str | None = Field(None, description="Stage 3 normalized name. None if is_food is not True.")
    quantity: float | None = Field(None, description="Quantity from Row Parser. None if is_food is not True.")
    category: str | None = Field(None, description="Stage 3 category assignment. None if is_food is not True.")
    predicted_expiry_date: date | None = Field(None, description="Stage 4 output. None if is_food is not True.")
    shelf_life_days: int | None = Field(None, description="Stage 4 output. None if is_food is not True.")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Stage 3 normalization confidence. None if is_food is not True.")
    storage_context: str | None = Field(None, description="Echoes the request's storage_context. None if is_food is not True.")

    @model_validator(mode="after")
    def _downstream_fields_only_when_food(self) -> "ExtractedItem":
        """Enforces the actual pipeline contract (API_Spec.md §2), not
        just documents it: if is_food is not True, every Stage 3/4
        field MUST be None. Catches a real pipeline bug at the API
        boundary (e.g. Stage 3 accidentally running on a non-food item)
        instead of silently serializing inconsistent data to the client."""
        if self.is_food is not True:
            downstream = {
                "canonical_name": self.canonical_name,
                "quantity": self.quantity,
                "category": self.category,
                "predicted_expiry_date": self.predicted_expiry_date,
                "shelf_life_days": self.shelf_life_days,
                "confidence": self.confidence,
                "storage_context": self.storage_context,
            }
            populated = [k for k, v in downstream.items() if v is not None]
            if populated:
                raise ValueError(
                    f"is_food={self.is_food!r} but downstream field(s) {populated} are populated - "
                    "Stage 3/4 must not run for non-food items (API_Spec.md §2)."
                )
        return self


class ReceiptUploadResponse(BaseModel):
    """Full response body for POST /receipts/upload, per API_Spec.md §2."""

    model_config = ConfigDict(from_attributes=True)

    receipt_id: str
    extracted_items: list[ExtractedItem]
    total_items_extracted: int = Field(..., description="len(extracted_items) - includes non-food items.")
    processing_time_ms: int = Field(..., description="Wall-clock pipeline latency for this receipt. See ML_Pipeline.md §9 for aggregate benchmarks.")
