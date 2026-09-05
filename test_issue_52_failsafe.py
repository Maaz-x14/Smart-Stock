"""
Regression test for Issue #52: verifies the batch is_food fail-safe
actually fires on every documented malformed/empty/truncated shape,
without hitting the real Groq API.

Run: pytest test_issue_52_failsafe.py -v
"""

import pytest
from unittest.mock import patch

from ml_service.item_extraction.food_classifier import (
    _parse_batch_response,
    classify_is_food_batch,
    ClassificationOutcome,
)


# --- _parse_batch_response: direct unit tests, no mocking needed ---

def test_none_content_returns_none():
    assert _parse_batch_response(None, expected_count=3) is None


def test_empty_string_content_returns_none():
    # The exact #52 report shape: finish_reason='length', empty content.
    assert _parse_batch_response("", expected_count=3) is None


def test_whitespace_only_content_returns_none():
    assert _parse_batch_response("   \n  ", expected_count=3) is None


def test_truncated_mid_object_returns_none():
    # Realistic finish_reason='length' shape: cut off mid-array.
    raw = '[{"index": 1, "is_food": true, "confidence": 0.9}, {"index": 2, "is_foo'
    assert _parse_batch_response(raw, expected_count=5) is None


def test_valid_json_but_wrong_length_returns_none():
    raw = '[{"index": 1, "is_food": true, "confidence": 0.9}]'
    assert _parse_batch_response(raw, expected_count=5) is None


def test_missing_index_in_range_returns_none():
    # 3 expected, only indices 1 and 3 present - index 2 never shows up.
    raw = (
        '[{"index": 1, "is_food": true, "confidence": 0.9}, '
        '{"index": 3, "is_food": false, "confidence": 0.2}]'
    )
    assert _parse_batch_response(raw, expected_count=3) is None


def test_duplicate_index_returns_none():
    raw = (
        '[{"index": 1, "is_food": true, "confidence": 0.9}, '
        '{"index": 1, "is_food": false, "confidence": 0.2}]'
    )
    assert _parse_batch_response(raw, expected_count=2) is None


def test_non_bool_is_food_returns_none():
    # int 1/0 instead of true/false - isinstance(1, bool) is False in
    # Python, so this must be rejected, not silently truthy-coerced.
    raw = '[{"index": 1, "is_food": 1, "confidence": 0.9}]'
    assert _parse_batch_response(raw, expected_count=1) is None


def test_confidence_out_of_range_returns_none():
    raw = '[{"index": 1, "is_food": true, "confidence": 1.5}]'
    assert _parse_batch_response(raw, expected_count=1) is None


def test_non_list_top_level_returns_none():
    raw = '{"items": [{"index": 1, "is_food": true, "confidence": 0.9}]}'
    assert _parse_batch_response(raw, expected_count=1) is None


def test_valid_response_parses_correctly():
    # Sanity check - the happy path must still work.
    raw = (
        '[{"index": 1, "is_food": true, "confidence": 0.9}, '
        '{"index": 2, "is_food": false, "confidence": 0.15}]'
    )
    result = _parse_batch_response(raw, expected_count=2)
    assert result == [(True, 0.9), (False, 0.15)]


# --- classify_is_food_batch: end-to-end fail-safe, Groq call mocked ---

def test_empty_content_from_groq_yields_all_unknown():
    """The actual #52 scenario end-to-end: Groq returns empty content
    (simulating finish_reason='length'), every item in the batch must
    resolve to UNKNOWN - not True, not a partial mix."""
    item_names = ["Kimtiaz", "Peek Frns Cocnt Crnch Farm Hose F/P", "ORG STRWBRY"]

    with patch(
        "ml_service.item_extraction.food_classifier._call_groq_batch",
        return_value=("", None),
    ):
        results = classify_is_food_batch(item_names)

    assert len(results) == 3
    for r in results:
        assert r.outcome == ClassificationOutcome.UNKNOWN
        assert r.is_food is None


def test_none_content_from_groq_yields_all_unknown():
    item_names = ["Kimtiaz", "Peek Frns Cocnt Crnch Farm Hose F/P"]

    with patch(
        "ml_service.item_extraction.food_classifier._call_groq_batch",
        return_value=(None, None),
    ):
        results = classify_is_food_batch(item_names)

    assert len(results) == 2
    for r in results:
        assert r.outcome == ClassificationOutcome.UNKNOWN


def test_truncated_response_from_groq_yields_all_unknown():
    item_names = ["A", "B", "C", "D", "E"]
    truncated = '[{"index": 1, "is_food": true, "confidence": 0.9}, {"index": 2, "is_foo'

    with patch(
        "ml_service.item_extraction.food_classifier._call_groq_batch",
        return_value=(truncated, "some reasoning trace"),
    ):
        results = classify_is_food_batch(item_names)

    assert len(results) == 5
    for r in results:
        assert r.outcome == ClassificationOutcome.UNKNOWN


def test_no_false_positive_is_food_true_on_any_malformed_shape():
    """Explicit regression for #52's actual symptom: malformed/empty
    content must NEVER resolve to is_food=True for any item."""
    malformed_shapes = ["", None, "not json at all", "[]", "[1, 2, 3]"]
    item_names = ["Kimtiaz", "Peek Frns Cocnt Crnch Farm Hose F/P"]

    for shape in malformed_shapes:
        with patch(
            "ml_service.item_extraction.food_classifier._call_groq_batch",
            return_value=(shape, None),
        ):
            results = classify_is_food_batch(item_names)
        assert all(r.is_food is not True for r in results), (
            f"False positive is_food=True on malformed shape: {shape!r}"
        )
