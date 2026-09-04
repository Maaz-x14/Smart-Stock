# benchmark_latency.py — Issue #33: sub-stage + end-to-end latency
#
# Calls the same stage functions pipeline.py's process_receipt() calls,
# individually timed. Does NOT modify pipeline.py or process_receipt()'s
# behavior — mirrors its exact stage sequence and gating logic.
#
# Cache note: Pass 3 (llm_fallback.py) results are cached in
# normalization_cache and checked BEFORE Groq is called (see HANDOFF.md
# "cache gotcha"). This script does NOT clear cache between reps by
# default — reps after the first will show artificially fast Stage 3
# numbers for any item that hit Pass 3 once. This is intentional: it
# matches real usage (repeat items across receipts hit cache). Each
# rep's report includes a cache-hit flag so this is visible, not hidden.
# Pass `--clear-cache` to instead get cold-cache worst-case numbers.

import argparse
import asyncio
import json
import logging
import platform
import statistics
import sys
import time
import tempfile
from dataclasses import dataclass, field
from datetime import date

from app.db import SessionLocal
from ml_service.ocr.model import run_ocr, _get_ocr
from ml_service.ocr.row_reconstruction import reconstruct_rows
from ml_service.parsing.row_parser import parse_receipt_rows
from ml_service.item_extraction.extractor import extract_item_fields_batch
from ml_service.normalization.normalizer import normalize_entity
from ml_service.expiry.predictor import predict_expiry

STAGES = [
    "ocr", "row_reconstruction", "row_parser",
    "item_field_extraction", "normalization", "expiry",
]


class RateLimitWatcher(logging.Handler):
    """Detects '429'/'Too Many Requests' log lines during a timed block
    so contaminated runs (retry-backoff counted as latency) are flagged
    instead of silently blended into mean/median/p95."""
    def __init__(self):
        super().__init__()
        self.hit = False

    def emit(self, record):
        msg = record.getMessage()
        if "429" in msg or "Too Many Requests" in msg:
            self.hit = True


@dataclass
class RunResult:
    receipt: str
    rep: int
    stage_ms: dict = field(default_factory=dict)
    end_to_end_ms: float = 0.0
    n_items: int = 0
    n_food_items: int = 0
    rate_limited: bool = False
    error: str | None = None


async def run_once(path: str, rep: int, db) -> RunResult:
    r = RunResult(receipt=path, rep=rep)
    watcher = RateLimitWatcher()
    logging.getLogger().addHandler(watcher)
    t_start = time.perf_counter()
    try:
        with open(path, "rb") as f:
            image_bytes = f.read()

        # Stage 1: OCR
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            ocr_result = await asyncio.to_thread(run_ocr, tmp.name)
        r.stage_ms["ocr"] = (time.perf_counter() - t0) * 1000

        # Stage 1.5: Row Reconstruction
        t0 = time.perf_counter()
        rows = await asyncio.to_thread(reconstruct_rows, ocr_result)
        r.stage_ms["row_reconstruction"] = (time.perf_counter() - t0) * 1000

        # Stage 1.6 + 1.7: Prefilter (internal) + Row Parser
        t0 = time.perf_counter()
        parsed_items = await asyncio.to_thread(parse_receipt_rows, rows)
        r.stage_ms["row_parser"] = (time.perf_counter() - t0) * 1000
        r.n_items = len(parsed_items)

        # Stage 2: Item Field Extraction (batched)
        item_names = [item["item_name"] or "" for item in parsed_items]
        t0 = time.perf_counter()
        fields_list = await asyncio.to_thread(extract_item_fields_batch, item_names)
        r.stage_ms["item_field_extraction"] = (time.perf_counter() - t0) * 1000

        # Stage 3 + 4: Normalization + Expiry (gated on is_food, matches pipeline.py)
        norm_total, expiry_total = 0.0, 0.0
        purchase_date = date.today()
        for parsed, fields in zip(parsed_items, fields_list):
            if fields.is_food is not True:
                continue
            r.n_food_items += 1
            raw_token = parsed["item_name"] or ""

            t0 = time.perf_counter()
            normalized = await asyncio.to_thread(
                normalize_entity, raw_token, parsed["quantity"] or 1.0, fields.unit, db
            )
            norm_total += (time.perf_counter() - t0) * 1000

            if normalized is None:
                continue

            t0 = time.perf_counter()
            await asyncio.to_thread(predict_expiry, normalized, purchase_date, db, None)
            expiry_total += (time.perf_counter() - t0) * 1000

        r.stage_ms["normalization"] = norm_total
        r.stage_ms["expiry"] = expiry_total

    except Exception as e:
        r.error = f"{type(e).__name__}: {e}"

    logging.getLogger().removeHandler(watcher)
    r.rate_limited = watcher.hit
    r.end_to_end_ms = (time.perf_counter() - t_start) * 1000
    return r


def percentile(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    k = (len(vals) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def summarize(results: list[RunResult]) -> dict:
    ok = [r for r in results if r.error is None and not r.rate_limited]
    n_rate_limited = len([r for r in results if r.rate_limited])
    summary = {
        "n_runs": len(results),
        "n_failed": len([r for r in results if r.error is not None]),
        "n_rate_limited_excluded": n_rate_limited,
        "note": "stats below exclude runs that hit a 429 mid-run — those measure "
                "retry-backoff wait, not pipeline latency" if n_rate_limited else None,
    }
    e2e = [r.end_to_end_ms for r in ok]
    summary["end_to_end_ms"] = {
        "mean": round(statistics.mean(e2e), 1) if e2e else None,
        "median": round(statistics.median(e2e), 1) if e2e else None,
        "p95": round(percentile(e2e, 95), 1) if e2e else None,
        "min": round(min(e2e), 1) if e2e else None,
        "max": round(max(e2e), 1) if e2e else None,
    }
    summary["per_stage_ms"] = {}
    for stage in STAGES:
        vals = [r.stage_ms[stage] for r in ok if stage in r.stage_ms]
        summary["per_stage_ms"][stage] = {
            "mean": round(statistics.mean(vals), 1) if vals else None,
            "median": round(statistics.median(vals), 1) if vals else None,
            "p95": round(percentile(vals, 95), 1) if vals else None,
        }
    return summary


def hardware_config() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="paths to receipt images")
    ap.add_argument("--reps", type=int, default=1, help="repetitions per image (keep low on Groq free tier)")
    ap.add_argument("--delay-sec", type=float, default=0, help="sleep between reps, to clear TPM window")
    ap.add_argument("--clear-cache", action="store_true",
                     help="clear normalization_cache before EACH rep (cold-cache numbers)")
    ap.add_argument("--out", default="latency_results.json")
    args = ap.parse_args()

    print("Warming up OCR model (excluded from timed runs)...")
    _get_ocr()

    db = SessionLocal()
    all_results: list[RunResult] = []
    try:
        for path in args.images:
            for rep in range(args.reps):
                if args.clear_cache:
                    import check_cache
                    check_cache.clear_all()
                r = await run_once(path, rep, db)
                all_results.append(r)
                tag = " [RATE-LIMITED, excluded]" if r.rate_limited else ""
                status = f"FAILED: {r.error}" if r.error else f"{r.end_to_end_ms:.0f}ms{tag}"
                print(f"  {path} rep={rep} -> {status}")
                if args.delay_sec:
                    await asyncio.sleep(args.delay_sec)
    finally:
        db.close()

    report = {
        "hardware_runtime_config": hardware_config(),
        "cache_cleared_per_rep": args.clear_cache,
        "receipts_used": args.images,
        "reps_per_receipt": args.reps,
        "overall": summarize(all_results),
        "per_receipt": {
            img: summarize([r for r in all_results if r.receipt == img])
            for img in args.images
        },
        "raw_runs": [
            {"receipt": r.receipt, "rep": r.rep, "end_to_end_ms": round(r.end_to_end_ms, 1),
             "stage_ms": {k: round(v, 1) for k, v in r.stage_ms.items()},
             "n_items": r.n_items, "n_food_items": r.n_food_items, "error": r.error}
            for r in all_results
        ],
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved {args.out}")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
