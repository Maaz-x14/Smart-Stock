# HANDOFF.md — Context Document for Next Chat Session

## Smart-Stock: AI-Powered Inventory & Waste Reduction System

---

## Writing style for this project (important)

- No over-explanation. Short, direct answers.
- Propose approach before code. Ask one clarifying question at a time, don't assume.
- Surgical edits to existing docs/code — never full rewrites unless explicitly asked.
- Push back on bad ideas. Distinguish facts from speculation.
- Individual per-file commits.
- Ruthless mentor mode: stress-test everything, don't sugarcoat.
- Present files at end of response as attachments, not pasted inline — so specific lines can be edited without re-pasting whole files. **Always the FULL file, never a diff/snippet** — the person applies it directly.

---

## Project Summary

Smart-Stock: portfolio/CV project. Reads grocery receipts (Pakistani retail), predicts expiry dates, reduces food waste. Full stack: React + TypeScript -> FastAPI -> PostgreSQL -> ML Pipeline (7 stages).

## Current state (as of the app build)

The ML pipeline (OCR -> row reconstruction -> prefilter -> row parser -> item extraction -> normalization -> expiry) was already done and validated before this build. This build added the application around it: backend API, frontend, CI/CD, Dockerfile.

**Backend — done, tested (37 pytest tests passing), linted clean:**
- Auth: register/login/refresh/logout, JWT access (30 min) + HttpOnly refresh cookie (7 days)
- Receipts: `/upload` (runs the ML pipeline, stores a pending extraction) and `/confirm` (writes to inventory, 404/410 handling)
- Inventory: CRUD, bulk-consume -> waste_log
- Alerts: `/alerts`, dismiss, and the daily 08:00 UTC expiry-scan job (APScheduler)
- Recipes: Spoonacular lookup with a Postgres-backed cache (150 req/day free tier)
- Schema: one reconciled Alembic migration (`f1b2c3d4e5a6`) — UUID keys, `receipts`/`alerts`/`alert_items`/`waste_log`/`recipe_cache` tables, matches `DB_Schema.md` v1.1
- Housekeeping: `requirements.txt`/`requirements-ci.txt`/`requirements-dev.txt` split (torch/transformers removed), lazy PaddleOCR import, env-safe `app/db.py`, debug row-dump removed from `pipeline.py`

**Frontend — done, tested (11 Vitest tests passing), typechecks, builds, lints clean:**
- Vite + React + TypeScript, TanStack Query, Zustand (upload/undo state only), React Router
- Fridge view (shelves by category), upload + mandatory confirm modal, alerts, recipes, waste tracker (Recharts)
- Design system per the build plan: receipt/shelf-tag visual concept, urgency tiers that never rely on color alone, one deliberate motion moment (the scan progress), reduced-motion fallback, keyboard-accessible confirm modal

**CI/CD — written, NOT run end-to-end (see "Known gaps" below):**
- `.github/workflows/ci.yml`: lint, test-backend (Postgres 15 service container, `alembic check`, mocked Groq/Paddle), frontend (lint/typecheck/test/build), docker build
- `.github/workflows/eval.yml`: separate manual/nightly workflow for live-Groq checks — **not wired up yet**, see its own header comment
- `Dockerfile` + `docker-compose.yml` for local dev — **not build-verified**, see "Known gaps"

## Known gaps — read before trusting any of the above blindly

1. **Dockerfile is unverified.** The sandbox this was built in blocks Docker Hub, so `docker build` never completed. Build and run it yourself before relying on it.
2. **CI workflows are YAML-valid but not run end-to-end on GitHub itself.** No `act`/`actionlint` available in the build sandbox, and the Postgres service container GitHub Actions needs also needs Docker Hub. However, everything the CI jobs *do* has now actually been run locally against real Postgres (not just SQLite) in this environment — see "What got caught by actually running things against Postgres" below. Push to a branch and watch the Actions tab for anything environment-specific that only GitHub's runners would surface (network, image pulls, secrets).
3. **`eval.yml` will fail if run as-is.** `scripts/test_pipeline_local.py` hardcodes your local machine's absolute paths to receipt images, which are no longer in the repo (removed for privacy — see below). Fix before relying on the nightly schedule (currently commented out for this reason). Its `SessionLocal` import bug (see below) is fixed, so once you point it at real images, only the path problem remains.
4. **Frontend types are hand-written**, not generated from the running API's OpenAPI schema (the plan's original intent) — there was no running backend instance to introspect during the build. Regenerate once the backend is deployed somewhere reachable.
5. **`inference/*.jpg` and `inference/local_inference.py` are still tracked and still physically present** — a previous pass in this same build said these had been "removed from the working tree," but that was wrong: `git status`/`git ls-files` both confirm they're still there, still committed, and `local_inference.py` still hardcodes `/home/maazahmad/Desktop/Smart-Stock/...`. `.gitignore` now excludes `/inference` and `result.txt` going forward, but gitignore only stops NEW changes from being tracked — it does nothing for files already committed. These are real receipt/invoice images (customer data) sitting in a public repo right now. Per your standing rule that you apply file changes yourself, run this instead of me doing it:
   ```
   git rm -r inference/
   git commit -m "Remove receipt images and local inference script (real customer data, public repo)"
   ```
   This removes them from the working tree and from the next commit, but **they remain recoverable from git history** (every prior commit that touched them) until you separately decide whether to rewrite history (`git filter-repo` or similar) — worth doing before calling this actually resolved, since the whole point is this data shouldn't be exposed.

## What got caught by actually running things against Postgres

The gap above (only SQLite had been tested) turned out to matter: running `alembic upgrade head` and `alembic check` against a real local Postgres 16 instance surfaced several real bugs that SQLite-only testing had hidden. All are now fixed, and the migration/model/seed data are verified consistent against actual Postgres, not just inferred from reading the code:

1. **`urgency_tier` as a `GENERATED ALWAYS ... STORED` column doesn't work.** DB_Schema.md and the migration both specced it using `CURRENT_DATE` inside the generation expression — Postgres requires generated columns to be IMMUTABLE, and `CURRENT_DATE` isn't. `alembic upgrade head` failed outright with `generation expression is not immutable`. Fixed by dropping the column entirely and computing `urgency_tier` in `app/routers/inventory.py`'s `_to_response()`, from the same `predicted_expiry_date` already used for `days_remaining` — simpler, and actually correct for "today" (a stored column would freeze urgency as of insert time, never update). `app/models.py`, the migration, and DB_Schema.md's underlying design are all affected by this — DB_Schema.md itself should probably be corrected too, not just the code.
2. **`shelf_life_reference` had real pre-existing drift**, from before this session: `app/models.py` declared `subcategory`, `source`, `created_at`, wider column widths, and a `UniqueConstraint("canonical_name", "storage_context")` that no migration had ever actually created in the database. `alembic check` caught this directly. Two consequences this had already caused, silently: `db/seeds/shelf_life_seed.py` called `ShelfLifeReference(notes=...)`, a field the model no longer had (`TypeError` at runtime — the seed script was simply broken); and nothing stopped a duplicate `(canonical_name, storage_context)` row from being inserted — which there in fact was (`"Raita"` was listed twice under different categories). All fixed: `notes` restored to the model, the missing columns/constraint added via a migration step, and the duplicate seed row removed.
3. **`SessionLocal` doesn't exist** in the rewritten `app/db.py` (it now lazily builds a session factory via `get_session_factory()`), but four pre-existing scripts still imported the old name: `db/seeds/shelf_life_seed.py`, `scripts/check_cache.py`, `scripts/benchmark_latency.py`, `scripts/test_pipeline_local.py`, and `ml_service/expiry/evaluate.py`. All five would have raised `ImportError` the moment anyone ran them. Fixed in all five.
4. **Index-naming drift**: `app/models.py` used SQLAlchemy's `unique=True`/`index=True` column shorthand in several places, which auto-generates index names like `ix_users_email`; the migrations instead create explicitly-named indexes like `idx_users_email`. Both exist, so the DB "worked," but `alembic check` reported permanent phantom drift (it looks like new indexes need creating every time, even though nothing's actually missing). Fixed by removing the shorthand and declaring each table's indexes explicitly via `__table_args__`, matching the names the migrations actually use.
5. **`receipts.extracted_items` and `recipe_cache.response_json` were `JSONB` in the migration but plain `JSON` in the model** — another `alembic check` type-drift finding. Nothing in the app uses JSONB-specific operators, so fixed by making the migration use plain `JSON` too, which is also portable to SQLite (JSONB isn't).

After all of the above, `alembic check` reports **zero drift** against a real Postgres 16 database (confirmed by actually running it, including a from-scratch `alembic upgrade head` → `alembic check` cycle), the seed script runs cleanly and is actually idempotent (confirmed by running it twice), and all 37 backend tests + ruff check/format still pass. None of this had been run against real Postgres before — only SQLite, in CI's test-backend job and locally. This is exactly the gap this section used to warn about; it's now closed for the schema/migration layer specifically (Docker and GitHub Actions itself remain unverified per item 1-2 above, since those need network access this sandbox doesn't have).

Also fixed while in here: `ruff format --check .` previously failed on 12 files in `app`/`tests` and, after scoping lint config correctly (excluding notebooks and one-off local scripts in `scripts/`, which were never meant to be style-gated), on 18 pre-existing `ml_service` files too — all purely cosmetic (import order, blank lines, line wrapping, one `isinstance(x, (int, float))` → `isinstance(x, int | float)`, a few `zip()` calls given an explicit `strict=False` to match their actual pre-existing behavior). Verified every touched `ml_service` module still imports cleanly and the full test suite still passes after the reformat — see `pyproject.toml`'s `exclude` comment for the scoping rationale.

## Pipeline (unchanged structure)

```
Receipt Image
  -> Stage 1: OCR (PaddleOCR, pretrained) - text + bounding boxes
  -> Stage 1.5: Row Reconstruction - deskew + y-position clustering
  -> Stage 1.6: Prefilter - drops metadata rows
  -> Stage 1.7: Row Parser - header-driven column mapping -> {item_name, quantity, price, discount, total}
  -> Stage 2: Item Field Extraction - regex unit + fuzzy brand lexicon + LLM is_food gate (BATCHED, chunk size 5) -> {unit, brand, is_food}
       is_food=false/unknown -> surfaced to user flagged "excluded", stops here
       is_food=true  |
  -> Stage 3: Normalization - 3-pass lookup (map -> fuzzy -> LLM) -> NormalizedItem
  -> Stage 4: Expiry Prediction - 3-tier rule-based lookup -> ExpiryPrediction
  -> Structured Inventory Item
```

Pipeline internals are unchanged by this build except: `ml_service/ocr/model.py`'s PaddleOCR import moved inside `_get_ocr()` (lazy), and the Issue #49 debug row-dump was removed from `pipeline.py`.

## Open ML issues (unaffected by the app build)

#16, #23, #34, #48 — all ML refinements, not app-blocking. See GitHub for details.

## Next steps

1. Fix the `eval.yml` gap (add non-sensitive sample receipts, or keep it local-only — your call).
2. Build and run the Dockerfile for real; fix whatever the sandbox couldn't catch.
3. Push and watch the CI workflows actually run once.
4. Decide on a deploy target (Render/Railway + Vercel, per `Architecture.md`) and wire up CD.
5. Regenerate frontend types from a live OpenAPI schema once deployed.
6. Decide whether to rewrite git history to purge the old receipt images.
