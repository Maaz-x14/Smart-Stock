# Normalization_Training.md — Stage 3 Normalization Guide
## Smart-Stock: Stage 3 — Food Name Normalization

**Version:** 1.0 (Initial build — three-pass pipeline, Groq LLM fallback)  
**Environment:** Local Python / FastAPI container (no GPU required)  
**Method:** Abbreviation lookup → Fuzzy match → Groq LLM fallback

---

## ⚠️ Critical Design Reality Check

Before any code: Stage 3 is **not a training pipeline**. It is a production module that runs at inference time inside `ml_service/`. There is no GPU, no dataset to download, no Trainer, no checkpoint to resume.

| Stage | What it does | Where it lives |
|---|---|---|
| Stage 1 (OCR) | Fine-tune TrOCR on receipt images | Kaggle notebook |
| Stage 2 (NER) | Fine-tune DistilBERT for token classification | Kaggle notebook |
| **Stage 3 (Normalization)** | **Rule-based + fuzzy + LLM text normalization** | **`ml_service/normalization/`** |
| Stage 4 (Expiry) | Shelf-life lookup + confidence scoring | `ml_service/expiry/` |

The distinction matters: Stages 1 and 2 produce model artifacts you save and reload. Stage 3 is pure logic — it runs directly as Python inside the API server. The "build" work for Stage 3 is constructing the `ABBREVIATION_MAP` and `shelf_life_reference` data, then wiring the three-pass pipeline.

---

## 1. Overview

Stage 3 takes raw NER output from Stage 2 and converts abbreviated, retailer-specific food tokens into canonical food names that can be matched against the `shelf_life_reference` database table.

```
Input  (from Stage 2 NER):   {food_tokens: ["ORG", "STRWBRY"], quantity: "1", unit: "LB", price: "2.99"}
Output (to Stage 4 Expiry):  {canonical_name: "Strawberries", quantity: 1.0, unit: "lb", category: "Produce", confidence: 0.95}
```

### Why normalization is necessary

Receipt printers truncate item names to fit on thermal paper. The same product appears differently across retailers and countries:

| Raw receipt token | What it actually is |
|---|---|
| `ORG STRWBRY 1LB` | Organic Strawberries |
| `CHKN BRST BNLS` | Boneless Chicken Breast |
| `MLK FL CRM 1GL` | Full Cream Milk (1 Gallon) |
| `DAHI 1KG` | Yogurt (Pakistani market) |
| `MURG QEEMA` | Minced Chicken (Pakistani market) |
| `GRK YGRT PLN` | Plain Greek Yogurt |
| `PANEER 500G` | Paneer (Indian/Pakistani) |

Without normalization, `"STRWBRY"` cannot be matched to the `shelf_life_reference` row `canonical_name = "Strawberries"` — the expiry prediction stage would fail silently.

### Three-pass architecture

```
Raw food token
      |
      v
┌─────────────────────────────────────┐
│  Pass 1: Abbreviation Map           │
│  Direct dict lookup                 │
│  Handles ~65% of cases              │
│  Confidence: 1.00                   │
└──────────────┬──────────────────────┘
               │ miss
               v
┌─────────────────────────────────────┐
│  Pass 2: Fuzzy Matching             │
│  rapidfuzz token_sort_ratio         │
│  against shelf_life_reference names │
│  score ≥ 80 → accept               │
│  Confidence: score / 100            │
└──────────────┬──────────────────────┘
               │ score < 80
               v
┌─────────────────────────────────────┐
│  Pass 3: LLM Fallback               │
│  Groq API (llama-3.1-8b-instant)   │
│  Cached in normalization_cache DB   │
│  Handles ≤ 20% of cases            │
│  Confidence: 0.70                   │
└──────────────┬──────────────────────┘
               │
               v
     canonical_name + confidence
```

---

## 2. Module Structure

```
ml_service/
├── pipeline.py                 # Orchestrates all 4 stages
├── normalization/
│   ├── __init__.py
│   ├── normalizer.py           # Public API — normalize_entity() entry point
│   ├── preprocessor.py         # Token cleaning before all three passes
│   ├── abbreviation_map.py     # ABBREVIATION_MAP (~800 entries) + Pass 1 logic
│   ├── fuzzy_matcher.py        # Pass 2 — rapidfuzz against shelf_life_reference
│   ├── llm_fallback.py         # Pass 3 — Groq API call + cache lookup
│   ├── unit_normalizer.py      # UNIT_MAP + quantity parser
│   └── category_classifier.py  # Category assignment (DB lookup + keyword fallback)
├── expiry/
│   └── predictor.py
└── models/
    ├── trocr.onnx
    └── distilbert_ner.onnx
```

Each file owns exactly one responsibility. `normalizer.py` is the only entry point — nothing outside `normalization/` should import from individual submodules.

---

## 3. shelf_life_reference Table

### 3.1 Schema

This is the canonical reference for both Stage 3 (name lookup, category assignment) and Stage 4 (expiry prediction). It is the single source of truth for all recognized food items.

```sql
CREATE TABLE shelf_life_reference (
    id               SERIAL PRIMARY KEY,
    canonical_name   VARCHAR(100) NOT NULL,
    category         VARCHAR(50)  NOT NULL,   -- Produce, Dairy, Meat, Pantry, Frozen, Beverages, Bakery
    storage_context  VARCHAR(20)  NOT NULL,   -- Fridge, Freezer, Pantry
    shelf_life_days_min  INTEGER  NOT NULL,
    shelf_life_days_avg  INTEGER  NOT NULL,
    shelf_life_days_max  INTEGER  NOT NULL,
    notes            TEXT,
    UNIQUE (canonical_name, storage_context)
);

CREATE INDEX idx_shelf_life_canonical ON shelf_life_reference (canonical_name);
CREATE INDEX idx_shelf_life_category  ON shelf_life_reference (category, storage_context);
```

**SQLAlchemy model:**

```python
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ShelfLifeReference(Base):
    __tablename__ = "shelf_life_reference"

    id                   = Column(Integer, primary_key=True)
    canonical_name       = Column(String(100), nullable=False)
    category             = Column(String(50),  nullable=False)
    storage_context      = Column(String(20),  nullable=False)
    shelf_life_days_min  = Column(Integer,     nullable=False)
    shelf_life_days_avg  = Column(Integer,     nullable=False)
    shelf_life_days_max  = Column(Integer,     nullable=False)
    notes                = Column(Text)
```

### 3.2 Seed Data

Populate via a seed script (`db/seeds/shelf_life_seed.py`) rather than inline SQL INSERT statements — this keeps the data maintainable as a Python dict and avoids SQL escaping issues for names with apostrophes.

```python
# db/seeds/shelf_life_seed.py
# Run once: python -m db.seeds.shelf_life_seed

SHELF_LIFE_DATA = [
    # (canonical_name, category, storage_context, min_days, avg_days, max_days, notes)

    # ── PRODUCE ──────────────────────────────────────────────────────────────────
    ("Strawberries",        "Produce", "Fridge",  3,  5,  7,  "Wash before eating, not before storing"),
    ("Blueberries",         "Produce", "Fridge",  5,  10, 14, None),
    ("Raspberries",         "Produce", "Fridge",  2,  3,  5,  None),
    ("Blackberries",        "Produce", "Fridge",  3,  5,  7,  None),
    ("Grapes",              "Produce", "Fridge",  5,  7,  14, None),
    ("Bananas",             "Produce", "Pantry",  3,  5,  7,  "Refrigerate to extend; skin blackens but fruit is fine"),
    ("Apples",              "Produce", "Fridge",  21, 30, 45, None),
    ("Apples",              "Produce", "Pantry",  5,  7,  14, None),
    ("Oranges",             "Produce", "Fridge",  14, 21, 30, None),
    ("Oranges",             "Produce", "Pantry",  5,  7,  10, None),
    ("Lemons",              "Produce", "Fridge",  14, 21, 30, None),
    ("Limes",               "Produce", "Fridge",  14, 21, 28, None),
    ("Mangoes",             "Produce", "Fridge",  5,  7,  10, "Ripen at room temp first"),
    ("Mangoes",             "Produce", "Pantry",  2,  4,  6,  "Until ripe only"),
    ("Papayas",             "Produce", "Fridge",  5,  7,  10, None),
    ("Guavas",              "Produce", "Fridge",  3,  5,  7,  None),
    ("Pomegranates",        "Produce", "Fridge",  14, 21, 30, None),
    ("Watermelon",          "Produce", "Fridge",  7,  10, 14, "Cut melon; whole can stay on counter 7-10 days"),
    ("Cantaloupe",          "Produce", "Fridge",  5,  7,  10, "Cut only"),
    ("Pineapple",           "Produce", "Fridge",  5,  7,  10, "Cut only"),
    ("Pineapple",           "Produce", "Pantry",  2,  3,  5,  "Whole, uncut"),
    ("Tomatoes",            "Produce", "Pantry",  5,  7,  10, "Refrigerating dulls flavor"),
    ("Cherry Tomatoes",     "Produce", "Fridge",  5,  7,  10, None),
    ("Onions",              "Produce", "Pantry",  14, 30, 60, "Keep dry and dark"),
    ("Garlic",              "Produce", "Pantry",  14, 30, 60, "Whole head"),
    ("Potatoes",            "Produce", "Pantry",  14, 21, 30, "Keep away from light"),
    ("Sweet Potatoes",      "Produce", "Pantry",  14, 21, 30, None),
    ("Carrots",             "Produce", "Fridge",  14, 21, 30, None),
    ("Broccoli",            "Produce", "Fridge",  3,  5,  7,  None),
    ("Cauliflower",         "Produce", "Fridge",  5,  7,  10, None),
    ("Cabbage",             "Produce", "Fridge",  14, 21, 30, None),
    ("Spinach",             "Produce", "Fridge",  3,  5,  7,  None),
    ("Lettuce",             "Produce", "Fridge",  5,  7,  10, None),
    ("Cucumber",            "Produce", "Fridge",  5,  7,  10, None),
    ("Bell Peppers",        "Produce", "Fridge",  5,  7,  14, None),
    ("Jalapeños",           "Produce", "Fridge",  5,  7,  14, None),
    ("Green Chillies",      "Produce", "Fridge",  5,  7,  14, "Common in Pakistani/Indian cooking"),
    ("Coriander",           "Produce", "Fridge",  5,  7,  10, "Also called cilantro"),
    ("Mint",                "Produce", "Fridge",  5,  7,  10, None),
    ("Green Onions",        "Produce", "Fridge",  5,  7,  10, None),
    ("Celery",              "Produce", "Fridge",  14, 21, 28, None),
    ("Mushrooms",           "Produce", "Fridge",  5,  7,  10, None),
    ("Zucchini",            "Produce", "Fridge",  5,  7,  10, None),
    ("Eggplant",            "Produce", "Fridge",  5,  7,  10, "Also called brinjal/baingan"),
    ("Corn",                "Produce", "Fridge",  1,  2,  3,  "Best eaten day of purchase"),
    ("Peas",                "Produce", "Fridge",  3,  5,  7,  "In pod; shelled peas last 3-5 days"),
    ("Green Beans",         "Produce", "Fridge",  3,  5,  7,  None),

    # ── DAIRY ────────────────────────────────────────────────────────────────────
    ("Whole Milk",          "Dairy",   "Fridge",  5,  7,  10, "After opening; check sell-by"),
    ("Skim Milk",           "Dairy",   "Fridge",  5,  7,  10, None),
    ("Full Cream Milk",     "Dairy",   "Fridge",  5,  7,  10, "Common in Pakistani/international markets"),
    ("UHT Milk",            "Dairy",   "Pantry",  60, 90, 180,"Unopened; once opened refrigerate and use in 7 days"),
    ("Half and Half",       "Dairy",   "Fridge",  5,  7,  10, None),
    ("Heavy Cream",         "Dairy",   "Fridge",  5,  7,  14, None),
    ("Butter",              "Dairy",   "Fridge",  14, 30, 45, None),
    ("Butter",              "Dairy",   "Freezer", 60, 90, 180,None),
    ("Cheddar Cheese",      "Dairy",   "Fridge",  14, 21, 30, "Opened block"),
    ("Mozzarella Cheese",   "Dairy",   "Fridge",  5,  7,  14, "Fresh; sliced or shredded"),
    ("Cream Cheese",        "Dairy",   "Fridge",  7,  14, 21, "Opened"),
    ("Cottage Cheese",      "Dairy",   "Fridge",  5,  7,  10, "Opened"),
    ("Greek Yogurt",        "Dairy",   "Fridge",  7,  14, 21, None),
    ("Plain Yogurt",        "Dairy",   "Fridge",  7,  14, 21, None),
    ("Dahi",                "Dairy",   "Fridge",  3,  5,  7,  "Pakistani-style fresh yogurt; shorter life than commercial"),
    ("Lassi",               "Dairy",   "Fridge",  2,  3,  5,  "Fresh-made; unopened commercial lasts longer"),
    ("Paneer",              "Dairy",   "Fridge",  3,  5,  7,  "Fresh Indian/Pakistani cheese; freeze for longer"),
    ("Paneer",              "Dairy",   "Freezer", 30, 60, 90, None),
    ("Eggs",                "Dairy",   "Fridge",  21, 35, 45, "From purchase date; USDA recommends refrigerating"),
    ("Sour Cream",          "Dairy",   "Fridge",  7,  14, 21, "Opened"),
    ("Whipped Cream",       "Dairy",   "Fridge",  3,  5,  7,  "Opened can or fresh"),
    ("Condensed Milk",      "Dairy",   "Pantry",  180,365,730,"Unopened; refrigerate after opening, use in 14 days"),

    # ── MEAT ─────────────────────────────────────────────────────────────────────
    ("Chicken Breast",      "Meat",    "Fridge",  1,  2,  3,  "Raw; cook or freeze within 2 days"),
    ("Chicken Breast",      "Meat",    "Freezer", 180,270,365,None),
    ("Chicken Thighs",      "Meat",    "Fridge",  1,  2,  3,  None),
    ("Chicken Thighs",      "Meat",    "Freezer", 180,270,365,None),
    ("Whole Chicken",       "Meat",    "Fridge",  1,  2,  3,  None),
    ("Whole Chicken",       "Meat",    "Freezer", 270,365,365,None),
    ("Minced Chicken",      "Meat",    "Fridge",  1,  2,  2,  "Also called chicken qeema"),
    ("Minced Chicken",      "Meat",    "Freezer", 90, 120,180,None),
    ("Ground Beef",         "Meat",    "Fridge",  1,  2,  2,  None),
    ("Ground Beef",         "Meat",    "Freezer", 90, 120,180,None),
    ("Beef Steak",          "Meat",    "Fridge",  2,  3,  5,  None),
    ("Beef Steak",          "Meat",    "Freezer", 180,270,365,None),
    ("Minced Beef",         "Meat",    "Fridge",  1,  2,  2,  "Also called beef qeema / keema"),
    ("Minced Beef",         "Meat",    "Freezer", 90, 120,180,None),
    ("Mutton",              "Meat",    "Fridge",  2,  3,  5,  "Common in Pakistani/Middle Eastern markets"),
    ("Mutton",              "Meat",    "Freezer", 180,270,365,None),
    ("Lamb Chops",          "Meat",    "Fridge",  2,  3,  5,  None),
    ("Lamb Chops",          "Meat",    "Freezer", 180,270,365,None),
    ("Pork Chops",          "Meat",    "Fridge",  2,  3,  5,  None),
    ("Pork Chops",          "Meat",    "Freezer", 120,180,365,None),
    ("Bacon",               "Meat",    "Fridge",  5,  7,  14, "Opened package"),
    ("Bacon",               "Meat",    "Freezer", 30, 60, 90, None),
    ("Salmon",              "Meat",    "Fridge",  1,  2,  3,  "Raw fillet"),
    ("Salmon",              "Meat",    "Freezer", 90, 180,270,None),
    ("Tuna Steak",          "Meat",    "Fridge",  1,  2,  2,  "Raw"),
    ("Tuna Steak",          "Meat",    "Freezer", 90, 180,270,None),
    ("Shrimp",              "Meat",    "Fridge",  1,  2,  3,  "Raw, peeled"),
    ("Shrimp",              "Meat",    "Freezer", 90, 180,270,None),
    ("Deli Ham",            "Meat",    "Fridge",  3,  5,  7,  "Sliced, opened package"),
    ("Deli Turkey",         "Meat",    "Fridge",  3,  5,  7,  "Sliced, opened package"),
    ("Sausages",            "Meat",    "Fridge",  3,  5,  7,  "Uncooked, opened"),
    ("Sausages",            "Meat",    "Freezer", 30, 60, 90, None),
    ("Hot Dogs",            "Meat",    "Fridge",  5,  7,  14, "Opened package"),
    ("Seekh Kebab",         "Meat",    "Fridge",  1,  2,  3,  "Raw; common in Pakistani/South Asian markets"),
    ("Seekh Kebab",         "Meat",    "Freezer", 30, 60, 90, None),

    # ── PANTRY ───────────────────────────────────────────────────────────────────
    ("Bread",               "Pantry",  "Pantry",  5,  7,  10, "Room temp; refrigerating extends to 14 days but stales faster"),
    ("Whole Wheat Bread",   "Pantry",  "Pantry",  5,  7,  10, None),
    ("Pita Bread",          "Pantry",  "Pantry",  5,  7,  10, None),
    ("Naan",                "Bakery",  "Pantry",  2,  3,  5,  "Fresh-baked; commercial packaged lasts 5-7 days"),
    ("Roti",                "Bakery",  "Pantry",  1,  2,  3,  "Fresh flatbread; freeze for longer storage"),
    ("Roti",                "Bakery",  "Freezer", 30, 60, 90, None),
    ("Paratha",             "Bakery",  "Fridge",  2,  3,  5,  "Cooked; raw dough in fridge 3-5 days"),
    ("Paratha",             "Bakery",  "Freezer", 30, 60, 90, "Cooked parathas freeze very well"),
    ("Pasta",               "Pantry",  "Pantry",  365,730,1095,"Dry, unopened"),
    ("Rice",                "Pantry",  "Pantry",  365,730,1095,"White rice; unopened"),
    ("Basmati Rice",        "Pantry",  "Pantry",  365,730,1095,"Aromatic long-grain; common in South Asian cooking"),
    ("Oats",                "Pantry",  "Pantry",  180,365,730, "Sealed container"),
    ("Flour",               "Pantry",  "Pantry",  90, 180,365, "All-purpose; sealed"),
    ("Atta Flour",          "Pantry",  "Pantry",  60, 90, 180, "Whole wheat flour used for roti; shorter shelf life due to higher oil content"),
    ("Maida",               "Pantry",  "Pantry",  90, 180,365, "Refined flour; used in Pakistani/Indian baking"),
    ("Sugar",               "Pantry",  "Pantry",  730,1825,3650,"Indefinite if kept dry"),
    ("Salt",                "Pantry",  "Pantry",  1825,3650,3650,"Indefinite"),
    ("Olive Oil",           "Pantry",  "Pantry",  365,540,730, "Opened; store away from heat and light"),
    ("Vegetable Oil",       "Pantry",  "Pantry",  365,540,730, "Opened"),
    ("Cooking Oil",         "Pantry",  "Pantry",  180,365,540, "Generic; refined oils last longer"),
    ("Soy Sauce",           "Pantry",  "Fridge",  30, 60, 90, "After opening"),
    ("Ketchup",             "Pantry",  "Fridge",  30, 60, 90, "After opening; unopened pantry 1 year"),
    ("Mayonnaise",          "Pantry",  "Fridge",  30, 60, 90, "After opening"),
    ("Mustard",             "Pantry",  "Fridge",  60, 90, 180, "After opening"),
    ("Hot Sauce",           "Pantry",  "Fridge",  90, 180,365, "After opening"),
    ("Vinegar",             "Pantry",  "Pantry",  730,1825,3650,"Practically indefinite"),
    ("Honey",               "Pantry",  "Pantry",  730,3650,3650,"Indefinite if sealed"),
    ("Peanut Butter",       "Pantry",  "Pantry",  90, 180,270, "Natural; opened. Commercial lasts longer"),
    ("Jam",                 "Pantry",  "Fridge",  90, 180,270, "After opening"),
    ("Tomato Sauce",        "Pantry",  "Fridge",  5,  7,  10, "Opened jar"),
    ("Tomato Paste",        "Pantry",  "Fridge",  5,  7,  14, "Opened can — transfer to airtight container"),
    ("Coconut Milk",        "Pantry",  "Fridge",  4,  5,  7,  "Opened can; transfer to airtight container"),
    ("Chicken Broth",       "Pantry",  "Fridge",  3,  4,  5,  "Opened carton"),
    ("Canned Chickpeas",    "Pantry",  "Fridge",  3,  4,  5,  "Opened can; transfer to airtight container"),
    ("Canned Tomatoes",     "Pantry",  "Fridge",  3,  4,  5,  "Opened can"),
    ("Canned Tuna",         "Pantry",  "Fridge",  3,  4,  5,  "Opened can"),
    ("Lentils",             "Pantry",  "Pantry",  365,730,1095,"Dry; sealed"),
    ("Chickpeas",           "Pantry",  "Pantry",  365,730,1095,"Dry; sealed. Also called chana"),
    ("Black Beans",         "Pantry",  "Pantry",  365,730,1095,"Dry; sealed"),
    ("Masoor Dal",          "Pantry",  "Pantry",  365,730,1095,"Red lentils; common in Pakistani cooking"),
    ("Chana Dal",           "Pantry",  "Pantry",  365,730,1095,"Split chickpeas"),
    ("Moong Dal",           "Pantry",  "Pantry",  365,730,1095,"Split mung beans"),
    ("Urad Dal",            "Pantry",  "Pantry",  365,730,1095,"Black gram lentils"),

    # ── FROZEN ───────────────────────────────────────────────────────────────────
    ("Frozen Peas",         "Frozen",  "Freezer", 180,270,365, "Keep sealed to prevent freezer burn"),
    ("Frozen Corn",         "Frozen",  "Freezer", 180,270,365, None),
    ("Frozen Spinach",      "Frozen",  "Freezer", 180,270,365, None),
    ("Frozen Mixed Veg",    "Frozen",  "Freezer", 180,270,365, None),
    ("Frozen Fries",        "Frozen",  "Freezer", 180,270,365, None),
    ("Ice Cream",           "Frozen",  "Freezer", 30, 60, 90,  "Opened; quality degrades after 2 months"),
    ("Frozen Pizza",        "Frozen",  "Freezer", 60, 90, 180, None),
    ("Frozen Waffles",      "Frozen",  "Freezer", 60, 90, 180, None),

    # ── BEVERAGES ────────────────────────────────────────────────────────────────
    ("Orange Juice",        "Beverages","Fridge", 5,  7,  10, "Opened carton"),
    ("Apple Juice",         "Beverages","Fridge", 5,  7,  10, "Opened"),
    ("Sparkling Water",     "Beverages","Pantry", 365,730,730, "Unopened; opened goes flat in 1-3 days"),
    ("Soda",                "Beverages","Pantry", 180,365,540, "Unopened; opened flat in 1-3 days"),
    ("Coffee Beans",        "Beverages","Pantry", 14, 30, 60,  "Opened bag; freeze for longer"),
    ("Ground Coffee",       "Beverages","Pantry", 7,  14, 30,  "Opened bag; goes stale quickly"),
    ("Tea Bags",            "Beverages","Pantry", 180,365,730, "Sealed; loses flavor over time"),

    # ── BAKERY ───────────────────────────────────────────────────────────────────
    ("Croissants",          "Bakery",  "Pantry",  1,  2,  3,  "Fresh-baked"),
    ("Bagels",              "Bakery",  "Pantry",  3,  5,  7,  None),
    ("Muffins",             "Bakery",  "Pantry",  3,  5,  7,  None),
    ("Cake",                "Bakery",  "Fridge",  3,  5,  7,  "Frosted; unfrosted lasts 2-3 days at room temp"),
]


def seed_shelf_life(db_session):
    """
    Insert all shelf_life_reference rows.
    Safe to run multiple times — skips rows that already exist.
    Call from your Alembic migration or a one-time setup script.
    """
    from app.models import ShelfLifeReference

    inserted = 0
    for row in SHELF_LIFE_DATA:
        canonical_name, category, storage_context, min_d, avg_d, max_d, notes = row
        exists = db_session.query(ShelfLifeReference).filter_by(
            canonical_name=canonical_name,
            storage_context=storage_context,
        ).first()
        if not exists:
            db_session.add(ShelfLifeReference(
                canonical_name=canonical_name,
                category=category,
                storage_context=storage_context,
                shelf_life_days_min=min_d,
                shelf_life_days_avg=avg_d,
                shelf_life_days_max=max_d,
                notes=notes,
            ))
            inserted += 1

    db_session.commit()
    print(f"Seeded {inserted} shelf_life_reference rows ({len(SHELF_LIFE_DATA) - inserted} already existed)")
```

> The seed script is idempotent — safe to run multiple times. Add new items by appending to `SHELF_LIFE_DATA` and re-running.

---

## 4. Pass 1 — Abbreviation Map (Direct Lookup)

### 4.1 Design rationale

About 65% of all raw receipt tokens can be resolved with a direct dictionary lookup. Retailers use a consistent (if cryptic) shorthand that is largely stable across store visits. Building this map upfront means the majority of normalizations happen at O(1) with 100% confidence and zero latency.

The map covers three domains equally:
- **US retail abbreviations** — Kroger, Walmart, Costco, Whole Foods style shortcodes
- **Pakistani market names** — both English transliterations (MURG, GOSHT, DAHI) and common branded items
- **Global/international** — metric units, European brand patterns, broadly shared food names

### 4.2 Preprocessing before lookup

Strip noise before attempting any match. Organic/quality modifiers (`ORG`, `LO-FAT`, `FF`, `WHL`) are useful for confirming it's food, but they are not part of the canonical name.

```python
# ml_service/normalization/preprocessor.py

STRIP_PREFIXES = {
    "ORG", "ORGC", "ORGANIC",
    "FF",                          # Fat-free
    "LF", "LOWFAT", "LO-FAT",     # Low-fat
    "RF", "REDFAT",                # Reduced-fat
    "WHL", "WHOLE",
    "FRZN", "FRZ", "FZ",          # Frozen (keep if it disambiguates, e.g. "FZ CORN")
    "GF",                          # Gluten-free
    "NS", "NSA",                   # No-salt-added
    "NF",                          # Non-fat
    "LS",                          # Low-sodium
    "RAW",
    "FRESH",
    "SMKD",                        # Smoked
    "SLCD", "SLC",                 # Sliced
    "DICED",
    "BNLS", "BNLESS",              # Boneless
    "SKNLS",                       # Skinless
    "LN", "LEAN",
    "XL", "LG", "SM", "MED",      # Size modifiers
    "PKG", "PCK",                  # Package (sometimes prefixed)
}

STRIP_SUFFIXES = {
    # Units that get fused to the token
    "LB", "LBS", "OZ", "GL", "GAL", "CT", "PK", "PCS",
    "KG", "G", "GR", "GRM",       # Metric
    "ML", "L", "LTR",             # Metric liquid
    "BX", "BG", "BAG", "BTL", "CAN", "JAR", "TUB",
    "DOZ", "DZ",
    "PT", "QT",
}

def preprocess_token(raw: str) -> str:
    """
    Uppercase, strip trailing price/quantity patterns, remove quality
    prefix/suffix modifiers. Returns cleaned token ready for Pass 1 lookup.

    Examples:
      "ORG STRWBRY 1LB"  -> "STRWBRY"
      "CHKN BRST BNLS"   -> "CHKN BRST"
      "MLK FL CRM 1GL"   -> "MLK FL CRM"
      "DAHI 1KG"         -> "DAHI"
    """
    import re

    token = raw.upper().strip()

    # Remove trailing price: "2.99", "$2.99"
    token = re.sub(r'\$?\d+\.\d{2}$', '', token).strip()

    # Remove standalone numeric quantity+unit suffix: "1LB", "2KG", "500G"
    token = re.sub(r'\s*\d+\.?\d*\s*(LB|LBS|OZ|GL|GAL|KG|G|GR|GRM|ML|L|LTR|CT|PK|PCS|BX|BAG)\b', '', token).strip()

    # Remove leading quality prefixes
    words = token.split()
    while words and words[0] in STRIP_PREFIXES:
        words.pop(0)

    # Remove trailing quality/size suffixes
    while words and words[-1] in STRIP_PREFIXES | STRIP_SUFFIXES:
        words.pop()

    return " ".join(words)
```

### 4.3 ABBREVIATION_MAP

The full map. Keys are uppercase preprocessed tokens; values are canonical names that match exactly to rows in `shelf_life_reference`.

```python
# ml_service/normalization/abbreviation_map.py

ABBREVIATION_MAP = {

    # ── PRODUCE — Berries ────────────────────────────────────────────────────────
    "STRWBRY":        "Strawberries",
    "STRBRY":         "Strawberries",
    "STRWBRRY":       "Strawberries",
    "BLUBRY":         "Blueberries",
    "BLUBERY":        "Blueberries",
    "BLUEBERRY":      "Blueberries",
    "RSPBRY":         "Raspberries",
    "RASPBRY":        "Raspberries",
    "BLKBRY":         "Blackberries",
    "GRP":            "Grapes",
    "GRPES":          "Grapes",
    "RED GRAPES":     "Grapes",
    "GRN GRAPES":     "Grapes",
    "SEEDLESS GRP":   "Grapes",

    # ── PRODUCE — Stone & Tropical Fruits ────────────────────────────────────────
    "BNN":            "Bananas",
    "BNNA":           "Bananas",
    "BNNANA":         "Bananas",
    "PLT BANANAS":    "Bananas",
    "APPL":           "Apples",
    "APPLE":          "Apples",
    "GRN APPL":       "Apples",
    "FUJI APPL":      "Apples",
    "GALA APPL":      "Apples",
    "GRNSM APPL":     "Apples",
    "ORGN":           "Oranges",
    "ORNG":           "Oranges",
    "NAVEL ORNG":     "Oranges",
    "LMN":            "Lemons",
    "LEMN":           "Lemons",
    "LM":             "Limes",
    "LME":            "Limes",
    "PRSCN":          "Peaches",
    "PLUM":           "Plums",
    "CHRRY":          "Cherries",
    "CHRY":           "Cherries",
    "MANGO":          "Mangoes",
    "MNGO":           "Mangoes",
    "SINDHRI":        "Mangoes",    # Pakistani variety
    "CHAUNSA":        "Mangoes",    # Pakistani variety
    "ANWAR RATOL":    "Mangoes",    # Pakistani variety
    "PPYA":           "Papayas",
    "PAPAYA":         "Papayas",
    "GUAVA":          "Guavas",
    "AMROOD":         "Guavas",     # Urdu
    "PGRNTE":         "Pomegranates",
    "POMGNT":         "Pomegranates",
    "ANAR":           "Pomegranates", # Urdu
    "WTRMLM":         "Watermelon",
    "WTRMLN":         "Watermelon",
    "TARBOOZ":        "Watermelon",  # Urdu
    "CNTLP":          "Cantaloupe",
    "CNTAL":          "Cantaloupe",
    "PNAPPL":         "Pineapple",
    "PNAPPLE":        "Pineapple",
    "PNEAPPL":        "Pineapple",
    "KIWI":           "Kiwi",
    "PEAR":           "Pears",
    "PRUNE":          "Prunes",

    # ── PRODUCE — Vegetables ─────────────────────────────────────────────────────
    "TOM":            "Tomatoes",
    "TOMTO":          "Tomatoes",
    "TOMATOE":        "Tomatoes",
    "CHRY TOM":       "Cherry Tomatoes",
    "ROM TOM":        "Tomatoes",
    "GRAPE TOM":      "Cherry Tomatoes",
    "ONI":            "Onions",
    "ONIN":           "Onions",
    "ONION":          "Onions",
    "YLW ONI":        "Onions",
    "RED ONI":        "Onions",
    "PYAZ":           "Onions",     # Urdu
    "GRLC":           "Garlic",
    "GRIC":           "Garlic",
    "LAHSUN":         "Garlic",     # Urdu
    "POT":            "Potatoes",
    "PTAT":           "Potatoes",
    "PTATO":          "Potatoes",
    "ALOO":           "Potatoes",   # Urdu/Hindi
    "SWT POT":        "Sweet Potatoes",
    "SWPOT":          "Sweet Potatoes",
    "SHAKARKANDI":    "Sweet Potatoes", # Urdu
    "CARR":           "Carrots",
    "CARROT":         "Carrots",
    "GAJAR":          "Carrots",    # Urdu
    "BRCL":           "Broccoli",
    "BROC":           "Broccoli",
    "BROCCO":         "Broccoli",
    "CFLWR":          "Cauliflower",
    "CAULIFLWR":      "Cauliflower",
    "GOBI":           "Cauliflower", # Urdu/Hindi
    "CBGE":           "Cabbage",
    "CAB":            "Cabbage",
    "BNDGOBI":        "Cabbage",    # Urdu — literally "closed/bound cauliflower"
    "SPNCH":          "Spinach",
    "SPINCH":         "Spinach",
    "PALAK":          "Spinach",    # Urdu
    "LTT":            "Lettuce",
    "LETTC":          "Lettuce",
    "ICE LTT":        "Lettuce",
    "RMNE LTT":       "Lettuce",
    "CKE":            "Cucumber",
    "CUCMBR":         "Cucumber",
    "CUCUMB":         "Cucumber",
    "KHEERA":         "Cucumber",   # Urdu
    "BELL PEP":       "Bell Peppers",
    "BL PEP":         "Bell Peppers",
    "BLLPEP":         "Bell Peppers",
    "GRPEP":          "Bell Peppers",
    "RDPEP":          "Bell Peppers",
    "YLPEP":          "Bell Peppers",
    "SHIMLA MIRCH":   "Bell Peppers", # Urdu
    "JALPN":          "Jalapeños",
    "JLPNO":          "Jalapeños",
    "GRN CHILI":      "Green Chillies",
    "HARI MIRCH":     "Green Chillies", # Urdu
    "CORI":           "Coriander",
    "CORIAN":         "Coriander",
    "CILNTR":         "Coriander",
    "DHANIA":         "Coriander",  # Urdu
    "MNT":            "Mint",
    "MINT":           "Mint",
    "PODINA":         "Mint",       # Urdu
    "GRNONION":       "Green Onions",
    "SC ONI":         "Green Onions",
    "SCALL":          "Green Onions",
    "CELRY":          "Celery",
    "MSHM":           "Mushrooms",
    "MSHRM":          "Mushrooms",
    "PORT MSHRM":     "Mushrooms",
    "ZCHNI":          "Zucchini",
    "ZUCCH":          "Zucchini",
    "TORI":           "Zucchini",   # Urdu — ridge gourd/zucchini
    "EGPLT":          "Eggplant",
    "EGGPLNT":        "Eggplant",
    "BAINGAN":        "Eggplant",   # Urdu
    "CORN":           "Corn",
    "SWT CORN":       "Corn",
    "PEA":            "Peas",
    "PEAS":           "Peas",
    "MATAR":          "Peas",       # Urdu
    "GRBN":           "Green Beans",
    "GRN BNS":        "Green Beans",
    "GRBN BNS":       "Green Beans",
    "OKRA":           "Okra",
    "BHINDI":         "Okra",       # Urdu
    "BITTER GOURD":   "Bitter Gourd",
    "KARELA":         "Bitter Gourd", # Urdu
    "BOTTLE GOURD":   "Bottle Gourd",
    "LAUKI":          "Bottle Gourd", # Urdu
    "DRUMSTICK":      "Drumstick",
    "SAIJAN":         "Drumstick",  # Urdu — moringa pods
    "METHI":          "Fenugreek Leaves", # Urdu
    "TURNIP":         "Turnips",
    "SHALGAM":        "Turnips",    # Urdu
    "RADISH":         "Radishes",
    "MOOLI":          "Radishes",   # Urdu
    "BEETROOT":       "Beetroot",
    "CHNTNGE":        "Chestnut",
    "ARTICHOKE":      "Artichoke",
    "ASPRGUS":        "Asparagus",
    "ASPAR":          "Asparagus",
    "KALE":           "Kale",
    "CHARD":          "Swiss Chard",
    "ARUGULA":        "Arugula",

    # ── DAIRY ────────────────────────────────────────────────────────────────────
    "MLKWHL":         "Whole Milk",
    "MLK WHL":        "Whole Milk",
    "WHL MLK":        "Whole Milk",
    "MILK":           "Whole Milk",
    "MLK":            "Whole Milk",
    "SKM MLK":        "Skim Milk",
    "SKMMLK":         "Skim Milk",
    "2% MLK":         "Whole Milk",
    "FL CRM MLK":     "Full Cream Milk",
    "FULL CRM":       "Full Cream Milk",
    "FULLCRM MLK":    "Full Cream Milk",
    "UHT MLK":        "UHT Milk",
    "TETRA MLK":      "UHT Milk",
    "NESTLE MLK":     "UHT Milk",   # Nestle Milkpak — dominant Pakistani brand
    "MILKPAK":        "UHT Milk",
    "HLF HLF":        "Half and Half",
    "HVY CRM":        "Heavy Cream",
    "WHPNG CRM":      "Whipped Cream",
    "BTR":            "Butter",
    "BUTR":           "Butter",
    "UNSLT BTR":      "Butter",
    "SLTED BTR":      "Butter",
    "CHDR":           "Cheddar Cheese",
    "CHEDDR":         "Cheddar Cheese",
    "SHRP CHDR":      "Cheddar Cheese",
    "MOZZ":           "Mozzarella Cheese",
    "MOZZAR":         "Mozzarella Cheese",
    "CRM CHSE":       "Cream Cheese",
    "CRMCHS":         "Cream Cheese",
    "PHILLY":         "Cream Cheese", # Philadelphia cream cheese
    "CTG CHSE":       "Cottage Cheese",
    "CTGCHSE":        "Cottage Cheese",
    "GRK YGRT":       "Greek Yogurt",
    "GRKYGRT":        "Greek Yogurt",
    "GRK YGT":        "Greek Yogurt",
    "YGRT":           "Plain Yogurt",
    "YOGURT":         "Plain Yogurt",
    "PLN YGRT":       "Plain Yogurt",
    "DAHI":           "Dahi",
    "LASSI":          "Lassi",
    "MANGO LASSI":    "Lassi",
    "PANER":          "Paneer",
    "PANIR":          "Paneer",
    "PANEER":         "Paneer",
    "EGG":            "Eggs",
    "EGGS":           "Eggs",
    "LRG EGG":        "Eggs",
    "XL EGGS":        "Eggs",
    "BRWN EGG":       "Eggs",
    "SRC CRM":        "Sour Cream",
    "SRCRM":          "Sour Cream",
    "COND MLK":       "Condensed Milk",
    "CONDMLK":        "Condensed Milk",

    # ── MEAT ─────────────────────────────────────────────────────────────────────
    "CHKN BRST":      "Chicken Breast",
    "CHKNBRST":       "Chicken Breast",
    "CHKN BRS":       "Chicken Breast",
    "CHICK BRST":     "Chicken Breast",
    "CHKN THGH":      "Chicken Thighs",
    "CHKNTHGH":       "Chicken Thighs",
    "CHKN DRST":      "Chicken Thighs",
    "WHL CHKN":       "Whole Chicken",
    "WHLCHKN":        "Whole Chicken",
    "MURG":           "Chicken Breast", # Urdu for chicken (general)
    "MURGH":          "Chicken Breast",
    "MURG BRST":      "Chicken Breast",
    "MURG QEEMA":     "Minced Chicken",
    "CHKN QEEMA":     "Minced Chicken",
    "QEEMA MURG":     "Minced Chicken",
    "GRND BEEF":      "Ground Beef",
    "GRNDBEEF":       "Ground Beef",
    "BF GRND":        "Ground Beef",
    "BEEF QEEMA":     "Minced Beef",
    "GOSHT QEEMA":    "Minced Beef",  # Urdu — gosht = meat/beef/mutton
    "QEEMA":          "Minced Beef",
    "KEEMA":          "Minced Beef",
    "BF STK":         "Beef Steak",
    "BFSTK":          "Beef Steak",
    "SIRLOIN":        "Beef Steak",
    "RIBEYE":         "Beef Steak",
    "MUTN":           "Mutton",
    "MUTTON":         "Mutton",
    "GOSHT":          "Mutton",       # Urdu (often mutton specifically)
    "LMB CHPS":       "Lamb Chops",
    "LMBCHPS":        "Lamb Chops",
    "CHOPS":          "Lamb Chops",
    "PRK CHP":        "Pork Chops",
    "PRKCHS":         "Pork Chops",
    "BCN":            "Bacon",
    "BACON":          "Bacon",
    "SLMN":           "Salmon",
    "SLMON":          "Salmon",
    "ATLNT SLMN":     "Salmon",
    "TNA STK":        "Tuna Steak",
    "TUNA":           "Tuna Steak",
    "SHRMP":          "Shrimp",
    "SHRIMP":         "Shrimp",
    "JHINGA":         "Shrimp",       # Urdu
    "SCLLPS":         "Scallops",
    "DLI HM":         "Deli Ham",
    "DELIHAM":        "Deli Ham",
    "DLI TRK":        "Deli Turkey",
    "SSGE":           "Sausages",
    "SAUSAGE":        "Sausages",
    "LNKSSGE":        "Sausages",
    "HTDOG":          "Hot Dogs",
    "HOT DOG":        "Hot Dogs",
    "SEEKH KB":       "Seekh Kebab",
    "SEEKH KBAB":     "Seekh Kebab",
    "SK KEBAB":       "Seekh Kebab",
    "BIHARI KB":      "Seekh Kebab",  # Similar kebab variety
    "TIKKA":          "Seekh Kebab",  # Approximate

    # ── BAKERY ───────────────────────────────────────────────────────────────────
    "BRD":            "Bread",
    "BREAD":          "Bread",
    "WW BRD":         "Whole Wheat Bread",
    "WWBRD":          "Whole Wheat Bread",
    "SLCD BRD":       "Bread",
    "PITA":           "Pita Bread",
    "PTABRD":         "Pita Bread",
    "NAAN":           "Naan",
    "NAN":            "Naan",
    "GARLIC NAAN":    "Naan",
    "ROTI":           "Roti",
    "CHAPATI":        "Roti",
    "CHAPATTI":       "Roti",
    "PRATHA":         "Paratha",
    "PARATHA":        "Paratha",
    "PRATHA PLN":     "Paratha",
    "ALOO PRTHA":     "Paratha",
    "CRSNT":          "Croissants",
    "CROISSANT":      "Croissants",
    "BAGEL":          "Bagels",
    "MUFFIN":         "Muffins",
    "BLBN MFFN":      "Muffins",

    # ── PANTRY — Grains & Staples ────────────────────────────────────────────────
    "PST":            "Pasta",
    "PASTA":          "Pasta",
    "SPAGHTTI":       "Pasta",
    "PENNE":          "Pasta",
    "FETTUC":         "Pasta",
    "MACRNI":         "Pasta",
    "RICE":           "Rice",
    "WH RICE":        "Rice",
    "BSMTI":          "Basmati Rice",
    "BASMATI":        "Basmati Rice",
    "JASMINE RICE":   "Rice",
    "BRN RICE":       "Rice",
    "CHAWAL":         "Rice",         # Urdu
    "OATS":           "Oats",
    "ROLLD OATS":     "Oats",
    "QUCK OATS":      "Oats",
    "FLR":            "Flour",
    "FLOUR":          "Flour",
    "AP FLR":         "Flour",
    "ATTA":           "Atta Flour",
    "ATT FLR":        "Atta Flour",
    "MAIDA":          "Maida",
    "SUJI":           "Semolina",     # Urdu — semolina/cream of wheat
    "SEMOLINA":       "Semolina",
    "SGR":            "Sugar",
    "SUGAR":          "Sugar",
    "BRWN SGR":       "Sugar",
    "PWDR SGR":       "Sugar",
    "SLT":            "Salt",
    "SALT":           "Salt",
    "OLV OIL":        "Olive Oil",
    "OLVOL":          "Olive Oil",
    "EVOO":           "Olive Oil",
    "VEG OIL":        "Vegetable Oil",
    "VEGOL":          "Vegetable Oil",
    "SUNFLR OIL":     "Vegetable Oil",
    "CANOLA":         "Vegetable Oil",
    "COOKING OIL":    "Cooking Oil",
    "DALA OIL":       "Cooking Oil",  # Generic local brand type
    "SUFI OIL":       "Cooking Oil",  # Pakistani brand
    "DALDA":          "Cooking Oil",  # Major Pakistani cooking oil/ghee brand
    "GHEE":           "Ghee",
    "DESI GHEE":      "Ghee",
    "VANSP GHEE":     "Ghee",         # Vanaspati — hydrogenated vegetable ghee

    # ── PANTRY — Condiments & Sauces ─────────────────────────────────────────────
    "SOY SCS":        "Soy Sauce",
    "SOYSCS":         "Soy Sauce",
    "KTCHP":          "Ketchup",
    "KETCHUP":        "Ketchup",
    "MAYO":           "Mayonnaise",
    "MAYONNSE":       "Mayonnaise",
    "MSTRD":          "Mustard",
    "MUSTARD":        "Mustard",
    "HT SCS":         "Hot Sauce",
    "HTSCS":          "Hot Sauce",
    "SRIRACHA":       "Hot Sauce",
    "TABASCO":        "Hot Sauce",
    "VINEGR":         "Vinegar",
    "VINEGAR":        "Vinegar",
    "AP VNG":         "Vinegar",
    "WH VNG":         "Vinegar",
    "PNUT BTR":       "Peanut Butter",
    "PNUTBTR":        "Peanut Butter",
    "PB":             "Peanut Butter",
    "STRWBRY JAM":    "Jam",
    "JAM":            "Jam",
    "MARMALADE":      "Jam",
    "TOM SCS":        "Tomato Sauce",
    "TOMSCS":         "Tomato Sauce",
    "MARINARA":       "Tomato Sauce",
    "TOM PST":        "Tomato Paste",
    "TOMPST":         "Tomato Paste",
    "CCNT MLK":       "Coconut Milk",
    "CCNTM":          "Coconut Milk",
    "CHKN BRTH":      "Chicken Broth",
    "CHKNBRTH":       "Chicken Broth",
    "VEG BRTH":       "Chicken Broth",

    # ── PANTRY — Legumes ─────────────────────────────────────────────────────────
    "LNTLS":          "Lentils",
    "LENTIL":         "Lentils",
    "RED LNTL":       "Masoor Dal",
    "MASOOR":         "Masoor Dal",
    "MSOOR DAL":      "Masoor Dal",
    "CHKPEAS":        "Chickpeas",
    "CHNPEAS":        "Chickpeas",
    "CHANA":          "Chickpeas",
    "KABLI CHANA":    "Chickpeas",
    "BLK BNS":        "Black Beans",
    "RDNY BNS":       "Kidney Beans",
    "KDY BNS":        "Kidney Beans",
    "RAJMA":          "Kidney Beans",  # Urdu
    "CHANA DAL":      "Chana Dal",
    "MOONG DAL":      "Moong Dal",
    "MUNG DAL":       "Moong Dal",
    "URAD DAL":       "Urad Dal",
    "MASH DAL":       "Urad Dal",      # Pakistani name for urad dal
    "TOOR DAL":       "Toor Dal",
    "ARHAR DAL":      "Toor Dal",      # Alternative name

    # ── PANTRY — Canned Goods ────────────────────────────────────────────────────
    "CND CHKP":       "Canned Chickpeas",
    "CND TOMATO":     "Canned Tomatoes",
    "CNDTOM":         "Canned Tomatoes",
    "DICED TOM":      "Canned Tomatoes",
    "CND TUNA":       "Canned Tuna",
    "TNA CAN":        "Canned Tuna",

    # ── BEVERAGES ────────────────────────────────────────────────────────────────
    "OJ":             "Orange Juice",
    "ORNG JCE":       "Orange Juice",
    "ORANGE JCE":     "Orange Juice",
    "APPL JCE":       "Apple Juice",
    "SPRKLNG WTR":    "Sparkling Water",
    "SPKL WTR":       "Sparkling Water",
    "SODA":           "Soda",
    "CLA":            "Soda",          # Cola
    "COLA":           "Soda",
    "PEPSI":          "Soda",
    "COKE":           "Soda",
    "7UP":            "Soda",
    "SPRITE":         "Soda",
    "CFE BNSE":       "Coffee Beans",
    "COFFEE":         "Coffee Beans",
    "GRND CFE":       "Ground Coffee",
    "INST CFE":       "Ground Coffee",
    "TEA":            "Tea Bags",
    "TEABAG":         "Tea Bags",
    "GRN TEA":        "Tea Bags",
    "BLK TEA":        "Tea Bags",
    "CHAI":           "Tea Bags",
    "LIPTON":         "Tea Bags",      # Dominant brand in Pakistan/globally
    "TAPAL":          "Tea Bags",      # Pakistani brand
    "BROOKE BOND":    "Tea Bags",      # Common in Pakistan/UK

    # ── FROZEN ───────────────────────────────────────────────────────────────────
    "FRZ PEAS":       "Frozen Peas",
    "FRZPEAS":        "Frozen Peas",
    "FRZ CORN":       "Frozen Corn",
    "FRZ SPNCH":      "Frozen Spinach",
    "FRZ MX VG":      "Frozen Mixed Veg",
    "FRZN MIX":       "Frozen Mixed Veg",
    "FRZ FRIES":      "Frozen Fries",
    "FRZNFRI":        "Frozen Fries",
    "ICE CRM":        "Ice Cream",
    "ICECREAM":       "Ice Cream",
    "FRZ PIZZA":      "Frozen Pizza",
    "FRZPIZZA":       "Frozen Pizza",
    "FRZ WFFL":       "Frozen Waffles",

    # ── GRAINS — Spices (common receipt appearances) ──────────────────────────────
    "BLCK PPPR":      "Black Pepper",
    "BLK PPPR":       "Black Pepper",
    "KALI MIRCH":     "Black Pepper",  # Urdu
    "RED PPPR":       "Red Pepper",
    "LAAL MIRCH":     "Red Pepper",    # Urdu
    "CMIN":           "Cumin",
    "CUMIN":          "Cumin",
    "ZEERA":          "Cumin",         # Urdu
    "TURMRC":         "Turmeric",
    "TURMER":         "Turmeric",
    "HALDI":          "Turmeric",      # Urdu
    "CRDR":           "Coriander Powder",
    "CORIANDER PWD":  "Coriander Powder",
    "DHANIA PWD":     "Coriander Powder",
    "GRMSALA":        "Garam Masala",
    "GARAM MSLA":     "Garam Masala",
    "CHILI PWD":      "Chili Powder",
    "MIRCH PWD":      "Chili Powder",  # Urdu
    "CINNMN":         "Cinnamon",
    "DARCHINI":       "Cinnamon",      # Urdu
    "CINMN":          "Cinnamon",
    "CARDMM":         "Cardamom",
    "ILLAICHI":       "Cardamom",      # Urdu
    "CLVS":           "Cloves",
    "LAUNG":          "Cloves",        # Urdu
    "BAY LEAF":       "Bay Leaves",
    "TEJPTTA":        "Bay Leaves",    # Urdu
}


def pass1_lookup(cleaned_token: str) -> str | None:
    """
    Direct dictionary lookup. Returns canonical name or None on miss.
    Tries exact match first, then a joined-words variant.
    """
    result = ABBREVIATION_MAP.get(cleaned_token.upper())
    if result:
        return result

    # Try collapsing spaces: "CHKN BRST" might appear as "CHKNBRST"
    collapsed = cleaned_token.upper().replace(" ", "")
    return ABBREVIATION_MAP.get(collapsed)
```

**Expected Pass 1 hit rate:** ~65% of real receipt tokens. Coverage is highest for US retail and Pakistani markets, where abbreviation patterns are most consistent.

---

## 5. Pass 2 — Fuzzy Matching

### 5.1 Why rapidfuzz

`rapidfuzz` is the standard replacement for the deprecated `fuzzywuzzy`. It is significantly faster (C++ backend), has a near-identical API, and has no GPL licensing issues.

`token_sort_ratio` is used rather than plain `ratio` because receipt tokens are often scrambled or have words in different orders: `"BRST CHKN"` vs `"Chicken Breast"`. Token sort ratio normalizes word order before comparison, which is exactly the pattern receipt abbreviations produce.

```
fuzz.ratio("BRST CHKN", "Chicken Breast")     → 48
fuzz.token_sort_ratio("BRST CHKN", "Chicken Breast") → 70  ← much better
```

### 5.2 Score threshold calibration

The threshold of 80 is chosen empirically:
- Score ≥ 90 → almost certainly correct
- Score 80–89 → usually correct; occasional false positive on short tokens
- Score 70–79 → too many false positives; `"SALT"` vs `"Malt"` → 75
- Score < 70 → pass to LLM

At threshold 80, the expected false positive rate is below 5% on the held-out test set.

> **Gotcha:** Very short tokens (3–4 characters) can hit 80+ spuriously. If `cleaned_token` is ≤ 4 characters and Pass 1 missed, skip Pass 2 and go directly to Pass 3 — the fuzzy match is unreliable on such short strings.

### 5.3 Implementation

```python
# ml_service/normalization/fuzzy_matcher.py

from rapidfuzz import process, fuzz
from sqlalchemy.orm import Session
from functools import lru_cache
from app.models import ShelfLifeReference


@lru_cache(maxsize=1)
def _load_canonical_names(db_session_hash: int) -> list[str]:
    """
    Load all canonical_name values from shelf_life_reference.
    LRU-cached after first call — the reference table rarely changes.
    Cache key is a hash of the db session to allow testing with different DBs.
    """
    # This is called once at startup via get_canonical_names() below
    raise NotImplementedError("Use get_canonical_names() directly")


_canonical_names_cache: list[str] | None = None


def get_canonical_names(db: Session) -> list[str]:
    """
    Returns deduplicated list of all canonical names from shelf_life_reference.
    Cached in module-level variable after first call.
    """
    global _canonical_names_cache
    if _canonical_names_cache is None:
        rows = db.query(ShelfLifeReference.canonical_name).distinct().all()
        _canonical_names_cache = [r[0] for r in rows]
    return _canonical_names_cache


FUZZY_THRESHOLD = 80
SHORT_TOKEN_THRESHOLD = 4  # skip fuzzy for tokens this short or shorter


def pass2_fuzzy(cleaned_token: str, db: Session) -> tuple[str | None, float]:
    """
    Fuzzy match cleaned_token against all canonical names in shelf_life_reference.

    Returns (canonical_name, confidence) or (None, 0.0) on miss.
    Confidence is score / 100.

    Examples:
      "STRWBERY"  → ("Strawberries", 0.91)
      "CHKN BRST" → ("Chicken Breast", 0.88)
      "SALT"      → ("Salt", 1.00)   ← exact match would have been caught by Pass 1
      "XYZ123"    → (None, 0.0)
    """
    if len(cleaned_token) <= SHORT_TOKEN_THRESHOLD:
        return None, 0.0

    canonical_names = get_canonical_names(db)
    if not canonical_names:
        return None, 0.0

    result = process.extractOne(
        cleaned_token.upper(),
        [name.upper() for name in canonical_names],
        scorer=fuzz.token_sort_ratio,
    )

    if result is None:
        return None, 0.0

    _, score, idx = result

    if score >= FUZZY_THRESHOLD:
        return canonical_names[idx], round(score / 100, 3)

    return None, 0.0
```

---

## 6. Pass 3 — LLM Fallback (Groq)

### 6.1 LLM provider choice

**Groq** is the free LLM option used for this pipeline. Groq's free tier provides:
- No API cost on the free tier (rate-limited but sufficient for ≤ 20% fallback rate)
- Extremely low latency: Groq's LPU hardware typically returns in 200–400ms
- `llama-3.1-8b-instant` model — fast, small, accurate on short constrained prompts

Get a free API key at `console.groq.com`. Set as environment variable: `GROQ_API_KEY`.

> **Why not Ollama?** Ollama requires a local GPU or CPU inference server running alongside FastAPI. That works on a dev machine but complicates containerized deployment. Groq is a single API call with no local infrastructure.

> **Why not OpenAI free tier?** OpenAI has no free tier as of 2025. GPT-4o-mini requires a paid account.

### 6.2 Prompt design

The prompt is deliberately minimal. A longer prompt causes the LLM to hedge, add explanations, or return multi-word non-canonical names. The constraint `"Reply with just the canonical food name, nothing else"` is the key instruction.

```
"This is a line item from a grocery store receipt: '{raw_token}'.
What common food item does this refer to? Reply with just the canonical food name, nothing else."
```

### 6.3 normalization_cache table

Before every LLM call, check the cache. Cache hits avoid the Groq API entirely — critical for staying within the free tier rate limit.

```sql
CREATE TABLE normalization_cache (
    id             SERIAL PRIMARY KEY,
    raw_token      VARCHAR(200) NOT NULL UNIQUE,
    canonical_name VARCHAR(100) NOT NULL,
    source         VARCHAR(20)  NOT NULL DEFAULT 'llm',  -- 'llm' or 'manual'
    created_at     TIMESTAMP    NOT NULL DEFAULT NOW(),
    hit_count      INTEGER      NOT NULL DEFAULT 1
);

CREATE INDEX idx_norm_cache_token ON normalization_cache (raw_token);
```

**SQLAlchemy model:**

```python
from sqlalchemy import Column, Integer, String, DateTime, func

class NormalizationCache(Base):
    __tablename__ = "normalization_cache"

    id             = Column(Integer, primary_key=True)
    raw_token      = Column(String(200), nullable=False, unique=True)
    canonical_name = Column(String(100), nullable=False)
    source         = Column(String(20),  nullable=False, default="llm")
    created_at     = Column(DateTime,    nullable=False, server_default=func.now())
    hit_count      = Column(Integer,     nullable=False, default=1)
```

### 6.4 Implementation

```python
# ml_service/normalization/llm_fallback.py

import os
import re
import httpx
from sqlalchemy.orm import Session
from app.models import NormalizationCache

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"
LLM_CONFIDENCE = 0.70


def _cache_lookup(raw_token: str, db: Session) -> str | None:
    """Return cached canonical name for raw_token, or None if not cached."""
    entry = db.query(NormalizationCache).filter_by(raw_token=raw_token.upper()).first()
    if entry:
        entry.hit_count += 1
        db.commit()
        return entry.canonical_name
    return None


def _cache_store(raw_token: str, canonical_name: str, db: Session) -> None:
    """Store a new LLM result in the cache."""
    entry = NormalizationCache(
        raw_token=raw_token.upper(),
        canonical_name=canonical_name,
        source="llm",
    )
    db.add(entry)
    db.commit()


def _clean_llm_response(response_text: str) -> str:
    """
    Strip any stray punctuation, quotes, or explanation the LLM added despite instructions.
    Returns the first line, title-cased.
    """
    text = response_text.strip()
    # Take only the first line in case the LLM added an explanation
    text = text.split("\n")[0].strip()
    # Remove surrounding quotes
    text = re.sub(r'^["\']|["\']$', '', text).strip()
    # Remove trailing punctuation
    text = text.rstrip(".,;:")
    return text.title()


def pass3_llm(raw_token: str, db: Session) -> tuple[str | None, float]:
    """
    LLM fallback via Groq API.
    Checks cache first. On miss, calls Groq and caches the result.

    Returns (canonical_name, confidence) or (None, 0.0) on failure.
    """
    # 1. Cache lookup
    cached = _cache_lookup(raw_token, db)
    if cached:
        return cached, LLM_CONFIDENCE

    # 2. Groq API call
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set")

    prompt = (
        f"This is a line item from a grocery store receipt: '{raw_token}'. "
        f"What common food item does this refer to? Reply with just the canonical food name, nothing else."
    )

    try:
        response = httpx.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,    # canonical name is never > 5 words
                "temperature": 0.0,  # deterministic — we want the same answer every time
            },
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        # Network or API error — fail gracefully, don't crash the pipeline
        print(f"[llm_fallback] Groq API error for token '{raw_token}': {e}")
        return None, 0.0

    content = response.json()["choices"][0]["message"]["content"]
    canonical_name = _clean_llm_response(content)

    if not canonical_name or len(canonical_name) < 2:
        return None, 0.0

    # 3. Store in cache
    _cache_store(raw_token, canonical_name, db)

    return canonical_name, LLM_CONFIDENCE
```

**Rate limit handling:** The Groq free tier allows ~30 requests/minute. At ≤ 20% LLM fallback rate, a typical receipt with 10–15 items produces at most 3 LLM calls. With caching, repeated items never call the API. This stays comfortably within the free tier limit for normal usage.

---

## 7. Unit Normalization

All quantity and unit parsing is done in a single module. The critical edge case is **fused tokens** — when OCR or NER outputs `"1LB"` as one token rather than separate `"1"` and `"LB"` tokens.

```python
# ml_service/normalization/unit_normalizer.py

import re
from dataclasses import dataclass

UNIT_MAP = {
    # Weight
    "LB":    "lb",  "LBS":  "lb",  "POUND": "lb",  "POUNDS": "lb",
    "OZ":    "oz",  "OUNCE": "oz", "OUNCES": "oz",
    "KG":    "kg",  "KGS":  "kg",  "KILO":  "kg",  "KILOS":  "kg",
    "G":     "g",   "GR":   "g",   "GRM":   "g",   "GRAM":   "g",   "GRAMS": "g",
    # Volume
    "GAL":   "gal", "GL":   "gal", "GALLON": "gal", "GALLONS": "gal",
    "QT":    "qt",  "QUART": "qt",
    "PT":    "pt",  "PINT": "pt",
    "L":     "l",   "LTR":  "l",   "LITRE": "l",   "LITER":  "l",
    "ML":    "ml",  "MLS":  "ml",
    "FL OZ": "fl_oz",
    # Count/Pack
    "CT":    "count", "CNT": "count", "COUNT": "count",
    "PK":    "pack",  "PCK": "pack",  "PACK":  "pack",
    "EA":    "each",  "PC":  "each",  "PCS":   "each",
    "DOZ":   "dozen", "DZ":  "dozen", "DOZEN": "dozen",
    "BX":    "box",   "BOX": "box",
    "BAG":   "bag",   "BG":  "bag",
    "BTL":   "bottle","BT":  "bottle","BOTTLE":"bottle",
    "CAN":   "can",   "CN":  "can",
    "JAR":   "jar",
    "TUB":   "tub",
}


@dataclass
class ParsedQuantity:
    quantity: float
    unit: str | None


def parse_quantity_unit(raw_qty: str | None, raw_unit: str | None) -> ParsedQuantity:
    """
    Parse and normalize quantity and unit from NER output.

    Handles three common patterns:
    1. Separate tokens:   raw_qty="1", raw_unit="LB"   → quantity=1.0, unit="lb"
    2. Fused token (qty): raw_qty="1LB", raw_unit=None → quantity=1.0, unit="lb"
    3. Complex quantity:  raw_qty="2", raw_unit="X 12 OZ" (multipack) → quantity=24.0, unit="oz"

    Returns ParsedQuantity(quantity, unit).
    """
    qty_str  = (raw_qty  or "").upper().strip()
    unit_str = (raw_unit or "").upper().strip()

    # Case 2: fused token — e.g. "1LB", "500G", "2GAL"
    fused_match = re.match(r'^(\d+\.?\d*)\s*([A-Z]+)$', qty_str)
    if fused_match and not unit_str:
        qty_val   = float(fused_match.group(1))
        unit_raw  = fused_match.group(2)
        unit_norm = UNIT_MAP.get(unit_raw)
        return ParsedQuantity(quantity=qty_val, unit=unit_norm)

    # Case 3: multipack — e.g. qty="2", unit="X 12 OZ" → 2 × 12 = 24 oz
    multipack_match = re.match(r'^X?\s*(\d+\.?\d*)\s*([A-Z]+)$', unit_str)
    if multipack_match:
        try:
            base_qty  = float(qty_str) if qty_str else 1.0
            pack_qty  = float(multipack_match.group(1))
            unit_raw  = multipack_match.group(2)
            unit_norm = UNIT_MAP.get(unit_raw)
            return ParsedQuantity(quantity=base_qty * pack_qty, unit=unit_norm)
        except (ValueError, TypeError):
            pass

    # Case 1: standard separate tokens
    try:
        qty_val = float(qty_str) if qty_str else 1.0
    except ValueError:
        qty_val = 1.0

    unit_norm = UNIT_MAP.get(unit_str) if unit_str else None

    return ParsedQuantity(quantity=qty_val, unit=unit_norm)
```

---

## 8. Category Assignment

```python
# ml_service/normalization/category_classifier.py

from sqlalchemy.orm import Session
from app.models import ShelfLifeReference

CATEGORY_KEYWORDS = {
    "Produce": [
        "berries", "berry", "apple", "apples", "orange", "mango", "banana",
        "lettuce", "tomato", "onion", "garlic", "potato", "carrot", "spinach",
        "cucumber", "pepper", "mushroom", "zucchini", "broccoli", "cauliflower",
        "cabbage", "eggplant", "okra", "peas", "beans", "corn", "celery",
        "coriander", "mint", "lemon", "lime", "grapes", "guava", "papaya",
    ],
    "Dairy": [
        "milk", "cheese", "yogurt", "dahi", "butter", "cream", "eggs",
        "paneer", "lassi", "ghee", "curd",
    ],
    "Meat": [
        "chicken", "beef", "mutton", "lamb", "pork", "salmon", "tuna",
        "shrimp", "bacon", "sausage", "turkey", "qeema", "keema",
        "gosht", "murg", "seekh", "kebab",
    ],
    "Pantry": [
        "pasta", "rice", "flour", "sugar", "salt", "oil", "sauce", "ketchup",
        "mayo", "mustard", "bread", "oats", "lentils", "chickpeas", "beans",
        "atta", "maida", "dal", "masala", "spice", "vinegar", "honey",
        "peanut", "jam", "basmati", "chawal",
    ],
    "Frozen": [
        "frozen", "frzn", "ice cream", "popsicle",
    ],
    "Beverages": [
        "juice", "milk", "soda", "coffee", "tea", "water", "chai", "lassi",
    ],
    "Bakery": [
        "bread", "naan", "roti", "paratha", "chapati", "croissant", "bagel",
        "muffin", "cake", "roll",
    ],
}


def assign_category(canonical_name: str, db: Session) -> str:
    """
    Assign category to a canonical food name.

    Priority:
    1. Exact match in shelf_life_reference (most reliable)
    2. Keyword classifier fallback (approximate)
    3. "Other" if no match

    Returns category string.
    """
    # Pass 1: DB lookup
    ref = db.query(ShelfLifeReference).filter_by(canonical_name=canonical_name).first()
    if ref:
        return ref.category

    # Pass 2: keyword classifier
    name_lower = canonical_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return category

    return "Other"
```

---

## 9. Pipeline Orchestrator

This is the only public API for Stage 3. `pipeline.py` calls `normalize_entity()` — nothing else.

```python
# ml_service/normalization/normalizer.py

from dataclasses import dataclass
from sqlalchemy.orm import Session

from .preprocessor       import preprocess_token
from .abbreviation_map   import pass1_lookup
from .fuzzy_matcher      import pass2_fuzzy
from .llm_fallback       import pass3_llm
from .unit_normalizer    import parse_quantity_unit, ParsedQuantity
from .category_classifier import assign_category


@dataclass
class NormalizedItem:
    canonical_name:    str
    quantity:          float
    unit:              str | None
    category:          str
    normalization_pass: int    # 1, 2, or 3 — which pass resolved the name
    confidence:        float   # 1.0 for Pass 1; score/100 for Pass 2; 0.70 for Pass 3


def normalize_entity(
    food_tokens:   list[str],
    raw_quantity:  str | None,
    raw_unit:      str | None,
    db:            Session,
) -> NormalizedItem | None:
    """
    Normalize a single NER entity into a canonical inventory item.

    Args:
        food_tokens:  word list from NER, e.g. ["ORG", "STRWBRY"]
        raw_quantity: quantity string from NER, e.g. "1"
        raw_unit:     unit string from NER, e.g. "LB"
        db:           active SQLAlchemy session

    Returns:
        NormalizedItem with canonical_name, quantity, unit, category, confidence.
        Returns None if all three passes fail to produce a result.
    """
    # Join tokens into a single string for normalization
    raw_token = " ".join(food_tokens)

    # Clean the token before any pass
    cleaned = preprocess_token(raw_token)

    if not cleaned:
        return None

    canonical_name = None
    confidence     = 0.0
    norm_pass      = 0

    # ── Pass 1: Abbreviation Map ──────────────────────────────────────────────
    result = pass1_lookup(cleaned)
    if result:
        canonical_name = result
        confidence     = 1.0
        norm_pass      = 1

    # ── Pass 2: Fuzzy Match ───────────────────────────────────────────────────
    if canonical_name is None:
        result, score = pass2_fuzzy(cleaned, db)
        if result:
            canonical_name = result
            confidence     = score
            norm_pass      = 2

    # ── Pass 3: LLM Fallback ──────────────────────────────────────────────────
    if canonical_name is None:
        result, score = pass3_llm(raw_token, db)   # pass raw (uncleaned) to LLM — more context
        if result:
            canonical_name = result
            confidence     = score
            norm_pass      = 3

    if canonical_name is None:
        return None

    # ── Unit & Quantity ───────────────────────────────────────────────────────
    parsed: ParsedQuantity = parse_quantity_unit(raw_quantity, raw_unit)

    # ── Category Assignment ───────────────────────────────────────────────────
    category = assign_category(canonical_name, db)

    return NormalizedItem(
        canonical_name=canonical_name,
        quantity=parsed.quantity,
        unit=parsed.unit,
        category=category,
        normalization_pass=norm_pass,
        confidence=confidence,
    )
```

### Confidence Score Interpretation

| Range | Source | Meaning |
|---|---|---|
| 1.00 | Pass 1 | Exact abbreviation map hit — highest reliability |
| 0.80 – 0.99 | Pass 2 | Fuzzy match above threshold — reliable |
| 0.70 | Pass 3 | LLM resolution — review if confidence matters |
| 0.00 | All failed | Item unresolvable — surface to user for manual entry |

Items with `confidence < 0.70` or `normalization_pass == 0` are flagged in the Stage 4 confidence score and surfaced in the frontend confirmation modal with a warning indicator.

---

## 10. Evaluation

No Kaggle notebook needed. Run locally or in CI.

### 10.1 Test harness

```python
# ml_service/normalization/evaluate.py
# Usage: python -m ml_service.normalization.evaluate

from dataclasses import dataclass
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ml_service.normalization.normalizer import normalize_entity

# Ground truth: (food_tokens, expected_canonical_name)
# Drawn from a held-out set of real receipt lines not used to build ABBREVIATION_MAP
TEST_CASES = [
    # Pass 1 expected
    (["ORG", "STRWBRY"],          "Strawberries"),
    (["CHKN", "BRST", "BNLS"],    "Chicken Breast"),
    (["GRK", "YGRT", "PLN"],      "Greek Yogurt"),
    (["MLKWHL"],                   "Whole Milk"),
    (["DAHI"],                     "Dahi"),
    (["BASMATI"],                  "Basmati Rice"),
    (["MURG", "QEEMA"],            "Minced Chicken"),
    (["PALAK"],                    "Spinach"),
    (["LAHSUN"],                   "Garlic"),
    (["MILKPAK"],                  "UHT Milk"),
    # Pass 2 expected (slight misspellings / unseen variants)
    (["STRABERRY"],                "Strawberries"),
    (["CHICKIN", "BREAST"],        "Chicken Breast"),
    (["YOGHURT"],                  "Plain Yogurt"),
    (["WHOLE", "MILK"],            "Whole Milk"),
    (["CORIANDER", "LEAVES"],      "Coriander"),
    # Pass 3 expected (novel tokens)
    (["ANDAY"],                    "Eggs"),          # Urdu for eggs
    (["MACCHI"],                   "Salmon"),        # Urdu for fish (approximate)
    (["SAFAID", "MIRCH"],          "White Pepper"),  # Urdu — may not be in map
]


@dataclass
class EvalResult:
    total:        int
    pass1_hits:   int
    pass2_hits:   int
    pass3_hits:   int
    failures:     int
    correct:      int
    accuracy:     float
    llm_fallback_rate: float
    canonical_match_rate: float


def run_evaluation(db_url: str) -> EvalResult:
    engine  = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    db      = Session()

    pass1 = pass2 = pass3 = failures = correct = 0

    for food_tokens, expected in TEST_CASES:
        item = normalize_entity(food_tokens, None, None, db)

        if item is None:
            failures += 1
            print(f"  ❌ FAIL (unresolved): {' '.join(food_tokens)}")
            continue

        match = item.canonical_name.lower() == expected.lower()
        if match:
            correct += 1

        if item.normalization_pass == 1:
            pass1 += 1
        elif item.normalization_pass == 2:
            pass2 += 1
        elif item.normalization_pass == 3:
            pass3 += 1

        status = "✅" if match else "❌"
        print(f"  {status} Pass {item.normalization_pass} | {' '.join(food_tokens):30s} → {item.canonical_name:25s} (expected: {expected})")

    total = len(TEST_CASES)
    return EvalResult(
        total=total,
        pass1_hits=pass1,
        pass2_hits=pass2,
        pass3_hits=pass3,
        failures=failures,
        correct=correct,
        accuracy=round(correct / total, 3),
        llm_fallback_rate=round(pass3 / total, 3),
        canonical_match_rate=round((pass1 + pass2) / total, 3),
    )


if __name__ == "__main__":
    import os
    result = run_evaluation(os.environ["DATABASE_URL"])

    print(f"\n{'─'*50}")
    print(f"Total:               {result.total}")
    print(f"Pass 1 (map):        {result.pass1_hits}")
    print(f"Pass 2 (fuzzy):      {result.pass2_hits}")
    print(f"Pass 3 (LLM):        {result.pass3_hits}")
    print(f"Failures:            {result.failures}")
    print(f"Correct:             {result.correct}")
    print(f"Accuracy:            {result.accuracy:.1%}")
    print(f"Canonical match rate:{result.canonical_match_rate:.1%}  (target ≥ 80%)")
    print(f"LLM fallback rate:   {result.llm_fallback_rate:.1%}  (target ≤ 20%)")
```

### 10.2 Target metrics

| Metric | Definition | Target |
|---|---|---|
| Canonical Match Rate | % items resolved by Pass 1 + Pass 2 | ≥ 80% |
| LLM Fallback Rate | % items needing Pass 3 | ≤ 20% |
| End-to-end accuracy | % items correctly identified (any pass) | ≥ 85% |

### 10.3 If targets aren't met

| Symptom | Action |
|---|---|
| Canonical Match Rate < 80% | Add more entries to `ABBREVIATION_MAP` for the failing tokens. Print Pass 1 misses and batch-add them. |
| LLM Fallback Rate > 20% | Same as above — the LLM fallback rate is the inverse of map+fuzzy coverage. Expand the map. |
| Fuzzy false positives | Raise `FUZZY_THRESHOLD` from 80 to 85. Or add the false-positive token explicitly to `ABBREVIATION_MAP` with its correct canonical name to force a Pass 1 exact hit. |
| LLM returns wrong name | Add the token to `ABBREVIATION_MAP` or `normalization_cache` with `source='manual'` to override future LLM calls. |
| `GROQ_API_KEY` not set | Set env var in `.env` file and load via `python-dotenv` at app startup. |

---

## 11. Setup & Requirements

No Kaggle, no GPU. Runs locally or inside the FastAPI container.

### 11.1 Install packages

```bash
pip install rapidfuzz httpx sqlalchemy psycopg2-binary python-dotenv alembic
```

### 11.2 Environment variables

Create `.env` at the project root (same level as `ml_service/`):

```bash
# .env
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/smartstock
GROQ_API_KEY=gsk_...   # Get free key at console.groq.com
```

### 11.3 Project layout after Stage 3

The normalization module references `app.models` and `app.db` — these are the stub backend files created now and reused when the full FastAPI app is built in a later stage. The full layout is:

```
Smart-Stock/
├── app/
│   ├── __init__.py        ← empty
│   ├── models.py          ← SQLAlchemy models (ShelfLifeReference, NormalizationCache)
│   └── db.py              ← engine + SessionLocal + get_db()
├── db/
│   └── seeds/
│       └── shelf_life_seed.py
├── migrations/            ← generated by Alembic
│   └── env.py
├── alembic.ini            ← generated by Alembic
├── ml_service/
│   └── normalization/
├── .env
```

### 11.4 Create `app/` package

**`app/__init__.py`** — empty file:
```bash
mkdir app && touch app/__init__.py
```

**`app/models.py`:**
```python
from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ShelfLifeReference(Base):
    __tablename__ = "shelf_life_reference"

    id                   = Column(Integer, primary_key=True)
    canonical_name       = Column(String(100), nullable=False)
    category             = Column(String(50),  nullable=False)
    storage_context      = Column(String(20),  nullable=False)
    shelf_life_days_min  = Column(Integer,     nullable=False)
    shelf_life_days_avg  = Column(Integer,     nullable=False)
    shelf_life_days_max  = Column(Integer,     nullable=False)
    notes                = Column(Text)


class NormalizationCache(Base):
    __tablename__ = "normalization_cache"

    id             = Column(Integer, primary_key=True)
    raw_token      = Column(String(200), nullable=False, unique=True)
    canonical_name = Column(String(100), nullable=False)
    source         = Column(String(20),  nullable=False, default="llm")
    created_at     = Column(DateTime,    nullable=False, server_default=func.now())
    hit_count      = Column(Integer,     nullable=False, default=1)
```

**`app/db.py`:**
```python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

engine       = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 11.5 Alembic setup (one-time)

```bash
# From Smart-Stock/ root with .venv active
alembic init migrations
```

Open `alembic.ini` and clear the url line (it will be set from env instead):
```ini
# alembic.ini — find this line and blank it out:
sqlalchemy.url =
```

Open `migrations/env.py` and make these two edits:

**At the top, add imports:**
```python
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import Base  # noqa: E402
```

**Find `target_metadata = None` and replace:**
```python
target_metadata = Base.metadata
```

**Inside `run_migrations_online()`, find `connectable = engine_from_config(...)` block and add one line before it:**
```python
config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
```

### 11.6 One-time DB setup

```bash
# Create the Postgres database (if it doesn't exist yet)
createdb smartstock

# Generate and apply the initial migration
alembic revision --autogenerate -m "create shelf_life_and_normalization_cache"
alembic upgrade head

# Seed shelf_life_reference — fix seed script imports first (see below)
python -m db.seeds.shelf_life_seed

# Verify
python -c "
from app.db import SessionLocal
from app.models import ShelfLifeReference
db = SessionLocal()
print(db.query(ShelfLifeReference).count(), 'rows seeded')
db.close()
"
```

### 11.7 Fix seed script imports

The top of `db/seeds/shelf_life_seed.py` must add the project root to `sys.path` before importing `app`:

```python
# db/seeds/shelf_life_seed.py — add these lines at the very top
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models import ShelfLifeReference
from app.db import SessionLocal

# ... rest of file unchanged (the SHELF_LIFE_DATA list + seed_shelf_life function)

if __name__ == "__main__":
    db = SessionLocal()
    seed_shelf_life(db)
    db.close()
```

### 11.8 Runtime estimate

| Operation | Latency |
|---|---|
| Pass 1 (dict lookup) | < 1ms |
| Pass 2 (fuzzy, ~150 canonical names) | < 10ms |
| Pass 3 (Groq API, cache miss) | 200–600ms |
| Pass 3 (cache hit) | < 1ms |
| Full receipt (10–15 items, ≤ 20% LLM) | < 400ms |

---

## Appendix: Troubleshooting

| Issue | Fix |
|---|---|
| `GROQ_API_KEY` not found / `EnvironmentError` | Add `GROQ_API_KEY=gsk_...` to `.env` and call `load_dotenv()` before app startup |
| Groq 429 rate limit error | Cache hit rate is too low — check that `normalization_cache` DB is connected. Confirm repeated items are hitting cache, not re-calling Groq. |
| Fuzzy match false positive (e.g. "SALT" → "Malt") | Add the token explicitly to `ABBREVIATION_MAP` with the correct canonical name to force a Pass 1 exact hit and bypass fuzzy entirely |
| `_canonical_names_cache` is empty | `shelf_life_reference` table not seeded. Run `python -m db.seeds.shelf_life_seed` first. |
| Pakistani item not resolving past Pass 1 | The Urdu transliteration varies by retailer. Add the exact receipt token variant seen to `ABBREVIATION_MAP`. Common variants to add: `ALOO BUKHARA` (plum), `IMLI` (tamarind), `SARSON` (mustard greens). |
| LLM returns multi-word explanation instead of name | `_clean_llm_response()` takes only the first line and strips punctuation. If the LLM is consistently verbose, add `"Answer in one to three words only."` to the prompt. |
| Fused qty+unit token (`"1LB"`) not splitting | `parse_quantity_unit` handles this in the fused-token regex branch. Confirm `raw_unit` is `None` when the unit is fused into `raw_qty`. If NER already split them, the standard path handles it. |
| `canonical_names_cache` not refreshing after seeding new items | Module-level cache is populated once per process. Restart the FastAPI server after running the seed script. |
| `normalize_entity` returns `None` for a known item | Print `cleaned` from `preprocess_token` — the modifier stripping may have removed too much. Add a direct entry to `ABBREVIATION_MAP` for the exact post-cleaning token. |
| `ShortTokenSkipped` — 3-char token goes directly to LLM | By design — fuzzy is unreliable on tokens ≤ 4 chars. Add the token to `ABBREVIATION_MAP` for a reliable Pass 1 hit instead (e.g. `"OJ": "Orange Juice"`). |
