# DB_Schema.md — Database Schema
## Smart-Stock PostgreSQL Schema

**Version:** 1.0  
**Engine:** PostgreSQL 15+  
**ORM:** SQLAlchemy 2.0

---

## Entity Relationship Overview

```
users
  │
  ├──< inventory_items >──── shelf_life_reference
  │         │
  │         └──< waste_log
  │
  └──< alerts >──< alert_items >── inventory_items
```

---

## 1. Table: `users`

Stores registered user accounts.

```sql
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    name            VARCHAR(255) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
```

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK | Auto-generated |
| `email` | VARCHAR(255) | UNIQUE, NOT NULL | Login identifier |
| `name` | VARCHAR(255) | NOT NULL | Display name |
| `hashed_password` | VARCHAR(255) | NOT NULL | bcrypt hash |
| `is_active` | BOOLEAN | DEFAULT TRUE | Soft disable account |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | — |
| `updated_at` | TIMESTAMPTZ | DEFAULT NOW() | Updated via trigger |

---

## 2. Table: `shelf_life_reference`

Master lookup table mapping food categories and subcategories to average shelf life in days per storage context. Populated at seed time; not user-editable.

```sql
CREATE TABLE shelf_life_reference (
    id                  SERIAL PRIMARY KEY,
    canonical_name      VARCHAR(255) NOT NULL,
    category            VARCHAR(100) NOT NULL,
    subcategory         VARCHAR(100),
    storage_context     VARCHAR(50) NOT NULL CHECK (
                            storage_context IN ('fridge', 'freezer', 'pantry')
                        ),
    shelf_life_days_min INTEGER NOT NULL,
    shelf_life_days_max INTEGER NOT NULL,
    shelf_life_days_avg INTEGER NOT NULL GENERATED ALWAYS AS (
                            (shelf_life_days_min + shelf_life_days_max) / 2
                        ) STORED,
    source              VARCHAR(255),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_shelf_life_lookup
    ON shelf_life_reference(canonical_name, storage_context);
CREATE INDEX idx_shelf_life_category ON shelf_life_reference(category);
```

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | PK |
| `canonical_name` | VARCHAR(255) | e.g., "Strawberries", "Whole Milk" |
| `category` | VARCHAR(100) | Produce, Dairy, Meat, Pantry, Frozen |
| `subcategory` | VARCHAR(100) | e.g., "Berries", "Leafy Greens" |
| `storage_context` | VARCHAR(50) | `fridge` / `freezer` / `pantry` |
| `shelf_life_days_min` | INTEGER | Lower bound |
| `shelf_life_days_max` | INTEGER | Upper bound |
| `shelf_life_days_avg` | INTEGER | Computed: (min+max)/2 |
| `source` | VARCHAR(255) | Reference (FDA, USDA, etc.) |

**Sample Data:**

| canonical_name | category | storage_context | min | max |
|---|---|---|---|---|
| Strawberries | Produce | fridge | 5 | 7 |
| Strawberries | Produce | freezer | 180 | 365 |
| Whole Milk | Dairy | fridge | 7 | 14 |
| Chicken Breast | Meat | fridge | 1 | 2 |
| Chicken Breast | Meat | freezer | 270 | 365 |
| Bread | Pantry | pantry | 5 | 7 |

---

## 3. Table: `inventory_items`

Core table. One row per distinct item in a user's inventory.

```sql
CREATE TYPE item_status AS ENUM ('ACTIVE', 'CONSUMED', 'WASTED');
CREATE TYPE urgency_tier AS ENUM ('green', 'yellow', 'red', 'expired');
CREATE TYPE storage_context AS ENUM ('fridge', 'freezer', 'pantry');

CREATE TABLE inventory_items (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                 UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    shelf_life_ref_id       INTEGER REFERENCES shelf_life_reference(id),

    canonical_name          VARCHAR(255) NOT NULL,
    raw_token               VARCHAR(500),
    category                VARCHAR(100) NOT NULL,
    quantity                NUMERIC(10, 2) NOT NULL DEFAULT 1.0,
    unit                    VARCHAR(50),
    storage_context         storage_context NOT NULL DEFAULT 'fridge',

    purchase_date           DATE NOT NULL DEFAULT CURRENT_DATE,
    predicted_expiry_date   DATE NOT NULL,
    expiry_confidence       NUMERIC(4, 3),

    status                  item_status NOT NULL DEFAULT 'ACTIVE',
    urgency_tier            urgency_tier GENERATED ALWAYS AS (
                                CASE
                                    WHEN predicted_expiry_date < CURRENT_DATE
                                        THEN 'expired'
                                    WHEN predicted_expiry_date <= CURRENT_DATE + INTERVAL '2 days'
                                        THEN 'red'
                                    WHEN predicted_expiry_date <= CURRENT_DATE + INTERVAL '5 days'
                                        THEN 'yellow'
                                    ELSE 'green'
                                END
                            ) STORED,

    source                  VARCHAR(50) DEFAULT 'receipt',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_inventory_user_id ON inventory_items(user_id);
CREATE INDEX idx_inventory_expiry ON inventory_items(predicted_expiry_date)
    WHERE status = 'ACTIVE';
CREATE INDEX idx_inventory_status ON inventory_items(user_id, status);
```

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `user_id` | UUID | FK → users |
| `shelf_life_ref_id` | INTEGER | FK → shelf_life_reference (nullable) |
| `canonical_name` | VARCHAR(255) | Normalized food name |
| `raw_token` | VARCHAR(500) | Original receipt text for audit |
| `category` | VARCHAR(100) | Food category |
| `quantity` | NUMERIC(10,2) | Amount |
| `unit` | VARCHAR(50) | lb, gal, container, etc. |
| `storage_context` | ENUM | fridge / freezer / pantry |
| `purchase_date` | DATE | Defaults to today |
| `predicted_expiry_date` | DATE | From expiry engine |
| `expiry_confidence` | NUMERIC(4,3) | 0.000–1.000 |
| `status` | ENUM | ACTIVE / CONSUMED / WASTED |
| `urgency_tier` | ENUM | Computed column |
| `source` | VARCHAR(50) | `receipt` or `manual` |

---

## 4. Table: `alerts`

One alert record per daily scheduler run per user. Links to at-risk items.

```sql
CREATE TYPE alert_status AS ENUM ('ACTIVE', 'DISMISSED');

CREATE TABLE alerts (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                 UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    triggered_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expiry_threshold_hours  INTEGER NOT NULL DEFAULT 48,
    status                  alert_status NOT NULL DEFAULT 'ACTIVE',
    dismissed_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_alerts_user_id ON alerts(user_id);
CREATE INDEX idx_alerts_status ON alerts(user_id, status);
```

### 4.1 Table: `alert_items` (Junction)

Links alerts to the specific inventory items that triggered them.

```sql
CREATE TABLE alert_items (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id        UUID NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    item_id         UUID NOT NULL REFERENCES inventory_items(id) ON DELETE CASCADE,
    UNIQUE(alert_id, item_id)
);

CREATE INDEX idx_alert_items_alert_id ON alert_items(alert_id);
```

---

## 5. Table: `waste_log`

Terminal state record for every item that leaves ACTIVE status.

```sql
CREATE TYPE waste_outcome AS ENUM ('CONSUMED', 'WASTED');

CREATE TABLE waste_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    item_id         UUID NOT NULL REFERENCES inventory_items(id),
    item_name       VARCHAR(255) NOT NULL,
    outcome         waste_outcome NOT NULL,
    recipe_used     VARCHAR(500),
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_waste_log_user_id ON waste_log(user_id);
CREATE INDEX idx_waste_log_outcome ON waste_log(user_id, outcome);
CREATE INDEX idx_waste_log_date ON waste_log(user_id, recorded_at);
```

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `user_id` | UUID | FK → users |
| `item_id` | UUID | FK → inventory_items (kept even if item deleted) |
| `item_name` | VARCHAR(255) | Denormalized for historical reporting |
| `outcome` | ENUM | CONSUMED or WASTED |
| `recipe_used` | VARCHAR(500) | Spoonacular recipe title if cooked |
| `recorded_at` | TIMESTAMPTZ | When the state was recorded |

---

## 6. Triggers

### Auto-update `updated_at` on row change

```sql
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_inventory_updated_at
    BEFORE UPDATE ON inventory_items
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
```

---

## 7. Key Queries

### Get at-risk items for alert scheduler
```sql
SELECT ii.*, u.email
FROM inventory_items ii
JOIN users u ON ii.user_id = u.id
WHERE ii.status = 'ACTIVE'
  AND ii.predicted_expiry_date <= CURRENT_DATE + INTERVAL '48 hours'
ORDER BY ii.user_id, ii.predicted_expiry_date ASC;
```

### Waste stats for a user (last 30 days)
```sql
SELECT
    outcome,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM waste_log
WHERE user_id = :user_id
  AND recorded_at >= NOW() - INTERVAL '30 days'
GROUP BY outcome;
```

### Inventory grouped by category
```sql
SELECT
    category,
    COUNT(*) AS item_count,
    COUNT(*) FILTER (WHERE urgency_tier = 'red') AS urgent_count
FROM inventory_items
WHERE user_id = :user_id
  AND status = 'ACTIVE'
GROUP BY category
ORDER BY category;
```
