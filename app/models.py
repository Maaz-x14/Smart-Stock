# app/models.py
"""
SQLAlchemy models, reconciled with DB_Schema.md v1.1.

Schema reconciliation notes (app build, see HANDOFF.md):
  - UUID primary keys everywhere except shelf_life_reference and
    normalization_cache, which DB_Schema.md keeps as SERIAL/Integer -
    they're ML-internal lookup/cache tables, never referenced by a
    user-facing FK, so there's no reconciliation need there.
  - users gains hashed_password, is_active, updated_at - required for
    real auth; none of this existed before this build.
  - inventory_items is rebuilt to match DB_Schema.md §3: UUID keys,
    raw_token, source ('receipt'/'manual'), expiry_confidence (renamed
    from `confidence` to match the spec's column name). urgency_tier
    is NOT a DB column: DB_Schema.md specs it as a Postgres GENERATED
    ALWAYS AS (... CURRENT_DATE ...) STORED column, but Postgres
    generated columns must be immutable and CURRENT_DATE isn't -
    confirmed by actually running the migration against real
    Postgres, which fails with "generation expression is not
    immutable". Computed instead at response time in
    app/routers/inventory.py from predicted_expiry_date. ML-only
    columns (expiry_confidence, source=receipt metadata like
    normalization_pass... see below) are nullable because a manually
    added item (POST /inventory, DB_Schema.md "Not persisted" note)
    never goes through Stage 2-4 and has no normalization/expiry
    pipeline metadata to report.
  - normalization_pass is NOT part of DB_Schema.md's inventory_items
    table - it's a pipeline-internal debugging field (ML_Pipeline.md),
    useful at /receipts/upload time to trace which Stage 3 pass
    resolved a name, but there's no product reason to persist it per
    inventory row once the item is confirmed. Dropped from the table;
    still present on ExtractedItem/the upload response.
  - New tables added, matching DB_Schema.md exactly: receipts (new -
    not in DB_Schema.md, see its own docstring below), alerts,
    alert_items, waste_log.
  - recipe_cache is new, not in DB_Schema.md v1.1 (added for the
    Spoonacular caching requirement in the app plan) - keyed by a
    stable hash of the sorted ingredient list.
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

from app.db_types import GUID

Base = declarative_base()


def _uuid_pk():
    return Column(GUID(), primary_key=True, default=uuid.uuid4)


# -- ML-internal tables (unchanged shape from the original models.py) --------


class ShelfLifeReference(Base):
    __tablename__ = "shelf_life_reference"

    id = Column(Integer, primary_key=True)
    # Reconciled against the actual DB (app schema reconciliation
    # migration f1b2c3d4e5a6, see its docstring): this model had
    # drifted from its own migrations before this session - it
    # declared subcategory/source/created_at and wider VARCHAR widths
    # that no migration ever created, while the original migration's
    # `notes` column (still used by db/seeds/shelf_life_seed.py) had
    # been dropped from the model entirely, silently breaking the
    # seed script (ShelfLifeReference(notes=...) raised TypeError -
    # caught by actually running `alembic check` against real
    # Postgres, not by inspection). `notes` is back; subcategory/
    # source/created_at are kept (they're genuinely useful and the
    # migration now creates them) since nothing used them.
    canonical_name = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    subcategory = Column(String(100), nullable=True)
    storage_context = Column(String(20), nullable=False)
    shelf_life_days_min = Column(Integer, nullable=False)
    shelf_life_days_avg = Column(Integer, nullable=False)
    shelf_life_days_max = Column(Integer, nullable=False)
    notes = Column(Text, nullable=True)
    source = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (UniqueConstraint("canonical_name", "storage_context", name="idx_shelf_life_lookup"),)


class NormalizationCache(Base):
    __tablename__ = "normalization_cache"

    id = Column(Integer, primary_key=True)
    raw_token = Column(String(200), nullable=False, unique=True)
    canonical_name = Column(String(100), nullable=False)
    source = Column(String(20), nullable=False, default="llm")
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    hit_count = Column(Integer, nullable=False, default=1)


# -- App tables ----------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id = _uuid_pk()
    # No unique=True/index=True here: the migration creates a named
    # UniqueConstraint("email") plus a separate idx_users_email index
    # explicitly, and SQLAlchemy's own unique=True/index=True shorthand
    # generates its OWN auto-named index (ix_users_email) instead of
    # reusing those - `alembic check` then reports permanent drift
    # (add ix_users_email / remove idx_users_email) even though the DB
    # is actually fine. Declaring the matching named index explicitly
    # below keeps the model in sync with what the migration actually
    # created, instead of letting SQLAlchemy's defaults diverge from it.
    email = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("email"),
        Index("idx_users_email", "email"),
    )

    inventory_items = relationship("InventoryItem", back_populates="user", cascade="all, delete-orphan")
    receipts = relationship("Receipt", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    waste_log_entries = relationship("WasteLog", back_populates="user", cascade="all, delete-orphan")


class Receipt(Base):
    """Holds one uploaded receipt's pipeline output between /upload and
    /confirm. Not in DB_Schema.md v1.1 (written before the app existed
    to persist anything) - added so /confirm has something durable to
    read from and can return 404/410 per API_Spec.md, instead of
    needing server-side session state or a client-echoed payload it
    cannot verify.

    extracted_items stores the ExtractedItem list as JSON exactly as
    produced by process_receipt() (dataclasses -> dicts), so /confirm
    can look up by receipt_id + validate the user is only confirming
    items that were actually extracted for them.
    """

    __tablename__ = "receipts"

    id = _uuid_pk()
    # index=True omitted on purpose everywhere in this file - see
    # User.email's comment above for why (matches each table's
    # explicitly-named idx_* index from the migration instead).
    user_id = Column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    extracted_items = Column(JSON, nullable=False)
    storage_context = Column(String(20), nullable=False, default="fridge")
    purchase_date = Column(Date, nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending | confirmed
    processing_time_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    confirmed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="receipts")

    __table_args__ = (Index("idx_receipts_user_id", "user_id"),)


class InventoryItem(Base):
    __tablename__ = "inventory_items"

    id = _uuid_pk()
    user_id = Column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    shelf_life_ref_id = Column(Integer, ForeignKey("shelf_life_reference.id"), nullable=True)

    canonical_name = Column(String(255), nullable=False)
    raw_token = Column(String(500), nullable=True)
    brand = Column(String(255), nullable=True)
    category = Column(String(100), nullable=False)
    quantity = Column(Numeric(10, 2), nullable=False, default=1.0)
    unit = Column(String(50), nullable=True)
    storage_context = Column(
        String(20), nullable=False, default="fridge"
    )  # fridge | freezer | pantry, lowercase

    purchase_date = Column(Date, nullable=False)
    predicted_expiry_date = Column(Date, nullable=False)
    # Nullable: a manually added item (source='manual') never ran
    # through Stage 4 and has no model confidence to report.
    expiry_confidence = Column(Numeric(4, 3), nullable=True)

    status = Column(String(20), nullable=False, default="ACTIVE")  # ACTIVE | CONSUMED | WASTED
    # No urgency_tier column: it was originally meant to be a Postgres
    # GENERATED ALWAYS STORED column (per DB_Schema.md), but Postgres
    # generated columns must be immutable and CURRENT_DATE isn't -
    # `alembic upgrade head` fails against real Postgres with
    # "generation expression is not immutable" if you try. Computed
    # instead in app/routers/inventory.py's _to_response(), from the
    # same predicted_expiry_date already used for days_remaining -
    # simpler, and correct for "today" in a way a stored column
    # computed at write time never could be.

    source = Column(String(50), nullable=False, default="receipt")  # receipt | manual
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    user = relationship("User", back_populates="inventory_items")
    alert_items = relationship("AlertItem", back_populates="item", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_inventory_user_id", "user_id"),
        Index("idx_inventory_status", "user_id", "status"),
        Index(
            "idx_inventory_expiry",
            "predicted_expiry_date",
            postgresql_where=(status == "ACTIVE"),
        ),
    )


class Alert(Base):
    __tablename__ = "alerts"

    id = _uuid_pk()
    user_id = Column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    triggered_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    expiry_threshold_hours = Column(Integer, nullable=False, default=48)
    status = Column(String(20), nullable=False, default="ACTIVE")  # ACTIVE | DISMISSED
    dismissed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    user = relationship("User", back_populates="alerts")
    alert_items = relationship("AlertItem", back_populates="alert", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_alerts_user_id", "user_id"),
        Index("idx_alerts_status", "user_id", "status"),
    )


class AlertItem(Base):
    __tablename__ = "alert_items"

    id = _uuid_pk()
    alert_id = Column(GUID(), ForeignKey("alerts.id", ondelete="CASCADE"), nullable=False)
    item_id = Column(GUID(), ForeignKey("inventory_items.id", ondelete="CASCADE"), nullable=False)

    __table_args__ = (
        UniqueConstraint("alert_id", "item_id", name="uq_alert_items_alert_item"),
        Index("idx_alert_items_alert_id", "alert_id"),
    )

    alert = relationship("Alert", back_populates="alert_items")
    item = relationship("InventoryItem", back_populates="alert_items")


class WasteLog(Base):
    __tablename__ = "waste_log"

    id = _uuid_pk()
    user_id = Column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    # No FK ondelete cascade target removal requirement: DB_Schema.md
    # keeps item_id even if the inventory item is later hard-deleted,
    # for historical reporting - so this FK has no ON DELETE clause
    # (defaults to RESTRICT, which is intentional: an item with waste
    # history cannot be hard-deleted; DELETE /inventory/{id} must
    # check for this, see services/inventory.py).
    item_id = Column(GUID(), ForeignKey("inventory_items.id"), nullable=False)
    item_name = Column(String(255), nullable=False)
    outcome = Column(String(20), nullable=False)  # CONSUMED | WASTED
    recipe_used = Column(String(500), nullable=True)
    recorded_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    user = relationship("User", back_populates="waste_log_entries")

    __table_args__ = (
        Index("idx_waste_log_user_id", "user_id"),
        Index("idx_waste_log_outcome", "user_id", "outcome"),
        Index("idx_waste_log_date", "user_id", "recorded_at"),
    )


class RecipeCache(Base):
    """Caches Spoonacular responses keyed by a stable hash of the
    sorted, lowercased ingredient list + limit. Not in DB_Schema.md
    v1.1 - added per the app plan's Spoonacular caching requirement
    (150 req/day free tier) rather than inventing an in-memory cache
    that would not survive a restart or span instances.
    """

    __tablename__ = "recipe_cache"

    id = Column(Integer, primary_key=True)
    cache_key = Column(String(128), nullable=False)
    response_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("cache_key", name="recipe_cache_cache_key_key"),
        Index("idx_recipe_cache_key", "cache_key"),
    )
