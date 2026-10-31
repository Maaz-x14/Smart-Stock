"""app schema reconciliation: UUID keys, auth, receipts, alerts, waste_log

Revision ID: f1b2c3d4e5a6
Revises: a3f9c1d84b02
Create Date: 2026-10-30 00:00:00.000000

Reconciles app/models.py with DB_Schema.md v1.1 (see plan doc, "Schema
reconciliation" / HANDOFF.md). The pre-app `users` and `inventory_items`
tables used integer PKs and held only test data from before this build
(no production deployment exists yet) - confirmed with Maaz before
writing this migration (plan doc, "Decisions and assumptions"). They
are dropped and recreated with UUID keys rather than converted in
place, which would need a data migration for no real data.

This is a destructive migration for `users` and `inventory_items`.
Do not run it against a database you want to keep rows from.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f1b2c3d4e5a6"
down_revision: Union[str, Sequence[str], None] = "a3f9c1d84b02"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')  # gen_random_uuid()

    # -- shelf_life_reference: reconcile pre-existing drift --
    # app/models.py had already drifted from this table's original
    # migration (c04767df306c) before this session's work: it added
    # subcategory/source/created_at and widened canonical_name/
    # category/storage_context, none of which any migration ever
    # created. Caught by running `alembic check` against real
    # Postgres (not previously tested - only SQLite had been used).
    op.add_column("shelf_life_reference", sa.Column("subcategory", sa.String(length=100), nullable=True))
    op.add_column("shelf_life_reference", sa.Column("source", sa.String(length=50), nullable=True))
    op.add_column(
        "shelf_life_reference",
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    # The model has always declared this as a UniqueConstraint, but no
    # migration ever actually created it - the real table has no such
    # constraint (confirmed via \d shelf_life_reference against a real
    # Postgres DB). Without it, nothing stops a duplicate
    # (canonical_name, storage_context) row from being seeded or
    # inserted twice.
    op.create_unique_constraint("idx_shelf_life_lookup", "shelf_life_reference", ["canonical_name", "storage_context"])

    # -- Drop the old int-keyed tables (test data only, see docstring) --
    op.drop_table("inventory_items")
    op.drop_table("users")

    # -- users --
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("email"),
    )
    op.create_index("idx_users_email", "users", ["email"])

    # -- receipts (new; holds pending extractions for /confirm, see models.py) --
    op.create_table(
        "receipts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("extracted_items", sa.JSON(), nullable=False),
        sa.Column("storage_context", sa.String(length=20), nullable=False, server_default="fridge"),
        sa.Column("purchase_date", sa.Date(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("processing_time_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'confirmed')", name="ck_receipts_status"),
    )
    op.create_index("idx_receipts_user_id", "receipts", ["user_id"])

    # -- inventory_items --
    op.create_table(
        "inventory_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("shelf_life_ref_id", sa.Integer(), sa.ForeignKey("shelf_life_reference.id"), nullable=True),
        sa.Column("canonical_name", sa.String(length=255), nullable=False),
        sa.Column("raw_token", sa.String(length=500), nullable=True),
        sa.Column("brand", sa.String(length=255), nullable=True),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("quantity", sa.Numeric(10, 2), nullable=False, server_default="1.0"),
        sa.Column("unit", sa.String(length=50), nullable=True),
        sa.Column("storage_context", sa.String(length=20), nullable=False, server_default="fridge"),
        sa.Column("purchase_date", sa.Date(), nullable=False, server_default=sa.text("CURRENT_DATE")),
        sa.Column("predicted_expiry_date", sa.Date(), nullable=False),
        sa.Column("expiry_confidence", sa.Numeric(4, 3), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="ACTIVE"),
        sa.Column("source", sa.String(length=50), nullable=False, server_default="receipt"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("storage_context IN ('fridge', 'freezer', 'pantry')", name="ck_inventory_storage_context"),
        sa.CheckConstraint("status IN ('ACTIVE', 'CONSUMED', 'WASTED')", name="ck_inventory_status"),
    )
    # No urgency_tier column here. DB_Schema.md originally specced it
    # as a Postgres GENERATED ALWAYS AS (... CURRENT_DATE ...) STORED
    # column, but Postgres generated columns must be immutable and
    # CURRENT_DATE isn't - running this against real Postgres (not
    # just SQLite) fails with "generation expression is not
    # immutable". Computed instead at response time in
    # app/routers/inventory.py from predicted_expiry_date - see
    # app/models.py's comment on InventoryItem for the full story.
    op.create_index("idx_inventory_user_id", "inventory_items", ["user_id"])
    op.create_index(
        "idx_inventory_expiry",
        "inventory_items",
        ["predicted_expiry_date"],
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )
    op.create_index("idx_inventory_status", "inventory_items", ["user_id", "status"])

    # -- alerts --
    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("expiry_threshold_hours", sa.Integer(), nullable=False, server_default="48"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="ACTIVE"),
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("status IN ('ACTIVE', 'DISMISSED')", name="ck_alerts_status"),
    )
    op.create_index("idx_alerts_user_id", "alerts", ["user_id"])
    op.create_index("idx_alerts_status", "alerts", ["user_id", "status"])

    # -- alert_items --
    op.create_table(
        "alert_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("alert_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("alerts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("item_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("inventory_items.id", ondelete="CASCADE"), nullable=False),
        sa.UniqueConstraint("alert_id", "item_id", name="uq_alert_items_alert_item"),
    )
    op.create_index("idx_alert_items_alert_id", "alert_items", ["alert_id"])

    # -- waste_log --
    op.create_table(
        "waste_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("item_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("inventory_items.id"), nullable=False),
        sa.Column("item_name", sa.String(length=255), nullable=False),
        sa.Column("outcome", sa.String(length=20), nullable=False),
        sa.Column("recipe_used", sa.String(length=500), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("outcome IN ('CONSUMED', 'WASTED')", name="ck_waste_log_outcome"),
    )
    op.create_index("idx_waste_log_user_id", "waste_log", ["user_id"])
    op.create_index("idx_waste_log_outcome", "waste_log", ["user_id", "outcome"])
    op.create_index("idx_waste_log_date", "waste_log", ["user_id", "recorded_at"])

    # -- recipe_cache (new; app-plan addition, not in DB_Schema.md v1.1) --
    op.create_table(
        "recipe_cache",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("cache_key", sa.String(length=128), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("cache_key"),
    )
    op.create_index("idx_recipe_cache_key", "recipe_cache", ["cache_key"])

    # -- updated_at triggers (DB_Schema.md §6) --
    op.execute(
        """
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        "CREATE TRIGGER trg_users_updated_at BEFORE UPDATE ON users "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at()"
    )
    op.execute(
        "CREATE TRIGGER trg_inventory_updated_at BEFORE UPDATE ON inventory_items "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_inventory_updated_at ON inventory_items")
    op.execute("DROP TRIGGER IF EXISTS trg_users_updated_at ON users")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at()")

    op.drop_table("recipe_cache")
    op.drop_table("waste_log")
    op.drop_table("alert_items")
    op.drop_table("alerts")
    op.drop_table("inventory_items")
    op.drop_table("receipts")
    op.drop_table("users")

    # Recreate the old int-keyed tables so downgrade is at least
    # structurally reversible (data is not - it was dropped upgrading).
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.UniqueConstraint("email"),
    )
    op.create_table(
        "inventory_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("canonical_name", sa.String(length=100), nullable=False),
        sa.Column("brand", sa.String(length=100), nullable=True),
        sa.Column("quantity", sa.Numeric(8, 2), nullable=False),
        sa.Column("unit", sa.String(length=20), nullable=True),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("storage_context", sa.String(length=20), nullable=False),
        sa.Column("purchase_date", sa.Date(), nullable=False),
        sa.Column("predicted_expiry", sa.Date(), nullable=False),
        sa.Column("shelf_life_days", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=False),
        sa.Column("expiry_source", sa.String(length=30), nullable=False),
        sa.Column("flag_for_review", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("normalization_pass", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="ACTIVE"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
    )

    op.drop_constraint("idx_shelf_life_lookup", "shelf_life_reference", type_="unique")
    op.drop_column("shelf_life_reference", "created_at")
    op.drop_column("shelf_life_reference", "source")
    op.drop_column("shelf_life_reference", "subcategory")
