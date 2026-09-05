"""add brand column to inventory_items

Revision ID: a3f9c1d84b02
Revises: 56bd7cdc6bbb
Create Date: 2026-09-06 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f9c1d84b02'
down_revision: Union[str, Sequence[str], None] = '56bd7cdc6bbb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable: Stage 2's brand_matcher.py returns None when no lexicon
    # match is found (a missing brand is a valid, expected outcome, not
    # a failure - see Item_Extraction.md §3). Not every inventory item
    # will have a brand.
    op.add_column(
        'inventory_items',
        sa.Column('brand', sa.String(length=100), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('inventory_items', 'brand')
