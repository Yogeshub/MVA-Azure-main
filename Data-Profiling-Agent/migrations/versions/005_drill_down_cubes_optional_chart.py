"""Make drill_down_cubes.chart_id nullable so cubes can actually be persisted

The profiling pipeline computes drill-down cubes on every run, but nothing ever
wrote them to this table: a cube is keyed by (hierarchy level, ancestor path)
and has no chart of its own, while chart_id was NOT NULL. The API worked around
that by passing an empty cube list to the drill-down service, so drill-down
silently returned nothing on every run.

Relaxing chart_id to nullable lets the run's cubes be stored as what they are —
run-scoped, hierarchy-scoped aggregates — and served back to the analyst UI.

Revision ID: 005
Revises: 004
Create Date: 2026-08-25 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = 'agent2'


def upgrade() -> None:
    op.alter_column(
        'drill_down_cubes',
        'chart_id',
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
        schema=SCHEMA,
    )


def downgrade() -> None:
    # Rows written under the new model have no chart to point at, so they
    # cannot survive a tightening of the constraint.
    op.execute(f'DELETE FROM {SCHEMA}.drill_down_cubes WHERE chart_id IS NULL')
    op.alter_column(
        'drill_down_cubes',
        'chart_id',
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
        schema=SCHEMA,
    )
