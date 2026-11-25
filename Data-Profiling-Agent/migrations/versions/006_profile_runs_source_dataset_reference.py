"""Record where a run's rows permanently live, so drill-down survives retention

A finished run could only reach its rows through the scratch copy Agent 2 wrote
under TEMP_STORAGE_DIR while profiling, which is swept after
RUN_FILE_RETENTION_HOURS. Once that copy was gone, any breakdown the run had not
pre-aggregated into a cube could not be computed at all — drill-down failed with
"the source rows for this run are no longer retained".

The bytes were never gone: the orchestrator's Dataset Registry stores every
upload content-addressed by SHA-256 and never purges it. What was missing was a
reference from the run to that permanent copy. These columns are that reference.

All three are nullable — a run created by a caller that supplies no reference
behaves exactly as it did before, and existing rows need no backfill.

Revision ID: 006
Revises: 005
Create Date: 2026-08-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '006'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = 'agent2'


def upgrade() -> None:
    op.add_column(
        'profile_runs',
        sa.Column('source_fingerprint', sa.String(length=64), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        'profile_runs',
        sa.Column('source_dataset_uri', sa.String(length=1024), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        'profile_runs',
        sa.Column('source_sheet_name', sa.String(length=255), nullable=True),
        schema=SCHEMA,
    )
    # Every run built from the same upload shares one fingerprint; looking runs
    # up by dataset is the natural question once the reference exists.
    op.create_index(
        'idx_profile_runs_source_fingerprint',
        'profile_runs',
        ['source_fingerprint'],
        unique=False,
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index('idx_profile_runs_source_fingerprint', table_name='profile_runs', schema=SCHEMA)
    op.drop_column('profile_runs', 'source_sheet_name', schema=SCHEMA)
    op.drop_column('profile_runs', 'source_dataset_uri', schema=SCHEMA)
    op.drop_column('profile_runs', 'source_fingerprint', schema=SCHEMA)
