"""create drift_metrics_snapshots table

Revision ID: 2026_10_11_0002
Revises: 2026_10_09_0001
Create Date: 2026-10-11

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "2026_10_11_0002"
down_revision: str | None = "2026_10_09_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "drift_metrics_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("records_evaluated", sa.Integer(), nullable=False),
        sa.Column("profile_version", sa.String(length=50), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column(
            "num_features_evaluated",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "num_drifted_features",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "prediction_drift_detected",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("prediction_psi", sa.Float(), nullable=True),
        sa.Column("prediction_p_value", sa.Float(), nullable=True),
        sa.Column(
            "prediction_status",
            sa.String(length=50),
            nullable=False,
            server_default="not_evaluated",
        ),
        sa.Column(
            "feature_metrics",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()), "postgresql"
            ),
            nullable=False,
        ),
        sa.Column(
            "summary_metadata",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()), "postgresql"
            ),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_drift_metrics_snapshots_created_at"),
        "drift_metrics_snapshots",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_drift_metrics_snapshots_created_at"),
        table_name="drift_metrics_snapshots",
    )
    op.drop_table("drift_metrics_snapshots")
