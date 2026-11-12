"""add embedding_status

Revision ID: b7c1e4a90f22
Revises: 389aefa10183
Create Date: 2026-11-12 15:20:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b7c1e4a90f22"
down_revision: str | None = "389aefa10183"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Allow chunks to be stored before an embedding exists."""
    op.add_column(
        "document_chunks",
        sa.Column(
            "embedding_status",
            sa.Text(),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
    )
    op.execute(
        """
        UPDATE public.document_chunks
        SET embedding_status = 'embedded'
        WHERE embedding IS NOT NULL
        """
    )
    op.execute("ALTER TABLE public.document_chunks ALTER COLUMN embedding DROP NOT NULL")
    op.create_check_constraint(
        "ck_document_chunks_embedding_status",
        "document_chunks",
        "embedding_status IN ('pending', 'embedded', 'failed')",
    )
    op.create_index(
        "ix_document_chunks_embedding_status",
        "document_chunks",
        ["embedding_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_embedding_status", table_name="document_chunks")
    op.drop_constraint(
        "ck_document_chunks_embedding_status",
        "document_chunks",
        type_="check",
    )
    op.execute("DELETE FROM public.document_chunks WHERE embedding IS NULL")
    op.execute(
        "ALTER TABLE public.document_chunks ALTER COLUMN embedding SET NOT NULL"
    )
    op.drop_column("document_chunks", "embedding_status")
