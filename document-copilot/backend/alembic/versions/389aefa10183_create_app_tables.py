"""create app tables

Revision ID: 389aefa10183
Revises:
Create Date: 2026-11-10 22:52:24.090096

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

from alembic import op

revision: str = "389aefa10183"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_TABLES = (
    "users",
    "source_documents",
    "document_chunks",
    "chat_threads",
    "chat_messages",
    "message_citations",
)


def upgrade() -> None:
    """Create the app schema, search indexes, and row-level security."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions")
    op.execute("SET search_path TO public, extensions")

    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["id"],
            ["auth.users.id"],
            name="fk_users_id_auth_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_table(
        "source_documents",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("company_name", sa.Text(), nullable=False),
        sa.Column("cik", sa.String(length=10), nullable=False),
        sa.Column("filing_type", sa.String(length=16), nullable=False),
        sa.Column("filing_date", sa.Date(), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=True),
        sa.Column("fiscal_year", sa.Integer(), nullable=False),
        sa.Column("accession_number", sa.String(length=32), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_source_documents"),
        sa.UniqueConstraint(
            "accession_number", name="uq_source_documents_accession_number"
        ),
    )
    op.create_index(
        "ix_source_documents_ticker", "source_documents", ["ticker"], unique=False
    )
    op.create_table(
        "document_chunks",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column(
            "search_vector",
            TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=False,
        ),
        sa.Column("metadata", JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["source_documents.id"],
            name="fk_document_chunks_document_id_source_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_chunks"),
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_id_chunk_index",
        ),
    )
    op.create_index(
        "ix_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
        unique=False,
    )
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding_hnsw "
        "ON public.document_chunks "
        "USING hnsw (embedding extensions.vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX ix_document_chunks_search_vector "
        "ON public.document_chunks USING gin (search_vector)"
    )
    op.execute(
        "CREATE INDEX ix_document_chunks_metadata "
        "ON public.document_chunks USING gin (metadata)"
    )
    op.create_table(
        "chat_threads",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_chat_threads_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chat_threads"),
    )
    op.create_index(
        "ix_chat_threads_user_id", "chat_threads", ["user_id"], unique=False
    )
    op.create_table(
        "chat_messages",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("thread_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("parts", JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("role in ('user', 'assistant')", name="role"),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["chat_threads.id"],
            name="fk_chat_messages_thread_id_chat_threads",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chat_messages"),
        sa.UniqueConstraint(
            "thread_id", "position", name="uq_chat_messages_thread_id_position"
        ),
    )
    op.create_index(
        "ix_chat_messages_thread_id", "chat_messages", ["thread_id"], unique=False
    )
    op.create_table(
        "message_citations",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("citation_index", sa.Integer(), nullable=False),
        sa.Column("excerpt", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["chat_messages.id"],
            name="fk_message_citations_message_id_chat_messages",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.id"],
            name="fk_message_citations_chunk_id_document_chunks",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_message_citations"),
        sa.UniqueConstraint(
            "message_id",
            "citation_index",
            name="uq_message_citations_message_id_citation_index",
        ),
    )
    op.create_index(
        "ix_message_citations_message_id",
        "message_citations",
        ["message_id"],
        unique=False,
    )
    op.create_index(
        "ix_message_citations_chunk_id",
        "message_citations",
        ["chunk_id"],
        unique=False,
    )

    _enable_row_level_security()
    _grant_table_access()


def downgrade() -> None:
    """Drop the app tables. The vector extension is left in place."""
    op.drop_index("ix_message_citations_chunk_id", table_name="message_citations")
    op.drop_index("ix_message_citations_message_id", table_name="message_citations")
    op.drop_table("message_citations")
    op.drop_index("ix_chat_messages_thread_id", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("ix_chat_threads_user_id", table_name="chat_threads")
    op.drop_table("chat_threads")
    op.execute("DROP INDEX IF EXISTS public.ix_document_chunks_metadata")
    op.execute("DROP INDEX IF EXISTS public.ix_document_chunks_search_vector")
    op.execute("DROP INDEX IF EXISTS public.ix_document_chunks_embedding_hnsw")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_index("ix_source_documents_ticker", table_name="source_documents")
    op.drop_table("source_documents")
    op.drop_table("users")


def _enable_row_level_security() -> None:
    for table in APP_TABLES:
        op.execute(f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY")

    op.execute(
        """
        CREATE POLICY users_select_own ON public.users
        FOR SELECT TO authenticated
        USING (id = auth.uid())
        """
    )
    op.execute(
        """
        CREATE POLICY users_insert_own ON public.users
        FOR INSERT TO authenticated
        WITH CHECK (id = auth.uid())
        """
    )
    op.execute(
        """
        CREATE POLICY users_update_own ON public.users
        FOR UPDATE TO authenticated
        USING (id = auth.uid())
        WITH CHECK (id = auth.uid())
        """
    )
    op.execute(
        """
        CREATE POLICY source_documents_select_authenticated
        ON public.source_documents
        FOR SELECT TO authenticated
        USING (true)
        """
    )
    op.execute(
        """
        CREATE POLICY document_chunks_select_authenticated
        ON public.document_chunks
        FOR SELECT TO authenticated
        USING (true)
        """
    )
    op.execute(
        """
        CREATE POLICY chat_threads_own ON public.chat_threads
        FOR ALL TO authenticated
        USING (user_id = auth.uid())
        WITH CHECK (user_id = auth.uid())
        """
    )
    op.execute(
        """
        CREATE POLICY chat_messages_own ON public.chat_messages
        FOR ALL TO authenticated
        USING (
            EXISTS (
                SELECT 1 FROM public.chat_threads
                WHERE chat_threads.id = chat_messages.thread_id
                AND chat_threads.user_id = auth.uid()
            )
        )
        WITH CHECK (
            EXISTS (
                SELECT 1 FROM public.chat_threads
                WHERE chat_threads.id = chat_messages.thread_id
                AND chat_threads.user_id = auth.uid()
            )
        )
        """
    )
    op.execute(
        """
        CREATE POLICY message_citations_own ON public.message_citations
        FOR ALL TO authenticated
        USING (
            EXISTS (
                SELECT 1
                FROM public.chat_messages
                JOIN public.chat_threads
                    ON chat_threads.id = chat_messages.thread_id
                WHERE chat_messages.id = message_citations.message_id
                AND chat_threads.user_id = auth.uid()
            )
        )
        WITH CHECK (
            EXISTS (
                SELECT 1
                FROM public.chat_messages
                JOIN public.chat_threads
                    ON chat_threads.id = chat_messages.thread_id
                WHERE chat_messages.id = message_citations.message_id
                AND chat_threads.user_id = auth.uid()
            )
        )
        """
    )


def _grant_table_access() -> None:
    for table in APP_TABLES:
        op.execute(f"REVOKE ALL ON TABLE public.{table} FROM PUBLIC, anon")
        op.execute(
            "GRANT SELECT, INSERT, UPDATE, DELETE "
            f"ON TABLE public.{table} TO authenticated, service_role"
        )
