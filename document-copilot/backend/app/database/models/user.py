from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, ForeignKey, MetaData, Table, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.models.base import Base

if TYPE_CHECKING:
    from app.database.models.chat_thread import ChatThread

# Supabase Auth owns this table. Kept off Base.metadata so Alembic does not create it.
auth_users = Table(
    "users",
    MetaData(),
    Column("id", Uuid, primary_key=True),
    schema="auth",
)


class User(Base):
    __tablename__ = "users"

    # Supabase Auth owns auth.users. This row is the app's record for that user.
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey(auth_users.c.id, name="fk_users_id_auth_users", ondelete="CASCADE"),
        primary_key=True,
    )
    email: Mapped[str] = mapped_column(Text, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    threads: Mapped[list[ChatThread]] = relationship(back_populates="user")
