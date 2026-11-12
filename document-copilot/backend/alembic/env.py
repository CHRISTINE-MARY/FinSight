from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine.url import make_url

from alembic import context
from app.config import settings
from app.database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _psycopg_url(url: str) -> str:
    """Use the psycopg 3 dialect. Plain postgresql:// selects psycopg2."""
    parsed = make_url(url)
    # Port 6543 is transaction mode. Migrations need a session, which is port 5432.
    if parsed.port == 6543:
        raise RuntimeError(
            "Alembic cannot use the Supabase transaction pooler on port 6543. "
            "Use the session pooler connection string on port 5432."
        )
    if parsed.drivername != "postgresql+psycopg":
        parsed = parsed.set(drivername="postgresql+psycopg")
    # ConfigParser treats % as interpolation, so percent-encoding must be doubled.
    return parsed.render_as_string(hide_password=False).replace("%", "%%")


config.set_main_option("sqlalchemy.url", _psycopg_url(settings.database_url))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
