# Backend

FastAPI service for Document Copilot. Run these commands from `backend/`.

## Setup

Copy `.env.example` to `.env` and fill in the values, then install dependencies:

```bash
uv sync
```

The app reads `.env` through `app/config.py` and exits on startup if a required value is missing.

## Run

```bash
uv run uvicorn app.main:app --reload
```

- Health: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs

## Manage

```bash
uv add <package>     # add a dependency
uv run pytest        # run tests
uv run ruff check .  # lint
```

## Migrations

`DATABASE_URL` must be the Supabase session connection on port 5432. On an IPv4 network, use the session pooler string from the Connect dialog. This applies the reviewed migration and changes the database:

```bash
uv run alembic upgrade head
```
