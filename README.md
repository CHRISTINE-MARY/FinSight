# FinSight

Ask questions in plain English about SEC filings and get answers that cite the source passage.

## Setup

Install Python 3.12+, [uv](https://docs.astral.sh/uv/), Node.js 20+, and pnpm. Create a Supabase project and an [NVIDIA API key](https://build.nvidia.com/settings), then copy the env examples and fill them in:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

Backend (http://127.0.0.1:8000):

```bash
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

Frontend (http://localhost:5173):

```bash
cd frontend
pnpm install
pnpm dev
```
