# FinSight

FinSight is a research assistant for reading SEC filings. Analysts ask questions in plain English about a corpus of 10-Ks. The app retrieves the relevant passages and answers with a citation back to the filing, so each claim can be checked against the source. If the filings do not support the question, it says so instead of guessing.

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
