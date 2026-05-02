# MediMatch AI (React + FastAPI)

Doctor matching and booking:
- **Frontend**: React + Vite + TypeScript (mobile-friendly UI)
- **Backend**: FastAPI (Python) with Cloud SQL / Chroma / Mistral integrations

## Project Structure

- `frontend/`: React app for auth, symptom matching, booking, appointment history, emergency vitals demo
- `backend/`: FastAPI API service
- `database.py`, `cloud_sql_appointments.py`, `zipcodes_ca.py`: shared helpers imported by the API (live at repo root on `PYTHONPATH`)
- `cloud_sql_schema.sql`: reference DDL for Cloud SQL (MySQL) — run in your instance when provisioning
- `docker-compose.yml`: local full-stack container run

## Database setup (optional)

- **Cloud SQL (MySQL)** — set `CLOUD_SQL_*` in `.env`. Apply `cloud_sql_schema.sql` in Cloud SQL Studio (or `mysql` CLI) against your database.
- **TimescaleDB (doctor availability)** — set `DB_CONNECTION` in `.env`. From repo root, initialize tables once:
  ```bash
  cd backend && PYTHONPATH=.. python -c "from database import sync_init_database; sync_init_database()"
  ```

## Local Run

1. Copy environment file and set values:
```bash
cp .env.example .env
```

2. Backend:
```bash
pip install -r requirements.txt
cd backend && PYTHONPATH="$(pwd)/.." uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Frontend:
```bash
cd frontend
npm install
npm run dev
```

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000/api/v1`

## Docker Run

```bash
docker compose up --build
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000/api/v1`

## API Endpoints (Core)

- `POST /api/v1/auth/signup`
- `POST /api/v1/auth/login`
- `POST /api/v1/match/symptoms`
- `POST /api/v1/match/recommendation`
- `GET /api/v1/doctors/{doctor_name}/slots?appointment_date=YYYY-MM-DD`
- `POST /api/v1/appointments`
- `GET /api/v1/appointments/me?email=...`
- `POST /api/v1/location/reverse-zip`
- `GET /api/v1/cases/similar?symptoms=...`
