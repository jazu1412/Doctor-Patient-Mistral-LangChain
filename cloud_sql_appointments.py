"""
Google Cloud SQL (MySQL) appointments: book slots 9 AM–5 PM, default user jas@gmail.com.
Set CLOUD_SQL_* in .env or leave unset to skip Cloud SQL (app still works with TimescaleDB only).
"""
import os
import logging
import sys
from typing import List, Dict, Optional, Tuple
from datetime import date, time, timedelta
from dotenv import load_dotenv
import threading

load_dotenv()

# Logger: writes to cloud_sql_appointments.log and to stderr (Streamlit console)
_logger = logging.getLogger("cloud_sql_appointments")
_logger.setLevel(logging.DEBUG)
if not _logger.handlers:
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    try:
        _fh = logging.FileHandler("cloud_sql_appointments.log", encoding="utf-8")
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(_fmt)
        _logger.addHandler(_fh)
    except Exception:
        pass
    _ch = logging.StreamHandler(sys.stderr)
    _ch.setLevel(logging.DEBUG)
    _ch.setFormatter(_fmt)
    _logger.addHandler(_ch)

# Cloud SQL MySQL connection (use Cloud SQL Auth Proxy or direct with SSL)
CLOUD_SQL_HOST = os.getenv("CLOUD_SQL_HOST", "")
CLOUD_SQL_PORT = int(os.getenv("CLOUD_SQL_PORT", "3306"))
CLOUD_SQL_USER = os.getenv("CLOUD_SQL_USER", "")
CLOUD_SQL_PASSWORD = os.getenv("CLOUD_SQL_PASSWORD", "")
CLOUD_SQL_DATABASE = os.getenv("CLOUD_SQL_DATABASE", "doctor_appointments_db")

DEFAULT_USER_EMAIL = "jas@gmail.com"

# 30-minute slots from 9:00 AM to 5:00 PM (last slot 16:30–17:00)
SLOT_STARTS = [
    time(9, 0), time(9, 30), time(10, 0), time(10, 30), time(11, 0), time(11, 30),
    time(12, 0), time(12, 30), time(13, 0), time(13, 30), time(14, 0), time(14, 30),
    time(15, 0), time(15, 30), time(16, 0), time(16, 30),
]

_connection = None
_connection_lock = threading.Lock()
_last_connection_error = None  # So UI can show why connection failed


def _get_conn(reconnect: bool = False):
    """Get MySQL connection; returns None if Cloud SQL not configured."""
    global _connection, _last_connection_error
    _last_connection_error = None
    if not CLOUD_SQL_HOST or not CLOUD_SQL_USER:
        _last_connection_error = "CLOUD_SQL_HOST or CLOUD_SQL_USER not set in .env"
        _logger.warning("Cloud SQL not configured: %s", _last_connection_error)
        return None
    try:
        import pymysql
    except ImportError:
        _last_connection_error = "pymysql not installed (pip install pymysql)"
        _logger.warning(_last_connection_error)
        return None
    with _connection_lock:
        if reconnect:
            _logger.debug("Reconnect requested, clearing connection")
            _connection = None
        if _connection is None or not getattr(_connection, "open", True):
            try:
                if _connection is not None:
                    try:
                        _connection.close()
                    except Exception:
                        pass
                    _connection = None
                _logger.info("Connecting to Cloud SQL: host=%s port=%s database=%s user=%s",
                             CLOUD_SQL_HOST, CLOUD_SQL_PORT, CLOUD_SQL_DATABASE, CLOUD_SQL_USER)
                _connection = pymysql.connect(
                    host=CLOUD_SQL_HOST,
                    port=int(CLOUD_SQL_PORT),
                    user=CLOUD_SQL_USER,
                    password=CLOUD_SQL_PASSWORD or "",
                    database=CLOUD_SQL_DATABASE or None,
                    charset="utf8mb4",
                    cursorclass=pymysql.cursors.DictCursor,
                    connect_timeout=15,
                )
                _logger.info("Connected to Cloud SQL successfully")
            except Exception as e:
                _connection = None
                _last_connection_error = str(e)
                _logger.exception("Cloud SQL connection failed: %s", e)
    return _connection


def get_or_create_user(email: str = DEFAULT_USER_EMAIL) -> Optional[int]:
    """Get user_id for email; insert user if not exists. Returns None if DB unavailable."""
    _logger.debug("get_or_create_user email=%s", email)
    conn = _get_conn()
    if not conn:
        _logger.warning("get_or_create_user: no connection")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
            if row:
                uid = row["user_id"]
                _logger.debug("get_or_create_user: found user_id=%s", uid)
                return uid
            _logger.debug("get_or_create_user: inserting new user")
            cur.execute("INSERT INTO users (email) VALUES (%s)", (email,))
            conn.commit()
            uid = cur.lastrowid
            _logger.info("get_or_create_user: created user_id=%s for %s", uid, email)
            return uid
    except Exception as e:
        _logger.exception("get_or_create_user failed: %s", e)
        if conn:
            conn.rollback()
        return None
    finally:
        pass  # keep connection open for reuse


def get_doctor_id_by_name(doctor_name: str) -> Optional[int]:
    """Map doctor name to doctor_id 1–200 (stable hash). Works even if Cloud SQL doctors table has placeholders."""
    if not doctor_name:
        return None
    # Stable mapping so same doctor always gets same id
    return 1 + (abs(hash(doctor_name)) % 200)


def _normalize_slot_time(value) -> Optional[time]:
    """Convert MySQL slot_start_time (timedelta, time, or str) to datetime.time for comparison."""
    if value is None:
        return None
    if isinstance(value, time):
        return value
    if isinstance(value, timedelta):
        # PyMySQL returns MySQL TIME as timedelta (seconds since midnight)
        total = int(value.total_seconds())
        if total < 0:
            total += 24 * 3600
        hours, r = divmod(total, 3600)
        minutes, secs = divmod(r, 60)
        return time(hours % 24, minutes, secs)
    if isinstance(value, str):
        # e.g. "09:00:00" or "09:00"
        parts = value.strip().split(":")
        if len(parts) >= 2:
            try:
                h, m = int(parts[0]), int(parts[1])
                s = int(parts[2]) if len(parts) > 2 else 0
                return time(h % 24, m % 60, s % 60)
            except (ValueError, IndexError):
                pass
    return None


def get_available_slots(doctor_id: int, appointment_date: date) -> List[time]:
    """Return list of slot_start times (9–5, 30-min) that are not booked for (doctor_id, date)."""
    _logger.debug("get_available_slots doctor_id=%s date=%s", doctor_id, appointment_date.isoformat())
    conn = _get_conn()
    if not conn:
        _logger.debug("get_available_slots: no connection, returning all slots")
        return list(SLOT_STARTS)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT slot_start_time FROM appointments
                WHERE doctor_id = %s AND appointment_date = %s AND status = 'booked'
                """,
                (doctor_id, appointment_date.isoformat()),
            )
            rows = cur.fetchall()
            booked = set()
            for r in rows:
                raw = r.get("slot_start_time")
                t = _normalize_slot_time(raw)
                if t is not None:
                    booked.add(t)
            available = [t for t in SLOT_STARTS if t not in booked]
            _logger.debug("get_available_slots: %s booked, %s available", len(booked), len(available))
            return available
    except Exception as e:
        _logger.warning("get_available_slots failed: %s, returning all slots", e)
        return list(SLOT_STARTS)


def book_appointment(
    doctor_id: int,
    user_id: int,
    appointment_date: date,
    slot_start_time: time,
) -> Tuple[bool, str]:
    """
    Book one slot. Returns (True, msg) on success, (False, msg) on failure.
    Uses a fresh connection so the write is not affected by prior read state.
    """
    date_str = appointment_date.isoformat()
    time_str = slot_start_time.strftime("%H:%M:%S")
    _logger.info("book_appointment called: doctor_id=%s user_id=%s date=%s slot=%s",
                 doctor_id, user_id, date_str, time_str)

    if slot_start_time not in SLOT_STARTS:
        _logger.warning("book_appointment: invalid slot %s", slot_start_time)
        return False, "Invalid slot (must be 9:00–16:30 in 30-min steps)."
    # Use fresh connection for write to avoid transaction/state issues
    conn = _get_conn(reconnect=True)
    if not conn:
        _logger.error("book_appointment: no connection after reconnect")
        return False, "Cloud SQL not configured or unavailable."
    try:
        with conn.cursor() as cur:
            params = (doctor_id, user_id, date_str, time_str)
            _logger.info("book_appointment: executing INSERT with params %s", params)
            cur.execute(
                """
                INSERT INTO appointments (doctor_id, user_id, appointment_date, slot_start_time, status)
                VALUES (%s, %s, %s, %s, 'booked')
                """,
                params,
            )
            _logger.debug("book_appointment: INSERT executed, committing")
            conn.commit()
            new_id = cur.lastrowid
            _logger.info("book_appointment: commit done, lastrowid=%s", new_id)
            if not new_id:
                _logger.error("book_appointment: lastrowid is empty after INSERT")
                return False, "INSERT succeeded but no appointment_id returned (check table and grants)."
            # Verify row is visible in this connection
            cur.execute(
                "SELECT appointment_id FROM appointments WHERE appointment_id = %s",
                (new_id,),
            )
            row = cur.fetchone()
            if not row:
                _logger.error("book_appointment: row not found after INSERT for appointment_id=%s", new_id)
                return False, "Row not found after INSERT. Check that your Cloud SQL user has INSERT privilege on doctor_appointments_db.appointments."
            _logger.info("book_appointment: success appointment_id=%s", new_id)
            return True, f"Appointment #{new_id} booked."
    except Exception as e:
        _logger.exception("book_appointment exception: %s", e)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        err = str(e).lower()
        if "duplicate" in err or "uq_doctor_date_slot" in err:
            return False, "This slot is already booked."
        return False, f"Booking failed: {e!r}"


def list_appointments_for_user(
    user_id: Optional[int] = None,
    email: Optional[str] = None,
    from_date: Optional[date] = None,
) -> List[Dict]:
    """List booked appointments for user (by user_id or email). Optionally filter from_date."""
    _logger.debug("list_appointments_for_user user_id=%s email=%s from_date=%s", user_id, email, from_date)
    conn = _get_conn()
    if not conn:
        _logger.debug("list_appointments_for_user: no connection")
        return []
    uid = user_id
    if uid is None and email:
        uid = get_or_create_user(email)
    if uid is None:
        _logger.warning("list_appointments_for_user: could not resolve user_id")
        return []
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT a.appointment_id, a.doctor_id, a.appointment_date, a.slot_start_time, a.status,
                       d.doctor_name, d.speciality
                FROM appointments a
                JOIN doctors d ON d.doctor_id = a.doctor_id
                WHERE a.user_id = %s AND a.status = 'booked'
                """
            params = [uid]
            if from_date is not None:
                sql += " AND a.appointment_date >= %s"
                params.append(from_date.isoformat())
            sql += " ORDER BY a.appointment_date, a.slot_start_time"
            cur.execute(sql, params)
            rows = cur.fetchall()
            out = []
            for r in rows:
                slot = _normalize_slot_time(r["slot_start_time"])
                slot = slot.strftime("%H:%M") if slot else str(r.get("slot_start_time", ""))
                out.append({
                    "appointment_id": r["appointment_id"],
                    "doctor_id": r["doctor_id"],
                    "doctor_name": r["doctor_name"],
                    "speciality": r["speciality"],
                    "appointment_date": str(r["appointment_date"]),
                    "slot_start_time": slot,
                    "status": r["status"],
                })
            _logger.debug("list_appointments_for_user: returning %s appointments for user_id=%s", len(out), uid)
            return out
    except Exception as e:
        _logger.warning("list_appointments_for_user failed: %s", e)
        return []


def sync_doctors_to_cloud_sql(doctors: List[Dict]) -> Tuple[int, str]:
    """
    Update Cloud SQL doctors table with real names/specialities from ChromaDB.
    First pass: map each ChromaDB doctor to doctor_id by hash; update that row. Names that
    collide (same hash as an earlier name) are collected as "unused".
    Second pass: fill remaining rows (doctor_name LIKE 'Doctor %') with unused names so
    all 200 rows get real names.
    Returns (updated_count, message).
    """
    conn = _get_conn(reconnect=True)
    if not conn:
        return 0, "Cloud SQL not available"
    updated = 0
    try:
        with conn.cursor() as cur:
            used_ids = {}  # doctor_id -> (name, speciality) first assigned
            unused_names = []  # (name, speciality) that collided with an already-used id
            for d in doctors:
                name = d.get("doctor_name") or d.get("name")
                speciality = d.get("speciality") or d.get("specialty") or "General"
                if not name or str(name).strip() in ("", "N/A"):
                    continue
                name = str(name).strip()
                doctor_id = get_doctor_id_by_name(name)
                cur.execute(
                    "UPDATE doctors SET doctor_name = %s, speciality = %s WHERE doctor_id = %s",
                    (name, speciality, doctor_id),
                )
                if cur.rowcount > 0:
                    updated += 1
                    if doctor_id in used_ids:
                        unused_names.append((name, speciality))
                    else:
                        used_ids[doctor_id] = (name, speciality)
                    _logger.debug("sync_doctors_to_cloud_sql: updated doctor_id=%s -> %s", doctor_id, name[:50])
            # Second pass: fill all rows that still have placeholder "Doctor %"
            gap_ids = sorted(set(range(1, 201)) - set(used_ids.keys()))
            # Pool: unused names first, then cycle through ChromaDB doctors so we have enough for every gap
            fill_pool = list(unused_names)
            if len(fill_pool) < len(gap_ids):
                fill_pool.extend(used_ids.values())
            j = 0
            while len(fill_pool) < len(gap_ids) and doctors and j < 2 * len(doctors):
                d = doctors[j % len(doctors)]
                name = d.get("doctor_name") or d.get("name")
                speciality = d.get("speciality") or d.get("specialty") or "General"
                if name and str(name).strip() not in ("", "N/A"):
                    fill_pool.append((str(name).strip(), speciality))
                j += 1
            filled = 0
            for i, gid in enumerate(gap_ids):
                if i < len(fill_pool):
                    name, spec = fill_pool[i]
                    cur.execute(
                        "UPDATE doctors SET doctor_name = %s, speciality = %s WHERE doctor_id = %s",
                        (name, spec, gid),
                    )
                    if cur.rowcount > 0:
                        updated += 1
                        filled += 1
                        _logger.debug("sync_doctors_to_cloud_sql: filled gap doctor_id=%s -> %s", gid, name[:50])
            conn.commit()
        _logger.info("sync_doctors_to_cloud_sql: updated %s rows (%s in fill pass)", updated, filled)
        return updated, f"Updated {updated} doctor(s) in Cloud SQL"
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        _logger.warning("sync_doctors_to_cloud_sql failed: %s", e)
        return 0, str(e)


def is_cloud_sql_available() -> bool:
    """Return True if Cloud SQL is configured, reachable, and schema (users table) exists."""
    conn = _get_conn()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
            # Verify we're in the right database (tables exist)
            cur.execute("SELECT 1 FROM users LIMIT 1")
            cur.fetchone()
            return True
    except Exception:
        return False


def get_cloud_sql_status() -> str:
    """Return a short status message for the UI (e.g. 'Connected' or error hint)."""
    global _last_connection_error
    if not CLOUD_SQL_HOST or not CLOUD_SQL_USER:
        return "Not configured (set CLOUD_SQL_HOST, CLOUD_SQL_USER in .env)"
    conn = _get_conn()
    if not conn:
        err = _last_connection_error or "Connection failed"
        # Shorten long messages; keep clues like "Access denied", "Unknown database", "timed out"
        if len(err) > 80:
            err = err[:77] + "..."
        return err
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DATABASE() AS db")
            row = cur.fetchone()
            db_name = row.get("db") or "(none)"
            cur.execute("SELECT 1 FROM users LIMIT 1")
            cur.fetchone()
            return f"Connected (DB: {db_name})"
    except Exception as e:
        return f"Schema error: {str(e)[:50]} (is CLOUD_SQL_DATABASE correct?)"
