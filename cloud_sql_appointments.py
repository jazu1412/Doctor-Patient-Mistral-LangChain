"""
Google Cloud SQL (MySQL) appointments: book slots 9 AM–5 PM, default user jas@gmail.com.
Set CLOUD_SQL_* in .env or leave unset to skip Cloud SQL (app still works with TimescaleDB only).
"""
import os
import logging
import sys
import hashlib
import hmac
import secrets
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

_PASSWORD_ALGO = "pbkdf2_sha256"
_PASSWORD_ITERATIONS = 240000


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
        # Validate existing connection before reuse.
        if _connection is not None:
            try:
                _connection.ping(reconnect=True)
            except Exception:
                try:
                    _connection.close()
                except Exception:
                    pass
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


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _PASSWORD_ITERATIONS,
    ).hex()
    return f"{_PASSWORD_ALGO}${_PASSWORD_ITERATIONS}${salt}${digest}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iters_s, salt, expected = encoded.split("$", 3)
        if algo != _PASSWORD_ALGO:
            return False
        iters = int(iters_s)
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iters,
    ).hex()
    return hmac.compare_digest(actual, expected)


def ensure_auth_tables() -> bool:
    """Create app_users table for login/signup if missing."""
    # Use a fresh connection to avoid stale-socket issues after long idle periods.
    conn = _get_conn(reconnect=True)
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_users (
                    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    full_name VARCHAR(255) NULL,
                    role ENUM('patient', 'doctor') NOT NULL,
                    password_hash VARCHAR(512) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        return True
    except Exception as e:
        _logger.warning("ensure_auth_tables failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def signup_auth_user(email: str, password: str, role: str, full_name: str = "") -> Tuple[bool, str]:
    """Create a doctor/patient login account in Cloud SQL."""
    conn = _get_conn(reconnect=True)
    if not conn:
        return False, "Cloud SQL not available"
    email_n = (email or "").strip().lower()
    role_n = (role or "").strip().lower()
    full_name_n = (full_name or "").strip()
    if role_n not in ("patient", "doctor"):
        return False, "Role must be patient or doctor"
    if "@" not in email_n:
        return False, "Please enter a valid email"
    if len(password or "") < 6:
        return False, "Password must be at least 6 characters"
    try:
        ensure_auth_tables()
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM app_users WHERE email = %s", (email_n,))
            if cur.fetchone():
                return False, "Email already registered"
            cur.execute(
                """
                INSERT INTO app_users (email, full_name, role, password_hash)
                VALUES (%s, %s, %s, %s)
                """,
                (email_n, full_name_n or None, role_n, _hash_password(password)),
            )
            conn.commit()
        # Keep appointments users table consistent for patient bookings.
        get_or_create_user(email_n)
        return True, "Signup successful"
    except Exception as e:
        _logger.warning("signup_auth_user failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return False, str(e)


def login_auth_user(email: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
    """Authenticate app user by email/password."""
    conn = _get_conn()
    if not conn:
        return False, "Cloud SQL not available", None
    email_n = (email or "").strip().lower()
    try:
        ensure_auth_tables()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, full_name, role, password_hash FROM app_users WHERE email = %s",
                (email_n,),
            )
            row = cur.fetchone()
            if not row:
                return False, "User not found", None
            if not _verify_password(password or "", row.get("password_hash", "")):
                return False, "Invalid email or password", None
            user = {
                "id": row["id"],
                "email": row["email"],
                "full_name": row.get("full_name") or "",
                "role": row["role"],
            }
            return True, "Login successful", user
    except Exception as e:
        _logger.warning("login_auth_user failed: %s", e)
        return False, str(e), None


def _normalize_doctor_name(name: str) -> str:
    return " ".join(str(name or "").strip().split())


def _is_placeholder_name(name: str) -> bool:
    n = _normalize_doctor_name(name).casefold()
    return n.startswith("doctor ")


def _deterministic_bucket_id(doctor_name: str) -> int:
    digest = hashlib.sha256(_normalize_doctor_name(doctor_name).casefold().encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % 200
    return bucket + 1


def _resolve_or_assign_doctor_id(doctor_name: str, speciality: Optional[str] = None) -> Optional[int]:
    """
    Return a stable doctor_id for a name, assigning a safe slot when needed.

    Strategy:
    1) If exact name already exists in doctors, reuse that doctor_id.
    2) Else probe ids deterministically (hash bucket + linear probing) and claim the
       first placeholder/empty slot or same-name slot.
    3) If Cloud SQL is unavailable, return deterministic fallback id.
    """
    normalized = _normalize_doctor_name(doctor_name)
    if not normalized:
        return None
    spec = (speciality or "General").strip() or "General"

    conn = _get_conn()
    if not conn:
        return _deterministic_bucket_id(normalized)

    try:
        with conn.cursor() as cur:
            # First, exact-name lookup (case-insensitive).
            cur.execute(
                """
                SELECT doctor_id
                FROM doctors
                WHERE LOWER(TRIM(doctor_name)) = LOWER(TRIM(%s))
                LIMIT 1
                """,
                (normalized,),
            )
            row = cur.fetchone()
            if row and row.get("doctor_id") is not None:
                did = int(row["doctor_id"])
                # Keep speciality fresh for known doctor rows.
                cur.execute(
                    "UPDATE doctors SET speciality = %s WHERE doctor_id = %s",
                    (spec, did),
                )
                conn.commit()
                return did

            base_id = _deterministic_bucket_id(normalized)

            # Deterministic linear probing to avoid collisions.
            for offset in range(200):
                cand = ((base_id - 1 + offset) % 200) + 1
                cur.execute(
                    "SELECT doctor_name FROM doctors WHERE doctor_id = %s LIMIT 1",
                    (cand,),
                )
                slot = cur.fetchone()

                if not slot:
                    cur.execute(
                        "INSERT INTO doctors (doctor_id, doctor_name, speciality) VALUES (%s, %s, %s)",
                        (cand, normalized, spec),
                    )
                    conn.commit()
                    return cand

                existing_name = _normalize_doctor_name(slot.get("doctor_name", ""))
                if existing_name.casefold() == normalized.casefold() or _is_placeholder_name(existing_name):
                    cur.execute(
                        "UPDATE doctors SET doctor_name = %s, speciality = %s WHERE doctor_id = %s",
                        (normalized, spec, cand),
                    )
                    conn.commit()
                    return cand

    except Exception as e:
        _logger.debug("_resolve_or_assign_doctor_id failed for %s: %s", normalized, e)
        try:
            conn.rollback()
        except Exception:
            pass

    return _deterministic_bucket_id(normalized)


def get_doctor_id_by_name(doctor_name: str) -> Optional[int]:
    """
    Resolve doctor_id from doctors table by name; fallback to deterministic id mapping (1..200).

    NOTE:
    We must not use Python's built-in hash(), because it is randomized per process.
    Randomized hashes can map the same doctor to a different doctor_id after restart,
    causing appointments to show the wrong doctor in JOIN queries.
    """
    if not doctor_name:
        return None
    return _resolve_or_assign_doctor_id(doctor_name)


def find_doctor_id_by_name(doctor_name: str) -> Optional[int]:
    """Lookup-only resolver: return doctor_id if name exists in Cloud SQL doctors table, else None."""
    normalized = _normalize_doctor_name(doctor_name)
    if not normalized:
        return None
    conn = _get_conn()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doctor_id
                FROM doctors
                WHERE LOWER(TRIM(doctor_name)) = LOWER(TRIM(%s))
                LIMIT 1
                """,
                (normalized,),
            )
            row = cur.fetchone()
            if row and row.get("doctor_id") is not None:
                return int(row["doctor_id"])
    except Exception as e:
        _logger.debug("find_doctor_id_by_name failed for %s: %s", normalized, e)
    return None


def is_doctor_bookable_in_cloud_sql(doctor_name: str, on_date: Optional[date] = None) -> bool:
    """
    Return True only if doctor exists in Cloud SQL and has at least one free slot on the date.
    """
    doctor_id = find_doctor_id_by_name(doctor_name)
    if doctor_id is None:
        return False
    target_date = on_date or date.today()
    slots = get_available_slots(doctor_id, target_date)
    return len(slots) > 0


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
    Uses deterministic assignment with collision handling so each doctor name
    gets a stable row and is not overwritten by another name.
    Returns (updated_count, message).
    """
    conn = _get_conn(reconnect=True)
    if not conn:
        return 0, "Cloud SQL not available"
    updated = 0
    try:
        # Deterministic order so doctor_id assignment doesn't depend on
        # the incoming list order (important for consistent appointment joins).
        doctors_sorted = sorted(
            doctors,
            key=lambda d: _normalize_doctor_name((d.get("doctor_name") or d.get("name") or "")),
        )
        for d in doctors_sorted:
            name = d.get("doctor_name") or d.get("name")
            speciality = d.get("speciality") or d.get("specialty") or "General"
            if not name or str(name).strip() in ("", "N/A"):
                continue
            doctor_id = _resolve_or_assign_doctor_id(str(name), str(speciality))
            if doctor_id is not None:
                updated += 1
                _logger.debug("sync_doctors_to_cloud_sql: assigned doctor_id=%s -> %s", doctor_id, str(name)[:50])
        _logger.info("sync_doctors_to_cloud_sql: updated/assigned %s doctors", updated)
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
