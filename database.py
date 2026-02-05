"""
Database module for managing doctor availability in TimescaleDB
Using synchronous psycopg2 for Streamlit compatibility
"""
import psycopg2
from psycopg2 import pool
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import threading

load_dotenv()

# Database connection string
DB_CONNECTION = os.getenv(
    "DB_CONNECTION",
    "postgres://tsdbadmin:g9heeciuyqfflr2z@lowkayf0dw.u827f1jcvc.tsdb.cloud.timescale.com:35981/tsdb?sslmode=require"
)

# Parse connection string
def parse_connection_string(conn_str: str) -> Dict[str, str]:
    """Parse PostgreSQL connection string"""
    if conn_str.startswith("postgres://"):
        # Parse postgres:// format
        conn_str = conn_str.replace("postgres://", "")
        if "@" in conn_str:
            auth, rest = conn_str.split("@", 1)
            if ":" in auth:
                user, password = auth.split(":", 1)
            else:
                user, password = auth, ""
            
            if "?" in rest:
                host_port_db, params = rest.split("?", 1)
            else:
                host_port_db, params = rest, ""
            
            if "/" in host_port_db:
                host_port, dbname = host_port_db.rsplit("/", 1)
            else:
                host_port, dbname = host_port_db, "tsdb"
            
            if ":" in host_port:
                host, port = host_port.split(":", 1)
            else:
                host, port = host_port, "5432"
            
            return {
                "host": host,
                "port": port,
                "dbname": dbname,
                "user": user,
                "password": password,
                "sslmode": "require"
            }
    return {}

# Connection pool
_connection_pool = None
_pool_lock = threading.Lock()

def get_connection():
    """Get database connection from pool"""
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                try:
                    conn_params = parse_connection_string(DB_CONNECTION)
                    _connection_pool = psycopg2.pool.SimpleConnectionPool(
                        1, 5,
                        host=conn_params.get("host"),
                        port=conn_params.get("port"),
                        database=conn_params.get("dbname"),
                        user=conn_params.get("user"),
                        password=conn_params.get("password"),
                        sslmode=conn_params.get("sslmode", "require")
                    )
                except Exception as e:
                    # Fallback: try direct connection string
                    try:
                        _connection_pool = psycopg2.pool.SimpleConnectionPool(
                            1, 5, DB_CONNECTION
                        )
                    except:
                        # If pool fails, return None - operations will handle gracefully
                        return None
    
    try:
        return _connection_pool.getconn()
    except:
        return None


def return_connection(conn):
    """Return connection to pool"""
    global _connection_pool
    if _connection_pool and conn:
        try:
            _connection_pool.putconn(conn)
        except:
            pass


def init_database():
    """Initialize database schema - create doctors table if it doesn't exist"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False
        
        cur = conn.cursor()
        
        # Create doctors table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id SERIAL PRIMARY KEY,
                doctor_name VARCHAR(255) NOT NULL UNIQUE,
                speciality VARCHAR(255) NOT NULL,
                chroma_id VARCHAR(255),
                is_available BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_doctors_name ON doctors(doctor_name)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_doctors_available ON doctors(is_available)
        """)
        
        # Create updated_at trigger
        cur.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        
        cur.execute("""
            DROP TRIGGER IF EXISTS update_doctors_updated_at ON doctors;
            CREATE TRIGGER update_doctors_updated_at
            BEFORE UPDATE ON doctors
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """)
        
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_connection(conn)


def sync_doctors_from_chroma(doctors: List[Dict]):
    """Sync doctors from ChromaDB to TimescaleDB"""
    if not doctors:
        return
    
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        
        cur = conn.cursor()
        
        for doctor in doctors:
            doctor_name = doctor.get('doctor_name')
            speciality = doctor.get('speciality', 'Unknown')
            chroma_id = doctor.get('id', '')
            
            if doctor_name:
                cur.execute("""
                    INSERT INTO doctors (doctor_name, speciality, chroma_id, is_available)
                    VALUES (%s, %s, %s, TRUE)
                    ON CONFLICT (doctor_name) 
                    DO UPDATE SET 
                        speciality = EXCLUDED.speciality,
                        chroma_id = EXCLUDED.chroma_id,
                        updated_at = CURRENT_TIMESTAMP
                """, (doctor_name, speciality, chroma_id))
        
        conn.commit()
        cur.close()
    except Exception as e:
        if conn:
            conn.rollback()
    finally:
        if conn:
            return_connection(conn)


def get_available_doctors(doctor_names: List[str]) -> List[str]:
    """Get list of available doctor names from database"""
    if not doctor_names:
        return []
    
    conn = None
    try:
        conn = get_connection()
        if not conn:
            # If no connection, return all names (fail-safe)
            return doctor_names
        
        cur = conn.cursor()
        cur.execute("""
            SELECT doctor_name 
            FROM doctors 
            WHERE doctor_name = ANY(%s) AND is_available = TRUE
        """, (doctor_names,))
        
        rows = cur.fetchall()
        result = [row[0] for row in rows]
        cur.close()
        return result
    except Exception:
        # If error, return all names (fail-safe)
        return doctor_names
    finally:
        if conn:
            return_connection(conn)


def check_doctor_availability(doctor_name: str) -> bool:
    """Check if a doctor is available"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return True  # Fail-safe: assume available
        
        cur = conn.cursor()
        cur.execute("""
            SELECT is_available 
            FROM doctors 
            WHERE doctor_name = %s
        """, (doctor_name,))
        
        row = cur.fetchone()
        cur.close()
        return row[0] if row else True  # Default to available if not found
    except Exception:
        return True  # Fail-safe: assume available
    finally:
        if conn:
            return_connection(conn)


def book_doctor(doctor_name: str) -> bool:
    """Mark a doctor as unavailable (booked)"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False
        
        cur = conn.cursor()
        cur.execute("""
            UPDATE doctors 
            SET is_available = FALSE, updated_at = CURRENT_TIMESTAMP
            WHERE doctor_name = %s AND is_available = TRUE
            RETURNING id
        """, (doctor_name,))
        
        result = cur.fetchone()
        conn.commit()
        cur.close()
        return result is not None
    except Exception:
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_connection(conn)


def release_doctor(doctor_name: str) -> bool:
    """Mark a doctor as available again"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False
        
        cur = conn.cursor()
        cur.execute("""
            UPDATE doctors 
            SET is_available = TRUE, updated_at = CURRENT_TIMESTAMP
            WHERE doctor_name = %s
            RETURNING id
        """, (doctor_name,))
        
        result = cur.fetchone()
        conn.commit()
        cur.close()
        return result is not None
    except Exception:
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_connection(conn)


def get_all_doctors() -> List[Dict]:
    """Get all doctors with their availability status"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return []
        
        cur = conn.cursor()
        cur.execute("""
            SELECT id, doctor_name, speciality, is_available, created_at, updated_at
            FROM doctors
            ORDER BY doctor_name
        """)
        
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        result = [dict(zip(columns, row)) for row in rows]
        cur.close()
        return result
    except Exception:
        return []
    finally:
        if conn:
            return_connection(conn)


def get_doctor_by_name(doctor_name: str) -> Optional[Dict]:
    """Get doctor information by name"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return None
        
        cur = conn.cursor()
        cur.execute("""
            SELECT id, doctor_name, speciality, is_available, created_at, updated_at
            FROM doctors
            WHERE doctor_name = %s
        """, (doctor_name,))
        
        row = cur.fetchone()
        cur.close()
        
        if row:
            columns = [desc[0] for desc in cur.description]
            return dict(zip(columns, row))
        return None
    except Exception:
        return None
    finally:
        if conn:
            return_connection(conn)


# Synchronous wrapper functions (for consistency with existing code)
sync_init_database = init_database
sync_sync_doctors = sync_doctors_from_chroma
sync_get_available_doctors = get_available_doctors
sync_check_doctor_availability = check_doctor_availability
sync_book_doctor = book_doctor
sync_release_doctor = release_doctor
sync_get_all_doctors = get_all_doctors
sync_get_doctor_by_name = get_doctor_by_name
