import streamlit as st
from mistralai import Mistral
import chromadb
import logging
import re
import hashlib
import time as time_module
from typing import List, Dict, Optional
import os
import json
import urllib.parse
import urllib.request
from dotenv import load_dotenv
from database import (
    sync_init_database,
    sync_sync_doctors,
    sync_get_available_doctors,
    sync_check_doctor_availability,
    sync_book_doctor,
    sync_release_doctor,
    sync_get_all_doctors,
)
from patient_processor import (
    process_patient_dataset,
    search_similar_cases,
    get_collection_stats,
    get_patient_collection,
)
from zipcodes_ca import ZIP_CODES_CA
from datetime import date, time
from streamlit_geolocation import streamlit_geolocation

# Cloud SQL appointments (optional)
try:
    from cloud_sql_appointments import (
        is_cloud_sql_available,
        get_cloud_sql_status,
        get_or_create_user,
        ensure_auth_tables,
        signup_auth_user,
        login_auth_user,
        get_doctor_id_by_name,
        get_available_slots,
        book_appointment,
        list_appointments_for_user,
        sync_doctors_to_cloud_sql,
        SLOT_STARTS,
        DEFAULT_USER_EMAIL,
    )
except ImportError:
    def is_cloud_sql_available(): return False
    def get_cloud_sql_status(): return "Not installed (pip install pymysql)"
    def get_or_create_user(*a, **k): return None
    def ensure_auth_tables(*a, **k): return False
    def signup_auth_user(*a, **k): return (False, "Cloud SQL not installed")
    def login_auth_user(*a, **k): return (False, "Cloud SQL not installed", None)
    def get_doctor_id_by_name(*a, **k): return None
    def get_available_slots(*a, **k): return []
    def book_appointment(*a, **k): return (False, "Cloud SQL not installed")
    def list_appointments_for_user(*a, **k): return []
    def sync_doctors_to_cloud_sql(*a, **k): return (0, "Cloud SQL not installed")
    SLOT_STARTS = []
    DEFAULT_USER_EMAIL = "jas@gmail.com"

# Use same logger as cloud_sql_appointments so logs go to cloud_sql_appointments.log
_log = logging.getLogger("cloud_sql_appointments")

# Pre-computed sorted ZIP list for stable grouping
ZIP_LIST = sorted(ZIP_CODES_CA)
ZIP_GROUP_SIZE = 5  # number of nearby ZIPs to associate with each doctor
ZIP_INDEX = {z: i for i, z in enumerate(ZIP_LIST)}

# Load environment variables
load_dotenv()

# Configuration - Load from environment variables
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "patient-doctor")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "doctor_embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")
CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral-small-latest")

# Validate required environment variables
required_vars = {
    "MISTRAL_API_KEY": MISTRAL_API_KEY,
    "CHROMA_API_KEY": CHROMA_API_KEY,
    "CHROMA_TENANT": CHROMA_TENANT
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file.")
    st.stop()

# Initialize clients
@st.cache_resource
def init_clients():
    """Initialize Mistral and ChromaDB clients"""
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    chroma_client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    # Initialize database (silently fail if DB unavailable)
    try:
        sync_init_database()
    except Exception:
        # Silently fail - app will work without database for demo
        pass
    
    return mistral_client, collection

# Initialize session state
if 'clients_initialized' not in st.session_state:
    st.session_state.clients_initialized = False

try:
    mistral_client, collection = init_clients()
    st.session_state.clients_initialized = True
except Exception as e:
    st.error(f"Failed to initialize clients: {str(e)}")
    st.stop()

def get_symptom_embedding(symptoms: str) -> List[float]:
    """Convert patient symptoms to embedding using Mistral"""
    try:
        embeddings_response = mistral_client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=[symptoms]
        )
        return embeddings_response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating embedding: {str(e)}")
        return None


def normalize_zip(zip_code: str) -> str:
    """Normalize ZIP code to 5-digit string."""
    digits = "".join(ch for ch in zip_code if ch.isdigit())
    return digits[:5] if len(digits) >= 5 else digits


def is_supported_zip(zip_code: str) -> bool:
    """Check if ZIP is in supported CA list."""
    return zip_code in ZIP_CODES_CA


def _zip_distance(a: str, b: str) -> Optional[int]:
    ia = ZIP_INDEX.get(a)
    ib = ZIP_INDEX.get(b)
    if ia is None or ib is None:
        return None
    return abs(ia - ib)


def _doctor_nearby_rank(patient_zip: str, doctor_zips: List[str]) -> int:
    """
    Lower is better.
    0 = exact service ZIP match
    otherwise = closest ZIP "distance" using supported ZIP ordering.
    """
    if not patient_zip or not doctor_zips:
        return 10**9
    pz = normalize_zip(patient_zip)
    if pz in doctor_zips:
        return 0
    best = None
    for dz in doctor_zips:
        d = _zip_distance(pz, dz)
        if d is None:
            continue
        best = d if best is None else min(best, d)
    return best if best is not None else 10**9


def get_browser_geolocation() -> Dict:
    """
    Fetch browser GPS coordinates via streamlit-geolocation component.
    This renders a small UI control; users may need to click it to allow/refresh location.
    """
    try:
        geo = streamlit_geolocation() or {}
        return geo if isinstance(geo, dict) else {}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def reverse_geocode_zip(lat: float, lon: float) -> Optional[str]:
    """
    Reverse geocode lat/lon -> postal code using OpenStreetMap Nominatim.
    """
    lat_r = round(float(lat), 5)
    lon_r = round(float(lon), 5)
    params = urllib.parse.urlencode(
        {"format": "jsonv2", "lat": str(lat_r), "lon": str(lon_r), "addressdetails": "1"}
    )
    url = f"https://nominatim.openstreetmap.org/reverse?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "DoctorPatientMatching/1.0 (local streamlit)",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        addr = data.get("address") or {}
        pc = addr.get("postcode")
        if not pc:
            return None
        return normalize_zip(str(pc))
    except Exception:
        return None


def assign_zips_to_doctor(doctor_name: str) -> List[str]:
    """
    Deterministically assign a small group of nearby ZIP codes to a doctor.
    - Uses a hash of the doctor name to pick a center index.
    - Then takes a window of ZIP_GROUP_SIZE around that index.
    This gives each doctor 4–5 nearby ZIPs in the supported area.
    """
    if not ZIP_LIST:
        return []
    n = len(ZIP_LIST)
    # Use a stable hash so ZIP assignment doesn't change between app runs.
    digest = re.sub(r"\\s+", " ", (doctor_name or "").strip()).encode("utf-8")
    center = int.from_bytes(hashlib.sha256(digest).digest()[:8], "big") % n
    half = ZIP_GROUP_SIZE // 2
    start = max(0, center - half)
    end = min(n, start + ZIP_GROUP_SIZE)
    # Adjust start if we're too close to the end
    start = max(0, end - ZIP_GROUP_SIZE)
    return ZIP_LIST[start:end]


def _clean_doctor_name(name: str) -> str:
    # Some upstream data includes the latin-1 artifact "Â".
    return " ".join(str(name or "").replace("Â", "").split())


def _is_pediatric_speciality(speciality: str) -> bool:
    text = _clean_doctor_name(speciality).casefold()
    pediatric_markers = [
        "pediatr",
        "paediatr",
        "child",
        "kids",
        "kid ",
        "neonat",
        "infant",
        "toddler",
        "adolescent",
        "teen",
        "juvenile",
    ]
    return any(m in text for m in pediatric_markers)


def _filter_doctors_by_age(doctors: List[Dict], patient_age: Optional[int]) -> List[Dict]:
    if patient_age is None:
        return doctors
    try:
        age = int(patient_age)
    except Exception:
        return doctors

    # Age-gating requirement: prevent pediatric doctors appearing for adult matches.
    if age >= 18:
        adult_doctors = [d for d in doctors if not _is_pediatric_speciality(d.get("speciality", ""))]
        return adult_doctors if adult_doctors else []  # strict: avoid pediatric matches
    else:
        pediatric_doctors = [d for d in doctors if _is_pediatric_speciality(d.get("speciality", ""))]
        return pediatric_doctors if pediatric_doctors else doctors  # fallback if none tagged


def _symptom_speciality_bonus(symptoms: str, speciality: str, document: str = "") -> float:
    """
    Lightweight clinical reranking bonus so obvious symptom-speciality pairs rise first.
    """
    text = f"{(symptoms or '').lower()} {(document or '').lower()}"
    spec = (speciality or "").lower()

    rules = [
        {
            "symptoms": ["diarrhea", "diarrhoea", "loose stool", "watery stool", "gastro", "stomach upset"],
            "preferred_specs": ["gastro", "infectious disease", "internal medicine", "family medicine", "general medicine"],
            "bonus": 0.22,
        },
        {
            "symptoms": ["chest pain", "palpitations", "shortness of breath", "heart"],
            "preferred_specs": ["cardio"],
            "bonus": 0.22,
        },
        {
            "symptoms": ["headache", "migraine", "seizure", "dizziness", "neurolog"],
            "preferred_specs": ["neuro"],
            "bonus": 0.2,
        },
        {
            "symptoms": ["rash", "itch", "skin", "eczema"],
            "preferred_specs": ["dermat"],
            "bonus": 0.2,
        },
        {
            "symptoms": ["cough", "wheez", "asthma", "breath", "lung"],
            "preferred_specs": ["pulmon", "respir"],
            "bonus": 0.2,
        },
    ]

    total = 0.0
    for rule in rules:
        if any(k in text for k in rule["symptoms"]) and any(p in spec for p in rule["preferred_specs"]):
            total += rule["bonus"]

    # Cap the bonus so vector similarity still matters.
    return min(total, 0.35)


def find_best_doctor(
    symptoms_embedding: List[float],
    symptoms_text: str = "",
    top_k: int = 3,
    patient_age: Optional[int] = None,
) -> List[Dict]:
    """Find the best matching doctors using vector similarity search, filtered by availability"""
    try:
        # Get more results than needed to account for availability filtering
        results = collection.query(
            query_embeddings=[symptoms_embedding],
            n_results=max(top_k * 6, 20),
            include=["documents", "metadatas", "distances"]
        )
        
        doctors = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                doctor_name = metadata.get("doctor_name", "N/A")
                speciality = metadata.get("speciality", "N/A")

                # Existing metadata may contain a single zip or list of zips
                meta_zips: List[str] = []
                if "zip_codes" in metadata and isinstance(metadata["zip_codes"], list):
                    meta_zips = [normalize_zip(z) for z in metadata["zip_codes"] if z]
                elif "zip_code" in metadata or "zipcode" in metadata:
                    z = metadata.get("zip_code") or metadata.get("zipcode")
                    if z:
                        meta_zips = [normalize_zip(str(z))]

                if meta_zips:
                    zip_codes = [z for z in meta_zips if is_supported_zip(z)]
                else:
                    # Deterministically assign 4–5 nearby ZIPs
                    zip_codes = assign_zips_to_doctor(doctor_name)

                doctor_info = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'doctor_name': doctor_name,
                    'speciality': speciality,
                    'zip_codes': zip_codes,
                    'distance': results['distances'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # used internally only
                }
                # Symptom-aware reranking: boost clinically likely specialities.
                doctor_info["clinical_bonus"] = _symptom_speciality_bonus(
                    symptoms_text,
                    speciality,
                    doctor_info.get("document", "")
                )
                doctor_info["rank_score"] = doctor_info["similarity_score"] + doctor_info["clinical_bonus"]
                doctors.append(doctor_info)
        
        # Sync doctors to database (silently fail if DB unavailable)
        if doctors:
            try:
                sync_sync_doctors(doctors)
            except Exception:
                # Silently fail - don't show error during demo
                pass
        
        # Filter by availability - only return available doctors
        if doctors:
            doctor_names = [doc['doctor_name'] for doc in doctors]
            try:
                available_names = sync_get_available_doctors(doctor_names)
                if available_names:  # Only filter if we got results
                    doctors = [doc for doc in doctors if doc['doctor_name'] in available_names]
                # If no available names returned, show all (fail-safe)
            except Exception:
                # Silently fail - show all doctors if DB unavailable
                pass

        # Age-gated matching (prevents pediatric providers for adult patients)
        if doctors:
            doctors = _filter_doctors_by_age(doctors, patient_age)
        
        doctors.sort(key=lambda d: d.get("rank_score", d.get("similarity_score", 0.0)), reverse=True)

        # Return only top_k available/bookable doctors
        return doctors[:top_k]
    except Exception as e:
        st.error(f"Error querying ChromaDB: {str(e)}")
        return []

def get_doctor_recommendation(symptoms: str) -> str:
    """Get AI-powered recommendation explanation using Mistral chat"""
    try:
        prompt = f"""Based on the patient's symptoms: "{symptoms}", 
provide a brief explanation of why this doctor and speciality would be a good match. 
Keep the response concise (2-3 sentences)."""
        
        chat_response = mistral_client.chat.complete(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        st.warning(f"Could not generate AI recommendation: {str(e)}")
        return ""


@st.dialog("Booking confirmed")
def booking_confirm_dialog(doctor_name_display: str, speciality: str, appt_date: str, slot_time: str, msg: str, appointment_id: Optional[int] = None):
    st.success("✅ Appointment booked successfully")
    st.markdown(f"**Doctor:** {doctor_name_display}")
    st.markdown(f"**Speciality:** {speciality}")
    st.markdown(f"**Date & time:** {appt_date} at {slot_time}")
    if appointment_id is not None:
        st.markdown(f"**Appointment ID:** `{appointment_id}`")
    st.caption(msg)

# Streamlit UI
st.set_page_config(
    page_title="Doctor-Patient Matching System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


def _parse_chroma_doctors_from_collection(col) -> List[Dict]:
    """
    Fetch all doctor metadata from Chroma and normalize into Cloud-SQL sync format.
    Expected keys: doctor_name, speciality (or speciality/specialty).
    """
    all_results = col.get(include=["metadatas"])
    ids = all_results.get("ids") or []
    metadatas = all_results.get("metadatas") or []

    # Chroma returns nested lists depending on version; normalize to flat.
    if ids and isinstance(ids[0], list):
        ids = ids[0]
        metadatas = metadatas[0] if metadatas else []

    doctors = []
    for i, chroma_id in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else None
        if not meta:
            continue
        name = meta.get("doctor_name") or meta.get("name") or ""
        speciality = meta.get("speciality") or meta.get("specialty") or "General"
        name = _clean_doctor_name(name)
        if not name or name.upper() in ("N/A", "NA"):
            continue
        doctors.append({"id": chroma_id, "doctor_name": name, "speciality": speciality})
    return doctors


def ensure_cloud_sql_doctor_present(name: str, speciality: str) -> bool:
    """
    Fast, on-demand Cloud SQL sync for a single doctor.
    Avoids heavy full-directory sync (which slows down UI and reruns).
    """
    if not is_cloud_sql_available():
        return False
    try:
        sync_doctors_to_cloud_sql([{"doctor_name": name, "speciality": speciality}])
        return True
    except Exception:
        return False


def _current_user_email() -> str:
    return (st.session_state.get("auth_user") or {}).get("email", DEFAULT_USER_EMAIL)


def _render_auth_page():
    st.title("🔐 Sign in to continue")
    st.caption("Login or create an account as Patient or Doctor.")
    if not is_cloud_sql_available():
        st.error("Cloud SQL is required for authentication. Please check Cloud SQL connectivity.")
        st.stop()

    ensure_auth_tables()
    tab_login, tab_signup = st.tabs(["Login", "Sign up"])

    with tab_login:
        login_email = st.text_input("Email", key="auth_login_email")
        login_password = st.text_input("Password", type="password", key="auth_login_password")
        if st.button("Login", type="primary", key="auth_login_btn", use_container_width=True):
            ok, msg, user = login_auth_user(login_email, login_password)
            if ok and user:
                st.session_state.auth_user = user
                st.success(f"Welcome, {user.get('full_name') or user.get('email')} ({user.get('role')}).")
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        signup_name = st.text_input("Full name", key="auth_signup_name")
        signup_email = st.text_input("Email", key="auth_signup_email")
        signup_role = st.selectbox("Role", ["patient", "doctor"], key="auth_signup_role")
        signup_password = st.text_input("Password (min 6 chars)", type="password", key="auth_signup_password")
        if st.button("Create account", key="auth_signup_btn", use_container_width=True):
            ok, msg = signup_auth_user(signup_email, signup_password, signup_role, signup_name)
            if ok:
                st.success("Signup successful. Please login.")
            else:
                st.error(msg)


if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if not st.session_state.get("auth_user"):
    _render_auth_page()
    st.stop()

# Navigation (top menu)
_auth_user = st.session_state.get("auth_user") or {}
_is_doctor = (_auth_user.get("role") == "doctor")
if _is_doctor:
    _nav_options = ["📅 My Appointments", "🔬 Clinical Decision Support"]
    _nav_default_index = 1
else:
    _nav_options = ["🏠 Home - Doctor Matching", "📅 My Appointments"]
    _nav_default_index = 0

page = st.radio(
    "Navigation",
    _nav_options,
    index=_nav_default_index,
    horizontal=True,
)

with st.sidebar:
    u = st.session_state.get("auth_user") or {}
    st.markdown("### 👤 Account")
    st.caption(f"{u.get('full_name') or u.get('email', 'Signed in')}")
    st.caption(f"Role: {u.get('role', 'unknown')}")
    if st.button("Logout", key="auth_logout_btn", use_container_width=True):
        st.session_state.auth_user = None
        st.rerun()

# Main content based on selected page
if page == "🏠 Home - Doctor Matching":
    if _is_doctor:
        st.info("Home doctor-matching page is hidden for doctor accounts.")
        st.stop()
    st.title("🏥 Doctor-Patient Matching System")
    st.markdown("Enter your symptoms below to find the best matching doctor for your needs.")
    # Show last booking result so it survives rerun and user can see confirmation
    if "last_booking_result" in st.session_state and st.session_state.last_booking_result:
        r = st.session_state.last_booking_result

        if r.get("ok"):
            appt_id = r.get("appointment_id")
            if st.session_state.get("last_dialog_appointment_id") != appt_id:
                booking_confirm_dialog(
                    doctor_name_display=r.get("doctor_name_display", r.get("doctor_name", "N/A")),
                    speciality=r.get("speciality", ""),
                    appt_date=r.get("date", ""),
                    slot_time=r.get("time", ""),
                    msg=r.get("msg", ""),
                    appointment_id=appt_id,
                )
                st.session_state.last_dialog_appointment_id = appt_id
            st.success(f"✅ Appointment booked for {r.get('date', '')} at {r.get('time', '')}.")
        else:
            st.error(f"❌ Booking failed: {r.get('msg', '')}")

        if st.button("Dismiss", key="dismiss_booking_result"):
            st.session_state.last_booking_result = None
            st.rerun()
        st.divider()

if page == "🏠 Home - Doctor Matching":
    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        symptoms_input = st.text_area(
            "Describe your symptoms:",
            height=150,
            placeholder="e.g., I have been experiencing persistent headaches, dizziness, and blurred vision for the past week..."
        )

    with col2:
        st.markdown("### Search Options")
        top_k = st.selectbox("Number of doctors to show", options=[1, 2, 3, 4, 5], index=2)
        patient_age = st.number_input("Patient age", min_value=0, max_value=120, value=30, step=1)
        show_ai_recommendation = st.checkbox("Show AI recommendation", value=True)
        if "zip_input" not in st.session_state:
            st.session_state.zip_input = ""

        use_geo = st.button("📍 Use my current location", use_container_width=True)
        if use_geo:
            st.session_state.geo_requested = True

        if st.session_state.get("geo_requested"):
            st.caption("Click the small location control below if prompted.")
            geo = get_browser_geolocation()

            if isinstance(geo, dict) and geo.get("error"):
                st.toast(f"Location error: {geo['error']}", icon="⚠️")
                st.session_state.geo_requested = False
            elif isinstance(geo, dict) and geo.get("latitude") is not None and geo.get("longitude") is not None:
                with st.spinner("Finding your ZIP code…"):
                    z = reverse_geocode_zip(float(geo["latitude"]), float(geo["longitude"]))
                if z and is_supported_zip(z):
                    st.session_state.zip_input = z
                    st.toast(f"Detected ZIP: {z}", icon="✅")
                    st.session_state.geo_requested = False
                    st.rerun()
                elif z:
                    st.toast(f"Detected ZIP {z} is outside supported area.", icon="⚠️")
                    st.session_state.geo_requested = False
                else:
                    st.toast("Could not detect ZIP from location.", icon="⚠️")
                    st.session_state.geo_requested = False

        zip_input = st.text_input(
            "Patient ZIP code (optional)",
            placeholder="e.g., 92103",
            max_chars=10,
            help="Used to rank nearby doctors. You can also auto-detect via GPS.",
            key="zip_input",
        )

    # Search button: store results in session state so "Confirm booking" is still rendered on next run
    if st.button("🔍 Find Doctor", type="primary", use_container_width=True):
        if not symptoms_input.strip():
            st.warning("Please enter your symptoms before searching.")
        else:
            with st.spinner("Analyzing your symptoms and finding the best doctor match..."):
                # Step 1: Convert symptoms to embedding
                symptoms_embedding = get_symptom_embedding(symptoms_input)
                
                if symptoms_embedding:
                    # Step 2: Find best matching doctors
                    doctors = find_best_doctor(
                        symptoms_embedding,
                        symptoms_text=symptoms_input,
                        top_k=top_k,
                        patient_age=patient_age,
                    )
                    
                    # Location-aware reranking (nearby doctors first)
                    zip_filter = normalize_zip(zip_input) if zip_input else ""
                    if zip_filter:
                        if not is_supported_zip(zip_filter):
                            st.warning(f"ZIP code {zip_filter} is outside the supported service area.")
                            doctors = []
                        else:
                            for d in doctors:
                                d["nearby_rank"] = _doctor_nearby_rank(zip_filter, d.get("zip_codes", []))
                            doctors.sort(
                                key=lambda d: (
                                    d.get("nearby_rank", 10**9),
                                    -float(d.get("rank_score", d.get("similarity_score", 0.0))),
                                )
                            )
                    
                    if doctors:
                        st.session_state.last_doctors = doctors
                        st.session_state.last_symptoms = symptoms_input
                        st.success(f"Found {len(doctors)} matching doctor(s)!")
                    else:
                        st.session_state.last_doctors = []
                        st.error("No matching doctors found. Please try rephrasing your symptoms.")
                else:
                    st.error("Failed to process your symptoms. Please try again.")

# Always show last search results (and booking UI) on Home when we have them so "Confirm booking" click is not lost
if page == "🏠 Home - Doctor Matching" and st.session_state.get("last_doctors"):
    doctors = st.session_state.last_doctors
    symptoms_for_recommendation = st.session_state.get("last_symptoms", "")

    # Don't sync Cloud SQL here: Streamlit reruns on every keystroke.
    # We sync lazily only when the user actually books (see booking UI).
    # Log selected doctors when recommendation is shown
    _log.info(
        "Recommendation shown for %s doctor(s): %s",
        len(doctors),
        [(d.get("doctor_name"), d.get("speciality")) for d in doctors],
    )
    # Display results (without explicit percentage match score)
    for idx, doctor in enumerate(doctors, 1):
        with st.container():
            col1, _ = st.columns([3, 1])
            with col1:
                doctor_name_display = _clean_doctor_name(doctor.get("doctor_name", "N/A"))
                st.markdown(f"### 👨‍⚕️ Doctor {idx}: {doctor_name_display}")
                st.markdown(f"**Speciality:** {doctor.get('speciality', '')}")
                # Keep ZIPs for nearby ranking, but don't display them in UI.
                cloud_sql_connected = is_cloud_sql_available()

                # Cloud SQL booking UI (independent of Timescale availability)
                if cloud_sql_connected:
                    st.success("✅ Available (book with date & time below)")
                    with st.expander(f"📅 Book slot with {doctor_name_display} (date & time → Cloud SQL)", expanded=True):
                        appt_date = st.date_input(
                            "Date",
                            value=date.today(),
                            min_value=date.today(),
                            key=f"date_{idx}_{doctor['doctor_name']}",
                        )
                        # Ensure THIS doctor exists in Cloud SQL (fast single-row upsert).
                        ensure_cloud_sql_doctor_present(
                            name=doctor.get("doctor_name", ""),
                            speciality=doctor.get("speciality", "General"),
                        )

                        doctor_id = get_doctor_id_by_name(doctor['doctor_name'])
                        if doctor_id is None:
                            st.info("Doctor ID could not be resolved yet. Try again in a few seconds.")
                        else:
                            slots_available = get_available_slots(doctor_id, appt_date)
                            slot_labels = [t.strftime("%I:%M %p") for t in slots_available]
                            if not slot_labels:
                                st.warning("No slots left on this date.")
                            else:
                                chosen_slot_label = st.selectbox(
                                    "Time (9 AM–5 PM)",
                                    slot_labels,
                                    key=f"slot_{idx}_{doctor['doctor_name']}",
                                )
                                if st.button("Confirm booking", key=f"confirm_{idx}_{doctor['doctor_name']}"):
                                    try:
                                        _log.info(
                                            "App: Confirm booking clicked doctor=%s doctor_id=%s date=%s chosen_slot=%s",
                                            doctor['doctor_name'], doctor_id, appt_date, chosen_slot_label
                                        )
                                        user_id = get_or_create_user(_current_user_email())
                                        if user_id is None:
                                            _log.warning("App: get_or_create_user returned None")
                                            st.session_state.last_booking_result = {"ok": False, "msg": "Could not resolve logged-in user."}
                                            st.error("Could not resolve logged-in user.")
                                            st.rerun()

                                        slot_time = slots_available[slot_labels.index(chosen_slot_label)]
                                        _log.info(
                                            "App: calling book_appointment doctor_id=%s user_id=%s date=%s slot_time=%s",
                                            doctor_id, user_id, appt_date, slot_time
                                        )
                                        ok, msg = book_appointment(doctor_id, user_id, appt_date, slot_time)
                                        _log.info("App: book_appointment returned ok=%s msg=%s", ok, msg)

                                        appointment_id = None
                                        m = re.search(r"#(\\d+)", msg or "")
                                        if m:
                                            appointment_id = int(m.group(1))

                                        st.session_state.last_booking_result = {
                                            "ok": ok,
                                            "msg": msg,
                                            "doctor_name": doctor['doctor_name'],
                                            "doctor_name_display": doctor_name_display,
                                            "speciality": doctor.get("speciality", ""),
                                            "date": str(appt_date),
                                            "time": chosen_slot_label,
                                            "appointment_id": appointment_id,
                                        }
                                        st.rerun()
                                    except Exception as e:
                                        st.session_state.last_booking_result = {"ok": False, "msg": str(e)}
                                        st.error(f"Booking error: {e}")
                                        st.rerun()

                # Timescale-only path (when Cloud SQL is not connected)
                else:
                    try:
                        is_available = sync_check_doctor_availability(doctor['doctor_name'])
                        if is_available:
                            st.success("✅ Available")
                            if st.button(f"📅 Book Appointment", key=f"book_{doctor['doctor_name']}_{idx}"):
                                try:
                                    if sync_book_doctor(doctor['doctor_name']):
                                        st.success(f"✅ Successfully booked appointment with {doctor_name_display}!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Doctor is no longer available. Please search again.")
                                except Exception:
                                    st.info("ℹ️ Booking service temporarily unavailable")
                        else:
                            st.warning("⏸️ Currently Unavailable")
                    except Exception:
                        st.success("✅ Available")
                        if st.button(f"📅 Book Appointment", key=f"book_{doctor['doctor_name']}_{idx}"):
                            st.info("ℹ️ Booking feature requires database connection")
                if show_ai_recommendation and idx == 1:
                    recommendation = get_doctor_recommendation(symptoms_for_recommendation)
                    if recommendation:
                        with st.expander("💡 AI Recommendation", expanded=True):
                            st.write(recommendation)
            st.divider()
    # Recommended doctor summary
    if doctors:
        best_doctor = doctors[0]
        best_doctor_display = _clean_doctor_name(best_doctor.get("doctor_name", "N/A"))
        st.markdown("---")
        st.markdown("### ✅ Recommended Doctor")
        st.info(
            f"**{best_doctor_display}** - {best_doctor.get('speciality', '')}\n\n"
            "Based on your symptoms, this doctor is recommended as the best available match."
        )

elif page == "📅 My Appointments":
    st.title("📅 My Appointments")
    st.markdown("View and manage your booked appointments (Cloud SQL).")
    if not is_cloud_sql_available():
        st.warning("Cloud SQL is not connected. Appointments are stored in Cloud SQL — check your connection in the sidebar.")
    else:
        appointments = list_appointments_for_user(email=_current_user_email())
        if not appointments:
            st.info("No appointments yet. Book a slot from **Home - Doctor Matching** after finding a doctor.")
        else:
            st.success(f"**{len(appointments)}** appointment(s) found for **{_current_user_email()}**.")
            # Optional: filter from date
            filter_future = st.checkbox("Show only today and future", value=True, key="my_appt_filter")
            if filter_future:
                today = date.today().isoformat()
                appointments = [a for a in appointments if a["appointment_date"] >= today]
            if not appointments:
                st.caption("No appointments match the filter.")
            else:
                view_tab = st.radio("View", ["List", "Table"], horizontal=True, key="my_appt_view")
                if view_tab == "List":
                    for a in appointments:
                        with st.container():
                            st.markdown(f"**{a['doctor_name']}** — {a['speciality']}")
                            st.caption(f"📅 {a['appointment_date']} at {a['slot_start_time']} · #{a['appointment_id']} · {a['status']}")
                            st.divider()
                else:
                    import pandas as pd
                    df = pd.DataFrame(appointments)
                    df = df[["appointment_id", "appointment_date", "slot_start_time", "doctor_name", "speciality", "status"]]
                    st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "🔬 Clinical Decision Support":
    if not _is_doctor:
        st.error("Clinical Decision Support is available only for doctor accounts.")
        st.stop()
    st.title("🔬 Clinical Decision Support")
    st.markdown("Search for similar past patient cases to aid in clinical decision making.")
    
    # Get collection stats
    stats = get_collection_stats()
    if stats['success']:
        st.info(f"📊 Patient Cases Database: {stats['count']} cases available")
    else:
        st.warning("⚠️ Patient cases database not available. Please upload dataset in Admin Panel.")
    
    # Search interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        patient_symptoms = st.text_area(
            "Enter Patient Symptoms:",
            height=150,
            placeholder="e.g., 45-year-old male presenting with chest pain, shortness of breath, and sweating for the past 2 hours..."
        )
    
    with col2:
        st.markdown("### Search Options")
        num_cases = st.slider("Number of similar cases", 1, 10, 5)
        show_details = st.checkbox("Show detailed case information", value=True)
    
    if st.button("🔍 Search Similar Cases", type="primary", use_container_width=True):
        if not patient_symptoms.strip():
            st.warning("Please enter patient symptoms before searching.")
        else:
            with st.spinner("Searching for similar past cases..."):
                similar_cases = search_similar_cases(patient_symptoms, mistral_client, top_k=num_cases)

                if similar_cases is None:
                    # Error already shown by search_similar_cases(); avoid misleading "no cases" message.
                    st.info("Search failed due to temporary API issue. Please try again in a few seconds.")
                elif similar_cases:
                    st.success(f"Found {len(similar_cases)} similar case(s)!")

                    # AI Analysis of similar cases
                    if len(similar_cases) > 0:
                        st.markdown("---")
                        st.markdown("### 💡 AI Analysis")
                        try:
                            # Use full document text for AI analysis (not truncated)
                            cases_summary = "\n".join([f"Case {i+1}: {case['document']}" for i, case in enumerate(similar_cases[:3])])
                            analysis_prompt = f"""Based on the current patient symptoms: "{patient_symptoms}"

And these similar past cases:
{cases_summary}

Provide a brief clinical analysis comparing the current case with past cases. Highlight:
1. Similarities in symptoms and presentation
2. Potential diagnosis considerations
3. Treatment approaches that worked in similar cases

Keep the response concise and clinically relevant."""
                            
                            analysis_response = mistral_client.chat.complete(
                                model=CHAT_MODEL,
                                messages=[{"role": "user", "content": analysis_prompt}]
                            )
                            
                            st.info(analysis_response.choices[0].message.content)
                        except Exception as e:
                            st.warning(f"Could not generate AI analysis: {str(e)}")

                    st.markdown("---")
                    for idx, case in enumerate(similar_cases, 1):
                        with st.container():
                            st.markdown(f"### 📋 Case {idx}")
                            
                            # Display case document - FULL TEXT, no truncation
                            with st.expander("📄 Case Details (Full Text)", expanded=True):
                                # Use text_area for better display of long text
                                st.text_area(
                                    "Complete Patient Case Information:",
                                    value=case['document'],
                                    height=300,
                                    key=f"case_{idx}_full",
                                    label_visibility="collapsed"
                                )
                                # Also show as markdown for better formatting
                                st.markdown(f"**Full Case Text:**")
                                st.markdown(f"```\n{case['document']}\n```")
                            
                            # Display metadata if available
                            if show_details and case['metadata']:
                                with st.expander("🔍 Additional Information"):
                                    metadata = case['metadata']
                                    # Display key fields
                                    key_fields = ['patient_id', 'diagnosis', 'treatment', 'age', 'gender', 'medical_history']
                                    for field in key_fields:
                                        if field in metadata:
                                            st.markdown(f"**{field.replace('_', ' ').title()}:** {metadata[field]}")
                                    
                                    # Show all metadata in a table
                                    if len(metadata) > len(key_fields):
                                        st.markdown("**All Metadata:**")
                                        import pandas as pd
                                        meta_df = pd.DataFrame([metadata])
                                        st.dataframe(meta_df.T, use_container_width=True)
                            
                            st.divider()
                else:
                    st.error("No similar cases found. The database may be empty or symptoms don't match any cases.")
    
    # Show collection info
    st.markdown("---")
    st.markdown("### 📚 About Clinical Decision Support")
    st.markdown("""
    This feature uses vector similarity search to find past patient cases with similar symptoms.
    It helps clinicians by:
    - Finding similar historical cases
    - Comparing current patient with past cases
    - Providing AI-powered analysis of similarities
    - Aiding in diagnosis and treatment decisions
    """)

elif page == "⚙️ Admin Panel":
    st.title("⚙️ Admin Panel")
    st.markdown("Manage patient dataset and system configuration.")
    
    # Admin tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Dataset", "📊 Collection Stats", "🔗 ChromaDB Info"])
    
    with tab1:
        st.markdown("### 📤 Upload PMC-Patients Dataset")
        st.markdown("""
        Upload the PMC-Patients dataset CSV file to create embeddings for clinical decision support.
        
        **Dataset Source:** [Kaggle - PMC-Patients Dataset](https://www.kaggle.com/datasets/priyamchoksi/pmc-patients-dataset-for-clinical-decision-support)
        """)
        
        # Option to use file path or upload
        use_file_path = st.checkbox("Use file path instead of upload (for large files)", value=False,
                                    help="If upload fails due to size limits, use this option to process file from your filesystem")
        
        uploaded_file = None
        file_path_input = None
        
        if use_file_path:
            file_path_input = st.text_input(
                "Enter full path to CSV file",
                placeholder="/path/to/PMC-Patients.csv",
                help="Enter the absolute path to your CSV file on your computer"
            )
            if file_path_input and os.path.exists(file_path_input):
                st.success(f"✅ File found: {file_path_input}")
            elif file_path_input:
                st.error(f"❌ File not found: {file_path_input}")
        else:
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="Upload the PMC-Patients.csv file from the Kaggle dataset"
            )
            
            if uploaded_file is not None:
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                if file_size_mb > 200:
                    st.warning(f"⚠️ **File is {file_size_mb:.1f} MB. If upload fails, check `.streamlit/config.toml` and restart Streamlit.**")
        
        # Determine which file source to use and get file info
        file_ready = False
        file_size_mb = 0
        
        if use_file_path and file_path_input and os.path.exists(file_path_input):
            file_ready = True
            file_size_mb = os.path.getsize(file_path_input) / (1024 * 1024)
        elif uploaded_file is not None:
            file_ready = True
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        if file_ready:
            
            # Show file info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("File Size", f"{file_size_mb:.1f} MB")
            with col2:
                st.metric("Status", "Ready to Process" if file_size_mb > 0 else "Empty")
            
            # Warning for large files
            if file_size_mb > 200:
                st.warning(f"⚠️ **Large File Detected ({file_size_mb:.1f} MB)**\n\n"
                          f"This file is quite large. Processing may take a significant amount of time and API costs.\n\n"
                          f"**Estimated Processing Time:** {file_size_mb * 0.5:.0f}-{file_size_mb * 2:.0f} minutes\n"
                          f"**Recommendation:** Consider processing a sample first to test.")
            
            # Processing options
            st.markdown("### ⚙️ Processing Options")
            
            col1, col2 = st.columns(2)
            with col1:
                process_sample = st.checkbox("Process Sample First", value=file_size_mb > 200, 
                                            help="Process only first N rows to test")
                if process_sample:
                    sample_size = st.number_input("Sample Size", min_value=100, max_value=10000, 
                                                 value=1000, step=100,
                                                 help="Number of rows to process in sample")
            
            with col2:
                process_full = st.checkbox("Process Full Dataset", value=not (file_size_mb > 200),
                                          help="Process entire dataset (may take hours for large files)")
            
            # Determine file path (will be set when processing starts)
            tmp_path = None
            
            # Process button
            if (process_sample or process_full) and st.button("🚀 Process Dataset", type="primary"):
                # Determine file path
                if use_file_path and file_path_input:
                    tmp_path = file_path_input
                else:
                    # Save uploaded file temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                
                max_rows = sample_size if process_sample and not process_full else None
                
                if max_rows:
                    st.info(f"📝 Processing sample: First {max_rows:,} rows")
                else:
                    st.info(f"📝 Processing full dataset (this may take a while for {file_size_mb:.1f} MB file)")
                
                # Create progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    result = process_patient_dataset(tmp_path, mistral_client, progress_bar, status_text, max_rows)
                    
                    status_text.empty()
                    
                    if result['success']:
                        st.success(f"""
                        ✅ **Processing Complete!**
                        - Processed: {result['processed']:,} records
                        - Failed: {result['failed']:,} records
                        - Total in collection: {result['collection_count']:,} cases
                        - File size: {result.get('file_size_mb', 0):.1f} MB
                        """)
                        
                        if max_rows and process_full:
                            st.info("💡 Sample processing complete. Uncheck 'Process Sample First' and check 'Process Full Dataset' to continue with the rest.")
                    else:
                        st.error(f"❌ Error processing dataset: {result.get('error', 'Unknown error')}")
                        if 'Memory' in result.get('error', ''):
                            st.warning("💡 **Memory Error**: Try processing a smaller sample first.")
                
                except Exception as e:
                    status_text.empty()
                    st.error(f"❌ Processing failed: {str(e)}")
                    if 'memory' in str(e).lower() or 'Memory' in str(e):
                        st.warning("💡 **Memory Issue**: The file is too large to process at once. Try processing a sample first.")
                
                finally:
                    # Clean up temp file (only if it was uploaded, not if using file path)
                    if not use_file_path and tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
        
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Download the PMC-Patients dataset from [Kaggle](https://www.kaggle.com/datasets/priyamchoksi/pmc-patients-dataset-for-clinical-decision-support)
        2. Extract the CSV file (PMC-Patients.csv)
        3. Upload it using the file uploader above
        4. **For large files (>200MB)**: 
           - Start with "Process Sample First" to test (recommended)
           - Then process the full dataset
        5. Click "Process Dataset" to create embeddings
        6. The embeddings will be stored in ChromaDB collection: `patient_cases`
        
        **Note:** Large files (500MB+) may take several hours to process. The system processes data in chunks to manage memory efficiently.
        """)
    
    with tab2:
        st.markdown("### 📊 Collection Statistics")
        
        # Patient collection stats
        patient_stats = get_collection_stats()
        if patient_stats['success']:
            st.metric("Patient Cases", patient_stats['count'])
            st.info(f"Collection: `{patient_stats['collection_name']}`")
        else:
            st.error(f"Error: {patient_stats.get('error', 'Unknown error')}")
        
        # Doctor collection stats
        st.markdown("---")
        st.markdown("### Doctor Collection")
        try:
            doctor_count = collection.count()
            st.metric("Doctor Records", doctor_count)
            st.info(f"Collection: `{COLLECTION_NAME}`")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with tab3:
        st.markdown("### 🔗 ChromaDB Connection Info")
        st.code(f"""
        Database: {CHROMA_DATABASE}
        Tenant: {CHROMA_TENANT}
        Doctor Collection: {COLLECTION_NAME}
        Patient Collection: patient_cases
        """)
        
        st.markdown("### 🌐 ChromaDB Web Interface")
        st.markdown(f"""
        Access your ChromaDB collections via the web interface:
        - [ChromaDB Console](https://www.trychroma.com/vmjs1412/aws-us-east-1/{CHROMA_DATABASE})
        - Doctor Embeddings: [View Collection](https://www.trychroma.com/vmjs1412/aws-us-east-1/{CHROMA_DATABASE}/collections/{COLLECTION_NAME})
        - Patient Cases: [View Collection](https://www.trychroma.com/vmjs1412/aws-us-east-1/{CHROMA_DATABASE}/collections/patient_cases)
        """)
        
        if st.button("🔄 Refresh Stats"):
            st.rerun()

# Sidebar with info (only show on home page)
if page == "🏠 Home - Doctor Matching":
    with st.sidebar:
        st.markdown("### ℹ️ How it works")
        st.markdown("""
        1. **Enter Symptoms**: Describe your symptoms in detail
        2. **AI Analysis**: Your symptoms are converted to embeddings
        3. **Vector Search**: We search for doctors with matching specialities
        4. **Best Match**: The system recommends the most suitable doctor
        
        The matching is based on semantic similarity between your symptoms 
        and doctor specialities using advanced AI embeddings.
        """)
        
        st.markdown("### 📊 System Info")
        st.code(f"""
        Model: {CHAT_MODEL}
        Embedding: {EMBEDDING_MODEL}
        Database: {CHROMA_DATABASE}
        """)
        
        st.markdown("### ☁️ Cloud SQL (appointments)")
        cloud_status = get_cloud_sql_status()
        if is_cloud_sql_available():
            st.success(f"✅ {cloud_status}")
        else:
            st.caption(f"❌ {cloud_status}")
        
        if st.button("🔄 Refresh Connection"):
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🗄️ Database Management")
        
        if st.button("📊 View All Doctors"):
            try:
                all_doctors = sync_get_all_doctors()
                if all_doctors:
                    import pandas as pd
                    df = pd.DataFrame(all_doctors)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No doctors in database yet.")
            except Exception as e:
                st.error(f"Error fetching doctors: {str(e)}")
        
        if is_cloud_sql_available():
            st.markdown("### 📅 My Appointments (Cloud SQL)")
            appointments = list_appointments_for_user(email=_current_user_email())
            if appointments:
                for a in appointments[:10]:
                    st.caption(f"{a['appointment_date']} {a['slot_start_time']} — {a['doctor_name']}")
                if len(appointments) > 10:
                    st.caption(f"... and {len(appointments) - 10} more")
            else:
                st.caption("No appointments yet.")
        
        if st.button("🔄 Sync Doctors from ChromaDB"):
            try:
                # Get all doctors from ChromaDB (request metadatas and documents so names are present)
                all_results = collection.get(include=["metadatas", "documents"])
                doctors_to_sync = []
                ids = all_results.get("ids") or []
                metadatas = all_results.get("metadatas") or []
                documents = all_results.get("documents") or []
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                    metadatas = metadatas[0] if metadatas else []
                    documents = documents[0] if documents else []
                if not metadatas:
                    metadatas = [None] * len(ids)
                if not documents:
                    documents = [""] * len(ids)
                for i, doc_id in enumerate(ids):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    if meta is None:
                        meta = {}
                    doc_text = documents[i] if i < len(documents) else ""
                    name = (meta.get("doctor_name") or meta.get("name") or "").strip() if meta else ""
                    if not name and doc_text:
                        name = (doc_text.split("\n")[0] or doc_text[:80] or "N/A").strip()
                    if not name:
                        name = "N/A"
                    speciality = (meta.get("speciality") or meta.get("specialty") or "General").strip() if meta else "General"
                    doctors_to_sync.append({
                        "id": doc_id,
                        "doctor_name": name,
                        "speciality": speciality,
                    })
                sync_sync_doctors(doctors_to_sync)
                st.success(f"✅ Synced {len(doctors_to_sync)} doctors to database!")
                # Update Cloud SQL doctors so My Appointments shows real names instead of "Doctor 132"
                if is_cloud_sql_available() and doctors_to_sync:
                    n, msg = sync_doctors_to_cloud_sql(doctors_to_sync)
                    if n > 0:
                        st.success(f"✅ {msg}")
                    else:
                        if msg and "not available" not in msg.lower():
                            st.error(f"Cloud SQL doctor names: {msg}")
                        else:
                            st.warning("Cloud SQL doctor names not updated. Check that your DB user has UPDATE on the doctors table.")
            except Exception as e:
                st.error(f"Error syncing doctors: {str(e)}")
else:
    # Sidebar info for other pages
    with st.sidebar:
        st.markdown("### 📊 System Info")
        st.code(f"""
        Model: {CHAT_MODEL}
        Embedding: {EMBEDDING_MODEL}
        Database: {CHROMA_DATABASE}
        """)
        
        if st.button("🔄 Refresh Connection"):
            st.cache_resource.clear()
            st.rerun()


