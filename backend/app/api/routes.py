from datetime import date, datetime
import urllib.parse
import urllib.request

from fastapi import APIRouter, HTTPException, Query

from cloud_sql_appointments import (
    book_appointment,
    get_available_slots,
    get_doctor_id_by_name,
    get_or_create_user,
    list_appointments_for_user,
    login_auth_user,
    signup_auth_user,
    sync_doctors_to_cloud_sql,
)
from database import sync_get_available_doctors, sync_sync_doctors
from app.core.config import get_settings
from app.schemas.appointment import (
    AppointmentCreateRequest,
    AppointmentCreateResponse,
    AppointmentsResponse,
)
from app.schemas.auth import AuthResponse, LoginRequest, SignupRequest
from app.schemas.clinical import ClinicalAnalysisRequest, ClinicalAnalysisResponse
from app.schemas.doctor import MatchRequest, RecommendationResponse, SlotsResponse
from app.services.dependencies import (
    get_doctor_collection,
    get_mistral_client,
    get_patient_collection,
)
from app.services.mailer import send_booking_confirmation_email
from app.services.matching import (
    assign_zips_to_doctor,
    doctor_nearby_rank,
    filter_doctors_by_age,
    filter_doctors_by_symptom_speciality,
    is_supported_zip,
    normalize_zip,
    symptom_speciality_bonus,
)

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.post("/auth/signup", response_model=AuthResponse)
def signup(payload: SignupRequest) -> AuthResponse:
    ok, msg = signup_auth_user(payload.email, payload.password, payload.role, payload.full_name)
    return AuthResponse(ok=ok, message=msg)


@router.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest) -> AuthResponse:
    ok, msg, user = login_auth_user(payload.email, payload.password)
    return AuthResponse(ok=ok, message=msg, user=user)


@router.post("/match/symptoms")
def match_symptoms(payload: MatchRequest) -> dict:
    settings = get_settings()
    mistral = get_mistral_client()
    collection = get_doctor_collection()
    symptom_text = (payload.symptoms or "").split("\nPatient age:")[0].strip() or (payload.symptoms or "")
    emb = mistral.embeddings.create(model=settings.embedding_model, inputs=[symptom_text])
    query_embedding = emb.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(payload.top_k * 6, 20),
        include=["documents", "metadatas", "distances"],
    )
    doctors = []
    if results.get("ids") and len(results["ids"][0]) > 0:
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i] or {}
            doctor_name = metadata.get("doctor_name", "N/A")
            speciality = metadata.get("speciality", "N/A")
            meta_zips = []
            if isinstance(metadata.get("zip_codes"), list):
                meta_zips = [normalize_zip(z) for z in metadata["zip_codes"] if z]
            elif metadata.get("zip_code") or metadata.get("zipcode"):
                z = metadata.get("zip_code") or metadata.get("zipcode")
                meta_zips = [normalize_zip(str(z))]
            zip_codes = [z for z in meta_zips if is_supported_zip(z)] if meta_zips else assign_zips_to_doctor(doctor_name)
            similarity_score = 1 - float(results["distances"][0][i])
            clinical_bonus = symptom_speciality_bonus(symptom_text, speciality, results["documents"][0][i])
            rank_score = similarity_score + clinical_bonus
            if settings.enable_zip_matching and payload.patient_zip:
                nearby_rank = doctor_nearby_rank(payload.patient_zip, zip_codes)
            else:
                # Zip-based ordering disabled by feature flag.
                nearby_rank = 0
            doctors.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "doctor_name": doctor_name,
                    "speciality": speciality,
                    "zip_codes": zip_codes,
                    "distance": results["distances"][0][i],
                    "similarity_score": similarity_score,
                    "clinical_bonus": clinical_bonus,
                    "rank_score": rank_score,
                    "nearby_rank": nearby_rank,
                }
            )

    if doctors:
        sync_sync_doctors(doctors)
        sync_doctors_to_cloud_sql(doctors)
        available_names = sync_get_available_doctors([d["doctor_name"] for d in doctors])
        doctors = [d for d in doctors if d["doctor_name"] in available_names] if available_names else doctors
        doctors = filter_doctors_by_age(doctors, payload.patient_age)
        doctors = filter_doctors_by_symptom_speciality(doctors, symptom_text)
        doctors.sort(key=lambda d: (d["nearby_rank"], -d["rank_score"]))
    out = {"doctors": doctors[: payload.top_k]}
    from app.orchestration.hooks import hook_patient_matchmaker

    hook_patient_matchmaker(
        {"route": "match/symptoms", "top_k": payload.top_k, "matched_count": len(out["doctors"])}
    )
    return out


@router.post("/match/recommendation", response_model=RecommendationResponse)
def recommendation(payload: MatchRequest) -> RecommendationResponse:
    settings = get_settings()
    mistral = get_mistral_client()
    prompt = (
        f"Based on symptoms: '{payload.symptoms}', explain in 2-3 sentences "
        "why the suggested specialty is a good match."
    )
    chat = mistral.chat.complete(
        model=settings.chat_model, messages=[{"role": "user", "content": prompt}]
    )
    content = chat.choices[0].message.content or ""
    from app.orchestration.hooks import hook_patient_matchmaker

    hook_patient_matchmaker({"route": "match/recommendation"})
    return RecommendationResponse(recommendation=content)


@router.get("/doctors/{doctor_name}/slots", response_model=SlotsResponse)
def doctor_slots(doctor_name: str, appointment_date: str = Query(...)) -> SlotsResponse:
    doctor_id = get_doctor_id_by_name(doctor_name)
    if doctor_id is None:
        raise HTTPException(status_code=404, detail="Doctor not found in Cloud SQL")
    dt = datetime.strptime(appointment_date, "%Y-%m-%d").date()
    slots = get_available_slots(doctor_id, dt)
    return SlotsResponse(
        doctor_id=doctor_id,
        date=appointment_date,
        available_slots=[slot.strftime("%H:%M") for slot in slots],
    )


@router.post("/appointments", response_model=AppointmentCreateResponse)
def create_appointment(payload: AppointmentCreateRequest) -> AppointmentCreateResponse:
    doctor_id = get_doctor_id_by_name(payload.doctor_name)
    if doctor_id is None:
        synced, _ = sync_doctors_to_cloud_sql(
            [{"doctor_name": payload.doctor_name, "speciality": payload.speciality}]
        )
        if synced == 0:
            raise HTTPException(status_code=400, detail="Doctor sync failed")
        doctor_id = get_doctor_id_by_name(payload.doctor_name)
    user_id = get_or_create_user(payload.email)
    if user_id is None or doctor_id is None:
        raise HTTPException(status_code=500, detail="User/doctor resolution failed")

    appt_date = datetime.strptime(payload.appointment_date, "%Y-%m-%d").date()
    slot = datetime.strptime(payload.slot_start_time, "%H:%M").time()
    ok, msg = book_appointment(doctor_id, user_id, appt_date, slot)
    if ok:
        sent, mail_msg = send_booking_confirmation_email(
            to_email=payload.email,
            doctor_name=payload.doctor_name,
            speciality=payload.speciality,
            appointment_date=payload.appointment_date,
            slot_start_time=payload.slot_start_time,
        )
        if not sent:
            msg = f"{msg} (Email not sent: {mail_msg})"
    return AppointmentCreateResponse(ok=ok, message=msg)


@router.get("/appointments/me", response_model=AppointmentsResponse)
def my_appointments(email: str) -> AppointmentsResponse:
    appointments = list_appointments_for_user(email=email, from_date=date.today())
    return AppointmentsResponse(appointments=appointments)


@router.post("/location/reverse-zip")
def reverse_zip(lat: float, lon: float) -> dict:
    params = urllib.parse.urlencode(
        {"format": "jsonv2", "lat": str(round(lat, 5)), "lon": str(round(lon, 5)), "addressdetails": "1"}
    )
    url = f"https://nominatim.openstreetmap.org/reverse?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "DoctorPatientMatching/2.0 (fastapi)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
    import json

    payload = json.loads(body)
    zip_code = (
        payload.get("address", {}).get("postcode")
        or payload.get("address", {}).get("postal_code")
    )
    return {"zip_code": normalize_zip(str(zip_code or ""))}


@router.get("/cases/similar")
def similar_cases(symptoms: str, top_k: int = 5) -> dict:
    settings = get_settings()
    mistral = get_mistral_client()
    patient_collection = get_patient_collection()
    emb = mistral.embeddings.create(model=settings.embedding_model, inputs=[symptoms])
    result = patient_collection.query(
        query_embeddings=[emb.data[0].embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    rows = []
    if result.get("ids") and result["ids"][0]:
        for i, item_id in enumerate(result["ids"][0]):
            rows.append(
                {
                    "id": item_id,
                    "document": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "distance": result["distances"][0][i],
                    "similarity_score": 1 - result["distances"][0][i],
                }
            )
    return {"cases": rows}


@router.post("/clinical/analysis", response_model=ClinicalAnalysisResponse)
def clinical_analysis(payload: ClinicalAnalysisRequest) -> ClinicalAnalysisResponse:
    settings = get_settings()
    mistral = get_mistral_client()
    docs = payload.case_documents[:3]
    if not docs:
        from app.orchestration.hooks import hook_doctor_support

        hook_doctor_support({"route": "clinical/analysis", "cases_used": 0})
        return ClinicalAnalysisResponse(
            analysis="No similar cases were available to analyze."
        )

    cases_summary = "\n".join([f"Case {idx + 1}: {doc}" for idx, doc in enumerate(docs)])
    prompt = f"""Based on the current patient symptoms: "{payload.symptoms}"

And these similar past cases:
{cases_summary}

Provide a brief clinical analysis comparing the current case with past cases. Highlight:
1. Similarities in symptoms and presentation
2. Potential diagnosis considerations
3. Treatment approaches that worked in similar cases

Formatting requirements:
- Use plain markdown headings and bullet points only.
- Do NOT use markdown tables.
- Do NOT use separator lines made of dashes, equals signs, or pipes.
- Do NOT use "||" or "---" style dividers in the content.

Keep the response concise and clinically relevant."""

    chat = mistral.chat.complete(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = chat.choices[0].message.content or ""
    from app.orchestration.hooks import hook_doctor_support

    hook_doctor_support({"route": "clinical/analysis", "cases_used": len(docs)})
    return ClinicalAnalysisResponse(analysis=content)
