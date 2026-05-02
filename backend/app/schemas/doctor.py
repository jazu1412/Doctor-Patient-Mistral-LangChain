from pydantic import BaseModel


class DoctorItem(BaseModel):
    id: str
    doctor_name: str
    speciality: str
    zip_codes: list[str]
    similarity_score: float
    rank_score: float
    clinical_bonus: float
    distance: float


class MatchRequest(BaseModel):
    symptoms: str
    top_k: int = 5
    patient_age: int | None = None
    patient_zip: str | None = None


class RecommendationResponse(BaseModel):
    recommendation: str


class SlotsResponse(BaseModel):
    doctor_id: int
    date: str
    available_slots: list[str]
