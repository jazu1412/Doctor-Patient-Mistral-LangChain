from pydantic import BaseModel, EmailStr


class AppointmentCreateRequest(BaseModel):
    doctor_name: str
    speciality: str
    email: EmailStr
    appointment_date: str
    slot_start_time: str


class AppointmentCreateResponse(BaseModel):
    ok: bool
    message: str


class AppointmentsResponse(BaseModel):
    appointments: list[dict]
