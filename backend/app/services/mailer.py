import logging
import smtplib
from email.message import EmailMessage

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def send_booking_confirmation_email(
    to_email: str,
    doctor_name: str,
    speciality: str,
    appointment_date: str,
    slot_start_time: str,
) -> tuple[bool, str]:
    settings = get_settings()
    if not settings.smtp_username or not settings.smtp_password:
        return False, "SMTP credentials are not configured"

    msg = EmailMessage()
    msg["Subject"] = "MediMatch Appointment Confirmation"
    msg["From"] = settings.smtp_from_email
    msg["To"] = to_email
    msg.set_content(
        f"""Your appointment is confirmed.

Doctor: {doctor_name}
Speciality: {speciality}
Date: {appointment_date}
Time: {slot_start_time}

Thank you,
MediMatch AI
"""
    )

    try:
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=15) as smtp:
            smtp.login(settings.smtp_username, settings.smtp_password)
            smtp.send_message(msg)
        return True, "Confirmation email sent"
    except Exception as exc:
        logger.exception("Failed to send booking confirmation email: %s", exc)
        return False, f"Failed to send email: {exc}"
