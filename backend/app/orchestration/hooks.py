"""
Optional LangChain orchestration hooks — failures never affect API responses.
"""

from __future__ import annotations

import logging

from app.orchestration.chains import (
    doctor_support_assistant_agent,
    emergency_handler_agent,
    patient_matchmaker_agent,
)

logger = logging.getLogger(__name__)


def hook_patient_matchmaker(metadata: dict) -> None:
    try:
        patient_matchmaker_agent.invoke(metadata)
    except Exception:
        logger.debug("LangChain PatientMatchmaker hook skipped", exc_info=True)


def hook_doctor_support(metadata: dict) -> None:
    try:
        doctor_support_assistant_agent.invoke(metadata)
    except Exception:
        logger.debug("LangChain DoctorSupportAssistant hook skipped", exc_info=True)


def hook_emergency(metadata: dict) -> None:
    try:
        emergency_handler_agent.invoke(metadata)
    except Exception:
        logger.debug("LangChain EmergencyHandler hook skipped", exc_info=True)
