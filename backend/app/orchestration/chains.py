"""
LangChain orchestration facades (architecture diagram alignment).

These runnables label the three logical agents; existing REST/WebSocket handlers
still perform Mistral/Chroma/SQL work — hooks invoke these chains with metadata only
so behaviour and responses stay unchanged.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def _tag_patient_matchmaker(state: dict) -> dict:
    return {
        **state,
        "orchestration_layer": "langchain",
        "agent_role": "Patient Matchmaker",
    }


def _tag_doctor_support(state: dict) -> dict:
    return {
        **state,
        "orchestration_layer": "langchain",
        "agent_role": "Doctor Support Assistant",
    }


def _tag_emergency_handler(state: dict) -> dict:
    return {
        **state,
        "orchestration_layer": "langchain",
        "agent_role": "Emergency Handler Agent",
    }


# Diagram: Patient Matchmaker — REST match / recommendation paths.
patient_matchmaker_agent = (
    RunnablePassthrough()
    | RunnableLambda(_tag_patient_matchmaker)
).with_config(run_name="PatientMatchmaker", tags=["PatientMatchmaker", "langchain"])

# Diagram: Doctor Support Assistant — clinical analysis & similar-case flows.
doctor_support_assistant_agent = (
    RunnablePassthrough()
    | RunnableLambda(_tag_doctor_support)
).with_config(run_name="DoctorSupportAssistant", tags=["DoctorSupportAssistant", "langchain"])

# Diagram: Emergency Handler Agent — WebSocket / vital-alert context.
emergency_handler_agent = (
    RunnablePassthrough()
    | RunnableLambda(_tag_emergency_handler)
).with_config(run_name="EmergencyHandlerAgent", tags=["EmergencyHandlerAgent", "langchain"])
