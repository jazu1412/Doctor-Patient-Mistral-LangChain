from pydantic import BaseModel


class ClinicalAnalysisRequest(BaseModel):
    symptoms: str
    case_documents: list[str]


class ClinicalAnalysisResponse(BaseModel):
    analysis: str
