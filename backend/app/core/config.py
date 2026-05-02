import os
from functools import lru_cache
from pydantic import BaseModel


class Settings(BaseModel):
    api_v1_prefix: str = "/api/v1"
    app_name: str = "Doctor Patient API"
    cors_origins_raw: str = os.getenv("CORS_ORIGINS", "*")
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    chroma_api_key: str = os.getenv("CHROMA_API_KEY", "")
    chroma_tenant: str = os.getenv("CHROMA_TENANT", "")
    chroma_database: str = os.getenv("CHROMA_DATABASE", "patient-doctor")
    collection_name: str = os.getenv("COLLECTION_NAME", "doctor_embeddings")
    patient_collection_name: str = os.getenv("PATIENT_COLLECTION_NAME", "patient_cases")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "mistral-embed")
    chat_model: str = os.getenv("CHAT_MODEL", "mistral-small-latest")
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "465"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from_email: str = os.getenv("SMTP_FROM_EMAIL", "medimatchai7@gmail.com")
    enable_zip_matching: bool = os.getenv("ENABLE_ZIP_MATCHING", "false").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    @property
    def cors_origins(self) -> list[str]:
        if self.cors_origins_raw.strip() == "*":
            return ["*"]
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
