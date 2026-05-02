from functools import lru_cache

import chromadb
from mistralai import Mistral

from app.core.config import get_settings


@lru_cache
def get_mistral_client() -> Mistral:
    settings = get_settings()
    return Mistral(api_key=settings.mistral_api_key)


@lru_cache
def get_chroma_client():
    settings = get_settings()
    return chromadb.CloudClient(
        api_key=settings.chroma_api_key,
        tenant=settings.chroma_tenant,
        database=settings.chroma_database,
    )


def get_doctor_collection():
    settings = get_settings()
    client = get_chroma_client()
    return client.get_or_create_collection(name=settings.collection_name)


def get_patient_collection():
    settings = get_settings()
    client = get_chroma_client()
    return client.get_or_create_collection(name=settings.patient_collection_name)
