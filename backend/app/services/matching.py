import hashlib
import re
from typing import Optional

from zipcodes_ca import ZIP_CODES_CA


ZIP_LIST = sorted(ZIP_CODES_CA)
ZIP_GROUP_SIZE = 5
ZIP_INDEX = {z: i for i, z in enumerate(ZIP_LIST)}


def normalize_zip(zip_code: str) -> str:
    digits = "".join(ch for ch in (zip_code or "") if ch.isdigit())
    return digits[:5] if len(digits) >= 5 else digits


def is_supported_zip(zip_code: str) -> bool:
    return zip_code in ZIP_CODES_CA


def assign_zips_to_doctor(doctor_name: str) -> list[str]:
    if not ZIP_LIST:
        return []
    n = len(ZIP_LIST)
    digest = re.sub(r"\s+", " ", (doctor_name or "").strip()).encode("utf-8")
    center = int.from_bytes(hashlib.sha256(digest).digest()[:8], "big") % n
    half = ZIP_GROUP_SIZE // 2
    start = max(0, center - half)
    end = min(n, start + ZIP_GROUP_SIZE)
    start = max(0, end - ZIP_GROUP_SIZE)
    return ZIP_LIST[start:end]


def _zip_distance(a: str, b: str) -> Optional[int]:
    ia = ZIP_INDEX.get(a)
    ib = ZIP_INDEX.get(b)
    if ia is None or ib is None:
        return None
    return abs(ia - ib)


def doctor_nearby_rank(patient_zip: str, doctor_zips: list[str]) -> int:
    if not patient_zip or not doctor_zips:
        return 10**9
    pz = normalize_zip(patient_zip)
    if pz in doctor_zips:
        return 0
    best = None
    for dz in doctor_zips:
        dist = _zip_distance(pz, dz)
        if dist is None:
            continue
        best = dist if best is None else min(best, dist)
    return best if best is not None else 10**9


def _is_pediatric_speciality(speciality: str) -> bool:
    text = str(speciality or "").casefold()
    return any(
        marker in text
        for marker in [
            "pediatr",
            "paediatr",
            "child",
            "kids",
            "neonat",
            "infant",
            "toddler",
            "adolescent",
            "teen",
            "juvenile",
        ]
    )


def filter_doctors_by_age(doctors: list[dict], patient_age: Optional[int]) -> list[dict]:
    if patient_age is None:
        return doctors
    if patient_age >= 18:
        return [d for d in doctors if not _is_pediatric_speciality(d.get("speciality", ""))]
    pediatric = [d for d in doctors if _is_pediatric_speciality(d.get("speciality", ""))]
    return pediatric if pediatric else doctors


def symptom_speciality_bonus(symptoms: str, speciality: str, document: str = "") -> float:
    text = f"{(symptoms or '').lower()} {(document or '').lower()}"
    spec = (speciality or "").lower()
    total = 0.0
    if any(k in text for k in ["diarrhea", "diarrhoea", "loose stool", "watery stool", "gastro"]):
        total += 0.22 if any(s in spec for s in ["gastro", "internal medicine", "family medicine"]) else 0.0
    if any(k in text for k in ["chest pain", "palpitations", "shortness of breath", "heart"]):
        total += 0.22 if "cardio" in spec else 0.0
    if any(k in text for k in ["headache", "migraine", "seizure", "dizziness"]):
        total += 0.20 if "neuro" in spec else 0.0
    if any(k in text for k in ["rash", "itch", "skin", "eczema"]):
        total += 0.20 if "dermat" in spec else 0.0
    if any(k in text for k in ["urinary", "dysuria", "prostate", "testicular", "genital"]):
        total += 0.30 if any(s in spec for s in ["urolog", "androlog", "genitourinary"]) else -0.08
    if any(
        k in text
        for k in [
            "knee pain",
            "knee",
            "joint pain",
            "arthritis",
            "hip pain",
            "back pain",
            "shoulder pain",
            "ankle pain",
            "sprain",
            "fracture",
            "ligament",
            "muscle pain",
        ]
    ):
        if any(
            s in spec
            for s in [
                "orthop",
                "sports medicine",
                "physiatr",
                "physical medicine",
                "rheumat",
                "pain medicine",
                "rehab",
            ]
        ):
            total += 0.34
        elif "neuro" in spec and not any(
            n in text
            for n in ["numbness", "tingling", "seizure", "migraine", "headache", "neuropath"]
        ):
            total -= 0.20
    return max(min(total, 0.45), -0.2)


def preferred_speciality_markers_for_symptoms(symptoms: str) -> list[str]:
    text = (symptoms or "").lower()
    if any(
        k in text
        for k in [
            "knee pain",
            "joint pain",
            "hip pain",
            "shoulder pain",
            "ankle pain",
            "arthritis",
            "sprain",
            "fracture",
            "ligament",
            "muscle pain",
        ]
    ):
        return [
            "orthop",
            "sports medicine",
            "physiatr",
            "physical medicine",
            "rheumat",
            "pain medicine",
            "rehab",
        ]
    return []


def filter_doctors_by_symptom_speciality(doctors: list[dict], symptoms: str) -> list[dict]:
    preferred = preferred_speciality_markers_for_symptoms(symptoms)
    if not preferred:
        return doctors
    preferred_doctors = [
        d for d in doctors if any(p in (d.get("speciality", "") or "").lower() for p in preferred)
    ]
    return preferred_doctors if preferred_doctors else doctors
