from pydantic import BaseModel


class ApiError(BaseModel):
    detail: str
