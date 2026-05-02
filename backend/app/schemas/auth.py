from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    role: str
    full_name: str = ""


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthUser(BaseModel):
    id: int
    email: EmailStr
    full_name: str = ""
    role: str


class AuthResponse(BaseModel):
    ok: bool
    message: str
    user: AuthUser | None = None
