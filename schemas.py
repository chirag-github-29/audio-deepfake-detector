from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# ── User Schemas ─────────────────────────────────
class UserCreate(BaseModel):
    username : str
    email    : str
    password : str

class UserResponse(BaseModel):
    id       : int
    username : str
    email    : str

    class Config:
        from_attributes = True

# ── Auth Schemas ─────────────────────────────────
class Token(BaseModel):
    access_token : str
    token_type   : str

class LoginRequest(BaseModel):
    username : str
    password : str

# ── Detection Schemas ─────────────────────────────
class DetectionResponse(BaseModel):
    id         : int
    filename   : str
    prediction : str
    confidence : float
    label      : int
    timestamp  : datetime

    class Config:
        from_attributes = True