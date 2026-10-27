from typing import Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str
    filename: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=2_000)
    history: list[ChatTurn] = []


Member = Literal["self", "spouse", "children", "parents"]
_Short = Field(default="", max_length=300)


class HealthProfile(BaseModel):
    """What the person tells the fit-score form. Used for one assessment only, never stored."""

    # Basics
    age: int = Field(ge=0, le=110)
    gender: Literal["female", "male", "other", "unspecified"] = "unspecified"
    city_tier: Literal["metro", "tier2", "tier3", "unspecified"] = "unspecified"
    covered_members: list[Member] = Field(default_factory=lambda: ["self"], min_length=1, max_length=4)
    desired_sum_insured: int | None = Field(default=None, ge=0, le=1_000_000_000)  # rupees, up to ₹100 crore
    annual_budget: int | None = Field(default=None, ge=0, le=100_000_000)  # rupees
    # Medical history
    conditions: list[str] = Field(default_factory=list, max_length=20)
    medications: str = _Short
    past_surgeries: str = _Short
    hospitalised_last_2y: bool = False
    # Lifestyle
    smoker: bool = False
    alcohol: Literal["none", "occasional", "regular"] = "none"
    height_cm: float | None = Field(default=None, ge=40, le=250)
    weight_kg: float | None = Field(default=None, ge=2, le=350)
    hazardous_activity: str = _Short
    # Plans and priorities
    planning_pregnancy: bool = False
    planned_procedures: list[str] = Field(default_factory=list, max_length=10)
    wants_opd: bool = False
    wants_dental: bool = False
    wants_ayush: bool = False
    avoid_copay: bool = False
    avoid_room_rent_limit: bool = False


class FitScoreRequest(BaseModel):
    session_id: str
    profile: HealthProfile
