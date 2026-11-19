#This defines what JSON looks like coming in and going out of the API. 
from pydantic import BaseModel
from typing import Optional, List
import datetime


class FaultSchema(BaseModel):
    fault_text: str

    class Config:
        from_attributes = True


class RepSchema(BaseModel):
    rep_number:    int
    max_angle:     float
    min_angle:     float
    rom:           float
    duration_s:    float
    quality_score: Optional[float] = None
    faults:        List[FaultSchema] = []

    class Config:
        from_attributes = True


class SessionCreate(BaseModel):
    patient_id:   int = 1
    exercise:     str
    total_reps:   int
    date:         str
    mean_rom:     Optional[float] = None
    mean_quality: Optional[float] = None
    reps:         List[RepSchema] = []
    summary:      Optional[dict]  = None


class SessionOut(BaseModel):
    id:           int
    patient_id:   int
    exercise:     str
    total_reps:   int
    mean_rom:     Optional[float]
    mean_quality: Optional[float]
    date:         str

    class Config:
        from_attributes = True


class PatientCreate(BaseModel):
    name:  str
    dob:   Optional[str] = None
    notes: Optional[str] = None


class PatientOut(BaseModel):
    id:    int
    name:  str
    dob:   Optional[str]
    notes: Optional[str]

    class Config:
        from_attributes = True


class PrescriptionCreate(BaseModel):
    patient_id:  int
    exercise:    str
    target_reps: int
    target_rom:  float


class PrescriptionOut(BaseModel):
    id:          int
    patient_id:  int
    exercise:    str
    target_reps: int
    target_rom:  float

    class Config:
        from_attributes = True


class LiveFrame(BaseModel):
    timestamp:   float
    joint_angle: Optional[float] = None
    state:       Optional[str]   = None
    rep_count:   int             = 0
    fault:       Optional[str]   = None