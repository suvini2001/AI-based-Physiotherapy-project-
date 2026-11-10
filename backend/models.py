from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base
import datetime

class Patient(Base):
    __tablename__ = "patients"
    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String, nullable=False)
    patient_id = Column(String, unique=True, index=True)   # ← ADD THIS LINE
    dob        = Column(String)
    notes      = Column(Text)
    sessions   = relationship("Session", back_populates="patient")
    prescriptions = relationship("Prescription", back_populates="patient")

class Prescription(Base):
    __tablename__ = "prescriptions"
    id          = Column(Integer, primary_key=True)
    patient_id  = Column(Integer, ForeignKey("patients.id"))
    exercise    = Column(String)
    target_reps = Column(Integer)
    target_rom  = Column(Float)
    created_at  = Column(DateTime, default=datetime.datetime.utcnow)
    patient     = relationship("Patient", back_populates="prescriptions")

class Session(Base):
    __tablename__ = "sessions"
    id           = Column(Integer, primary_key=True, index=True)
    patient_id   = Column(Integer, ForeignKey("patients.id"))
    exercise     = Column(String)
    total_reps   = Column(Integer)
    mean_rom     = Column(Float)
    mean_quality = Column(Float)
    date         = Column(String)
    raw_json     = Column(Text)
    patient      = relationship("Patient", back_populates="sessions")
    reps         = relationship("Rep", back_populates="session", cascade="all, delete-orphan")

class Rep(Base):
    __tablename__ = "reps"
    id            = Column(Integer, primary_key=True)
    session_id    = Column(Integer, ForeignKey("sessions.id"))
    rep_number    = Column(Integer)
    max_angle     = Column(Float)
    min_angle     = Column(Float)
    rom           = Column(Float)
    duration_s    = Column(Float)
    quality_score = Column(Float)
    session       = relationship("Session", back_populates="reps")
    faults        = relationship("Fault", back_populates="rep", cascade="all, delete-orphan")

class Fault(Base):
    __tablename__ = "faults"
    id         = Column(Integer, primary_key=True)
    rep_id     = Column(Integer, ForeignKey("reps.id"))
    fault_text = Column(String)
    rep        = relationship("Rep", back_populates="faults")