
import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    ForeignKey,
    DateTime,
)

from sqlalchemy.orm import relationship

from database import Base


# ==================================================
# PATIENT MODEL
# ==================================================

class Patient(Base):
    __tablename__ = "patients"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    name = Column(
        String,
        nullable=False,
    )

    patient_id = Column(
        String,
        unique=True,
        index=True,
    )

    # NEW:
    # Stores a secure password hash, never the password.
    #
    # Nullable temporarily to preserve existing
    # development patients during migration.
    password_hash = Column(
        String(255),
        nullable=True,
    )

    dob = Column(String)

    notes = Column(Text)

    sessions = relationship(
        "Session",
        back_populates="patient",
    )

    prescriptions = relationship(
        "Prescription",
        back_populates="patient",
    )


# ==================================================
# PRESCRIPTION MODEL
# ==================================================

class Prescription(Base):
    __tablename__ = "prescriptions"

    id = Column(
        Integer,
        primary_key=True,
    )

    patient_id = Column(
        Integer,
        ForeignKey("patients.id"),
    )

    exercise = Column(String)

    target_reps = Column(Integer)

    target_rom = Column(Float)

    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
    )

    patient = relationship(
        "Patient",
        back_populates="prescriptions",
    )

    contract_version = Column(
        Integer,
        default=1,
        nullable=False,
        server_default="1",
    )


# ==================================================
# EXERCISE SESSION MODEL
# ==================================================

class Session(Base):
    __tablename__ = "sessions"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    patient_id = Column(
        Integer,
        ForeignKey("patients.id"),
    )

    exercise = Column(String)

    total_reps = Column(Integer)

    mean_rom = Column(Float)

    mean_quality = Column(Float)

    date = Column(String)

    raw_json = Column(Text)

    patient = relationship(
        "Patient",
        back_populates="sessions",
    )

    reps = relationship(
        "Rep",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    session_uuid = Column(
        String(36),
        unique=True,
        nullable=True,   # nullable so old sessions aren't broken
        index=True,
    )


# ==================================================
# REPETITION MODEL
# ==================================================

class Rep(Base):
    __tablename__ = "reps"

    id = Column(
        Integer,
        primary_key=True,
    )

    session_id = Column(
        Integer,
        ForeignKey("sessions.id"),
    )

    rep_number = Column(Integer)

    max_angle = Column(Float)

    min_angle = Column(Float)

    rom = Column(Float)

    duration_s = Column(Float)

    quality_score = Column(Float)

    session = relationship(
        "Session",
        back_populates="reps",
    )

    faults = relationship(
        "Fault",
        back_populates="rep",
        cascade="all, delete-orphan",
    )


# ==================================================
# MOVEMENT FAULT MODEL
# ==================================================

class Fault(Base):
    __tablename__ = "faults"

    id = Column(
        Integer,
        primary_key=True,
    )

    rep_id = Column(
        Integer,
        ForeignKey("reps.id"),
    )

    fault_text = Column(String)

    rep = relationship(
        "Rep",
        back_populates="faults",
    )

# ==================================================
# PHYSIO MODEL
# (replaces the hardcoded PHYSIO_ACCOUNTS dict)
# ==================================================

class Physio(Base):
    __tablename__ = "physios"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    physio_id = Column(
        String(20),
        unique=True,
        index=True,
        nullable=False,
    )

    name = Column(
        String(150),
        nullable=False,
    )

    specialty = Column(String(150))

    password_hash = Column(
        String(255),
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
    )

    patients = relationship(
        "PhysioPatient",
        back_populates="physio",
    )


# ==================================================
# DEVICE MODEL
# (one row per patient device — holds the device token)
# ==================================================

class Device(Base):
    __tablename__ = "devices"

    id = Column(
        Integer,
        primary_key=True,
    )

    patient_id = Column(
        Integer,
        ForeignKey("patients.id"),
        nullable=False,
    )

    device_token = Column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    # human-readable label e.g. "Chamodi's phone"
    label = Column(String(100))

    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
    )

    last_seen_at = Column(
        DateTime,
        nullable=True,
    )

    # set when the device is revoked (stolen phone)
    revoked_at = Column(
        DateTime,
        nullable=True,
    )

    patient = relationship("Patient")


# ==================================================
# PHYSIO-PATIENT ASSIGNMENT
# (which physio looks after which patient)
# ==================================================

class PhysioPatient(Base):
    __tablename__ = "physio_patient"

    id = Column(
        Integer,
        primary_key=True,
    )

    physio_id = Column(
        Integer,
        ForeignKey("physios.id"),
        nullable=False,
    )

    patient_id = Column(
        Integer,
        ForeignKey("patients.id"),
        nullable=False,
    )

    assigned_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
    )

    physio = relationship(
        "Physio",
        back_populates="patients",
    )

    patient = relationship("Patient")
