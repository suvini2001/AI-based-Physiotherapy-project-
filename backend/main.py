
import json
import os
import random
import secrets
import string
import subprocess
import sys

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta, timezone
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import (
    VerifyMismatchError,
    VerificationError,
    InvalidHashError,
)

from fastapi import (
    FastAPI,
    WebSocket,
    Depends,
    HTTPException,
    status,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
    HTTPBearer,
    HTTPAuthorizationCredentials,
)

from jose import JWTError, jwt
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from database import get_db, engine
from models import (
    Base,
    Patient,
    Prescription,
    Session as SessionModel,
    Rep,
    Fault,
)


# ============================================================
# DATABASE
# ============================================================

# Creates missing tables, but does NOT migrate existing tables.
# The password_hash column was added through PostgreSQL.
Base.metadata.create_all(bind=engine)


# ============================================================
# CONFIGURATION
# ============================================================

# Existing clinician demo authentication is retained.
# It must be replaced before production use.
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480

# Separate signing key for new patient authentication.
PATIENT_JWT_SECRET = os.getenv("PATIENT_JWT_SECRET")
PATIENT_TOKEN_EXPIRE_MINUTES = 60

password_hasher = PasswordHasher()

app = FastAPI(
    title="PhysioTrack API",
    version="1.1-dev",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# EXISTING CLINICIAN AUTHENTICATION
# ============================================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/login"
)

PHYSIO_ACCOUNTS = {
    "DR001": {
        "id": "DR001",
        "name": "Dr. Priya Nair",
        "password": "physio2026",
        "specialty": "Orthopaedic Rehabilitation",
    },
    "DR002": {
        "id": "DR002",
        "name": "Dr. Kasun Perera",
        "password": "physio2026",
        "specialty": "Neurological Rehabilitation",
    },
}


def verify_password(plain: str, stored: str) -> bool:
    # Legacy clinician-only verification.
    # This is not used for patient passwords.
    return secrets.compare_digest(plain, stored)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
):
    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=15)
    )

    to_encode.update({"exp": expire})

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def get_current_physio(
    token: str = Depends(oauth2_scheme),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )

        physio_id = payload.get("sub")

        if (
            physio_id is None
            or physio_id not in PHYSIO_ACCOUNTS
        ):
            raise credentials_exception

        return PHYSIO_ACCOUNTS[physio_id]

    except JWTError:
        raise credentials_exception


@app.post("/auth/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    physio = PHYSIO_ACCOUNTS.get(
        form_data.username.upper()
    )

    if (
        not physio
        or not verify_password(
            form_data.password,
            physio["password"],
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect ID or password",
        )

    token = create_access_token(
        data={"sub": physio["id"]},
        expires_delta=timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        ),
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "physio": {
            "id": physio["id"],
            "name": physio["name"],
            "specialty": physio["specialty"],
        },
    }


@app.get("/auth/me")
def get_me(
    current=Depends(get_current_physio),
):
    return current


# ============================================================
# NEW PATIENT AUTHENTICATION HELPERS
# ============================================================

patient_bearer = HTTPBearer(auto_error=False)


class SecurePatientRegistration(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    password: str = Field(min_length=12, max_length=128)


class SecurePatientLogin(BaseModel):
    patient_id: str
    password: str


def require_patient_secret():
    if (
        not PATIENT_JWT_SECRET
        or len(PATIENT_JWT_SECRET) < 32
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Patient authentication is not configured. "
                "Set PATIENT_JWT_SECRET."
            ),
        )

    return PATIENT_JWT_SECRET


def hash_patient_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_patient_password(
    password: str,
    stored_hash: str | None,
) -> bool:
    if not stored_hash:
        return False

    try:
        return password_hasher.verify(
            stored_hash,
            password,
        )
    except (
        VerifyMismatchError,
        VerificationError,
        InvalidHashError,
    ):
        return False


def create_patient_access_token(
    patient: Patient,
) -> str:
    signing_secret = require_patient_secret()

    expires = datetime.now(timezone.utc) + timedelta(
        minutes=PATIENT_TOKEN_EXPIRE_MINUTES
    )

    claims = {
        "sub": str(patient.id),
        "role": "patient",
        "exp": expires,
    }

    return jwt.encode(
        claims,
        signing_secret,
        algorithm=ALGORITHM,
    )


def get_current_patient(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        patient_bearer
    ),
    db: Session = Depends(get_db),
) -> Patient:
    unauthorized = HTTPException(
        status_code=401,
        detail="Patient authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
    ):
        raise unauthorized

    signing_secret = require_patient_secret()

    try:
        payload = jwt.decode(
            credentials.credentials,
            signing_secret,
            algorithms=[ALGORITHM],
        )

        if payload.get("role") != "patient":
            raise unauthorized

        subject = payload.get("sub")

        if (
            not isinstance(subject, str)
            or not subject.isdigit()
        ):
            raise unauthorized

        patient = db.query(Patient).filter(
            Patient.id == int(subject)
        ).first()

        if patient is None or not patient.password_hash:
            raise unauthorized

        return patient

    except JWTError:
        raise unauthorized


def latest_prescription(
    db: Session,
    patient_id: int,
):
    return db.query(Prescription).filter(
        Prescription.patient_id == patient_id
    ).order_by(
        Prescription.id.desc()
    ).first()


def patient_response(
    patient: Patient,
    db: Session,
):
    presc = latest_prescription(db, patient.id)

    return {
        "id": patient.id,
        "patient_id": patient.patient_id,
        "name": patient.name,
        "exercise": (
            presc.exercise
            if presc
            else "elbow_flexion"
        ),
        "target_reps": (
            presc.target_reps
            if presc
            else 10
        ),
    }


# ============================================================
# NEW SECURE PATIENT REGISTRATION
# ============================================================

@app.post(
    "/patient/register-secure",
    status_code=201,
)
def patient_register_secure(
    data: SecurePatientRegistration,
    db: Session = Depends(get_db),
):
    require_patient_secret()

    name = data.name.strip()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Name is required",
        )

    password_hash = hash_patient_password(
        data.password
    )

    # Retry to handle an unlikely patient-ID collision.
    for _ in range(10):
        patient_id = (
            "PT" +
            str(secrets.randbelow(1000000)).zfill(6)
        )

        patient = Patient(
            name=name,
            patient_id=patient_id,
            password_hash=password_hash,
            notes="Secure self-registration",
        )

        try:
            db.add(patient)
            db.commit()
            db.refresh(patient)

            token = create_patient_access_token(
                patient
            )

            return {
                "access_token": token,
                "token_type": "bearer",
                "patient": patient_response(
                    patient,
                    db,
                ),
            }

        except IntegrityError:
            db.rollback()

    raise HTTPException(
        status_code=503,
        detail="Could not generate patient ID",
    )


# ============================================================
# NEW SECURE PATIENT LOGIN
# ============================================================

@app.post("/patient/login-secure")
def patient_login_secure(
    data: SecurePatientLogin,
    db: Session = Depends(get_db),
):
    require_patient_secret()

    patient_id = data.patient_id.strip().upper()

    patient = db.query(Patient).filter(
        Patient.patient_id == patient_id
    ).first()

    if (
        patient is None
        or not verify_patient_password(
            data.password,
            patient.password_hash,
        )
    ):
        # Same error for unknown ID and wrong password.
        raise HTTPException(
            status_code=401,
            detail="Incorrect patient ID or password",
        )

    token = create_patient_access_token(
        patient
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "patient": patient_response(
            patient,
            db,
        ),
    }


# ============================================================
# PROTECTED PATIENT PROFILE
# ============================================================

@app.get("/patient/me")
def patient_me(
    patient: Patient = Depends(get_current_patient),
    db: Session = Depends(get_db),
):
    return patient_response(patient, db)


# ============================================================
# PROTECTED PATIENT SESSION HISTORY
# ============================================================

@app.get("/patient/me/last-session")
def patient_my_last_session(
    patient: Patient = Depends(get_current_patient),
    db: Session = Depends(get_db),
):
    last = db.query(SessionModel).filter(
        SessionModel.patient_id == patient.id
    ).order_by(
        SessionModel.id.desc()
    ).first()

    if not last:
        return {
            "total_reps": 0,
            "mean_rom": None,
            "date": None,
        }

    return {
        "total_reps": last.total_reps,
        "mean_rom": last.mean_rom,
        "date": last.date,
    }


# ============================================================
# EXISTING CLINICIAN PATIENT MANAGEMENT
# ============================================================

@app.get("/api/patients")
def list_patients(
    db: Session = Depends(get_db),
    _=Depends(get_current_physio),
):
    patients = db.query(Patient).all()

    result = []

    for p in patients:
        last = db.query(SessionModel).filter(
            SessionModel.patient_id == p.id
        ).order_by(
            SessionModel.id.desc()
        ).first()

        presc = latest_prescription(db, p.id)

        result.append({
            "id": p.id,
            "name": p.name,
            "dob": p.dob,
            "notes": p.notes,
            "last_session": (
                last.date if last else None
            ),
            "last_rom": (
                last.mean_rom if last else None
            ),
            "total_sessions": (
                db.query(SessionModel).filter(
                    SessionModel.patient_id == p.id
                ).count()
            ),
            "prescription": {
                "exercise": presc.exercise,
                "target_reps": presc.target_reps,
                "target_rom": presc.target_rom,
            } if presc else None,
        })

    return result


@app.post("/api/patients")
def create_patient(
    data: dict,
    db: Session = Depends(get_db),
    _=Depends(get_current_physio),
):
    p = Patient(
        name=data["name"],
        dob=data.get("dob"),
        notes=data.get("notes"),
    )

    db.add(p)
    db.commit()
    db.refresh(p)

    return {
        "id": p.id,
        "name": p.name,
    }


@app.get("/api/patients/{patient_id}/sessions")
def get_sessions(
    patient_id: int,
    db: Session = Depends(get_db),
    _=Depends(get_current_physio),
):
    sessions = db.query(SessionModel).filter(
        SessionModel.patient_id == patient_id
    ).order_by(
        SessionModel.id.desc()
    ).all()

    return [
        {
            "id": s.id,
            "exercise": s.exercise,
            "total_reps": s.total_reps,
            "mean_rom": s.mean_rom,
            "mean_quality": s.mean_quality,
            "date": s.date,
        }
        for s in sessions
    ]


@app.get("/api/sessions/{session_id}/reps")
def get_reps(
    session_id: int,
    db: Session = Depends(get_db),
    _=Depends(get_current_physio),
):
    reps = db.query(Rep).filter(
        Rep.session_id == session_id
    ).all()

    result = []

    for r in reps:
        faults = db.query(Fault).filter(
            Fault.rep_id == r.id
        ).all()

        result.append({
            "rep_number": r.rep_number,
            "max_angle": r.max_angle,
            "min_angle": r.min_angle,
            "rom": r.rom,
            "duration_s": r.duration_s,
            "quality_score": r.quality_score,
            "faults": [
                f.fault_text for f in faults
            ],
        })

    return result


# ============================================================
# EXISTING SESSION UPLOAD
#
# WARNING:
# This legacy endpoint is unauthenticated.
# It must be secured before real patient use.
# ============================================================

@app.post("/api/sessions")
def upload_session(
    data: dict,
    db: Session = Depends(get_db),
):
    summary = data.get("summary", {})

    s = SessionModel(
        patient_id=data.get("patient_id", 1),
        exercise=data["exercise"],
        total_reps=data["total_reps"],
        mean_rom=summary.get("mean_rom"),
        mean_quality=None,
        date=data["date"],
        raw_json=json.dumps(data),
    )

    db.add(s)
    db.commit()
    db.refresh(s)

    for rep in data.get("reps", []):
        r = Rep(
            session_id=s.id,
            rep_number=rep["rep_number"],
            max_angle=rep["max_angle"],
            min_angle=rep["min_angle"],
            rom=rep["rom"],
            duration_s=rep["duration_s"],
            quality_score=rep.get("quality_score"),
        )

        db.add(r)
        db.flush()

        for fault in rep.get("faults", []):
            db.add(
                Fault(
                    rep_id=r.id,
                    fault_text=fault,
                )
            )

    db.commit()

    return {
        "status": "ok",
        "session_id": s.id,
    }


# ============================================================
# EXISTING PRESCRIPTION MANAGEMENT
# ============================================================

@app.post("/api/prescriptions")
def set_prescription(
    data: dict,
    db: Session = Depends(get_db),
    _=Depends(get_current_physio),
):
    p = Prescription(
        patient_id=data["patient_id"],
        exercise=data["exercise"],
        target_reps=data["target_reps"],
        target_rom=data.get("target_rom"),
    )

    db.add(p)
    db.commit()

    return {"status": "ok"}


# ============================================================
# EXISTING LIVE MONITORING
#
# WARNING:
# These legacy endpoints are not authenticated.
# ============================================================

active_ws: list[WebSocket] = []


@app.websocket("/ws/live/{patient_id}")
async def ws_live(
    ws: WebSocket,
    patient_id: int,
):
    await ws.accept()
    active_ws.append(ws)

    try:
        while True:
            await ws.receive_text()
    except Exception:
        if ws in active_ws:
            active_ws.remove(ws)


@app.post("/api/live_frame")
async def push_frame(data: dict):
    dead = []

    for ws in active_ws:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)

    for ws in dead:
        if ws in active_ws:
            active_ws.remove(ws)

    return {"ok": True}


# ============================================================
# EXISTING DEVELOPMENT SEED
#
# WARNING:
# Disable/remove before deployment.
# ============================================================

@app.post("/api/seed")
def seed(
    db: Session = Depends(get_db),
):
    if db.query(Patient).count() > 0:
        return {"status": "already seeded"}

    patients = [
        Patient(
            name="Amara Silva",
            dob="1985-03-12",
            notes="Post-op elbow fracture",
        ),
        Patient(
            name="Nimal Fernando",
            dob="1972-07-29",
            notes="Rotator cuff repair",
        ),
        Patient(
            name="Sanduni Perera",
            dob="1990-11-05",
            notes="Frozen shoulder",
        ),
    ]

    for p in patients:
        db.add(p)

    db.commit()

    return {
        "status": "seeded",
        "patients": 3,
    }


# ============================================================
# EXISTING LEGACY PATIENT AUTHENTICATION
#
# WARNING:
# These endpoints are insecure and retained only to
# prevent disruption during development.
#
# The React frontend still uses these endpoints.
# ============================================================

def generate_patient_id(db):
    while True:
        pid = "PT" + "".join(
            random.choices(
                string.digits,
                k=4,
            )
        )

        if not db.query(Patient).filter(
            Patient.patient_id == pid
        ).first():
            return pid


@app.post("/patient/register")
def patient_register(
    data: dict,
    db: Session = Depends(get_db),
):
    name = data.get("name", "").strip()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Name is required",
        )

    pid = generate_patient_id(db)

    p = Patient(
        name=name,
        patient_id=pid,
        notes="Self-registered",
    )

    db.add(p)
    db.commit()
    db.refresh(p)

    return {
        "patient_id": pid,
        "name": p.name,
        "id": p.id,
    }


@app.post("/patient/login")
def patient_login(
    data: dict,
    db: Session = Depends(get_db),
):
    pid = data.get(
        "patient_id", ""
    ).strip().upper()

    name = data.get(
        "name", ""
    ).strip().lower()

    p = db.query(Patient).filter(
        Patient.patient_id == pid
    ).first()

    if not p:
        raise HTTPException(
            status_code=404,
            detail="Patient ID not found",
        )

    if p.name.strip().lower() != name:
        raise HTTPException(
            status_code=401,
            detail="Name does not match",
        )

    return patient_response(p, db)


@app.get("/patient/{patient_id}/last-session")
def patient_last_session(
    patient_id: int,
    db: Session = Depends(get_db),
):
    last = db.query(SessionModel).filter(
        SessionModel.patient_id == patient_id
    ).order_by(
        SessionModel.id.desc()
    ).first()

    if not last:
        return {
            "total_reps": 0,
            "mean_rom": None,
            "date": None,
        }

    return {
        "total_reps": last.total_reps,
        "mean_rom": last.mean_rom,
        "date": last.date,
    }


# ============================================================
# EXISTING DESKTOP AI LAUNCHER
#
# Preserved for compatibility.
#
# WARNING:
# These endpoints are not authenticated.
# The original launcher uses a Windows path.
# The browser-based simulation does not use it.
# ============================================================

active_process = {}


@app.post("/api/launch-session")
def launch_session(
    data: dict,
    db: Session = Depends(get_db),
):
    patient_id = data.get("patient_id")
    exercise = data.get(
        "exercise",
        "elbow",
    )
    side = data.get(
        "side",
        "right",
    )

    exercise_arg = (
        "shoulder"
        if exercise == "shoulder_abduction"
        else "elbow"
    )

    project_root = (
        r"D:\3YP\AI-based-Physiotherapy-project-"
    )

    script_path = os.path.join(
        project_root,
        "demo_cv.py",
    )

    python_exe = os.path.join(
        project_root,
        ".venv",
        "Scripts",
        "python.exe",
    )

    log_path = os.path.join(
        project_root,
        "session_launch.log",
    )

    if not os.path.exists(script_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"demo_cv.py not found at {script_path}"
            ),
        )

    if not os.path.exists(python_exe):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Python not found at {python_exe}"
            ),
        )

    if str(patient_id) in active_process:
        try:
            active_process[
                str(patient_id)
            ].terminate()
        except Exception:
            pass

    log_file = open(log_path, "w")

    print(
        f"Launching: {python_exe} {script_path} "
        f"--exercise {exercise_arg} "
        f"--side {side} "
        f"--patient-id {patient_id}"
    )

    proc = subprocess.Popen(
        [
            python_exe,
            script_path,
            "--exercise",
            exercise_arg,
            "--side",
            side,
            "--patient-id",
            str(patient_id),
        ],
        cwd=project_root,
        stdout=log_file,
        stderr=log_file,
    )

    log_file.close()

    active_process[str(patient_id)] = proc

    print(f"Launched PID {proc.pid}")

    return {
        "status": "launched",
        "pid": proc.pid,
        "exercise": exercise_arg,
    }


@app.post("/api/stop-session")
def stop_session(data: dict):
    patient_id = str(
        data.get("patient_id")
    )

    if patient_id in active_process:
        try:
            active_process[
                patient_id
            ].terminate()

            del active_process[patient_id]

        except Exception:
            pass

    return {"status": "stopped"}


@app.get("/api/session-status/{patient_id}")
def session_status(patient_id: str):
    proc = active_process.get(patient_id)

    if proc is None:
        return {"running": False}

    poll = proc.poll()

    if poll is not None:
        del active_process[patient_id]

        return {
            "running": False,
            "exit_code": poll,
        }

    return {
        "running": True,
        "pid": proc.pid,
    }
