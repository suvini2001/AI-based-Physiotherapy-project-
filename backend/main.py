
import json
import os
import random
import secrets
import string
import subprocess
import sys
import live_log

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
    PhysioPatient,
    AuditLog,
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

_PHYSIO_DEMO_PASSWORD = os.getenv("PHYSIO_DEMO_PASSWORD", "")

PHYSIO_ACCOUNTS = {
    "DR001": {
        "id": "DR001",
        "name": "Dr. Priya Nair",
        "password":  _PHYSIO_DEMO_PASSWORD,
        "specialty": "Orthopaedic Rehabilitation",
    },
    "DR002": {
        "id": "DR002",
        "name": "Dr. Kasun Perera",
        "password":  _PHYSIO_DEMO_PASSWORD,
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
# LIVE MONITORING — ROOMS + TICKETS + FRAME-AGE LOGGING
# ============================================================

import uuid as _uuid
from collections import defaultdict

# One set of WebSocket connections per patient id
# { patient_id (int) -> set of WebSocket }
_rooms: dict[int, set] = defaultdict(set)

# Short-lived ws-tickets
# { ticket (str) -> {"patient_id": int, "physio_id": str, "expires": datetime} }
_ws_tickets: dict[str, dict] = {}

WS_TICKET_TTL_SECONDS = 30


# ── Ticket endpoint ──────────────────────────────────────────

@app.post("/api/ws-ticket")
def issue_ws_ticket(
    data: dict,
    db: Session = Depends(get_db),
    physio=Depends(get_current_physio),
):
    """
    A physio requests a one-time ticket to open a WebSocket
    for a specific patient's live feed.

    Returns a ticket string valid for 30 seconds.
    The ticket is consumed when the WebSocket handshake succeeds.
    """
    patient_id = data.get("patient_id")

    if not isinstance(patient_id, int):
        raise HTTPException(
            status_code=400,
            detail="patient_id must be an integer",
        )

    # Check the patient exists
    patient = db.query(Patient).filter(
        Patient.id == patient_id
    ).first()

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found",
        )

    # Check the physio is assigned to this patient
    assignment = db.query(PhysioPatient).filter(
        PhysioPatient.patient_id == patient_id,
        PhysioPatient.physio_id == physio["id"],
    ).first()

    if assignment is None:
        raise HTTPException(
            status_code=403,
            detail="Not assigned to this patient",
        )

    ticket = str(_uuid.uuid4())
    expires = datetime.now(timezone.utc) + timedelta(
        seconds=WS_TICKET_TTL_SECONDS
    )

    _ws_tickets[ticket] = {
        "patient_id": patient_id,
        "physio_id": physio["id"],
        "expires": expires,
    }

    return {
        "ticket": ticket,
        "expires_in_seconds": WS_TICKET_TTL_SECONDS,
    }


# ── WebSocket endpoint ───────────────────────────────────────

@app.websocket("/ws/live/{patient_id}")
async def ws_live(
    ws: WebSocket,
    patient_id: int,
    ticket: str = "",
):
    """
    Doctor connects to a patient's live room using a ticket.
    Ticket is consumed on successful connect.
    """
    # Validate ticket before accepting the connection
    ticket_data = _ws_tickets.get(ticket)

    if ticket_data is None:
        await ws.close(code=4401)
        return

    if ticket_data["patient_id"] != patient_id:
        await ws.close(code=4403)
        return

    if datetime.now(timezone.utc) > ticket_data["expires"]:
        del _ws_tickets[ticket]
        await ws.close(code=4401)
        return

    # Consume the ticket — single use
    del _ws_tickets[ticket]

    await ws.accept()
    _rooms[patient_id].add(ws)

    try:
        while True:
            await ws.receive_text()   # keep alive; client sends heartbeats
    except Exception:
        pass
    finally:
        _rooms[patient_id].discard(ws)
        if not _rooms[patient_id]:
            del _rooms[patient_id]


# ── Frame push with frame-age logging ───────────────────────

@app.post("/api/live_frame")
async def push_frame(data: dict):
    """
    Patient device pushes one live frame.
    Server stamps it with server_ts and computes frame_age_ms.
    Broadcasts only to doctors in that patient's room.
    """
    server_ts = datetime.now(timezone.utc).timestamp()

    patient_id = data.get("patient_id")

    # frame_age_ms: how old the frame is by the time the server sees it
    client_ts = data.get("timestamp")
    frame_age_ms = None
    if isinstance(client_ts, (int, float)):
        frame_age_ms = round((server_ts - client_ts) * 1000, 1)

    # Attach server timestamp and age to the payload before broadcasting
    enriched = {
        **data,
        "server_ts": round(server_ts, 3),
        "frame_age_ms": frame_age_ms,
    }

    # Log frame age for the experiment (Week 8)
    if frame_age_ms is not None:
        print(
            f"[live] patient={patient_id} "
            f"frame_age={frame_age_ms}ms "
            f"angle={data.get('joint_angle')} "
            f"state={data.get('state')}"
        )
        
        live_log.log_frame(
            patient_id=patient_id,
            server_ts=server_ts,
            client_ts=client_ts,
            frame_age_ms=frame_age_ms,
            joint_angle=data.get("joint_angle"),
            state=data.get("state"),
            rep_count=data.get("rep_count"),
            fault=data.get("fault"),
        )

    # Broadcast only to this patient's room
    if patient_id is None:
        # Legacy: broadcast to everyone (keeps old behaviour working)
        targets = {ws for room in _rooms.values() for ws in room}
    else:
        targets = set(_rooms.get(int(patient_id), set()))

    dead = set()
    for ws in targets:
        try:
            await ws.send_json(enriched)
        except Exception:
            dead.add(ws)

    # Clean up dead connections
    for ws in dead:
        for room in _rooms.values():
            room.discard(ws)

    return {"ok": True, "frame_age_ms": frame_age_ms}

# ── RTT report from client ───────────────────────────────────

class RTTReport(BaseModel):
    patient_id: int
    server_ts: float   # the server_ts the client received in the frame
    rtt_ms: float = Field(gt=0)      # client measured: (now - original_client_ts) * 1000


@app.post("/api/live_rtt")
async def receive_rtt(report: RTTReport):
    """
    Client reports the round-trip time for a frame.
    The client measures: time it sent the frame → time it received
    the enriched frame back from the server via WebSocket.
    """
    print(
        f"[rtt] patient={report.patient_id} "
        f"rtt={report.rtt_ms}ms "
        f"server_ts={report.server_ts}"
    )

    live_log.log_rtt(
        patient_id=report.patient_id,
        server_ts=report.server_ts,
        rtt_ms=report.rtt_ms,
    )

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

# ============================================================
# AUDIT LOGGING HELPER
# ============================================================

def audit(
    db: Session,
    physio_id: str,
    action: str,
    patient_id: int = None,
    detail: str = None,
):
    """Record one audit log entry. Call this on every doctor view/export."""
    entry = AuditLog(
        physio_id=physio_id,
        action=action,
        patient_id=patient_id,
        detail=detail,
    )
    db.add(entry)
    db.commit()


# ============================================================
# WEEK 6 — NEEDS ATTENTION + LAST SYNCED API
# ============================================================

@app.get("/api/patients/{patient_id}/summary")
def patient_summary(
    patient_id: int,
    db: Session = Depends(get_db),
    physio=Depends(get_current_physio),
):
    """
    Returns the last 30 days of sessions, adherence score,
    and a needs_attention flag for this patient.
    Also writes an audit log entry.
    """
    from datetime import date
    import json

    patient = db.query(Patient).filter(
        Patient.id == patient_id
    ).first()

    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Audit: doctor viewed this patient
    audit(
        db,
        physio_id=physio["id"],
        action="view_patient",
        patient_id=patient_id,
    )

    # Last 30 days of sessions
    thirty_days_ago = (
        datetime.now(timezone.utc).date() -
        timedelta(days=30)
    ).isoformat()

    sessions = db.query(SessionModel).filter(
        SessionModel.patient_id == patient_id,
        SessionModel.date >= thirty_days_ago,
    ).order_by(SessionModel.date.desc()).all()

    presc = latest_prescription(db, patient_id)
    target_reps = presc.target_reps if presc else 10

    # Adherence: percentage of sessions where total_reps >= target
    if sessions:
        met = sum(
            1 for s in sessions
            if s.total_reps is not None and s.total_reps >= target_reps
        )
        adherence_pct = round(met / len(sessions) * 100)
    else:
        adherence_pct = 0

    # Last synced
    last_session = sessions[0] if sessions else None

    # Needs attention rules:
    # 1. No session in the last 7 days
    # 2. Adherence below 60%
    # 3. Mean ROM dropping (last session ROM < average of previous 5)
    needs_attention = False
    attention_reasons = []

    if not sessions:
        needs_attention = True
        attention_reasons.append("no_sessions_in_30_days")
    else:
        seven_days_ago = (
            datetime.now(timezone.utc).date() -
            timedelta(days=7)
        ).isoformat()
        recent = [s for s in sessions if s.date >= seven_days_ago]
        if not recent:
            needs_attention = True
            attention_reasons.append("no_session_in_7_days")

    if adherence_pct < 60:
        needs_attention = True
        attention_reasons.append("low_adherence")

    if len(sessions) >= 6:
        last_rom = sessions[0].mean_rom
        prev_roms = [
            s.mean_rom for s in sessions[1:6]
            if s.mean_rom is not None
        ]
        if last_rom is not None and prev_roms:
            avg_prev = sum(prev_roms) / len(prev_roms)
            if last_rom < avg_prev * 0.85:   # ROM dropped >15%
                needs_attention = True
                attention_reasons.append("rom_declining")

    return {
        "patient_id": patient_id,
        "name": patient.name,
        "needs_attention": needs_attention,
        "attention_reasons": attention_reasons,
        "adherence_pct": adherence_pct,
        "sessions_last_30_days": len(sessions),
        "last_synced": last_session.date if last_session else None,
        "last_rom": last_session.mean_rom if last_session else None,
        "sessions": [
            {
                "id": s.id,
                "date": s.date,
                "total_reps": s.total_reps,
                "mean_rom": s.mean_rom,
                "mean_quality": s.mean_quality,
            }
            for s in sessions
        ],
    }


@app.get("/api/patients/needs-attention")
def patients_needing_attention(
    db: Session = Depends(get_db),
    physio=Depends(get_current_physio),
):
    """
    Returns all patients assigned to this physio
    who currently need attention.
    """
    # Get patients assigned to this physio
    assignments = db.query(PhysioPatient).filter(
        PhysioPatient.physio_id == physio["id"],
    ).all()

    result = []
    for a in assignments:
        patient = db.query(Patient).filter(
            Patient.id == a.patient_id
        ).first()
        if patient is None:
            continue

        last = db.query(SessionModel).filter(
            SessionModel.patient_id == a.patient_id
        ).order_by(SessionModel.date.desc()).first()

        seven_days_ago = (
            datetime.now(timezone.utc).date() -
            timedelta(days=7)
        ).isoformat()

        needs_attention = (
            last is None or
            last.date < seven_days_ago
        )

        if needs_attention:
            result.append({
                "patient_id": patient.id,
                "name": patient.name,
                "last_synced": last.date if last else None,
                "reason": (
                    "no_sessions_ever" if last is None
                    else "no_session_in_7_days"
                ),
            })

    return result


@app.get("/api/sessions/{session_id}/reps-audited")
def get_reps_audited(
    session_id: int,
    db: Session = Depends(get_db),
    physio=Depends(get_current_physio),
):
    """
    Same as get_reps but writes an audit log entry.
    Use this endpoint from the doctor dashboard.
    """
    session = db.query(SessionModel).filter(
        SessionModel.id == session_id
    ).first()

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Audit: doctor viewed this session's reps
    audit(
        db,
        physio_id=physio["id"],
        action="view_session",
        patient_id=session.patient_id,
        detail=f"session_id={session_id}",
    )

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
            "faults": [f.fault_text for f in faults],
        })

    return result


@app.get("/api/audit-log")
def get_audit_log(
    patient_id: int = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    physio=Depends(get_current_physio),
):
    """
    Returns audit log entries for this physio.
    Optionally filter by patient_id.
    """
    query = db.query(AuditLog).filter(
        AuditLog.physio_id == physio["id"]
    )

    if patient_id is not None:
        query = query.filter(
            AuditLog.patient_id == patient_id
        )

    entries = query.order_by(
        AuditLog.created_at.desc()
    ).limit(limit).all()

    return [
        {
            "id": e.id,
            "action": e.action,
            "patient_id": e.patient_id,
            "detail": e.detail,
            "created_at": e.created_at.isoformat(),
        }
        for e in entries
    ]
