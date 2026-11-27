
import os
import json
from contextlib import asynccontextmanager
from ai_adapter import adapter
import secrets


from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta, timezone
from typing import Optional, Literal
from sync import router as sync_router

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

# Explicit local configuration; missing credentials fail closed.
SECRET_KEY = os.getenv("CLINICIAN_JWT_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480

# Separate signing key for new patient authentication.
PATIENT_JWT_SECRET = os.getenv("PATIENT_JWT_SECRET")
PATIENT_TOKEN_EXPIRE_MINUTES = 60

password_hasher = PasswordHasher()

@asynccontextmanager
async def lifespan(app):
    yield
    adapter.shutdown()


app = FastAPI(
    lifespan=lifespan,
    title="PhysioTrack API",
    version="1.1-dev",
)

app.include_router(sync_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174,http://localhost:3000",
    ).split(','),
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


# Local demo accounts are explicitly configured; passwords are Argon2 hashes.
PHYSIO_ACCOUNTS = {}
if os.getenv("CLINICIAN_PASSWORD_HASH"):
    account_id = os.getenv("CLINICIAN_ID", "DR001").upper()
    PHYSIO_ACCOUNTS[account_id] = {
        "id": account_id,
        "name": os.getenv("CLINICIAN_NAME", "Demo Clinician"),
        "specialty": "Physiotherapy demo",
        "password_hash": os.environ["CLINICIAN_PASSWORD_HASH"],
    }


def require_clinician_config():
    if not SECRET_KEY or len(SECRET_KEY) < 32 or not PHYSIO_ACCOUNTS:
        raise HTTPException(503, "Clinician authentication is not configured")
    return SECRET_KEY


def public_physio(account):
    return {key: account[key] for key in ("id", "name", "specialty")}


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
):
    require_clinician_config()
    to_encode = {**data, "role": "clinician"}

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

    require_clinician_config()
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"require_exp": True, "require_sub": True},
        )

        physio_id = payload.get("sub")

        if (
            payload.get("role") != "clinician"
            or physio_id is None
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
    require_clinician_config()
    physio = PHYSIO_ACCOUNTS.get(
        form_data.username.upper()
    )

    if (
        not physio
        or not verify_patient_password(
            form_data.password,
            physio["password_hash"],
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
    return public_physio(current)


# ============================================================
# NEW PATIENT AUTHENTICATION HELPERS
# ============================================================

patient_bearer = HTTPBearer(auto_error=False)


class SecurePatientRegistration(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    password: str = Field(min_length=12, max_length=128)


class SecurePatientLogin(BaseModel):
    patient_id: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=128)


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
            options={"require_exp": True, "require_sub": True},
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
        "has_prescription": presc is not None,
        "exercise": presc.exercise if presc else None,
        "target_reps": (
            presc.target_reps
            if presc
            else None
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

def measurement_provenance(session):
    try:
        raw = json.loads(session.raw_json or '{}')
        source = raw.get('source', {})
        if isinstance(source, dict) and source.get('pipeline') == 'research_pipeline':
            return {'measurement_source': 'research_pipeline', 'side': raw.get('side'), 'clinically_validated': False}
    except (ValueError, TypeError, AttributeError):
        pass
    return {'measurement_source': 'legacy_unverified', 'side': None, 'clinically_validated': False}


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
        **measurement_provenance(last),
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
            "patient_id": p.patient_id,
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
            **measurement_provenance(s),
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
# SESSION UPLOAD — CLOSED UNTIL TRUSTED INFERENCE IS VERIFIED
# ============================================================

@app.post("/api/sessions")
def upload_session(_=Depends(get_current_patient)):
    # Only a trusted, tested inference adapter may write measurements.
    raise HTTPException(503, "Direct measurement uploads are disabled. Only server AI sessions can save research results.")


# ============================================================
# EXISTING PRESCRIPTION MANAGEMENT
# ============================================================

class PrescriptionInput(BaseModel):
    patient_id: int = Field(gt=0)
    exercise: Literal["elbow_flexion", "shoulder_abduction"]
    target_reps: int = Field(ge=1, le=50)
    target_rom: float | None = Field(default=None, ge=0, le=180)


@app.post("/api/prescriptions")
def set_prescription(
    data: PrescriptionInput,
    db: Session = Depends(get_db),
    _=Depends(get_current_physio),
):
    if db.get(Patient, data.patient_id) is None:
        raise HTTPException(404, "Patient not found")
    p = Prescription(**data.model_dump())

    db.add(p)
    db.commit()

    return {"status": "ok"}


 
# ============================================================
# LEGACY ACQUISITION ROUTES — DISABLED
# The browser AI uses authenticated routes in ai_routes.py.
# ============================================================

@app.websocket("/ws/live/{patient_id}")
async def ws_live(ws: WebSocket, patient_id: int):
    await ws.close(
        code=1008,
        reason="Legacy WebSocket unavailable"
    )


@app.post("/api/live_frame")
def push_frame(_=Depends(get_current_patient)):
    raise HTTPException(
        503,
        "Legacy live frame uploads are disabled"
    )


@app.post("/api/seed")
def seed(_=Depends(get_current_physio)):
    raise HTTPException(
        410,
        "Register fictional patients through the patient portal"
    )


@app.post("/patient/register")
@app.post("/patient/login")
def legacy_patient_auth():
    raise HTTPException(
        410,
        "Use password-based patient authentication"
    )


@app.get("/patient/{patient_id}/last-session")
def patient_last_session(
    patient_id: int,
    patient: Patient = Depends(get_current_patient),
    db: Session = Depends(get_db),
):
    if patient_id != patient.id:
        raise HTTPException(
            403,
            "Cannot access another patient's records"
        )
    return patient_my_last_session(patient, db)


@app.post("/api/launch-session")
@app.post("/api/stop-session")
def desktop_session(_=Depends(get_current_patient)):
    raise HTTPException(
        503,
        "Desktop launcher disabled; use the browser AI session"
    )


@app.get("/api/session-status/{patient_id}")
def session_status(
    patient_id: int,
    patient: Patient = Depends(get_current_patient),
):
    if patient_id != patient.id:
        raise HTTPException(
            403,
            "Cannot access another patient's records"
        )
    return {
        "running": False,
        "available": False,
    }


@app.get("/patient/me/ai-status")
def ai_status(_=Depends(get_current_patient)):
    return adapter.status()



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
from ai_routes import install_ai_routes
install_ai_routes(app, get_current_patient, get_current_physio, get_db, latest_prescription)
