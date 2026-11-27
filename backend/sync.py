"""
backend/sync.py — Person B, Weeks 1 & 3.
Offline-sync backbone: device-token login, idempotent session sync, resumable
chunk upload. Add to the app with:   app.include_router(sync_router)   in main.py.

Reuses the existing tables (Session.session_uuid unique, Device.device_token) and
the same PATIENT_JWT_SECRET, so tokens it issues are accepted by main.get_current_patient.
"""
import os
import json
import hashlib
import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from jose import JWTError, jwt

from database import get_db
from models import Session as SessionModel, Rep, Fault, Device

router = APIRouter(tags=["sync"])

ALGORITHM = "HS256"
ACCESS_TTL_MIN = 15               # short-lived access token
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads_chunks"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _patient_secret() -> str:
    secret = os.getenv("PATIENT_JWT_SECRET")
    if not secret or len(secret) < 32:
        raise HTTPException(500, "PATIENT_JWT_SECRET missing or too short (need 32+ chars).")
    return secret


def _issue_access_token(patient_id: int) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {"sub": str(patient_id), "role": "patient",
               "exp": now + datetime.timedelta(minutes=ACCESS_TTL_MIN)}
    return jwt.encode(payload, _patient_secret(), algorithm=ALGORITHM)


def current_patient_id(authorization: Optional[str] = Header(None)) -> int:
    """Decode the same patient JWT main.py issues. Returns the patient id."""
    bad = HTTPException(401, "Patient authentication required",
                        headers={"WWW-Authenticate": "Bearer"})
    if not authorization or not authorization.lower().startswith("bearer "):
        raise bad
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, _patient_secret(), algorithms=[ALGORITHM])
    except JWTError:
        raise bad
    if payload.get("role") != "patient":
        raise bad
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub.isdigit():
        raise bad
    return int(sub)


# ----------------------------------------------------------------------------
# Concept 5: device-token login  ->  short access token
# A phone offline for days keeps its long-lived device_token; it trades it for a
# fresh 15-minute access token whenever it comes online. A stolen phone is cut
# off by setting devices.revoked_at (its token then stops working here).
# ----------------------------------------------------------------------------
class DeviceLogin(BaseModel):
    device_token: str


@router.post("/api/device/login")
def device_login(body: DeviceLogin, db=Depends(get_db)):
    dev = db.query(Device).filter(Device.device_token == body.device_token).first()
    if dev is None or dev.revoked_at is not None:
        raise HTTPException(401, "Unknown or revoked device")
    dev.last_seen_at = datetime.datetime.utcnow()
    db.commit()
    return {"access_token": _issue_access_token(dev.patient_id),
            "token_type": "bearer", "patient_id": dev.patient_id,
            "expires_in": ACCESS_TTL_MIN * 60}


# ----------------------------------------------------------------------------
# Concept 4: idempotent sync
# Each session carries a session_uuid the phone generated once. The server
# inserts-or-ignores by that uuid, so a retried upload (after a 500, a timeout,
# a killed connection) never creates a duplicate. That uuid is the "receipt
# number": same receipt -> same row.
# ----------------------------------------------------------------------------
class RepIn(BaseModel):
    rep_number: int
    max_angle: Optional[float] = None
    min_angle: Optional[float] = None
    rom: Optional[float] = None
    duration_s: Optional[float] = None
    quality_score: Optional[float] = None
    faults: List[str] = Field(default_factory=list)


class SessionIn(BaseModel):
    session_uuid: str = Field(..., min_length=8, max_length=36)
    exercise: str
    total_reps: int
    mean_rom: Optional[float] = None
    mean_quality: Optional[float] = None
    date: str
    reps: List[RepIn] = Field(default_factory=list)


class SyncRequest(BaseModel):
    sessions: List[SessionIn] = Field(default_factory=list)


@router.post("/api/sync")
def sync(body: SyncRequest, patient_id: int = Depends(current_patient_id), db=Depends(get_db)):
    results = []
    for s in body.sessions:
        existing = db.query(SessionModel).filter(
            SessionModel.session_uuid == s.session_uuid).first()
        if existing is not None:
            results.append({"session_uuid": s.session_uuid, "status": "duplicate",
                            "session_id": existing.id})
            continue

        sess = SessionModel(
            patient_id=patient_id,            # trust the token, not the payload
            exercise=s.exercise, total_reps=s.total_reps,
            mean_rom=s.mean_rom, mean_quality=s.mean_quality, date=s.date,
            raw_json=json.dumps(s.model_dump()), session_uuid=s.session_uuid,
        )
        db.add(sess)
        try:
            db.flush()                        # fires the unique constraint now
        except IntegrityError:                # a concurrent upload won the race
            db.rollback()
            dup = db.query(SessionModel).filter(
                SessionModel.session_uuid == s.session_uuid).first()
            results.append({"session_uuid": s.session_uuid, "status": "duplicate",
                            "session_id": dup.id if dup else None})
            continue

        for rep in s.reps:
            r = Rep(session_id=sess.id, rep_number=rep.rep_number,
                    max_angle=rep.max_angle, min_angle=rep.min_angle, rom=rep.rom,
                    duration_s=rep.duration_s, quality_score=rep.quality_score)
            db.add(r)
            db.flush()
            for ftext in rep.faults:
                db.add(Fault(rep_id=r.id, fault_text=ftext))

        results.append({"session_uuid": s.session_uuid, "status": "stored",
                        "session_id": sess.id})

    db.commit()
    return {"results": results}


# ----------------------------------------------------------------------------
# Concept 3 (B): resumable chunk upload with sha256
# A big upload (e.g. a recorded clip) is sent in chunks. If the connection drops,
# the phone asks /status which chunks arrived and resumes from there. On
# /complete the server concatenates in order and checks the sha256, so a
# corrupted or partial upload is rejected instead of silently stored.
# ----------------------------------------------------------------------------
def _upload_path(upload_id: str) -> Path:
    safe = "".join(c for c in upload_id if c.isalnum() or c in "-_")[:64]
    if not safe:
        raise HTTPException(400, "Bad upload_id")
    p = UPLOAD_DIR / safe
    p.mkdir(exist_ok=True)
    return p


@router.put("/api/upload/{upload_id}/chunk/{index}")
async def put_chunk(upload_id: str, index: int, body: bytes = b"",
                    patient_id: int = Depends(current_patient_id)):
    if index < 0:
        raise HTTPException(400, "Bad chunk index")
    (_upload_path(upload_id) / f"{index:06d}.part").write_bytes(body)
    return {"upload_id": upload_id, "received_index": index, "bytes": len(body)}


@router.get("/api/upload/{upload_id}/status")
def upload_status(upload_id: str, patient_id: int = Depends(current_patient_id)):
    parts = sorted(int(p.stem) for p in _upload_path(upload_id).glob("*.part"))
    return {"upload_id": upload_id, "received": parts, "count": len(parts)}


class CompleteUpload(BaseModel):
    total_chunks: int
    sha256: str


@router.post("/api/upload/{upload_id}/complete")
def complete_upload(upload_id: str, body: CompleteUpload,
                    patient_id: int = Depends(current_patient_id)):
    folder = _upload_path(upload_id)
    have = sorted(int(p.stem) for p in folder.glob("*.part"))
    expected = list(range(body.total_chunks))
    missing = [i for i in expected if i not in have]
    if missing:
        raise HTTPException(409, {"error": "missing_chunks", "missing": missing})

    h = hashlib.sha256()
    out = folder / "assembled.bin"
    with out.open("wb") as w:
        for i in expected:
            data = (folder / f"{i:06d}.part").read_bytes()
            h.update(data)
            w.write(data)
    if h.hexdigest() != body.sha256.lower():
        out.unlink(missing_ok=True)
        raise HTTPException(400, {"error": "sha256_mismatch",
                                  "got": h.hexdigest(), "expected": body.sha256.lower()})
    return {"upload_id": upload_id, "bytes": out.stat().st_size, "sha256": h.hexdigest(),
            "status": "complete"}
