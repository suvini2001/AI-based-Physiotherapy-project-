from fastapi import FastAPI, WebSocket, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import Optional
import json
import random
import string
import subprocess
import sys
import os

from database import get_db, engine
from models import Base, Patient, Prescription, Session as SessionModel, Rep, Fault

Base.metadata.create_all(bind=engine)

SECRET_KEY = "physio-secret-key-change-in-production-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480

app = FastAPI(title="PhysioTrack API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

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
    return plain == stored

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_physio(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        physio_id: str = payload.get("sub")
        if physio_id is None or physio_id not in PHYSIO_ACCOUNTS:
            raise credentials_exception
        return PHYSIO_ACCOUNTS[physio_id]
    except JWTError:
        raise credentials_exception

@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    physio = PHYSIO_ACCOUNTS.get(form_data.username.upper())
    if not physio or not verify_password(form_data.password, physio["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect ID or password",
        )
    token = create_access_token(
        data={"sub": physio["id"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "physio": {"id": physio["id"], "name": physio["name"],
                   "specialty": physio["specialty"]},
    }

@app.get("/auth/me")
def get_me(current=Depends(get_current_physio)):
    return current

@app.get("/api/patients")
def list_patients(db: Session = Depends(get_db), _=Depends(get_current_physio)):
    patients = db.query(Patient).all()
    result = []
    for p in patients:
        last = db.query(SessionModel).filter(
            SessionModel.patient_id == p.id
        ).order_by(SessionModel.id.desc()).first()
        presc = db.query(Prescription).filter(
            Prescription.patient_id == p.id
        ).order_by(Prescription.id.desc()).first()
        result.append({
            "id": p.id, "name": p.name, "dob": p.dob, "notes": p.notes,
            "last_session": last.date if last else None,
            "last_rom": last.mean_rom if last else None,
            "total_sessions": db.query(SessionModel).filter(
                SessionModel.patient_id == p.id).count(),
            "prescription": {
                "exercise": presc.exercise,
                "target_reps": presc.target_reps,
                "target_rom": presc.target_rom,
            } if presc else None,
        })
    return result

@app.post("/api/patients")
def create_patient(data: dict, db: Session = Depends(get_db),
                   _=Depends(get_current_physio)):
    p = Patient(name=data["name"], dob=data.get("dob"), notes=data.get("notes"))
    db.add(p)
    db.commit()
    db.refresh(p)
    return {"id": p.id, "name": p.name}

@app.get("/api/patients/{patient_id}/sessions")
def get_sessions(patient_id: int, db: Session = Depends(get_db),
                 _=Depends(get_current_physio)):
    sessions = db.query(SessionModel).filter(
        SessionModel.patient_id == patient_id
    ).order_by(SessionModel.id.desc()).all()
    return [{"id": s.id, "exercise": s.exercise, "total_reps": s.total_reps,
             "mean_rom": s.mean_rom, "mean_quality": s.mean_quality,
             "date": s.date} for s in sessions]

@app.get("/api/sessions/{session_id}/reps")
def get_reps(session_id: int, db: Session = Depends(get_db),
             _=Depends(get_current_physio)):
    reps = db.query(Rep).filter(Rep.session_id == session_id).all()
    result = []
    for r in reps:
        faults = db.query(Fault).filter(Fault.rep_id == r.id).all()
        result.append({
            "rep_number": r.rep_number, "max_angle": r.max_angle,
            "min_angle": r.min_angle, "rom": r.rom,
            "duration_s": r.duration_s, "quality_score": r.quality_score,
            "faults": [f.fault_text for f in faults],
        })
    return result

@app.post("/api/sessions")
def upload_session(data: dict, db: Session = Depends(get_db)):
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
            db.add(Fault(rep_id=r.id, fault_text=fault))
    db.commit()
    return {"status": "ok", "session_id": s.id}

@app.post("/api/prescriptions")
def set_prescription(data: dict, db: Session = Depends(get_db),
                     _=Depends(get_current_physio)):
    p = Prescription(
        patient_id=data["patient_id"],
        exercise=data["exercise"],
        target_reps=data["target_reps"],
        target_rom=data.get("target_rom"),
    )
    db.add(p)
    db.commit()
    return {"status": "ok"}

active_ws: list[WebSocket] = []

@app.websocket("/ws/live/{patient_id}")
async def ws_live(ws: WebSocket, patient_id: int):
    await ws.accept()
    active_ws.append(ws)
    try:
        while True:
            await ws.receive_text()
    except:
        active_ws.remove(ws)

@app.post("/api/live_frame")
async def push_frame(data: dict):
    dead = []
    for ws in active_ws:
        try:
            await ws.send_json(data)
        except:
            dead.append(ws)
    for ws in dead:
        active_ws.remove(ws)
    return {"ok": True}

@app.post("/api/seed")
def seed(db: Session = Depends(get_db)):
    if db.query(Patient).count() > 0:
        return {"status": "already seeded"}
    patients = [
        Patient(name="Amara Silva", dob="1985-03-12", notes="Post-op elbow fracture"),
        Patient(name="Nimal Fernando", dob="1972-07-29", notes="Rotator cuff repair"),
        Patient(name="Sanduni Perera", dob="1990-11-05", notes="Frozen shoulder"),
    ]
    for p in patients:
        db.add(p)
    db.commit()
    return {"status": "seeded", "patients": 3}

def generate_patient_id(db):
    while True:
        pid = "PT" + "".join(random.choices(string.digits, k=4))
        if not db.query(Patient).filter(Patient.patient_id == pid).first():
            return pid

@app.post("/patient/register")
def patient_register(data: dict, db: Session = Depends(get_db)):
    name = data.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    pid = generate_patient_id(db)
    p = Patient(name=name, patient_id=pid, notes="Self-registered")
    db.add(p)
    db.commit()
    db.refresh(p)
    return {"patient_id": pid, "name": p.name, "id": p.id}

@app.post("/patient/login")
def patient_login(data: dict, db: Session = Depends(get_db)):
    pid = data.get("patient_id", "").strip().upper()
    name = data.get("name", "").strip().lower()
    p = db.query(Patient).filter(Patient.patient_id == pid).first()
    if not p:
        raise HTTPException(status_code=404, detail="Patient ID not found")
    if p.name.strip().lower() != name:
        raise HTTPException(status_code=401, detail="Name does not match")
    presc = db.query(Prescription).filter(
        Prescription.patient_id == p.id
    ).order_by(Prescription.id.desc()).first()
    return {
        "id": p.id,
        "patient_id": pid,
        "name": p.name,
        "exercise": presc.exercise if presc else "elbow_flexion",
        "target_reps": presc.target_reps if presc else 10,
    }

@app.get("/patient/{patient_id}/last-session")
def patient_last_session(patient_id: int, db: Session = Depends(get_db)):
    last = db.query(SessionModel).filter(
        SessionModel.patient_id == patient_id
    ).order_by(SessionModel.id.desc()).first()
    if not last:
        return {"total_reps": 0, "mean_rom": None, "date": None}
    return {
        "total_reps": last.total_reps,
        "mean_rom": last.mean_rom,
        "date": last.date,
    }

# ── Launch demo_cv.py as subprocess ───────────────────────────────────
active_process = {}   # patient_id → subprocess

@app.post("/api/launch-session")
def launch_session(data: dict, db: Session = Depends(get_db)):
    patient_id = data.get("patient_id")
    exercise = data.get("exercise", "elbow")
    side = data.get("side", "right")

    exercise_arg = "shoulder" if exercise == "shoulder_abduction" else "elbow"

    project_root = r"D:\3YP\AI-based-Physiotherapy-project-"
    script_path = os.path.join(project_root, "demo_cv.py")
    python_exe = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    log_path = os.path.join(project_root, "session_launch.log")

    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail=f"demo_cv.py not found at {script_path}")

    if not os.path.exists(python_exe):
        raise HTTPException(status_code=404, detail=f"Python not found at {python_exe}")

    if str(patient_id) in active_process:
        try:
            active_process[str(patient_id)].terminate()
        except:
            pass

    log_file = open(log_path, "w")

    print(f"Launching: {python_exe} {script_path} --exercise {exercise_arg} --side {side} --patient-id {patient_id}")

    proc = subprocess.Popen(
        [python_exe, script_path,
         "--exercise", exercise_arg,
         "--side", side,
         "--patient-id", str(patient_id)],
        cwd=project_root,
        stdout=log_file,
        stderr=log_file,
    )
    active_process[str(patient_id)] = proc
    print(f"Launched PID {proc.pid}")

    return {"status": "launched", "pid": proc.pid, "exercise": exercise_arg}
@app.post("/api/stop-session")
def stop_session(data: dict):
    patient_id = str(data.get("patient_id"))
    if patient_id in active_process:
        try:
            active_process[patient_id].terminate()
            del active_process[patient_id]
        except:
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
        return {"running": False, "exit_code": poll}
    return {"running": True, "pid": proc.pid}