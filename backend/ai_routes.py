"""Authenticated browser inference. Persistence only accepts private worker output."""
import asyncio
from datetime import datetime, timezone
import json
from typing import Literal

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ai_adapter import adapter, AIError, MAX_FRAME_BYTES
from models import Patient, Session as SessionModel, Rep, Fault


class StartAI(BaseModel):
    model_config = ConfigDict(extra='forbid')
    side: Literal['left', 'right'] = 'right'


def install_ai_routes(app, current_patient, current_physio, get_db, latest_prescription):
    @app.exception_handler(AIError)
    async def ai_error_handler(request, error):
        headers = {'Retry-After': '1'} if error.status == 429 else None
        return JSONResponse(status_code=error.status, content={'detail': error.detail}, headers=headers)

    @app.post('/patient/me/ai-sessions', status_code=201)
    def start_ai(data: StartAI, patient: Patient = Depends(current_patient), db: Session = Depends(get_db)):
        prescription = latest_prescription(db, patient.id)
        if prescription is None:
            raise HTTPException(409, 'A clinician prescription is required')
        if prescription.exercise not in ('elbow_flexion', 'shoulder_abduction') or not isinstance(prescription.target_reps, int) or not 1 <= prescription.target_reps <= 50:
            raise HTTPException(409, 'Ask your clinician to update this prescription')
        return adapter.start(patient.id, dict(id=prescription.id, exercise=prescription.exercise,
                                             target_reps=prescription.target_reps, target_rom=prescription.target_rom), data.side)

    @app.post('/patient/me/ai-sessions/{session_id}/frame')
    async def frame(session_id: str, request: Request, patient: Patient = Depends(current_patient)):
        # Hold ownership lock while reading: at most one bounded image body per session.
        with adapter.owned(session_id, patient.id) as entry:
            if request.headers.get('content-type', '').split(';')[0] != 'image/jpeg':
                raise HTTPException(415, 'Content-Type must be image/jpeg')
            raw = bytearray()
            try:
                async with asyncio.timeout(5):
                    async for chunk in request.stream():
                        raw.extend(chunk)
                        if len(raw) > MAX_FRAME_BYTES:
                            raise HTTPException(413, 'Frame exceeds 256 KiB')
            except TimeoutError:
                raise HTTPException(408, 'Frame upload timed out')
            return await run_in_threadpool(adapter.frame_locked, entry, bytes(raw))

    @app.delete('/patient/me/ai-sessions/{session_id}')
    def discard(session_id: str, patient: Patient = Depends(current_patient)):
        return adapter.discard(session_id, patient.id)

    @app.post('/patient/me/ai-sessions/{session_id}/finish')
    def finish(session_id: str, patient: Patient = Depends(current_patient), db: Session = Depends(get_db)):
        with adapter.owned(session_id, patient.id, wait=True) as entry:
            if entry.saved_id is not None:
                return {'saved': True, 'session_id': entry.saved_id, 'total_reps': entry.summary['total_reps']}
            if entry.summary is None:
                if entry.worker.closed:
                    raise HTTPException(409, 'Session was stopped without results')
                entry.summary = entry.worker.call({'op': 'finish'})
                entry.worker.close()
            summary = entry.summary
            if not summary['calibrated'] or not summary['reps']:
                return {'saved': False, 'reason': 'No calibrated complete repetitions to save', 'total_reps': 0}
            # This data originates only from our private process, never from the HTTP body.
            if summary['source']['pipeline'] != 'research_pipeline' or summary['source']['is_simulation']:
                raise HTTPException(409, 'Untrusted inference result')
            payload = {**summary, 'prescription': entry.prescription, 'inference_session_id': entry.id}
            saved = SessionModel(patient_id=patient.id, exercise=entry.prescription['exercise'],
                                 total_reps=summary['total_reps'], mean_rom=summary['mean_rom'],
                                 mean_quality=summary['mean_quality'], date=datetime.now(timezone.utc).isoformat(),
                                 raw_json=json.dumps(payload, allow_nan=False))
            try:
                db.add(saved)
                db.flush()
                for measurement in summary['reps']:
                    rep = Rep(session_id=saved.id, **{key: measurement.get(key) for key in
                              ('rep_number', 'max_angle', 'min_angle', 'rom', 'duration_s', 'quality_score')})
                    db.add(rep)
                    db.flush()
                    for fault in measurement.get('faults', []):
                        db.add(Fault(rep_id=rep.id, fault_text=fault))
                db.commit()
            except Exception:
                db.rollback()
                raise
            entry.saved_id = saved.id
            return {'saved': True, 'session_id': saved.id, 'total_reps': summary['total_reps']}

    @app.get('/api/patients/{patient_id}/ai-live')
    def live(patient_id: int, _=Depends(current_physio)):
        return adapter.live(patient_id)
