"""Start a completely separate fictional-data demo; never uses DATABASE_URL/.env.local.

Credentials are generated once and kept in a private ignored local file. Both
services bind only to localhost. Ctrl-C stops the demo without deleting its DB.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def private_write(path, data):
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as output:
        output.write(data)


def configure():
    from argon2 import PasswordHasher
    folder = ROOT / '.local-ai-demo'
    folder.mkdir(mode=0o700, exist_ok=True)
    config_path = folder / 'config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        clinician_password, patient_password = secrets.token_urlsafe(18), secrets.token_urlsafe(18)
        config = {'PATIENT_JWT_SECRET': secrets.token_urlsafe(48),
                  'CLINICIAN_JWT_SECRET': secrets.token_urlsafe(48),
                  'CLINICIAN_PASSWORD_HASH': PasswordHasher().hash(clinician_password),
                  'demo_patient_password_hash': PasswordHasher().hash(patient_password)}
        private_write(config_path, json.dumps(config))
        private_write(folder / 'access.txt',
                      'PRIVATE LOCAL DEMO — FICTIONAL DATA ONLY\n\n'
                      f'Clinician: http://localhost:5180/login\nID: DRDEMO\nPassword: {clinician_password}\n\n'
                      f'Patient: http://localhost:5180/patient\nID: PTDEMOAI\nPassword: {patient_password}\n\n'
                      'These credentials apply only to the isolated demo database. Do not commit or share this file.\n')
    # Force a new, dedicated DB location; ignore any inherited production/dev URL.
    env = {**os.environ, **{k: v for k, v in config.items() if not k.startswith('demo_')},
           'DATABASE_URL': f'sqlite:///{folder / "demo.sqlite"}',
           'CLINICIAN_ID': 'DRDEMO', 'CLINICIAN_NAME': 'Fictional Demo Clinician',
           'AI_PYTHON': str(ROOT / '.venv-ai/bin/python'),
           'CORS_ORIGINS': 'http://localhost:5180,http://127.0.0.1:5180', 'WEB_CONCURRENCY': '1'}
    os.environ['DATABASE_URL'] = env['DATABASE_URL']
    sys.path.insert(0, str(ROOT / 'backend'))
    from database import engine, SessionLocal
    from models import Base, Patient, Prescription
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        patient = db.query(Patient).filter(Patient.patient_id == 'PTDEMOAI').first()
        if patient is None:
            patient = Patient(name='Fictional AI Demo Patient', patient_id='PTDEMOAI',
                              password_hash=config['demo_patient_password_hash'], notes='Isolated research demo')
            db.add(patient); db.flush()
            db.add(Prescription(patient_id=patient.id, exercise='elbow_flexion', target_reps=3, target_rom=90))
            db.commit()
    engine.dispose()
    return folder, env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-only', action='store_true')
    args = parser.parse_args()
    folder, env = configure()
    print(f'Private login details: {folder / "access.txt"}', flush=True)
    if args.configure_only:
        return
    if not Path(env['AI_PYTHON']).is_file():
        raise SystemExit('Run scripts/setup_web_ai.py first.')
    if not (ROOT / 'dashboard/node_modules').is_dir():
        raise SystemExit('Install frontend dependencies with npm ci in dashboard first.')
    children = []
    try:
        children.append(subprocess.Popen([sys.executable, '-m', 'uvicorn', 'main:app', '--app-dir', str(ROOT / 'backend'),
                                          '--host', '127.0.0.1', '--port', '8002', '--workers', '1'], cwd=ROOT, env=env))
        children.append(subprocess.Popen(['npm', 'run', 'dev', '--', '--host', 'localhost', '--port', '5180', '--strictPort'],
                                          cwd=ROOT / 'dashboard', env={**{k: v for k, v in os.environ.items() if k not in ('DATABASE_URL', 'PATIENT_JWT_SECRET', 'CLINICIAN_JWT_SECRET', 'CLINICIAN_PASSWORD_HASH')}, 'VITE_API_URL': 'http://127.0.0.1:8002'}))
        print('Isolated AI demo: http://localhost:5180 — Ctrl-C to stop both services.', flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(.5)
        raise SystemExit('A demo service stopped. Check whether ports 8002/5180 are already occupied.')
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill(); child.wait()


if __name__ == '__main__':
    main()
