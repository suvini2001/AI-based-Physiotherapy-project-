"""Disposable API for browser tests. Never uses the developer's database."""
import os
from pathlib import Path
import secrets
import sys
import tempfile
from argon2 import PasswordHasher
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='physiotrack-browser-') as directory:
        os.environ['CORS_ORIGINS'] = 'http://localhost:5189'
        os.environ['DATABASE_URL'] = f'sqlite:///{directory}/test.sqlite'
        os.environ['PATIENT_JWT_SECRET'] = secrets.token_urlsafe(48)
        os.environ['CLINICIAN_JWT_SECRET'] = secrets.token_urlsafe(48)
        os.environ['CLINICIAN_ID'] = 'DRTEST'
        os.environ['CLINICIAN_PASSWORD_HASH'] = PasswordHasher().hash(os.environ['WEB_TEST_PASSWORD'])
        uvicorn.run('main:app', host='127.0.0.1', port=18001, log_level='warning')
