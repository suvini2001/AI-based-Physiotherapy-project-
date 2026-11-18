"""Create private local demo configuration without printing credentials."""
import getpass
import os
from pathlib import Path
import secrets

from argon2 import PasswordHasher


def main():
    path = Path(__file__).resolve().parents[1] / '.env.local'
    if path.exists():
        raise SystemExit('.env.local already exists; edit it locally rather than overwriting it.')
    database_url = os.getenv('DATABASE_URL') or getpass.getpass('PostgreSQL DATABASE_URL (hidden): ')
    if not database_url.startswith(('postgresql://', 'postgresql+psycopg2://')):
        raise SystemExit('Use the existing PostgreSQL development database URL.')
    password = getpass.getpass('Choose a local clinician demo password (12+ characters): ')
    if len(password) < 12 or len(password) > 128:
        raise SystemExit('Password must contain 12–128 characters.')
    if password != getpass.getpass('Confirm password: '):
        raise SystemExit('Passwords do not match.')
    values = {
        'DATABASE_URL': database_url,
        'PATIENT_JWT_SECRET': secrets.token_urlsafe(48),
        'CLINICIAN_JWT_SECRET': secrets.token_urlsafe(48),
        'CLINICIAN_ID': 'DR001',
        'CLINICIAN_NAME': 'Fictional Demo Clinician',
        'CLINICIAN_PASSWORD_HASH': PasswordHasher().hash(password),
    }
    def quote(value):
        return "'" + value.replace('\\', '\\\\').replace("'", "\\'") + "'"
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as output:
        for key, value in values.items():
            output.write(f'{key}={quote(value)}\n')
    print('Created private .env.local. Clinician ID: DR001. Use the password you chose.')


if __name__ == '__main__':
    main()
