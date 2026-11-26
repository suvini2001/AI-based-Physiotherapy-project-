"""
scripts/backup.py
Nightly encrypted backup of the PostgreSQL database.
Run this script on a schedule (Windows Task Scheduler).

Usage:
  python scripts/backup.py backup   -- create a backup
  python scripts/backup.py restore <file>  -- restore from a backup file
  python scripts/backup.py list     -- list available backups
"""
import os
import sys
import subprocess
import datetime
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path="backend/.env")

# ── Config ────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")
BACKUP_DIR   = Path(os.environ.get("BACKUP_DIR", "backups"))
# A second copy goes here (different drive ideally)
BACKUP_DIR_2 = Path(os.environ.get("BACKUP_DIR_2", "backups/copy2"))
BACKUP_PASS  = os.environ.get("BACKUP_PASSWORD", "")


def _parse_pg_url(url: str) -> dict:
    """Parse postgresql://user:pass@host:port/dbname into parts."""
    # strip scheme
    rest = url.replace("postgresql://", "").replace("postgres://", "")
    user_pass, host_db = rest.split("@", 1)
    user, password = user_pass.split(":", 1) if ":" in user_pass else (user_pass, "")
    host_port, dbname = host_db.split("/", 1)
    host, port = host_port.split(":", 1) if ":" in host_port else (host_port, "5432")
    return {"user": user, "password": password,
            "host": host, "port": port, "dbname": dbname}


def _checksum(path: Path) -> str:
    """SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_backup():
    """Create an encrypted backup and a second copy."""
    if not BACKUP_PASS:
        print("[backup] ERROR: BACKUP_PASSWORD not set in .env")
        sys.exit(1)

    if not DATABASE_URL:
        print("[backup] ERROR: DATABASE_URL not set in .env")
        sys.exit(1)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR_2.mkdir(parents=True, exist_ok=True)

    db = _parse_pg_url(DATABASE_URL)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sql_file  = BACKUP_DIR / f"physio_{stamp}.sql"
    enc_file  = BACKUP_DIR / f"physio_{stamp}.sql.enc"
    meta_file = BACKUP_DIR / f"physio_{stamp}.meta.json"

    # 1. pg_dump
    env = os.environ.copy()
    env["PGPASSWORD"] = db["password"]

    print(f"[backup] Dumping database {db['dbname']} ...")
    result = subprocess.run(
        [
            "pg_dump",
            "-h", db["host"],
            "-p", db["port"],
            "-U", db["user"],
            "-F", "p",   # plain SQL
            "-f", str(sql_file),
            db["dbname"],
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[backup] pg_dump failed:\n{result.stderr}")
        sys.exit(1)

    sql_size = sql_file.stat().st_size
    print(f"[backup] Dump complete ({sql_size:,} bytes)")

    # 2. Encrypt with openssl AES-256-CBC
    print("[backup] Encrypting ...")
    result = subprocess.run(
        [
            "openssl", "enc", "-aes-256-cbc",
            "-pbkdf2", "-iter", "100000",
            "-pass", f"pass:{BACKUP_PASS}",
            "-in",  str(sql_file),
            "-out", str(enc_file),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[backup] Encryption failed:\n{result.stderr}")
        sql_file.unlink(missing_ok=True)
        sys.exit(1)

    # 3. Remove plain SQL (keep only encrypted)
    sql_file.unlink()

    # 4. Checksum
    checksum = _checksum(enc_file)

    # 5. Write metadata
    meta = {
        "timestamp": stamp,
        "db": db["dbname"],
        "host": db["host"],
        "file": enc_file.name,
        "size_bytes": enc_file.stat().st_size,
        "sha256": checksum,
    }
    meta_file.write_text(json.dumps(meta, indent=2))

    # 6. Second copy
    import shutil
    enc_copy  = BACKUP_DIR_2 / enc_file.name
    meta_copy = BACKUP_DIR_2 / meta_file.name
    shutil.copy2(enc_file, enc_copy)
    shutil.copy2(meta_file, meta_copy)

    print(f"[backup] Done.")
    print(f"  Primary : {enc_file}")
    print(f"  Copy    : {enc_copy}")
    print(f"  SHA-256 : {checksum}")


def cmd_restore(enc_path: str):
    """Decrypt and restore a backup file."""
    if not BACKUP_PASS:
        print("[restore] ERROR: BACKUP_PASSWORD not set in .env")
        sys.exit(1)

    enc_file = Path(enc_path)
    if not enc_file.exists():
        print(f"[restore] ERROR: file not found: {enc_file}")
        sys.exit(1)

    # Verify checksum against metadata if available
    meta_name = enc_file.name.replace(".sql.enc", ".meta.json")
    meta_file = enc_file.parent / meta_name
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        actual = _checksum(enc_file)
        if actual != meta["sha256"]:
            print(f"[restore] ERROR: checksum mismatch — file may be corrupted")
            print(f"  Expected : {meta['sha256']}")
            print(f"  Actual   : {actual}")
            sys.exit(1)
        print(f"[restore] Checksum OK")

    # Decrypt
    sql_file = enc_file.with_suffix("")   # remove .enc
    print(f"[restore] Decrypting to {sql_file} ...")
    result = subprocess.run(
        [
            "openssl", "enc", "-d", "-aes-256-cbc",
            "-pbkdf2", "-iter", "100000",
            "-pass", f"pass:{BACKUP_PASS}",
            "-in",  str(enc_file),
            "-out", str(sql_file),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[restore] Decryption failed:\n{result.stderr}")
        sys.exit(1)

    db = _parse_pg_url(DATABASE_URL)
    env = os.environ.copy()
    env["PGPASSWORD"] = db["password"]

    # Restore
    print(f"[restore] Restoring to {db['dbname']} ...")
    result = subprocess.run(
        [
            "psql",
            "-h", db["host"],
            "-p", db["port"],
            "-U", db["user"],
            "-d", db["dbname"],
            "-f", str(sql_file),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    # Remove decrypted SQL
    sql_file.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"[restore] psql failed:\n{result.stderr}")
        sys.exit(1)

    print(f"[restore] Done — database restored from {enc_file.name}")


def cmd_list():
    """List available backups."""
    if not BACKUP_DIR.exists():
        print("[backup] No backups found.")
        return
    files = sorted(BACKUP_DIR.glob("*.meta.json"), reverse=True)
    if not files:
        print("[backup] No backups found.")
        return
    print(f"{'Timestamp':<20} {'Size':>12}  {'File'}")
    print("-" * 60)
    for f in files:
        meta = json.loads(f.read_text())
        size_mb = meta["size_bytes"] / 1024 / 1024
        print(f"{meta['timestamp']:<20} {size_mb:>10.2f}MB  {meta['file']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backup"
    if cmd == "backup":
        cmd_backup()
    elif cmd == "restore":
        if len(sys.argv) < 3:
            print("Usage: python scripts/backup.py restore <file>")
            sys.exit(1)
        cmd_restore(sys.argv[2])
    elif cmd == "list":
        cmd_list()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)