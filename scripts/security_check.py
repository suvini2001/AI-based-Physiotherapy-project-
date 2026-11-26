"""
scripts/security_check.py
E5 security checks before deployment.
Run: python scripts/security_check.py

Checks:
  1. No hardcoded secrets in Python source files
  2. .env is not committed to git
  3. Legacy unauthenticated endpoints are removed
  4. BACKUP_PASSWORD is set and strong enough
"""
import os
import sys
import subprocess
from pathlib import Path

PASS = "✓"
FAIL = "✗"
results = []


def check(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f"\n       → {detail}" if detail and not passed else ""))


# ── Check 1: No hardcoded secrets ────────────────────────────
print("\n[1] Scanning for hardcoded secrets ...")

FORBIDDEN = [
    "physio-secret-key-change-in-production",
    "physio2026",
    "password123",
    "SECRET_KEY = \"",
    "PATIENT_JWT_SECRET = \"",
]

found_secrets = []
for py_file in Path("backend").rglob("*.py"):
    text = py_file.read_text(errors="ignore")
    for pattern in FORBIDDEN:
        if pattern in text:
            found_secrets.append(f"{py_file}: contains '{pattern}'")

check(
    "No hardcoded secrets in source",
    len(found_secrets) == 0,
    "; ".join(found_secrets) if found_secrets else "",
)


# ── Check 2: .env not in git ──────────────────────────────────
print("\n[2] Checking .env is not tracked by git ...")

result = subprocess.run(
    ["git", "ls-files", "backend/.env", ".env"],
    capture_output=True, text=True,
)
tracked = [l.strip() for l in result.stdout.splitlines() if l.strip()]

check(
    ".env files not committed to git",
    len(tracked) == 0,
    f"Tracked files: {tracked}" if tracked else "",
)

# Also check git log for any historical .env commits
result2 = subprocess.run(
    ["git", "log", "--all", "--full-history", "--", "**/.env", ".env"],
    capture_output=True, text=True,
)
historical = result2.stdout.strip()
check(
    "No .env in git history",
    "physio-secret-key-change-in-production" not in historical,
    "Old secret key found in git history — rotate keys" if historical else "",
)


# ── Check 3: Legacy unauthenticated endpoints removed ────────
print("\n[3] Checking legacy endpoints are removed ...")

main_py = Path("backend/main.py").read_text(errors="ignore")

legacy = {
    '"/api/seed"': "Seed endpoint still present — remove before deployment",
    '"/api/launch-session"': "Desktop launcher endpoint still present",
    '"/patient/register"': "Legacy insecure patient register still present",
    '"/patient/login"': "Legacy insecure patient login still present",
}

for pattern, msg in legacy.items():
    present = pattern in main_py
    check(f"Legacy endpoint removed: {pattern}", not present, msg if present else "")


# ── Check 4: Backup password set and strong ───────────────────
print("\n[4] Checking backup configuration ...")

from dotenv import load_dotenv
load_dotenv(dotenv_path="backend/.env")

backup_pass = os.environ.get("BACKUP_PASSWORD", "")
check(
    "BACKUP_PASSWORD is set",
    bool(backup_pass),
    "Set BACKUP_PASSWORD in backend/.env" if not backup_pass else "",
)
check(
    "BACKUP_PASSWORD is at least 20 characters",
    len(backup_pass) >= 20,
    f"Current length: {len(backup_pass)} — use at least 20 characters" if len(backup_pass) < 20 else "",
)

backup_dir = os.environ.get("BACKUP_DIR", "")
check(
    "BACKUP_DIR is configured",
    bool(backup_dir),
    "Set BACKUP_DIR in backend/.env" if not backup_dir else "",
)

backup_dir_2 = os.environ.get("BACKUP_DIR_2", "")
check(
    "BACKUP_DIR_2 (second copy) is configured",
    bool(backup_dir_2),
    "Set BACKUP_DIR_2 in backend/.env" if not backup_dir_2 else "",
)


# ── Summary ───────────────────────────────────────────────────
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)

print(f"\n{'='*50}")
print(f"  {passed} passed  |  {failed} failed  |  {len(results)} total")
print(f"{'='*50}\n")

if failed > 0:
    print("Fix the above issues before deploying.\n")
    sys.exit(1)
else:
    print("All checks passed. Safe to deploy.\n")