"""Install an isolated inference environment without changing research environments."""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default=shutil.which('python3.11'))
    args = parser.parse_args()
    if not args.python:
        raise SystemExit('Python 3.11 is required. Install it, then rerun with --python /path/to/python3.11.')
    version = subprocess.check_output([args.python, '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'], text=True).strip()
    if version != '3.11':
        raise SystemExit('This tested runtime uses Python 3.11. Existing environments will not be changed.')
    env = ROOT / '.venv-ai'
    if not env.exists():
        subprocess.run([args.python, '-m', 'venv', str(env)], check=True)
    python = env / 'bin/python'
    installed_version = subprocess.check_output([str(python), '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'], text=True).strip()
    if installed_version != '3.11':
        raise SystemExit('Existing .venv-ai has a different Python version; no changes made.')
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'backend/requirements-ai-lock.txt')], check=True)
    print('Isolated AI environment ready. Next: .venv-web/bin/python scripts/run_web_ai.py --check')


if __name__ == '__main__':
    main()
