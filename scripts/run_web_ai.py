"""Run the web API with the separate AI interpreter; never rewrites .env.local."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Run real model/blank-frame diagnostic, no camera or DB')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    python = ROOT / '.venv-ai/bin/python'
    if not python.is_file():
        raise SystemExit('Run scripts/setup_web_ai.py first.')
    if args.check:
        cache = ROOT / '.local-ai-cache'
        cache.mkdir(mode=0o700, exist_ok=True)
        env = {**os.environ, 'MPLCONFIGDIR': str(cache), 'XDG_CACHE_HOME': str(cache)}
        subprocess.run([str(python), str(ROOT / 'backend/ai_worker.py'), '--probe'], cwd=ROOT, env=env, check=True)
        return
    web = ROOT / '.venv-web/bin/python'
    if not web.is_file():
        raise SystemExit('Set up the web environment with backend/requirements-web.txt first.')
    if not (ROOT / '.env.local').is_file() and not os.getenv('DATABASE_URL'):
        raise SystemExit('Configure the local database and authentication using scripts/configure_web_demo.py first.')
    env = {**os.environ, 'AI_PYTHON': str(python), 'WEB_CONCURRENCY': '1'}
    command = [str(web), '-m', 'uvicorn', 'main:app', '--app-dir', str(ROOT / 'backend'),
               '--host', '127.0.0.1', '--port', str(args.port), '--workers', '1']
    if (ROOT / '.env.local').is_file():
        command += ['--env-file', str(ROOT / '.env.local')]
    os.chdir(ROOT)
    os.execve(str(web), command, env)


if __name__ == '__main__':
    main()
