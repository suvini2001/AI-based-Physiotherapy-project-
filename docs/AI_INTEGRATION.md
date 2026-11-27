# Browser AI integration — research prototype

## Scope and protection

Implemented on `dev/prabash-physiotrack-web`, starting at `3135bef`, in an isolated checkout with separate `.venv-ai` and `.venv-web` environments and disposable SQLite databases. Existing PostgreSQL records were not accessed or changed. Research sources, models, templates, notebooks and config remain byte-for-byte unchanged (103 protected tracked files checked against the starting manifest). Commit this integration only on the development branch; pushing requires the owner's confirmation. No merge into another branch is part of this work.

The shareable [technical report and walkthrough](../output/pdf/PhysioTrack_Technical_Progress_and_Walkthrough.pdf) records the pre-commit snapshot from 27 November 2026. Its statements about uncommitted changes describe that snapshot; use Git for the current commit and push status.

## Quick isolated demo (no existing database required)

```sh
.venv-web/bin/python scripts/start_ai_demo.py
```

This creates a **separate** ignored SQLite database in `.local-ai-demo/`, one fictional patient with a three-repetition elbow prescription, and a fictional clinician. Random passwords are written only to the owner-readable `.local-ai-demo/access.txt` file. No existing `DATABASE_URL`, PostgreSQL records, `.env.local`, or accounts are used. Services run at frontend `http://localhost:5180` and API port 8002; Ctrl-C stops both. The local DB and credentials persist for the next demo run. Read the private file locally to log in; do not share or commit it.

## Start locally against your configured database

From the repository root:

```sh
# One-time setup: Python 3.11, separate from existing environments.
python3 scripts/setup_web_ai.py --python /opt/homebrew/bin/python3.11

# Diagnostic: runs the shipped model and MediaPipe on a blank frame, not your camera.
.venv-web/bin/python scripts/run_web_ai.py --check

# Existing private .env.local / DATABASE_URL and authentication settings are retained.
.venv-web/bin/python scripts/run_web_ai.py
```

If no private web configuration exists, first run `.venv-web/bin/python scripts/configure_web_demo.py` as described in `WEB_DEMO.md`. No credentials are embedded in these scripts. Start the frontend in another terminal with `cd dashboard && npm run dev`.

The launcher sets `AI_PYTHON` to the separate interpreter and starts **one** Uvicorn worker on localhost. Do not enable multiple Uvicorn workers: session coordination is in-memory. To disable real inference, start the normal API without `AI_PYTHON`; simulation remains available. On other systems, provide an installed Python 3.11 path to the setup command. The lockfile records the tested Apple Silicon environment, not a guarantee of cross-platform support.

## Demo

1. Register a fictional patient and prescribe elbow flexion or shoulder abduction from the clinician portal.
2. Reload the patient dashboard. **Open Real AI Session** becomes enabled after the runtime startup check.
3. Select left/right arm and explicitly agree to send camera frames to the configured AI server.
4. Start real analysis. Frames are letterboxed to 640×480 JPEG without mirroring. The preview is not a recording.
5. Observe READY → CALIB → COUNT. Calibration asks for comfortable movement through the prescribed range and return; it is bounded to 15 seconds after readiness. Failed calibration stops counting and cannot save results. Shoulder analysis requires a front view.
6. The target repetition count comes from a server-side snapshot of the latest prescription. The prescribed ROM is retained for clinician review; it does not override the team's calibrated FSM thresholds.
7. At completion, the camera stops. Choose **Finish and save results** to save the server's research measurements; save promptly (idle sessions expire in 60 seconds). Stop/discard or navigating away discards unsaved work. Completed reps can be saved early; incomplete motion is not flushed into an extra rep.
8. Clinicians can monitor the selected patient on `/live`, then review stored sessions and repetition faults. Research results are labelled separately from older, unverified records.

## Architecture

- Browser → authenticated FastAPI routes → private JSON-lines subprocess → existing MediaPipe, AnglePipeline, FSM and scorer → measurement response.
- One subprocess per active session isolates MediaPipe state, feature/filter history, FSM state and the scorer's module-global template.
- `backend/ai_worker.py` reads the existing `AnglePipeline`, resolver and two constants from the AST of `demo_cv.py`. It executes those original definitions unchanged with their dependencies. It does **not** execute the desktop main loop, audio initialisation or unauthenticated uploader. A changed/missing entry-point structure fails closed.
- The worker imports `pose/landmarks.py`, `exercise_state.py`, scoring, and configured models normally. Elbow uses the configured shipped ML model, with the existing warm-up fallback. Shoulder uses the existing geometric method and correct side-specific template.
- Measurements contain a current-frame tracking flag and angle source. Missing current landmarks do not expose an imputed angle as current. Prolonged tracking loss during counting stops the session to prevent a rep bridging a missing observation interval.
- Research outputs are not medically validated. The adapter does not retrain models, change joint-angle formulas, edit templates, or claim accuracy beyond the existing research evidence.

## API and data ownership

| Route | Access and behaviour |
| --- | --- |
| `GET /patient/me/ai-status` | Patient JWT; cached startup probe, real/simulation availability |
| `POST /patient/me/ai-sessions` | Patient JWT, side only; prescription and patient identity come from the server |
| `POST /patient/me/ai-sessions/{id}/frame` | Owner only; raw `image/jpeg`; no client timestamps, angles or counts |
| `POST /patient/me/ai-sessions/{id}/finish` | Owner only; saves cached worker output in one DB transaction; repeat calls return the same saved ID during retention |
| `DELETE /patient/me/ai-sessions/{id}` | Owner only; stops worker and discards unsaved data |
| `GET /api/patients/{id}/ai-live` | Clinician JWT; selected patient's recent measurement only, no video |

Unknown/non-owned session IDs both return 404. Extra start-body fields are rejected. Direct `/api/sessions` uploads remain disabled, including mock payloads. Legacy desktop launch and unauthenticated live/WebSocket routes remain disabled.

The existing DB schema is reused: Session, Rep, Fault. No schema migration is needed. `raw_json` stores the inference-session ID, prescription snapshot, arm, calibration/completion status, source counts, research/model/template SHA-256 values, and research summary. No JPEGs, video, or per-frame landmark histories are persisted. Legacy records are not rewritten or retroactively classified as verified.

A private ignored `.local-ai-cache/` directory reuses library/font caches between workers. A fresh install may take longer on its first launch; the UI allows time for that startup.

## Bounds and lifecycle

- Maximum 2 active inference processes, one per patient, and 64 retained session entries.
- Maximum 20 submitted FPS; one request at a time per session. Browser backpressure waits for a response before capturing again.
- Initial research filter sampling configuration: 20 Hz (8 Hz is invalid for the existing 6 Hz Butterworth cutoff). The adapter passes measured cadence to the existing geometric smoother. The research ML feature filter retains its initial sampling rate as designed; variable network/processing cadence still needs real-movement validation.
- Fixed 640×480 JPEG; maximum compressed body 256 KiB. Header dimensions/format are checked before OpenCV decoding.
- Model-start timeout 90 seconds; frame IO timeout 20 seconds; malformed images return 422; crashed/timed-out workers fail closed.
- Idle expiry 60 seconds; absolute session lifetime 15 minutes; daemon reaper every 5 seconds.
- Cleanup on stop, reset, navigation, logout, pagehide, token-expiry unmount, server shutdown, or TTL. A disconnected client may leave a worker until the idle timeout, but its camera is released locally.
- Worker environment excludes API signing keys and database credentials. Worker processes have no application database connection.
- Live clinician polling once per second; measurements older than 3 seconds are not displayed as current. Existing demo clinician accounts still share the roster; clinician assignment is not implemented.
- Single-process prototype: active inference is not resumable after API restart. Saved DB records persist; session-finalisation retry state does not survive a restart.

## Verification and limits

Run independently:

```sh
.venv-web/bin/python -m pytest backend/tests/test_web_api.py backend/tests/test_ai_api.py -q
.venv-ai/bin/python -m pytest backend/tests/test_ai_worker.py -q
RUN_AI_RUNTIME_TESTS=1 .venv-web/bin/python -m pytest backend/tests/test_ai_runtime.py -q
.venv-ai/bin/python -m pytest tests -q
cd dashboard
AI_PYTHON="$(pwd)/../.venv-ai/bin/python" npm run test:e2e
npm run build
npm run lint
```

The real runtime/browser tests on macOS need normal OS graphics access even for CPU inference; MediaPipe's graph initialisation cannot run in the restricted execution sandbox. They use blank/synthetic camera frames, not the physical camera.

Verified layers:

- Reverified in the original project checkout: 37 API, 4 worker unit, 3 real subprocess and 8 browser tests passed (52 total). Frontend production build passed; lint reports four existing warnings. The running isolated demo also passed authenticated availability, actual blank-frame inference, discard, unchanged empty fictional history, and frontend HTTP checks. The persistent cache and separate startup timeout resolve the observed fresh-install startup failure.
- 37 web API tests: original authentication plus session ownership, size/type/rate bounds, capacity, server-owned saving, idempotence, history, and patient-specific monitoring. Persistence tests use an explicitly named test worker double, never a production mock fallback.
- 4 adapter/research unit tests: exact definition loading without desktop side effects, actual shipped LightGBM warm-up, image bounds, and a synthetic-landmark sequence through real shoulder geometry/calibration/FSM/scoring.
- Real subprocess tests: elbow and shoulder startup, malformed-frame recovery, blank-frame calibration failure with zero reps, private environment, and crash cleanup.
- 8 browser tests: previous workflow plus real-worker frame transport, consent/no-pose/discard, completion/save UI contract fixture, and cleanup of delayed startup.
- Actual MediaPipe positive-pose inference and shipped LightGBM inference succeeded on the public MediaPipe `pose.jpg` sample downloaded temporarily from `https://storage.googleapis.com/mediapipe-assets/pose.jpg`. Static input produced zero repetitions. This sample is not stored in the repository.

The original research suite has **151 passing tests and one pre-existing failure**: `test_shipped_template_is_marked_synthetic` requires `n_people >= 3`, but the supplied `template.json` says `n_people = 1`. This exact assertion/template mismatch also exists at baseline commit `3135bef`. The research test and template were intentionally not changed. Scores must remain labelled as research outputs.

The installed MediaPipe 0.10.14 universal2 wheel runs on this Apple Silicon machine, but its internal WHEEL metadata declares x86_64, causing `pip check` to report an unsupported-platform warning. We did not edit package metadata to hide it. Real startup and image-inference tests provide runtime evidence; validate on each deployment machine.

A full human exercise session, end-to-end repetition accuracy under actual camera cadence, and deployment against the existing PostgreSQL database have **not** been validated by these automated tests. No real patient data was used. These are the remaining acceptance checks, not grounds to claim medical validation.
