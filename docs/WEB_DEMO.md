# PhysioTrack local web prototype

Use branch `dev/prabash-physiotrack-web`. Fictional data only. This is a development prototype, not medically validated or production-ready. No research pipeline files are changed.

## Local setup (macOS)

From the repository root:

```sh
.venv-web/bin/python -m pip install -r backend/requirements-web.txt
.venv-web/bin/python scripts/configure_web_demo.py
.venv-web/bin/python -m uvicorn main:app --app-dir backend --env-file .env.local --host 127.0.0.1 --port 8000
```

The configuration helper prompts privately for the existing PostgreSQL development URL (database `physiotrack_prabash_dev`) and a new clinician password. It writes an ignored, owner-readable `.env.local` containing distinct random signing keys and an Argon2id password hash. It never prints passwords or overwrites existing configuration. If `DATABASE_URL` is already exported it uses that value. Existing environment variables take precedence over Uvicorn's env file: avoid stale exported authentication variables.

The clinician ID is `DR001`; use the password you chose. There are no built-in passwords. Missing signing keys or clinician account configuration disable authentication with HTTP 503. Keep `.env.local` private. Restart the API after configuration changes; previous tokens signed with other keys become invalid.

The existing database must already contain `patients.password_hash`, as in commit a3b98a5. Table creation does not migrate existing schemas. Existing records and NULL hashes are preserved. Old patients such as PT7585 cannot use name-only login; register a new fictional account for the demo. No automatic password reset or migration is performed.

In a second terminal:

```sh
cd dashboard
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Open http://localhost:5173. The default API URL is http://localhost:8000. Override with `VITE_API_URL` at frontend startup/build if needed; the API CORS list permits local development origins on 5173 and 5174. Camera access requires localhost or HTTPS.

## Exact demo flow

1. Open `/patient`, choose **Register**, and enter a fictional name and a unique password of at least 12 characters. Record the generated patient ID. Continue to the patient dashboard; it should say it is awaiting a prescription.
2. In a second tab open `/login`. Sign in as the configured clinician. Open **Patients**, select the fictional patient, then **Prescription**. Choose Elbow Flexion or Shoulder Abduction and set a small target such as 3. Save.
3. Return to the patient dashboard and reload. The prescription is fetched from `/patient/me`, not from browser-stored profile information. Open **Browser Exercise Demo**.
4. Click **Start Camera**, allow camera access, and verify the local preview. Click **Start Simulation**. Counts and angles are explicitly simulated, independent of camera movement. Three simulated repetitions complete in approximately 12 seconds.
5. Test **Stop Simulation**, restart, completion, and **Reset Simulation**. Test **Stop Camera** and starting it again. Use **Patient dashboard** to navigate back; the camera and timer are released when leaving the exercise page.
6. Sign out from the dashboard or exercise page. Opening the protected URL again must return to login. Log in with the recorded ID and password. Session history must remain unchanged after simulation.

Patient tokens are held in sessionStorage; no cached patient profile is trusted. Patient routes verify the token with the server on navigation, redirect on HTTP 401, and unmount at token expiry. Logout removes local credentials but does not revoke an already copied JWT before its expiration. Clinician tokens remain in localStorage for this prototype. All configured clinicians share the demo roster; there is no clinician assignment model.

## Disabled legacy paths

- `/patient/login` and `/patient/register`: HTTP 410; use password-based endpoints.
- Patient history: authenticated `/patient/me/last-session`; the compatibility ID route checks ownership.
- Session uploads, desktop launch/stop, and live frame writes: authentication required, then HTTP 503. There is no accepted client payload that saves simulated measurements.
- Live websocket: closed with policy violation, no shared cross-patient broadcast.
- Seed route: clinician authentication required, then HTTP 410; register fictional accounts normally.

The unsafe desktop launcher is unavailable instead of starting a process with an unauthenticated uploader. Existing stored clinical session read views remain available to authenticated clinicians.

## AI integration boundary and blocker

Read-only inspection found:

- `pose/landmarks.py`: `LandmarkExtractor.process_with_world(frame_bgr)` returns normalized and world landmark arrays, each `(33, 4)`; `close()` releases the extractor.
- `demo_cv.py`: `AnglePipeline.step(norm, world)` returns cleaned landmarks and an angle or `None`. Pipeline history is stateful and must be isolated per session.
- The main desktop loop also controls capture, timing/calibration, FSM, scoring, audio, overlays, and session uploads. Importing it imports `audio_feedback`, which initializes audio at module import. Scoring has shared module state.
- `session_packager.py` uploads without authentication; live frames carry no patient ID. It cannot safely feed the newly protected endpoints unchanged.
- `.venv-web` is Python 3.13. The research requirements pin MediaPipe 0.10.14 and explicitly require Python 3.10–3.12. Research dependencies are absent from the web environment.

`backend/ai_adapter.py` defines a per-session interface and an unavailable adapter. `/patient/me/ai-status` honestly reports simulation-only mode. It does not import research code, accept frames, invent measurements, or write results. A future worker in a compatible research environment must own an extractor, pipeline and FSM per patient, with bounded frame sizes, request rate limits, authentication, cleanup, calibrated timestamps and real fixture tests. No real browser AI integration is claimed.

## Verification

```sh
.venv-web/bin/python -m pytest backend/tests -q
cd dashboard
npm run build
npm run lint
npm run test:e2e
```

API tests generate temporary credentials and use isolated in-memory SQLite, never the existing PostgreSQL database. They cover registration, login, hashing, latest prescriptions, ownership, role separation, token expiry, disabled endpoints, and rejected simulation uploads. Browser tests start an isolated API on 18001 and frontend on 5189, generate temporary clinician credentials, and use headless Chrome with synthetic camera input. They verify real API-backed registration and prescribing, latest prescription display, simulation stop/reset/completion, zero uploads, invalid tokens, camera denial, pending camera cancellation, navigation/logout cleanup and token-expiry cleanup. They require Chrome at its macOS default path (or set `CHROME_PATH`) and permission to bind localhost ports. They do not touch your existing database or camera.

Verified in this checkout: **30 API tests passed, 5 browser tests passed, production build passed, lint exited successfully with 4 warnings**. The build warns about a large JavaScript chunk. Lint warnings concern fast refresh exports in the auth contexts and existing Dashboard/PatientDetail patterns. PostgreSQL deployment and physical camera behavior still require the local demo smoke test above.

The repository's original `pytest` suite requires the research environment. In `.venv-web`, collection currently fails on missing NumPy and pandas. Do not change research dependencies or algorithms merely to make that environment match the web prototype.

Existing secrets in old Git history are not erased by this change. The known hardcoded signing key is no longer accepted; configure fresh local values. No commit or push is performed as part of this handoff.

## Remaining dependency findings

`npm audit` reported 12 high-severity findings in development/build dependencies, propagated from the picomatch ReDoS advisory GHSA-fc5q-39xc-vjpf. Its proposed remediation includes major changes to Vite/Tailwind and is not applied in this targeted prototype update. Keep the development server local and review the build-tool dependency upgrade separately. This is another reason not to deploy this checkout as production-ready.
