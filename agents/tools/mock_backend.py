
import os
import json
import hmac
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from jsonschema import Draft202012Validator, ValidationError

app = FastAPI(title="PhysioTrack Mock Backend")

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "out"
SCHEMA_PATH = (
    BASE_DIR / "contract" / "agent_insights.v1.schema.json"
)

# Create output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the expected JSON schema
with SCHEMA_PATH.open("r", encoding="utf-8") as f:
    SCHEMA = json.load(f)

Draft202012Validator.check_schema(SCHEMA)
validator = Draft202012Validator(SCHEMA)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/sessions/{session_id}/insights")
def receive_insights(
    session_id: int,
    payload: dict,
    x_agent_key: str | None = Header(default=None)
):
    # 1. Verify API key
    expected_key = os.getenv("AGENT_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=500,
            detail="AGENT_KEY is not configured"
        )

    if not x_agent_key or not hmac.compare_digest(
        x_agent_key, expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    # 2. Validate JSON against output contract
    try:
        validator.validate(payload)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid insights JSON: {e.message}"
        )

    # 3. Verify session ID matches URL
    if payload.get("session_id") != session_id:
        raise HTTPException(
            status_code=422,
            detail="Session ID mismatch"
        )

    # 4. Validate request ID
    try:
        request_id = str(uuid.UUID(payload["request_id"]))
    except (KeyError, ValueError, TypeError, AttributeError):
        raise HTTPException(
            status_code=422,
            detail="Invalid or missing request_id"
        )

    # 5. Save the validated AI insights
    output_file = (
        OUT_DIR / f"{session_id}_{request_id}.json"
    )

    # Prevent duplicate saves
    if output_file.exists():
        return {
            "status": "already_saved",
            "session_id": session_id,
            "request_id": request_id
        }

    try:
        with output_file.open("x", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except FileExistsError:
        return {
            "status": "already_saved",
            "session_id": session_id,
            "request_id": request_id
        }

    return {
        "status": "saved",
        "session_id": session_id,
        "request_id": request_id
    }
