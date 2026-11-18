"""Future per-patient inference boundary. No research imports or camera/network IO.

A real implementation must own its extractor, AnglePipeline and FSM, accept
bounded frames and close resources. Transport stays disabled until fixture tests
verify real outputs and patient isolation.
"""
from dataclasses import dataclass
from typing import Protocol


class AIUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameMeasurement:
    timestamp: float
    joint_angle: float | None
    rep_count: int
    movement_state: str
    source: str = "research_pipeline"


class InferenceSession(Protocol):
    def process_frame(self, frame: bytes, timestamp: float) -> FrameMeasurement: ...
    def close(self) -> None: ...


class UnavailableAdapter:
    def status(self):
        return {
            "available": False,
            "mode": "simulation_only",
            "reason": "Research dependencies and isolated frame inference are not verified in the web environment",
            "clinical_uploads_enabled": False,
        }

    def open_session(self, *, patient_id: int, exercise: str) -> InferenceSession:
        raise AIUnavailable(self.status()["reason"])


adapter = UnavailableAdapter()
