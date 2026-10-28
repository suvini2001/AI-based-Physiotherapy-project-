# camera.py
"""
Root-level module proxy redirecting to capture.camera.
Enables 'from camera import FrameSource, FpsCounter' to resolve cleanly in IDEs and legacy scripts.
"""
from capture.camera import FrameSource, FpsCounter

__all__ = ["FrameSource", "FpsCounter"]
