# landmarks.py
"""
Root-level module proxy redirecting to pose.landmarks.
Enables 'from landmarks import LandmarkExtractor' to resolve cleanly in IDEs and legacy scripts.
"""
from pose.landmarks import LandmarkExtractor

__all__ = ["LandmarkExtractor"]
