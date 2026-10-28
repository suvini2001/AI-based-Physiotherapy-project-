# gap_fill.py
"""
Root-level module proxy redirecting to vision.gap_fill.
Enables 'from gap_fill import fill_gaps' to resolve cleanly in IDEs and legacy scripts.
"""
from vision.gap_fill import fill_gaps

__all__ = ["fill_gaps"]
