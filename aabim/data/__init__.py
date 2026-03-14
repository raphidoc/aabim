"""aabim.data — Spectral and atmospheric data access.

Subpackages
-----------
atmospheric   Atmospheric LUTs (AerLUT, GasLUT).
"""

from .atmospheric import AerLUT, GasLUT

__all__ = ["AerLUT", "GasLUT"]
