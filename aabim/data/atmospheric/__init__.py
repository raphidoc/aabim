"""aabim.data.atmospheric — Atmospheric look-up tables.

Classes
-------
BaseLUT   Abstract base; subclass to add new atmospheric LUT types.
AerLUT    Aerosol scattering / transmittance LUT.
GasLUT    Gas transmittance LUT.
"""

from .atmospheric import BaseLUT, AerLUT, GasLUT

__all__ = ["BaseLUT", "AerLUT", "GasLUT"]
