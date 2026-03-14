"""aabim.image — Hyperspectral image data structure and processing mixins.

Classes
-------
Image   Base image class; extended by sensor-specific converters.
"""

from .image import Image

__all__ = ["Image"]
