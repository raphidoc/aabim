"""aabim.inversion — Bayesian inversion of hyperspectral imagery.

Public API
----------
.. code-block:: python

    from aabim.inversion import (
        InversionData,
        AquaModel,
        AtmAquaModel,
        FelzenszwalbSegmenter,
        SLICSegmenter,
        NoSegmenter,
    )
"""
from aabim.inversion.segment import (
    Segmenter,
    NoSegmenter,
    FelzenszwalbSegmenter,
    SLICSegmenter,
)
from aabim.inversion.data   import InversionData
from aabim.inversion.model  import AquaModel, AtmAquaModel
from aabim.inversion.result import InversionResult

__all__ = [
    "Segmenter",
    "NoSegmenter",
    "FelzenszwalbSegmenter",
    "SLICSegmenter",
    "InversionData",
    "AquaModel",
    "AtmAquaModel",
    "InversionResult",
]
