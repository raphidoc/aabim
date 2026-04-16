"""Segmentation strategies for the inversion pipeline.

Design
------
All segmenters implement the :class:`Segmenter` ABC whose single method
``segment(image)`` returns a 2-D integer label array (n_rows, n_cols).
Pixels labelled ``-1`` are excluded (non-water or masked).

Built-in segmenters
-------------------
:class:`NoSegmenter`
    One label per water pixel — full pixel-wise inversion.
:class:`FelzenszwalbSegmenter`
    Graph-based segmentation (Felzenszwalb & Huttenlocher 2004) via
    ``scikit-image``.
:class:`SLICSegmenter`
    Simple Linear Iterative Clustering superpixels via ``scikit-image``.

Extension
---------
Subclass :class:`Segmenter` and implement ``segment()``.  No other
changes are required; pass an instance to
:func:`aabim.inversion.data.InversionData.from_image`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Segmenter(ABC):
    """Abstract base class for image segmenters."""

    @abstractmethod
    def segment(
        self,
        rgb: np.ndarray,
        water_mask: np.ndarray,
    ) -> np.ndarray:
        """Return an integer label array of shape (n_rows, n_cols).

        Parameters
        ----------
        rgb : np.ndarray, shape (n_rows, n_cols, 3)
            Three-band composite used for spatial segmentation (e.g. red,
            green, NIR bands of the L2R image normalised to [0, 1]).
        water_mask : np.ndarray, shape (n_rows, n_cols), dtype bool
            ``True`` for water pixels.  Non-water pixels are set to ``-1``
            in the returned label array.

        Returns
        -------
        np.ndarray, shape (n_rows, n_cols), dtype int
            Segment labels ≥ 0 for water pixels, ``-1`` elsewhere.
        """


class NoSegmenter(Segmenter):
    """One segment per water pixel (pixel-wise inversion).

    Each water pixel receives a unique integer label so that every pixel
    is inverted independently.  Recommended with ``method="optimize"``
    (L-BFGS) for tractable runtime on large images.
    """

    def segment(
        self,
        rgb: np.ndarray,
        water_mask: np.ndarray,
    ) -> np.ndarray:
        labels = np.full(water_mask.shape, -1, dtype=np.int32)
        water_idx = np.argwhere(water_mask)
        for i, (r, c) in enumerate(water_idx):
            labels[r, c] = i
        return labels


class FelzenszwalbSegmenter(Segmenter):
    """Graph-based segmentation (Felzenszwalb & Huttenlocher 2004).

    Requires ``scikit-image``.

    Parameters
    ----------
    scale : float
        Free parameter, larger values give larger segments.
    sigma : float
        Width of the Gaussian pre-smoothing kernel.
    min_size : int
        Minimum component size (pixels).
    """

    def __init__(
        self,
        scale: float = 50.0,
        sigma: float = 0.8,
        min_size: int = 20,
    ) -> None:
        self.scale    = scale
        self.sigma    = sigma
        self.min_size = min_size

    def segment(
        self,
        rgb: np.ndarray,
        water_mask: np.ndarray,
    ) -> np.ndarray:
        try:
            from skimage.segmentation import felzenszwalb
        except ImportError as exc:
            raise ImportError(
                "FelzenszwalbSegmenter requires scikit-image: "
                "conda install -c conda-forge scikit-image"
            ) from exc

        raw = felzenszwalb(
            rgb,
            scale=self.scale,
            sigma=self.sigma,
            min_size=self.min_size,
        )
        labels = np.where(water_mask, raw, -1).astype(np.int32)
        # Re-label contiguously from 0 to keep label space compact
        unique = np.unique(labels[labels >= 0])
        remap = np.full(int(unique.max()) + 1 if unique.size else 1, -1, dtype=np.int32)
        remap[unique] = np.arange(len(unique), dtype=np.int32)
        return remap[np.where(labels >= 0, labels, 0)] * (labels >= 0) + (-1) * (labels < 0)


class SLICSegmenter(Segmenter):
    """SLIC superpixel segmentation.

    Requires ``scikit-image``.

    Parameters
    ----------
    n_segments : int
        Approximate number of superpixels.
    compactness : float
        Balances colour proximity and spatial proximity.
    sigma : float
        Width of the Gaussian smoothing kernel applied before segmentation.
    """

    def __init__(
        self,
        n_segments: int = 200,
        compactness: float = 10.0,
        sigma: float = 1.0,
    ) -> None:
        self.n_segments  = n_segments
        self.compactness = compactness
        self.sigma       = sigma

    def segment(
        self,
        rgb: np.ndarray,
        water_mask: np.ndarray,
    ) -> np.ndarray:
        try:
            from skimage.segmentation import slic
        except ImportError as exc:
            raise ImportError(
                "SLICSegmenter requires scikit-image: "
                "conda install -c conda-forge scikit-image"
            ) from exc

        raw = slic(
            rgb,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            start_label=0,
        )
        labels = np.where(water_mask, raw, -1).astype(np.int32)
        unique = np.unique(labels[labels >= 0])
        remap = np.full(int(unique.max()) + 1 if unique.size else 1, -1, dtype=np.int32)
        remap[unique] = np.arange(len(unique), dtype=np.int32)
        return remap[np.where(labels >= 0, labels, 0)] * (labels >= 0) + (-1) * (labels < 0)
