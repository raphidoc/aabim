"""Inversion model classes.

Design
------
Both :class:`AquaModel` (aquatic-only, from ρ_w) and
:class:`AtmAquaModel` (joint atmospheric + aquatic, from ρ_at_sensor)
share the same interface through :class:`InversionModel`.

Each model has a single class method ``fit(inv_data, ...)`` that:

1. Compiles the Stan model once per worker process.
2. Dispatches per-segment inversions in parallel via
   :class:`~concurrent.futures.ProcessPoolExecutor` (``method="sample"``
   or ``method="optimize"``).
3. Collects posterior summaries and returns an
   :class:`~aabim.inversion.result.InversionResult`.

Methods
-------
``"sample"``
    Full HMC/NUTS posterior sampling.  Returns mean, std, and quantiles
    per parameter.  Best for segmented images (manageable number of
    Stan calls).
``"optimize"``
    L-BFGS point estimates.  No posterior uncertainty.  Suitable for
    pixel-wise inversion of large images.
"""
from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from aabim.inversion.data import InversionData

log = logging.getLogger(__name__)

_STAN_DIR = Path(__file__).parent.parent / "model" / "stan"

# Parameters extracted from the posterior (same for both models)
_AQUA_PARAMS   = ["chl", "a_g_440", "spm", "bb_p_gamma", "a_g_slope", "h_w"]
_ATM_EXTRA     = ["aod550"]
_RB_PARAM      = "r_b"    # spectral, shape (n_wl,)


# ---------------------------------------------------------------------------
# Worker-level globals (initialised once per process)
# ---------------------------------------------------------------------------

_worker_model:   Any = None   # CmdStanModel
_worker_header:  str = ""


def _worker_init(stan_file: str, user_header: str) -> None:
    global _worker_model, _worker_header
    from cmdstanpy import CmdStanModel
    import cmdstanpy
    cmdstanpy.utils.get_logger().setLevel(logging.ERROR)

    _worker_model  = CmdStanModel(
        stan_file=stan_file,
        user_header=user_header,
    )
    _worker_header = user_header


def _worker_sample(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run HMC sampling for one segment and return summary statistics."""
    seg_id  = args["seg_id"]
    data    = args["data"]
    cfg     = args["cfg"]

    fit = _worker_model.sample(
        data=data,
        chains=cfg.get("n_chains", 4),
        iter_warmup=cfg.get("iter_warmup", 1000),
        iter_sampling=cfg.get("n_samples", 1000),
        show_progress=False,
        show_console=False,
    )
    return {"seg_id": seg_id, "fit": fit, "method": "sample"}


def _worker_optimize(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run L-BFGS optimisation for one segment and return MAP estimates."""
    seg_id = args["seg_id"]
    data   = args["data"]

    fit = _worker_model.optimize(
        data=data,
        show_console=False,
    )
    return {"seg_id": seg_id, "fit": fit, "method": "optimize"}


def _summarise(fit: Any, params: List[str], n_wl: int, method: str) -> Dict[str, Any]:
    """Extract posterior summary statistics from a CmdStanPy fit object."""
    summary: Dict[str, Any] = {}

    if method == "sample":
        df = fit.summary()
        for p in params:
            row = df.loc[p] if p in df.index else None
            if row is not None:
                summary[f"{p}_mean"] = float(row["Mean"])
                summary[f"{p}_std"]  = float(row["StdDev"])
                summary[f"{p}_q05"]  = float(row["5%"])
                summary[f"{p}_q95"]  = float(row["95%"])

        # Spectral bottom reflectance r_b[1..n_wl]
        rb_mean = np.array([
            float(df.loc[f"r_b[{i+1}]"]["Mean"]) for i in range(n_wl)
        ])
        rb_std  = np.array([
            float(df.loc[f"r_b[{i+1}]"]["StdDev"]) for i in range(n_wl)
        ])
        summary["r_b_mean"] = rb_mean
        summary["r_b_std"]  = rb_std

    else:  # optimize
        mle = fit.optimized_params_dict
        for p in params:
            if p in mle:
                summary[f"{p}_mean"] = float(mle[p])

        rb_mean = np.array([
            float(mle.get(f"r_b[{i+1}]", np.nan)) for i in range(n_wl)
        ])
        summary["r_b_mean"] = rb_mean

    return summary


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class InversionModel(ABC):
    """Abstract base for Stan inversion models."""

    #: Subclasses set this to the .stan file name inside ``aabim/model/stan/``
    _stan_file: str = ""

    @classmethod
    def fit(
        cls,
        inv_data: InversionData,
        *,
        method:      str = "sample",
        n_chains:    int = 4,
        n_samples:   int = 1000,
        iter_warmup: int = 1000,
        n_workers:   int = -1,
    ) -> "InversionResult":   # noqa: F821
        """Run the inversion on all segments.

        Parameters
        ----------
        inv_data :
            Prepared :class:`~aabim.inversion.data.InversionData`.
        method :
            ``"sample"`` for HMC/NUTS; ``"optimize"`` for L-BFGS.
        n_chains :
            Number of HMC chains (ignored for ``"optimize"``).
        n_samples :
            HMC samples per chain (ignored for ``"optimize"``).
        iter_warmup :
            HMC warm-up iterations (ignored for ``"optimize"``).
        n_workers :
            Worker processes (``-1`` = all logical cores).
        """
        from aabim.inversion.result import InversionResult

        if method not in ("sample", "optimize"):
            raise ValueError(f"method must be 'sample' or 'optimize', got {method!r}")

        cls._check_mode(inv_data.mode)

        stan_file   = str(_STAN_DIR / cls._stan_file)
        user_header = str(_STAN_DIR / "fct_rtm_stan.hpp")

        if n_workers == -1:
            n_workers = os.cpu_count() or 1

        params  = _AQUA_PARAMS + (_ATM_EXTRA if inv_data.mode == "atm_aqua" else [])
        n_wl    = len(inv_data.wavelength)
        n_segs  = len(inv_data)

        worker_fn = _worker_sample if method == "sample" else _worker_optimize
        cfg       = {"n_chains": n_chains, "n_samples": n_samples,
                     "iter_warmup": iter_warmup}

        work = [
            {"seg_id": i, "data": inv_data.segments[i], "cfg": cfg}
            for i in range(n_segs)
        ]

        log.info(
            "%s.fit: %d segments  method=%s  workers=%d",
            cls.__name__, n_segs, method, n_workers,
        )

        summaries: List[Optional[Dict[str, Any]]] = [None] * n_segs

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(stan_file, user_header),
        ) as pool:
            futures = {pool.submit(worker_fn, w): w["seg_id"] for w in work}
            for fut in tqdm(
                as_completed(futures),
                total=n_segs,
                desc=f"{cls.__name__} ({method})",
            ):
                seg_id = futures[fut]
                try:
                    result = fut.result()
                    summaries[seg_id] = _summarise(
                        result["fit"], params, n_wl, method
                    )
                except Exception as exc:
                    log.warning("Segment %d failed: %s", seg_id, exc)
                    summaries[seg_id] = {}

        return InversionResult(
            summaries  = summaries,
            labels     = inv_data.labels,
            gdf        = inv_data.gdf,
            wavelength = inv_data.wavelength,
            params     = params,
            method     = method,
        )

    @classmethod
    @abstractmethod
    def _check_mode(cls, mode: str) -> None:
        """Raise if inv_data.mode is incompatible with this model."""


# ---------------------------------------------------------------------------
# Concrete models
# ---------------------------------------------------------------------------

class AquaModel(InversionModel):
    """Aquatic-only inversion from water-leaving reflectance (ρ_w).

    Uses ``model_aqua_rb.stan``.
    Requires ``inv_data.mode == "aqua"``.
    """

    _stan_file = "model_aqua_rb.stan"

    @classmethod
    def _check_mode(cls, mode: str) -> None:
        if mode != "aqua":
            raise ValueError(
                "AquaModel requires InversionData with mode='aqua'. "
                f"Got mode={mode!r}."
            )


class AtmAquaModel(InversionModel):
    """Joint atmospheric + aquatic inversion from ρ_at_sensor.

    Uses ``model_atmaqua_rb.stan``.
    Requires ``inv_data.mode == "atm_aqua"``.
    """

    _stan_file = "model_atmaqua_rb.stan"

    @classmethod
    def _check_mode(cls, mode: str) -> None:
        if mode != "atm_aqua":
            raise ValueError(
                "AtmAquaModel requires InversionData with mode='atm_aqua'. "
                f"Got mode={mode!r}."
            )
