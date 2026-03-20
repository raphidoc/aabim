# AABIM — Atmospheric and Aquatic Bayesian Inversion Model

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AABIM is a Python package for the atmospheric correction and water-leaving reflectance retrieval of airborne hyperspectral imagery over optically shallow water.
It was developed as part of Raphael Mabit's PhD thesis:

> Mabit, R. (2026). *Réflectances d'Anticosti : radiométrie in situ, imagerie hyperspectrale aéroportée et inversion bayésienne en eaux optiquement peu profondes.*

---

## Features

- **Image conversion** — reads WISE `.dpix` (ENVI) format; extensible to other sensors
- **Ancillary data** — automatic download of MERRA-2 atmospheric variables (AOD, pressure, water vapour, ozone) via NASA EarthData
- **Vicarious calibration** — SMA, OLS, and ratio models fit from in-situ water-leaving reflectance matchups
- **Atmospheric correction** — 6S-based LUT interpolation with per-band Gaussian SRF convolution of the Coddington solar spectrum; G21 sky-glint correction
- **Pixel extraction** — spatial matchup extraction with configurable window averaging
- **CPU and GPU backends** — tiled parallel processing with `ProcessPoolExecutor` (CPU) or CuPy (GPU)

---

## Installation

AABIM requires a conda environment. All core dependencies are available on [conda-forge](https://conda-forge.org/).

```bash
mamba env create -f environment.yml
conda activate aabim
pip install -e .
```

> **GPU support** (optional): the GPU backend uses [CuPy](https://cupy.dev/). It is included in `environment.yml` by default. Remove the `cupy` line if you do not have an NVIDIA GPU with CUDA drivers installed.

> **SMA calibration** (`SMAModel`) requires [`pylr2`](https://github.com/amaurs/then-what), installed via pip as listed in `environment.yml`. A conda-forge feedstock is planned.

---

## Quick start

```python
import aabim
from aabim import Image, SMAModel
from aabim.ancillary.get_ancillary import add_ancillary

# Load an L1C image (converted from WISE .dpix format)
img = Image.from_aabim_nc("ACI13_bbox_l1c.nc")

# Attach MERRA-2 ancillary data
add_ancillary(img, anc_dir="anc/")

# Load a vicarious calibration model and clip image to calibrated range
sma = SMAModel.load("sma_calibration_model.csv")
img.mask_wavelength([float(sma.wavelength[0]), float(sma.wavelength[-1])])

# Atmospheric correction → water-leaving reflectance
img_l2r = img.to_l2r("output.zarr", cal_model=sma, backend="cpu", n_workers=8)
```

See [`aabim/example/l2r_sma_calibration.ipynb`](aabim/example/l2r_sma_calibration.ipynb) for a full worked example including spectrum extraction and RGB visualisation.

---

## Sensor conversion

The converter reads a WISE `.dpix` ENVI image and writes a self-contained CF-1.0 NetCDF:

```python
from aabim.converter.wise.read_pix import Pix

pix = Pix("ACI13.dpix")
pix.to_aabim_nc("ACI13_l1c.nc")
```

---

## Vicarious calibration workflow

```python
from aabim import InSitu, CalibrationData, SMAModel

in_situ  = InSitu.from_csv("rho_w.csv")
cal_data = CalibrationData.compute(image_l1c, in_situ)
cal_data = CalibrationData.concat([cal_data_1, cal_data_2])   # multi-image

cal_model = SMAModel.fit(cal_data)
cal_model.save("sma_calibration_model.csv")
```

---

## Project structure

```
aabim/
├── ancillary/      MERRA-2 ancillary data download and attachment
├── calibration/    Vicarious calibration (SMA, OLS, ratio models)
├── converter/      Sensor-specific L0→L1C converters (WISE, …)
├── data/           Static data: solar irradiance spectrum, 6S atmospheric LUTs
├── example/        Jupyter notebook vignettes
├── image/          Core Image class with IO, correction, calibration, extract mixins
└── model/          Stan models for Bayesian inversion (in development)
```

---

## Citation

If you use AABIM in your research, please cite:

> Mabit, R. (2026). *Réflectances d'Anticosti : radiométrie in situ, imagerie hyperspectrale aéroportée et inversion bayésienne en eaux optiquement peu profondes.* PhD thesis, Université du Québec à Rimouski.

---

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
