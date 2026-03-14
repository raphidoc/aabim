import pooch

# Update hash when LUT files change
_REGISTRY = pooch.create(
    # Local cache — respects XDG on Linux, AppData on Windows
    path=pooch.os_cache("aabim"),
    base_url="https://drive.google.com/drive/folders/1dmvZKAHxS_eC8ba2SLrZhW5_J3h9rKdr?usp=sharing",
    registry={
        "lut_aerosol.nc": "sha256:09b8323ecae6c3b8f05d875fd4ef9bc8684d50b95dd7a8f0322a32bfe23bc4f4",
        "lut_gas.nc": "sha256:3c49165e2e98deb9ab75f184c271431a36aebca63589736305d826242cb88a89",
        "hybrid_reference_spectrum_c2022-11-30_with_unc.nc": "sha256:820d5d656c927d4973739800ce8ebf93c4277e107266a2ca9b9f7992cec0f8f4",
    },
    # Allow user to override cache location via environment variable
    env="AABIM_DATA_DIR",
)

# _AER_LUT_FILES = {
#     "WISE":    "atmospheric/aer_lut_wise.nc",
#     "default": "atmospheric/aer_lut.nc",
# }

def fetch_aer_lut() -> str:
    """Return local path to aerosol LUT, downloading if necessary."""
    # key = _AER_LUT_FILES.get(sensor_name, _AER_LUT_FILES["default"])
    return _REGISTRY.fetch("lut_aerosol.nc", progressbar=True)


def fetch_gas_lut() -> str:
    """Return local path to gas LUT, downloading if necessary."""
    return _REGISTRY.fetch("lut_gas.nc", progressbar=True)


def fetch_solar_irradiance() -> str:
    """Return local path to solar irradiance file."""
    return _REGISTRY.fetch(
        "hybrid_reference_spectrum_c2022-11-30_with_unc.nc", progressbar=True
    )


if __name__ == "__main__":
    fetch_aer_lut()
