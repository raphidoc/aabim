"""Tests for aabim.data.atmospheric (AerLUT, GasLUT)."""

import numpy as np

from aabim.data.atmospheric import AerLUT, GasLUT


def test_aer_lut_from_image(aci13_lut):
    lut = AerLUT.from_image(aci13_lut)
    assert len(lut.wavelength) == len(aci13_lut.wavelength)
    np.testing.assert_array_equal(lut.wavelength, aci13_lut.wavelength)


def test_gas_lut_from_image(aci13_lut):
    lut = GasLUT.from_image(aci13_lut)
    assert len(lut.wavelength) == len(aci13_lut.wavelength)
    np.testing.assert_array_equal(lut.wavelength, aci13_lut.wavelength)


def test_aer_lut_cache(aci13_lut, tmp_path):
    """Second call with same cache_dir should return identical wavelengths."""
    lut1 = AerLUT.from_image(aci13_lut, cache_dir=tmp_path)
    lut2 = AerLUT.from_image(aci13_lut, cache_dir=tmp_path)
    np.testing.assert_array_equal(lut1.wavelength, lut2.wavelength)
