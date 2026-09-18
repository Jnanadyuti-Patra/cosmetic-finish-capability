"""Colour pipeline verification against published reference values."""

import numpy as np
import pytest

from finishsim import colourimetry as cm

# Sharma, Wu & Dalal (2005), Color Research & Application 30(1), 21-30,
# Table 1 -- the standard CIEDE2000 implementation test set.  These pairs are
# chosen to exercise the hue-wraparound and the RT rotation term, which are
# exactly the parts a naive implementation gets wrong.
SHARMA_PAIRS = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize("lab1,lab2,expected", SHARMA_PAIRS)
def test_ciede2000_reference_pairs(lab1, lab2, expected):
    assert cm.delta_E_2000(lab1, lab2) == pytest.approx(expected, abs=1e-4)


def test_ciede2000_is_symmetric():
    a, b = (48.0, 12.0, -7.0), (51.0, 9.0, -3.0)
    assert cm.delta_E_2000(a, b) == pytest.approx(cm.delta_E_2000(b, a))


def test_ciede2000_self_distance_is_zero():
    assert cm.delta_E_2000((40.0, 5.0, -5.0), (40.0, 5.0, -5.0)) == pytest.approx(0.0)


def test_perfect_diffuser_is_reference_white():
    """A 100 % reflector under D65 must land on L*=100, a*=b*=0."""
    lab = cm.spectrum_to_Lab(np.ones_like(cm.WAVELENGTHS))
    assert lab[0] == pytest.approx(100.0, abs=1e-6)
    assert lab[1] == pytest.approx(0.0, abs=1e-6)
    assert lab[2] == pytest.approx(0.0, abs=1e-6)


def test_d65_white_point_matches_cie():
    """D65 white point is X=95.047, Y=100, Z=108.883 (CIE 1931 2-degree)."""
    wp = cm.white_point()
    assert wp[0] == pytest.approx(95.047, abs=0.05)
    assert wp[1] == pytest.approx(100.0, abs=1e-6)
    assert wp[2] == pytest.approx(108.883, abs=0.05)


def test_neutral_greys_stay_neutral():
    for r in (0.2, 0.5, 0.8):
        lab = cm.spectrum_to_Lab(np.full_like(cm.WAVELENGTHS, r))
        assert lab[1] == pytest.approx(0.0, abs=1e-6)
        assert lab[2] == pytest.approx(0.0, abs=1e-6)


def test_lightness_is_monotonic_in_reflectance():
    ls = [cm.spectrum_to_Lab(np.full_like(cm.WAVELENGTHS, r))[0]
          for r in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(b > a for a, b in zip(ls, ls[1:]))


def test_cache_does_not_change_results():
    """The memo cache must be transparent."""
    first = cm.spectrum_to_Lab(np.full_like(cm.WAVELENGTHS, 0.42)).copy()
    cm._TABLE_CACHE.clear()
    second = cm.spectrum_to_Lab(np.full_like(cm.WAVELENGTHS, 0.42))
    assert np.allclose(first, second)


def test_hex_swatch_format():
    assert cm.hex_swatch((50.0, 20.0, -30.0)).startswith("#")
    assert len(cm.hex_swatch((50.0, 20.0, -30.0))) == 7
