"""Transfer-matrix method: conservation, limits, and known-material checks."""

import numpy as np
import pytest

from finishsim import colourimetry as cm
from finishsim import materials, pvd


def test_reflectance_is_physical():
    """0 <= R <= 1 everywhere, for every preset, at every angle.

    This is the check that catches the n+ik / n-ik convention error: feeding
    the wrong sign into the Macleod matrix turns absorbing layers into gain
    media and drives R above 1.
    """
    for stack in pvd.PRESET_STACKS.values():
        for angle in (0.0, 8.0, 45.0, 70.0):
            R = pvd.reflectance(stack, angle_deg=angle)
            assert np.all(R >= 0.0)
            assert np.all(R <= 1.0)


def test_absorbing_layers_are_lossy_not_gaining():
    """A thick absorbing film must reflect less than a perfect mirror."""
    thick_tin = pvd.Stack(layers=[pvd.Layer("TiN", 500.0)], substrate="Al")
    bare_al = pvd.Stack(layers=[], substrate="Al")
    assert pvd.reflectance(thick_tin).mean() < pvd.reflectance(bare_al).mean()


def test_zero_thickness_layer_is_transparent():
    """A layer of zero thickness must not change the result at all."""
    bare = pvd.Stack(layers=[], substrate="SS304")
    with_null = pvd.Stack(layers=[pvd.Layer("TiN", 0.0)], substrate="SS304")
    assert np.allclose(pvd.reflectance(bare), pvd.reflectance(with_null), atol=1e-12)


def test_opaque_film_hides_its_substrate():
    """Past the absorption depth, the substrate stops mattering.

    TiN has k ~ 2 in the visible, so the 1/e absorption depth is tens of nm.
    At 400 nm the stack should read the same on steel as on aluminium.
    """
    on_ss = pvd.Stack(layers=[pvd.Layer("TiN", 400.0)], substrate="SS304")
    on_al = pvd.Stack(layers=[pvd.Layer("TiN", 400.0)], substrate="Al")
    assert cm.delta_E_2000(pvd.lab(on_ss), pvd.lab(on_al)) < 0.1


def test_bulk_nitride_colour_is_thickness_insensitive():
    """Once opaque, thickness is the wrong colour knob -- stoichiometry is."""
    ref = pvd.lab(pvd.Stack(layers=[pvd.Layer("TiN", 300.0)], substrate="SS304"))
    thick = pvd.lab(pvd.Stack(layers=[pvd.Layer("TiN", 360.0)], substrate="SS304"))
    assert cm.delta_E_2000(ref, thick) < 0.05

    off_stoich = pvd.lab(
        pvd.Stack(layers=[pvd.Layer("TiN", 300.0, 0.94)], substrate="SS304"))
    assert cm.delta_E_2000(ref, off_stoich) > 1.0


def test_aluminium_is_a_bright_neutral_mirror():
    lab = pvd.lab(pvd.Stack(layers=[], substrate="Al"))
    assert lab[0] > 90.0
    assert abs(lab[1]) < 3.0
    assert abs(lab[2]) < 3.0


def test_tin_is_gold():
    """TiN's signature: low blue reflectance, high red, strongly positive b*."""
    R = pvd.reflectance(pvd.Stack(layers=[pvd.Layer("TiN", 320.0)], substrate="SS304"))
    assert np.interp(450.0, cm.WAVELENGTHS, R) < np.interp(650.0, cm.WAVELENGTHS, R)
    assert pvd.lab(pvd.Stack(layers=[pvd.Layer("TiN", 320.0)], substrate="SS304"))[2] > 30.0


def test_interference_stack_flops_more_than_bulk():
    """An interference stack shifts colour with angle; a bulk absorber barely does."""
    bulk = pvd.PRESET_STACKS["TiN gold on stainless"]
    interference = pvd.PRESET_STACKS["TiN + SiO2 overcoat (interference)"]
    assert (pvd.angular_colour_shift(interference, (8.0, 30.0))[30.0]
            > pvd.angular_colour_shift(bulk, (8.0, 30.0))[30.0])


def test_stoichiometry_moves_colour_monotonically_below_one():
    """Going metal-rich should steadily walk the colour away from the master."""
    ref = pvd.lab(pvd.Stack(layers=[pvd.Layer("TiN", 300.0, 1.00)], substrate="SS304"))
    des = [
        cm.delta_E_2000(
            ref, pvd.lab(pvd.Stack(layers=[pvd.Layer("TiN", 300.0, x)], substrate="SS304")))
        for x in (0.98, 0.94, 0.90, 0.86)
    ]
    assert all(b > a for a, b in zip(des, des[1:]))


def test_unknown_material_raises():
    with pytest.raises(KeyError):
        materials.index("unobtainium", cm.WAVELENGTHS)


def test_dielectric_has_negligible_absorption():
    k = materials.index("SiO2", cm.WAVELENGTHS).imag
    assert np.allclose(k, 0.0, atol=1e-9)


def test_index_cache_is_transparent():
    a = materials.index("TiN", cm.WAVELENGTHS, 0.97).copy()
    materials._INDEX_CACHE.clear()
    b = materials.index("TiN", cm.WAVELENGTHS, 0.97)
    assert np.allclose(a, b)
