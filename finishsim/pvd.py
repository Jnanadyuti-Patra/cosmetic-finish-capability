"""PVD thin-film stack -> reflectance spectrum, by the transfer-matrix method.

This is the calculation that Essential Macleod and TFCALC perform.  For a
stack of j = 1..N absorbing layers on a substrate, each layer contributes a
characteristic matrix

    M_j = [[  cos(d_j),          i*sin(d_j)/eta_j ],
           [  i*eta_j*sin(d_j),  cos(d_j)         ]]

with phase thickness and tilted optical admittance

    d_j   = 2*pi * t_j * N_j * cos(th_j) / lambda
    eta_j = N_j * cos(th_j)          (s-polarised)
    eta_j = N_j / cos(th_j)          (p-polarised)

where N_j = n_j - i*k_j (Macleod's sign convention -- see ``_macleod``) and the
layer angle follows from Snell's law with complex indices.  The assembled
matrix maps the substrate admittance to the front surface, and the amplitude
reflection coefficient is

    [B, C]^T = (M_1 M_2 ... M_N) [1, eta_sub]^T
    Y = C / B ,   r = (eta_0 - Y)/(eta_0 + Y) ,   R = |r|^2

Layers are listed **outermost first** -- the order you would read off a
deposition recipe in reverse, and the order Macleod prints.

Unpolarised reflectance is the mean of the s and p results, which is what a
spectrophotometer measures at non-normal incidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import colourimetry as cm
from . import materials


@dataclass
class Layer:
    """One deposited layer.

    ``stoichiometry`` is x in MN_x for a reactively sputtered nitride, set on
    the tool by nitrogen partial pressure.  It is the main colour control for
    a bulk nitride film; see :func:`finishsim.materials._drude_lorentz`.
    """

    material: str
    thickness_nm: float
    stoichiometry: float = 1.0


@dataclass
class Stack:
    """A coating stack on a substrate.

    ``layers`` is ordered outermost (ambient side) first.
    """

    layers: list[Layer] = field(default_factory=list)
    substrate: str = "SS304"
    ambient: str = "air"

    def total_thickness_nm(self) -> float:
        return sum(l.thickness_nm for l in self.layers)

    def describe(self) -> str:
        chain = " / ".join(f"{l.material} {l.thickness_nm:.0f} nm" for l in self.layers)
        return f"{self.ambient} / {chain} / {self.substrate}"


def _macleod(N):
    """Convert n + i*k (optical-constants convention) to n - i*k (Macleod).

    Tabulated optical constants -- refractiveindex.info, ellipsometry output,
    and :mod:`finishsim.materials` -- are published as N = n + i*k with k > 0,
    which pairs with a time dependence exp(-i*w*t).  The characteristic-matrix
    form used below is Macleod's, which pairs with exp(+i*w*t) and therefore
    expects N = n - i*k.  Feeding n + i*k into it silently turns every
    absorbing layer into a gain medium and yields |r| > 1.

    One conjugation at the boundary keeps ``materials`` in the convention its
    data sources actually use.
    """
    return np.conj(N)


def _tilted_cos(N, n0, sin_th0):
    """cos(theta) inside a layer of index N, from Snell with complex indices."""
    return np.sqrt(1.0 - (n0 * sin_th0 / N) ** 2)


def reflectance(
    stack: Stack,
    wl_nm=cm.WAVELENGTHS,
    angle_deg: float = 8.0,
    polarisation: str = "unpolarised",
) -> np.ndarray:
    """Specular reflectance factor (0-1) of ``stack`` on the grid ``wl_nm``.

    ``angle_deg`` defaults to 8 degrees, the near-normal geometry of a d/8
    integrating-sphere spectrophotometer -- the instrument a finishing line
    actually uses for colour QC.
    """
    wl_nm = np.asarray(wl_nm, dtype=float)
    n0 = _macleod(materials.index(stack.ambient, wl_nm))
    nsub = _macleod(materials.index(stack.substrate, wl_nm))
    sin_th0 = np.sin(np.radians(angle_deg))

    modes = ("s", "p") if polarisation == "unpolarised" else (polarisation,)
    out = np.zeros_like(wl_nm)

    for mode in modes:
        def admittance(N, _mode=mode):
            # _mode is bound at definition time on purpose: a bare closure over
            # the loop variable would silently evaluate both polarisations with
            # whichever mode the loop ended on.
            c = _tilted_cos(N, n0, sin_th0)
            return N * c if _mode == "s" else N / c

        eta0 = admittance(n0)
        etasub = admittance(nsub)

        # Assemble the characteristic matrix, outermost layer first.
        m11 = np.ones_like(wl_nm, dtype=complex)
        m12 = np.zeros_like(wl_nm, dtype=complex)
        m21 = np.zeros_like(wl_nm, dtype=complex)
        m22 = np.ones_like(wl_nm, dtype=complex)

        for layer in stack.layers:
            N = _macleod(
                materials.index(layer.material, wl_nm, layer.stoichiometry)
            )
            c = _tilted_cos(N, n0, sin_th0)
            delta = 2.0 * np.pi * layer.thickness_nm * N * c / wl_nm
            eta = admittance(N)
            a11, a22 = np.cos(delta), np.cos(delta)
            a12 = 1j * np.sin(delta) / eta
            a21 = 1j * eta * np.sin(delta)
            m11, m12, m21, m22 = (
                m11 * a11 + m12 * a21,
                m11 * a12 + m12 * a22,
                m21 * a11 + m22 * a21,
                m21 * a12 + m22 * a22,
            )

        B = m11 + m12 * etasub
        C = m21 + m22 * etasub
        Y = C / B
        r = (eta0 - Y) / (eta0 + Y)
        out = out + np.abs(r) ** 2

    return np.clip(out / len(modes), 0.0, 1.0)


def lab(stack: Stack, wl_nm=cm.WAVELENGTHS, angle_deg: float = 8.0) -> np.ndarray:
    """CIELAB of a stack under D65 / 2-degree observer."""
    return cm.spectrum_to_Lab(reflectance(stack, wl_nm, angle_deg), wl_nm)


def angular_colour_shift(
    stack: Stack, angles_deg=(8.0, 30.0, 45.0, 60.0), wl_nm=cm.WAVELENGTHS
):
    """dE00 of the stack at each viewing angle, referenced to the first angle.

    Angular colour shift ("flop") is a genuine cosmetic reject mode on curved
    enclosures: the part passes at normal incidence and fails at the chamfer.
    Interference stacks flop hard; bulk-absorbing single layers barely do.
    """
    ref = lab(stack, wl_nm, angles_deg[0])
    return {
        a: cm.delta_E_2000(ref, lab(stack, wl_nm, a)) for a in angles_deg
    }


# ---------------------------------------------------------------------------
# Reference recipes -- realistic decorative PVD stacks to start a study from.
# ---------------------------------------------------------------------------
PRESET_STACKS = {
    "TiN gold on stainless": Stack(
        layers=[Layer("TiN", 320.0), Layer("Ti", 80.0)], substrate="SS304"
    ),
    "ZrN light gold on stainless": Stack(
        layers=[Layer("ZrN", 300.0), Layer("Ti", 80.0)], substrate="SS304"
    ),
    "CrN graphite on aluminium": Stack(
        layers=[Layer("CrN", 400.0), Layer("Ti", 60.0)], substrate="Al"
    ),
    "TiN + SiO2 overcoat (interference)": Stack(
        layers=[Layer("SiO2", 90.0), Layer("TiN", 300.0), Layer("Ti", 80.0)],
        substrate="SS304",
    ),
}
