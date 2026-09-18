"""Optical constants n(lambda) + i*k(lambda) for the substrates and coatings.

Two dispersion forms cover everything used here:

* **Drude-Lorentz** for metals and the metallic nitrides (TiN, ZrN, CrN, Al,
  stainless).  A free-electron Drude term sets the infrared metallic response;
  Lorentz oscillators add the interband transitions that give TiN and ZrN
  their gold colour.

      eps(w) = eps_inf - wp^2/(w^2 + i*G*w) + SUM_j  f_j*w_j^2/(w_j^2 - w^2 - i*g_j*w)

* **Sellmeier** for the transparent dielectrics (anodic Al2O3, SiO2), which
  have negligible absorption across the visible.

PARAMETER PROVENANCE -- read this before quoting any absolute number.
The oscillator parameters below are *representative literature-fit values*
for the material class, not a measurement of a specific supplier's coating.
They reproduce the correct colour family and the correct qualitative
dispersion, which is what a process-capability study needs: the study's
conclusions are about *sensitivity of colour to process variation*, and those
are governed by the stack geometry and the shape of n(lambda), not by the
third decimal place of eps_inf.

For a production-grade answer, drop a tabulated n,k file into ``data/`` and
load it with :func:`tabulated`; the API is identical and every downstream
calculation is unchanged.  refractiveindex.info exports in exactly this form.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# eV <-> nm.  E[eV] = 1239.841984 / lambda[nm]
_HC_EV_NM = 1239.841984

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Drude-Lorentz parameter sets.  Energies in eV.
#   eps_inf, (wp, gamma),  [(f, w0, gamma0), ...]
# ---------------------------------------------------------------------------
DRUDE_LORENTZ = {
    # Titanium nitride -- the classic decorative "PVD gold".  Metallic in the
    # red, interband absorption in the blue produces the yellow reflectance.
    "TiN": dict(eps_inf=4.86, wp=7.38, gamma=0.19, lorentz=[(2.70, 4.50, 2.10)]),
    # Zirconium nitride -- brighter, more saturated "light gold" than TiN.
    "ZrN": dict(eps_inf=3.40, wp=8.02, gamma=0.17, lorentz=[(2.20, 5.20, 2.30)]),
    # Chromium nitride -- darker, greyer, more absorbing.
    "CrN": dict(eps_inf=4.20, wp=5.10, gamma=0.55, lorentz=[(3.60, 3.20, 2.60)]),
    # Aluminium -- broadband mirror with the well-known 1.5 eV interband dip.
    "Al": dict(eps_inf=1.00, wp=14.98, gamma=0.60, lorentz=[(5.60, 1.55, 0.60)]),
    # Austenitic stainless (304-class) -- lossy, low-reflectance metal.
    "SS304": dict(eps_inf=2.70, wp=5.90, gamma=1.60, lorentz=[(2.10, 2.60, 3.00)]),
    # Titanium -- common adhesion / colour-tuning underlayer.
    "Ti": dict(eps_inf=2.20, wp=7.30, gamma=1.30, lorentz=[(3.00, 2.20, 2.60)]),
}

# Sellmeier coefficients: n^2 - 1 = SUM B_i * L^2 / (L^2 - C_i), L in microns.
SELLMEIER = {
    # Amorphous/porous anodic alumina.  Slightly below dense sapphire because
    # the anodic film is porous and hydrated.
    "Al2O3": dict(
        B=[1.4313493, 0.65054713, 5.3414021],
        C=[0.0052799, 0.0142382, 325.017834],
        scale=0.97,
    ),
    "SiO2": dict(
        B=[0.6961663, 0.4079426, 0.8974794],
        C=[0.0046791, 0.0135121, 97.9340025],
        scale=1.0,
    ),
}


def _drude_lorentz(name: str, wl_nm: np.ndarray, stoichiometry: float = 1.0) -> np.ndarray:
    """Complex index from the Drude-Lorentz model, optionally off-stoichiometric.

    ``stoichiometry`` is x in MN_x for the nitrides -- the quantity a PVD line
    actually controls, through nitrogen partial pressure during reactive
    sputtering.  It is the dominant colour knob for a bulk nitride coating,
    because film thickness stops mattering once the layer is optically opaque
    (past roughly 150 nm for TiN).

    Physical picture as x falls below 1 (nitrogen-starved, metal-rich):

    * more free carriers, so the plasma frequency rises  -> the reflectance
      edge moves blue and the film drifts from gold toward metallic silver;
    * the interband oscillator that creates the gold colour weakens;
    * point defects broaden every feature.

    Above x = 1 the film is nitrogen-stuffed, more insulating and darker.
    """
    p = DRUDE_LORENTZ[name]
    w = _HC_EV_NM / np.asarray(wl_nm, dtype=float)

    x = float(stoichiometry)
    if x == 1.0:
        wp, gamma, lorentz = p["wp"], p["gamma"], p["lorentz"]
    else:
        wp = p["wp"] * (1.0 + 0.35 * (1.0 - x))
        gamma = p["gamma"] * (1.0 + 0.80 * abs(1.0 - x))
        lorentz = [(f * max(x, 0.0) ** 1.5, w0, g0 * (1.0 + 0.5 * abs(1.0 - x)))
                   for f, w0, g0 in p["lorentz"]]

    eps = p["eps_inf"] - wp**2 / (w**2 + 1j * gamma * w)
    for f, w0, g0 in lorentz:
        eps = eps + f * w0**2 / (w0**2 - w**2 - 1j * g0 * w)
    return np.sqrt(eps)


def _sellmeier(name: str, wl_nm: np.ndarray) -> np.ndarray:
    p = SELLMEIER[name]
    L2 = (np.asarray(wl_nm, dtype=float) / 1000.0) ** 2
    n2 = 1.0
    for B, C in zip(p["B"], p["C"]):
        n2 = n2 + B * L2 / (L2 - C)
    return (np.sqrt(n2) * p["scale"]).astype(complex)


def tabulated(path, wl_nm: np.ndarray) -> np.ndarray:
    """Load measured n,k from a CSV with columns ``wavelength_nm,n,k``.

    Interpolates onto ``wl_nm``.  Use this to replace any model above with
    real supplier or refractiveindex.info data.
    """
    arr = np.genfromtxt(path, delimiter=",", names=True)
    n = np.interp(wl_nm, arr["wavelength_nm"], arr["n"])
    k = np.interp(wl_nm, arr["wavelength_nm"], arr["k"])
    return n + 1j * k


# Monte-Carlo runs re-evaluate the same dispersion on the same grid millions
# of times.  Memoise on (name, grid, stoichiometry); the returned arrays are
# treated as read-only by every caller.
_INDEX_CACHE: dict = {}


def index(name: str, wl_nm: np.ndarray, stoichiometry: float = 1.0) -> np.ndarray:
    """Complex refractive index of ``name`` on the wavelength grid ``wl_nm``.

    Resolution order: a CSV in ``data/<name>.csv`` wins over the built-in
    model, so dropping in measured data needs no code change.  A non-unity
    ``stoichiometry`` forces the Drude-Lorentz path, since a fixed table
    cannot describe an off-stoichiometric film.
    """
    wl_nm = np.ascontiguousarray(wl_nm, dtype=float)
    key = (name, wl_nm.tobytes(), round(float(stoichiometry), 9))
    hit = _INDEX_CACHE.get(key)
    if hit is not None:
        return hit

    csv = DATA_DIR / f"{name}.csv"
    if csv.exists() and stoichiometry == 1.0:
        out = tabulated(csv, wl_nm)
    elif name in DRUDE_LORENTZ:
        out = _drude_lorentz(name, wl_nm, stoichiometry)
    elif name in SELLMEIER:
        out = _sellmeier(name, wl_nm)
    elif name in ("air", "vacuum"):
        out = np.ones_like(wl_nm, dtype=complex)
    else:
        raise KeyError(
            f"unknown material {name!r}; known: "
            f"{sorted(set(DRUDE_LORENTZ) | set(SELLMEIER) | {'air'})}"
        )

    # Bound the cache: stoichiometry is continuous, so a long Monte-Carlo run
    # would otherwise accumulate one entry per sample.
    if len(_INDEX_CACHE) > 20000:
        _INDEX_CACHE.clear()
    _INDEX_CACHE[key] = out
    return out


def available() -> list[str]:
    """Material names this module can supply."""
    built_in = set(DRUDE_LORENTZ) | set(SELLMEIER) | {"air"}
    if DATA_DIR.exists():
        built_in |= {p.stem for p in DATA_DIR.glob("*.csv")}
    return sorted(built_in)


def source_of(name: str) -> str:
    """Where ``name``'s optical constants came from -- for report provenance."""
    if (DATA_DIR / f"{name}.csv").exists():
        return f"tabulated data/{name}.csv"
    if name in DRUDE_LORENTZ:
        return "Drude-Lorentz model (representative literature parameters)"
    if name in SELLMEIER:
        return "Sellmeier model"
    return "n = 1 (non-absorbing ambient)"
