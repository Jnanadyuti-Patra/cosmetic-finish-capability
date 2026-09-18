"""Type II sulfuric-acid anodizing of aluminium, then dyeing -> reflectance.

PHYSICS
-------
Anodic oxide grows by Faraday's law from the anodizing current, and is eaten
back chemically by the acid at the same time.  Net film growth is the
difference of the two:

    dx/dt  =  M_ox * j * eta_cur / (z * F * rho_ox)        (electrochemical growth)
            - k0 * exp(-Ea / R T) * [H2SO4]^m              (chemical back-dissolution)

with M_ox = 101.96 g/mol for Al2O3, z = 6 equivalents (2 Al -> Al2O3 transfers
six electrons), and rho_ox ~ 3.1 g/cm3 for the porous anodic film -- below
dense sapphire because the film is porous and partly hydrated.

Sanity check built into the defaults: 1.5 A/dm2 for 30 min at 20 C, 180 g/L
gives ~10 um of oxide, which is exactly the Type II architectural spec.

The back-dissolution term is why **bath temperature is the dominant cosmetic
lever** in anodizing.  Its Arrhenius factor roughly doubles per +10 C, so a
bath that drifts warm both thins the film and opens the pores, and the part
comes out lighter and duller.  That single coupling is the main thing this
module exists to quantify.

Porosity rises with temperature and acid strength.  It matters because the
dye lives in the pores: dye loading scales with pore volume, i.e. with
porosity x thickness.

COLOUR
------
A dyed anodic film is a bulk absorber, not an interference filter.  Light
crosses the film, reflects off the aluminium, and crosses back -- so
Beer-Lambert applies over a double pass:

    R(lambda) = R_surface + (1 - R_surface)^2 * R_Al(lambda) * exp(-2 * A(lambda))
    A(lambda) = eps(lambda) * c_dye * x

Because there is no interference term, dyed anodize barely shifts colour with
viewing angle -- the opposite of a PVD interference stack.  The two routes
fail in different ways, which is the point of modelling both.

Dye absorptivity eps(lambda) is represented as a sum of Gaussian bands; the
band positions are what make a dye read as blue, red or black.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import colourimetry as cm
from . import materials

# Physical constants
F = 96485.33212      # C/mol
R_GAS = 8.314462618  # J/(mol K)

# Al2O3 anodic film
M_OX = 0.10196       # kg/mol
Z_OX = 6.0           # equivalents per Al2O3
RHO_OX = 3100.0      # kg/m3, porous anodic alumina

# Back-dissolution kinetics.  k0 calibrated so that 20 C / 180 g/L gives
# ~0.05 um/min of chemical attack, the accepted figure for Type II baths.
K0_DISS = 1.8        # m/s per (g/L)
EA_DISS = 65_000.0   # J/mol


@dataclass
class DyeBand:
    """One Gaussian absorption band of a dye."""

    peak_nm: float
    width_nm: float
    strength: float


@dataclass
class Dye:
    """An absorbing dye, as a set of bands plus a bath concentration scale."""

    name: str
    bands: list[DyeBand] = field(default_factory=list)

    def absorptivity(self, wl_nm) -> np.ndarray:
        """Relative molar absorptivity eps(lambda), peak-normalised."""
        wl_nm = np.asarray(wl_nm, dtype=float)
        eps = np.zeros_like(wl_nm)
        for b in self.bands:
            eps += b.strength * np.exp(-0.5 * ((wl_nm - b.peak_nm) / b.width_nm) ** 2)
        return eps


# A dye absorbs the COMPLEMENT of the colour it appears.  A blue dye absorbs
# in the orange-red; a red dye absorbs in the green.
DYES = {
    "none (clear/silver)": Dye("none", []),
    "black": Dye(
        "black",
        [DyeBand(450, 120, 1.00), DyeBand(560, 120, 1.00), DyeBand(670, 120, 1.00)],
    ),
    "blue": Dye("blue", [DyeBand(600, 60, 1.00), DyeBand(660, 70, 0.70)]),
    "red": Dye("red", [DyeBand(510, 55, 1.00), DyeBand(560, 45, 0.55)]),
    "gold/bronze": Dye("gold", [DyeBand(440, 50, 1.00), DyeBand(400, 45, 0.60)]),
    "graphite grey": Dye(
        "graphite",
        [DyeBand(460, 130, 0.62), DyeBand(570, 130, 0.62), DyeBand(680, 130, 0.62)],
    ),
}


@dataclass
class AnodizeProcess:
    """The controllable process window of an anodize + dye line.

    Defaults are a standard Type II sulfuric decorative recipe.
    """

    current_density_A_dm2: float = 1.5
    bath_temp_C: float = 20.0
    acid_conc_gL: float = 180.0
    anodize_time_min: float = 30.0
    dissolved_Al_gL: float = 5.0      # bath age indicator
    dye_conc_gL: float = 3.0
    dye_temp_C: float = 55.0
    dye_time_min: float = 12.0
    dye_pH: float = 5.5
    dye: str = "black"
    sealed: bool = True

    # ---- derived process physics -------------------------------------
    def current_efficiency(self) -> float:
        """Fraction of charge that ends up as retained oxide.

        Falls with temperature and acid strength because more of the freshly
        grown oxide is re-dissolved at the pore base.
        """
        eta = 0.72 - 0.010 * (self.bath_temp_C - 20.0) - 0.0006 * (self.acid_conc_gL - 180.0)
        return float(np.clip(eta, 0.30, 0.95))

    def growth_rate_m_s(self) -> float:
        j = self.current_density_A_dm2 * 100.0  # A/dm2 -> A/m2
        return M_OX * j * self.current_efficiency() / (Z_OX * F * RHO_OX)

    def dissolution_rate_m_s(self) -> float:
        T = self.bath_temp_C + 273.15
        # Dissolved aluminium mildly inhibits further attack.
        inhibition = 1.0 / (1.0 + 0.020 * self.dissolved_Al_gL)
        return K0_DISS * np.exp(-EA_DISS / (R_GAS * T)) * self.acid_conc_gL * inhibition

    def thickness_um(self) -> float:
        """Net oxide thickness after anodizing."""
        t_s = self.anodize_time_min * 60.0
        net = (self.growth_rate_m_s() - self.dissolution_rate_m_s()) * t_s
        return float(max(net, 0.0) * 1e6)

    def porosity(self) -> float:
        """Volume fraction of pores in the film."""
        p = 0.12 + 0.0060 * (self.bath_temp_C - 20.0) + 0.00035 * (self.acid_conc_gL - 180.0)
        return float(np.clip(p, 0.04, 0.45))

    def cell_voltage_V(self) -> float:
        """Approximate steady anodizing voltage.

        Rises with current density, falls with temperature and acid strength
        (both raise bath conductivity and thin the barrier layer).  Reported
        because voltage is the variable an operator actually watches.
        """
        j = max(self.current_density_A_dm2, 1e-3)
        v = (
            16.0
            + 6.0 * np.log(j / 1.5)
            - 0.32 * (self.bath_temp_C - 20.0)
            - 0.020 * (self.acid_conc_gL - 180.0)
        )
        return float(np.clip(v, 4.0, 40.0))

    def barrier_thickness_nm(self) -> float:
        """Barrier layer at the pore base -- ~1.4 nm per volt for sulfuric."""
        return 1.4 * self.cell_voltage_V()

    def dye_loading(self) -> float:
        """Dimensionless dye concentration-path product, c_dye * x.

        Pore volume available is porosity x thickness.  Uptake saturates with
        immersion time, is faster in a hotter bath, and peaks near pH 5.5 where
        acid dyes are best adsorbed by the amphoteric oxide.
        """
        if self.dye == "none (clear/silver)":
            return 0.0
        tau = 8.0 * np.exp(-0.035 * (self.dye_temp_C - 55.0))  # minutes
        saturation = 1.0 - np.exp(-self.dye_time_min / max(tau, 0.5))
        ph_factor = np.exp(-0.5 * ((self.dye_pH - 5.5) / 1.4) ** 2)
        pore_volume = self.porosity() * self.thickness_um()
        return float(2.6 * pore_volume * saturation * ph_factor * (self.dye_conc_gL / 3.0))

    # ---- optics ------------------------------------------------------
    def reflectance(self, wl_nm=cm.WAVELENGTHS) -> np.ndarray:
        """Reflectance factor (0-1) of the finished, dyed film."""
        wl_nm = np.asarray(wl_nm, dtype=float)

        n_ox = materials.index("Al2O3", wl_nm).real
        n_al = materials.index("Al", wl_nm)

        # Front-surface Fresnel loss at air/oxide.
        r_surf = ((n_ox - 1.0) / (n_ox + 1.0)) ** 2
        # Sealing hydrates the surface and slightly reduces haze/gloss loss.
        if not self.sealed:
            r_surf = r_surf + 0.012

        # Aluminium substrate reflectance.
        r_al = np.abs((n_al - 1.0) / (n_al + 1.0)) ** 2

        # Double-pass Beer-Lambert absorption by the dye.
        eps = DYES[self.dye].absorptivity(wl_nm)
        absorbance = eps * self.dye_loading()

        # A porous film scatters a little; more porosity -> more haze, which
        # lifts the dark end and desaturates.
        haze = 0.010 + 0.05 * self.porosity()

        R = r_surf + (1.0 - r_surf) ** 2 * r_al * np.exp(-2.0 * absorbance) + haze * (
            1.0 - r_surf
        )
        return np.clip(R, 0.0, 1.0)

    def lab(self, wl_nm=cm.WAVELENGTHS) -> np.ndarray:
        """CIELAB of the finished part under D65 / 2-degree observer."""
        return cm.spectrum_to_Lab(self.reflectance(wl_nm), wl_nm)

    def summary(self) -> dict:
        """Everything a process engineer would want on one line."""
        L, a, b = self.lab()
        return {
            "thickness_um": self.thickness_um(),
            "porosity": self.porosity(),
            "voltage_V": self.cell_voltage_V(),
            "barrier_nm": self.barrier_thickness_nm(),
            "current_efficiency": self.current_efficiency(),
            "growth_um_min": self.growth_rate_m_s() * 60e6,
            "dissolution_um_min": self.dissolution_rate_m_s() * 60e6,
            "dye_loading": self.dye_loading(),
            "L*": L,
            "a*": a,
            "b*": b,
        }
