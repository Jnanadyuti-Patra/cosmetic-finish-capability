"""Run the full study and regenerate every number and figure in the report.

    python scripts/run_study.py

Writes CSVs and a results.json to ``outputs/``, and publication figures to
``docs/figures/``.  The LaTeX report reads those figures directly, so the
document and the code can never drift apart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Imports follow the sys.path insert above so the package resolves when this
# script is run directly from a checkout; ruff's E402 is waived in pyproject.
from finishsim import anodize as an
from finishsim import capability as cap
from finishsim import colourimetry as cm
from finishsim import doe, pvd

OUT = ROOT / "outputs"
FIG = ROOT / "docs" / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Cosmetic specification used throughout.
DE_SPEC = 1.0          # dE00 upper limit vs the approved colour master
THICK_LSL, THICK_USL = 8.0, 12.0   # um, Type II decorative window
TARGET_CPK = 1.33

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)


def swatch_row(ax, labs, labels, title):
    for i, (lab, lb) in enumerate(zip(labs, labels)):
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=cm.hex_swatch(lab)))
        ax.text(i + 0.46, -0.16, lb, ha="center", va="top", fontsize=7.5)
    ax.set_xlim(-0.1, len(labs))
    ax.set_ylim(-0.5, 1.1)
    ax.axis("off")
    ax.set_title(title, fontsize=9)


# ===========================================================================
# PART 1 -- ANODIZE
# ===========================================================================
def anodize_forward(state: dict) -> dict:
    p = an.AnodizeProcess(
        current_density_A_dm2=state.get("current_density", 1.5),
        bath_temp_C=state.get("bath_temp", 20.0),
        acid_conc_gL=state.get("acid_conc", 180.0),
        anodize_time_min=state.get("anodize_time", 30.0),
        dye_time_min=state.get("dye_time", 12.0),
        dye_temp_C=state.get("dye_temp", 55.0),
        dye_conc_gL=state.get("dye_conc", 3.0),
        dye_pH=state.get("dye_pH", 5.5),
        dye="black",
    )
    lab = p.lab()
    return {
        "thickness_um": p.thickness_um(),
        "voltage_V": p.cell_voltage_V(),
        "porosity": p.porosity(),
        "L": lab[0],
        "a": lab[1],
        "b": lab[2],
        "dE00": cm.delta_E_2000(ANODIZE_MASTER, lab),
    }


ANODIZE_MASTER = an.AnodizeProcess(dye="black").lab()


def run_anodize():
    print("\n=== ANODIZE: Type II sulfuric + black dye ===")
    print(f"colour master L*a*b* = {np.round(ANODIZE_MASTER, 3).tolist()}")

    factors = [
        doe.Factor("current_density", 1.2, 1.8, "A/dm2"),
        doe.Factor("bath_temp", 17.0, 23.0, "C"),
        doe.Factor("acid_conc", 165.0, 195.0, "g/L"),
        doe.Factor("dye_time", 9.0, 15.0, "min"),
    ]
    design = doe.build_design(factors, "box-behnken", n_centre=3)
    results = doe.run_design(design, factors, anodize_forward)
    results.to_csv(OUT / "anodize_doe.csv", index=False)
    print(f"Box-Behnken: {len(results)} runs")

    fits = {}
    for resp in ("dE00", "thickness_um"):
        f = doe.fit_response_surface(results, factors, resp)
        fits[resp] = f
        print(f"  RSM {resp:14s} R2={f['r2']:.4f}  adj={f['r2_adj']:.4f}  RMSE={f['rmse']:.4f}")
        print("    top terms:", doe.effect_table(f).head(3)["term"].tolist())

    jmp = doe.export_jmp(results, factors, ["dE00", "thickness_um"], OUT, "anodize_doe")

    # ---- Monte-Carlo capability at the nominal process window -------------
    variables = [
        cap.ProcessVariable("current_density", 1.5, 0.10, units="A/dm2"),
        cap.ProcessVariable("bath_temp", 20.0, 1.50, units="C"),
        cap.ProcessVariable("acid_conc", 180.0, 10.0, units="g/L"),
        cap.ProcessVariable("anodize_time", 30.0, 1.0, units="min"),
        cap.ProcessVariable("dye_time", 12.0, 1.0, units="min"),
        cap.ProcessVariable("dye_temp", 55.0, 2.0, units="C"),
        cap.ProcessVariable("dye_conc", 3.0, 0.3, units="g/L"),
        cap.ProcessVariable("dye_pH", 5.5, 0.3, units="-"),
    ]
    mc = cap.monte_carlo(anodize_forward, variables, n=6000)
    mc.to_csv(OUT / "anodize_montecarlo.csv", index=False)

    names = [v.name for v in variables]
    specs = {"dE00": (None, DE_SPEC), "thickness_um": (THICK_LSL, THICK_USL)}
    report = cap.capability_report(mc, specs, names)

    print(f"  baseline yield        {report['overall_yield']*100:.2f} %")
    print(f"  dE00  mean={report['responses']['dE00']['mean']:.3f} "
          f"sigma={report['responses']['dE00']['sigma']:.3f} "
          f"Cpk={report['responses']['dE00']['cpk']:.3f}")
    print(f"  thick mean={report['responses']['thickness_um']['mean']:.3f} "
          f"sigma={report['responses']['thickness_um']['sigma']:.3f} "
          f"Cpk={report['responses']['thickness_um']['cpk']:.3f}")

    sens = cap.sensitivity(mc, "dE00", names)
    print("  dE00 drivers:", sens.head(3)[["variable", "contribution_pct"]].to_dict("records"))

    tightened, history = cap.tighten_to_target(
        anodize_forward, variables, "dE00", DE_SPEC, TARGET_CPK
    )
    print(f"  tightening rounds: {len(history)}, final Cpk={history[-1]['cpk']:.3f}")
    for v0, v1 in zip(variables, tightened):
        if abs(v1.tolerance - v0.tolerance) > 1e-9:
            print(f"    {v0.name:18s} +/-{v0.tolerance:6.3f} -> +/-{v1.tolerance:6.3f} {v0.units}")

    # ---- is the colour itself a capability decision? ----------------------
    # Black sits deep in Beer-Lambert saturation, so dye-bath noise barely
    # moves it.  Pale and chromatic finishes sit on the steep part of the
    # curve.  If that is true, the colour a designer picks sets the yield
    # before the process is even specified -- worth testing explicitly.
    dye_rows = []
    for dye_name in ("black", "graphite grey", "blue", "red", "gold/bronze"):
        master = an.AnodizeProcess(dye=dye_name).lab()

        def fwd(state, _d=dye_name, _m=master):
            p = an.AnodizeProcess(
                current_density_A_dm2=state["current_density"],
                bath_temp_C=state["bath_temp"],
                acid_conc_gL=state["acid_conc"],
                anodize_time_min=state["anodize_time"],
                dye_time_min=state["dye_time"],
                dye_temp_C=state["dye_temp"],
                dye_conc_gL=state["dye_conc"],
                dye_pH=state["dye_pH"],
                dye=_d,
            )
            return {"dE00": cm.delta_E_2000(_m, p.lab())}

        m = cap.monte_carlo(fwd, variables, n=2500)
        dye_rows.append(
            {
                "dye": dye_name,
                "L": master[0], "a": master[1], "b": master[2],
                "mean_dE00": float(m["dE00"].mean()),
                "sigma_dE00": float(m["dE00"].std(ddof=1)),
                "cpk": cap.cpk(m["dE00"], None, DE_SPEC),
                "yield_pct": 100.0 * cap.yield_fraction(m, {"dE00": (None, DE_SPEC)}),
            }
        )
    dye_df = pd.DataFrame(dye_rows)
    dye_df.to_csv(OUT / "anodize_dye_robustness.csv", index=False)
    print("  colour-choice robustness (same process window, different dye):")
    for r in dye_rows:
        print(f"    {r['dye']:15s} mean dE={r['mean_dE00']:.3f} "
              f"sigma={r['sigma_dE00']:.3f} Cpk={r['cpk']:7.2f} "
              f"yield={r['yield_pct']:6.2f} %")

    # ---- figures ----------------------------------------------------------
    wl = cm.WAVELENGTHS

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    order = dye_df.sort_values("cpk")
    bars = ax.barh(order["dye"], order["cpk"].clip(upper=6.0), color="C0")
    ax.axvline(TARGET_CPK, color="C3", lw=1.5, label=f"target Cpk {TARGET_CPK}")
    for bar, c in zip(bars, order["cpk"]):
        ax.text(min(c, 6.0) + 0.08, bar.get_y() + bar.get_height() / 2,
                f"{c:.2f}" if c < 6 else f"{c:.0f}", va="center", fontsize=8)
    ax.set(xlabel="colour capability $C_{pk}$ (clipped at 6)",
           title="Identical process window, different dye")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG / "anodize_dye_robustness.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for T in (17, 20, 23):
        p = an.AnodizeProcess(bath_temp_C=T)
        axes[0].plot(wl, p.reflectance() * 100, label=f"{T} $^\\circ$C")
    axes[0].set(xlabel="wavelength (nm)", ylabel="reflectance (%)",
                title="Dyed anodic film: bath temperature")
    axes[0].legend(frameon=False)

    Ts = np.linspace(16, 26, 60)
    th = [an.AnodizeProcess(bath_temp_C=t).thickness_um() for t in Ts]
    po = [an.AnodizeProcess(bath_temp_C=t).porosity() for t in Ts]
    axes[1].plot(Ts, th, color="C0")
    axes[1].set(xlabel="bath temperature ($^\\circ$C)", ylabel="oxide thickness ($\\mu$m)")
    ax2 = axes[1].twinx()
    ax2.plot(Ts, po, color="C3", ls="--")
    ax2.set_ylabel("porosity", color="C3")
    ax2.grid(False)
    axes[1].set_title("Growth vs back-dissolution")
    fig.savefig(FIG / "anodize_physics.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].hist(mc["dE00"], bins=60, color="C0", alpha=0.85)
    axes[0].axvline(DE_SPEC, color="C3", lw=1.5, label=f"spec {DE_SPEC}")
    axes[0].set(xlabel="$\\Delta E_{00}$", ylabel="count", title="Colour capability")
    axes[0].legend(frameon=False)

    axes[1].hist(mc["thickness_um"], bins=60, color="C2", alpha=0.85)
    for s in (THICK_LSL, THICK_USL):
        axes[1].axvline(s, color="C3", lw=1.5)
    axes[1].set(xlabel="thickness ($\\mu$m)", title="Thickness capability")

    s = sens.head(6).iloc[::-1]
    axes[2].barh(s["variable"], s["contribution_pct"], color="C4")
    axes[2].set(xlabel="contribution to $\\Delta E_{00}$ (%)", title="Sensitivity Pareto")
    fig.savefig(FIG / "anodize_capability.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    labs, labels = [], []
    for T in (17, 19, 20, 21, 23):
        p = an.AnodizeProcess(bath_temp_C=T)
        labs.append(p.lab())
        labels.append(f"{T}$^\\circ$C\n$\\Delta E$={cm.delta_E_2000(ANODIZE_MASTER, p.lab()):.2f}")
    swatch_row(ax, labs, labels, "Rendered colour vs bath temperature (black dye)")
    fig.savefig(FIG / "anodize_swatches.png")
    plt.close(fig)

    return {
        "master_lab": ANODIZE_MASTER.tolist(),
        "doe_runs": int(len(results)),
        "rsm": {k: {"r2": v["r2"], "r2_adj": v["r2_adj"], "rmse": v["rmse"],
                    "coefficients": v["coefficients"]} for k, v in fits.items()},
        "capability": report,
        "sensitivity": sens.to_dict("records"),
        "tightening": history,
        "final_tolerances": {v.name: v.tolerance for v in tightened},
        "dye_robustness": dye_rows,
        "jmp": jmp,
    }


# ===========================================================================
# PART 2 -- PVD
# ===========================================================================
PVD_MASTER_STACK = pvd.Stack(
    layers=[pvd.Layer("SiO2", 90.0), pvd.Layer("TiN", 300.0, 1.0), pvd.Layer("Ti", 80.0)],
    substrate="SS304",
)
PVD_MASTER = pvd.lab(PVD_MASTER_STACK)


def pvd_forward(state: dict) -> dict:
    stack = pvd.Stack(
        layers=[
            pvd.Layer("SiO2", state.get("overcoat_nm", 90.0)),
            pvd.Layer("TiN", state.get("tin_nm", 300.0), state.get("stoichiometry", 1.0)),
            pvd.Layer("Ti", 80.0),
        ],
        substrate="SS304",
    )
    lab = pvd.lab(stack, angle_deg=8.0)
    flop = pvd.angular_colour_shift(stack, (8.0, 45.0))[45.0]
    return {
        "L": lab[0], "a": lab[1], "b": lab[2],
        "dE00": cm.delta_E_2000(PVD_MASTER, lab),
        "flop45": flop,
    }


def run_pvd():
    print("\n=== PVD: SiO2 / TiN / Ti on stainless ===")
    print(f"colour master L*a*b* = {np.round(PVD_MASTER, 3).tolist()}")

    factors = [
        doe.Factor("stoichiometry", 0.96, 1.04, "x in TiN_x"),
        doe.Factor("overcoat_nm", 82.0, 98.0, "nm"),
        doe.Factor("tin_nm", 280.0, 320.0, "nm"),
    ]
    design = doe.build_design(factors, "box-behnken", n_centre=3)
    results = doe.run_design(design, factors, pvd_forward)
    results.to_csv(OUT / "pvd_doe.csv", index=False)
    print(f"Box-Behnken: {len(results)} runs")

    fits = {}
    for resp in ("dE00", "flop45"):
        f = doe.fit_response_surface(results, factors, resp)
        fits[resp] = f
        print(f"  RSM {resp:8s} R2={f['r2']:.4f}  adj={f['r2_adj']:.4f}  RMSE={f['rmse']:.4f}")
        print("    top terms:", doe.effect_table(f).head(3)["term"].tolist())

    jmp = doe.export_jmp(results, factors, ["dE00", "flop45"], OUT, "pvd_doe")

    variables = [
        cap.ProcessVariable("stoichiometry", 1.0, 0.04, units="x"),
        cap.ProcessVariable("overcoat_nm", 90.0, 8.0, units="nm"),
        cap.ProcessVariable("tin_nm", 300.0, 20.0, units="nm"),
    ]
    mc = cap.monte_carlo(pvd_forward, variables, n=6000)
    mc.to_csv(OUT / "pvd_montecarlo.csv", index=False)

    names = [v.name for v in variables]
    specs = {"dE00": (None, DE_SPEC)}
    report = cap.capability_report(mc, specs, names)
    print(f"  baseline yield        {report['overall_yield']*100:.2f} %")
    print(f"  dE00 mean={report['responses']['dE00']['mean']:.3f} "
          f"sigma={report['responses']['dE00']['sigma']:.3f} "
          f"Cpk={report['responses']['dE00']['cpk']:.3f}")

    sens = cap.sensitivity(mc, "dE00", names)
    print("  dE00 drivers:", sens[["variable", "contribution_pct"]].to_dict("records"))

    tightened, history = cap.tighten_to_target(
        pvd_forward, variables, "dE00", DE_SPEC, TARGET_CPK
    )
    print(f"  tightening rounds: {len(history)}, final Cpk={history[-1]['cpk']:.3f}")
    for v0, v1 in zip(variables, tightened):
        if abs(v1.tolerance - v0.tolerance) > 1e-9:
            print(f"    {v0.name:16s} +/-{v0.tolerance:7.4f} -> +/-{v1.tolerance:7.4f} {v0.units}")

    # ---- figures ----------------------------------------------------------
    wl = cm.WAVELENGTHS
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for x in (0.96, 1.00, 1.04):
        st = pvd.Stack(layers=[pvd.Layer("SiO2", 90.0), pvd.Layer("TiN", 300.0, x),
                               pvd.Layer("Ti", 80.0)], substrate="SS304")
        axes[0].plot(wl, pvd.reflectance(st) * 100, label=f"x = {x:.2f}")
    axes[0].set(xlabel="wavelength (nm)", ylabel="reflectance (%)",
                title="TiN$_x$ stoichiometry")
    axes[0].legend(frameon=False)

    for ang in (8, 30, 45, 60):
        axes[1].plot(wl, pvd.reflectance(PVD_MASTER_STACK, angle_deg=ang) * 100,
                     label=f"{ang}$^\\circ$")
    axes[1].set(xlabel="wavelength (nm)", ylabel="reflectance (%)",
                title="Angular flop of the interference stack")
    axes[1].legend(frameon=False)
    fig.savefig(FIG / "pvd_physics.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].hist(mc["dE00"], bins=60, color="C1", alpha=0.85)
    axes[0].axvline(DE_SPEC, color="C3", lw=1.5, label=f"spec {DE_SPEC}")
    axes[0].set(xlabel="$\\Delta E_{00}$", ylabel="count", title="PVD colour capability")
    axes[0].legend(frameon=False)

    s = sens.iloc[::-1]
    axes[1].barh(s["variable"], s["contribution_pct"], color="C4")
    axes[1].set(xlabel="contribution to $\\Delta E_{00}$ (%)", title="Sensitivity Pareto")

    xs = np.linspace(0.94, 1.06, 90)
    de = [cm.delta_E_2000(PVD_MASTER, pvd.lab(pvd.Stack(
        layers=[pvd.Layer("SiO2", 90.0), pvd.Layer("TiN", 300.0, float(x)),
                pvd.Layer("Ti", 80.0)], substrate="SS304"))) for x in xs]
    axes[2].plot(xs, de, color="C1")
    axes[2].axhline(DE_SPEC, color="C3", lw=1.2)
    axes[2].set(xlabel="stoichiometry $x$ in TiN$_x$", ylabel="$\\Delta E_{00}$",
                title="Colour vs nitrogen control")
    fig.savefig(FIG / "pvd_capability.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    labs, labels = [], []
    for x in (0.92, 0.96, 1.00, 1.04, 1.08):
        st = pvd.Stack(layers=[pvd.Layer("SiO2", 90.0), pvd.Layer("TiN", 300.0, x),
                               pvd.Layer("Ti", 80.0)], substrate="SS304")
        lab = pvd.lab(st)
        labs.append(lab)
        labels.append(f"x={x:.2f}\n$\\Delta E$={cm.delta_E_2000(PVD_MASTER, lab):.1f}")
    swatch_row(ax, labs, labels, "Rendered PVD colour vs TiN$_x$ stoichiometry")
    fig.savefig(FIG / "pvd_swatches.png")
    plt.close(fig)

    return {
        "master_lab": PVD_MASTER.tolist(),
        "master_stack": PVD_MASTER_STACK.describe(),
        "doe_runs": int(len(results)),
        "rsm": {k: {"r2": v["r2"], "r2_adj": v["r2_adj"], "rmse": v["rmse"],
                    "coefficients": v["coefficients"]} for k, v in fits.items()},
        "capability": report,
        "sensitivity": sens.to_dict("records"),
        "tightening": history,
        "final_tolerances": {v.name: v.tolerance for v in tightened},
        "jmp": jmp,
    }


def main():
    print("=" * 74)
    print("Cosmetic finish process-capability study")
    print(f"CIE data source: {cm.CMF_SOURCE}")
    print(f"spec: dE00 <= {DE_SPEC}, thickness {THICK_LSL}-{THICK_USL} um, "
          f"target Cpk {TARGET_CPK}")
    print("=" * 74)

    results = {
        "spec": {"dE00_usl": DE_SPEC, "thickness_lsl": THICK_LSL,
                 "thickness_usl": THICK_USL, "target_cpk": TARGET_CPK},
        "cie_source": cm.CMF_SOURCE,
        "anodize": run_anodize(),
        "pvd": run_pvd(),
    }

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))

    (OUT / "results.json").write_text(
        json.dumps(results, indent=2, default=default), encoding="utf-8"
    )
    print(f"\nWrote {OUT/'results.json'} and figures to {FIG}")


if __name__ == "__main__":
    main()
