"""Interactive cosmetic-finish capability dashboard.

    streamlit run streamlit_app.py

Every Monte-Carlo run is wrapped in st.cache_data keyed on the scalar inputs.
Without that, each slider movement re-runs the main simulation plus five more
for the dye comparison, which is roughly 12000 forward physics evaluations and
makes the app unusable on Streamlit Community Cloud's free tier. With caching,
only the settings that actually changed are recomputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finishsim import anodize as an
from finishsim import capability as cap
from finishsim import colourimetry as cm
from finishsim import pvd

st.set_page_config(
    page_title="Cosmetic Finish Capability",
    page_icon="\U0001F3AF",
    layout="wide",
)

DE_SPEC_DEFAULT = 1.0
THICK_LSL, THICK_USL = 8.0, 12.0
ANODIZE_TOL_ORDER = (
    "current_density", "bath_temp", "acid_conc",
    "dye_time", "dye_temp", "dye_conc", "dye_pH",
)


# ===========================================================================
# Cached simulation layer
# ===========================================================================
def _anodize_process(j, T, C, t, dye, dt, dT, dC, pH, **over):
    return an.AnodizeProcess(
        current_density_A_dm2=over.get("current_density", j),
        bath_temp_C=over.get("bath_temp", T),
        acid_conc_gL=over.get("acid_conc", C),
        anodize_time_min=t,
        dye=dye,
        dye_time_min=over.get("dye_time", dt),
        dye_temp_C=over.get("dye_temp", dT),
        dye_conc_gL=over.get("dye_conc", dC),
        dye_pH=over.get("dye_pH", pH),
    )


def _anodize_vars(j, T, C, dt, dT, dC, pH, tol):
    nominal = dict(zip(ANODIZE_TOL_ORDER, (j, T, C, dt, dT, dC, pH)))
    return [cap.ProcessVariable(k, nominal[k], tol[i])
            for i, k in enumerate(ANODIZE_TOL_ORDER)]


@st.cache_data(show_spinner=False, max_entries=48)
def anodize_mc(j, T, C, t, dye, dt, dT, dC, pH, tol, n):
    """Monte-Carlo the anodizing route. ``tol`` is a tuple in ANODIZE_TOL_ORDER."""
    master = _anodize_process(j, T, C, t, dye, dt, dT, dC, pH).lab()

    def forward(state):
        p = _anodize_process(j, T, C, t, dye, dt, dT, dC, pH, **state)
        return {"dE00": cm.delta_E_2000(master, p.lab()),
                "thickness_um": p.thickness_um()}

    rows = cap.monte_carlo(forward, _anodize_vars(j, T, C, dt, dT, dC, pH, tol), n=n)
    return rows, list(master)


@st.cache_data(show_spinner=False, max_entries=24)
def dye_comparison(j, T, C, t, dt, dT, dC, pH, tol, n, de_spec):
    """Capability of every dye under one fixed process window."""
    out = []
    variables = _anodize_vars(j, T, C, dt, dT, dC, pH, tol)
    for d in an.DYES:
        if d == "none (clear/silver)":
            continue
        master = _anodize_process(j, T, C, t, d, dt, dT, dC, pH).lab()

        def forward(state, _d=d, _m=master):
            p = _anodize_process(j, T, C, t, _d, dt, dT, dC, pH, **state)
            return {"dE00": cm.delta_E_2000(_m, p.lab())}

        mm = cap.monte_carlo(forward, variables, n=n)
        out.append({
            "dye": d,
            "colour": cm.hex_swatch(master),
            "mean dE00": float(mm["dE00"].mean()),
            "Cpk": cap.cpk(mm["dE00"], None, de_spec),
            "yield %": 100 * cap.yield_fraction(mm, {"dE00": (None, de_spec)}),
        })
    return pd.DataFrame(out)


def _pvd_stack(oc, tn, x, substrate="SS304"):
    return pvd.Stack(
        layers=[pvd.Layer("SiO2", oc), pvd.Layer("TiN", tn, x), pvd.Layer("Ti", 80.0)],
        substrate=substrate,
    )


@st.cache_data(show_spinner=False, max_entries=48)
def pvd_mc(oc, tn, x, tol_oc, tol_tn, tol_x, n):
    master = pvd.lab(_pvd_stack(oc, tn, x), angle_deg=8.0)

    # Only dE00 is simulated here. Angular flop is reported for the nominal
    # stack alone, so computing it per sample would double the transfer-matrix
    # work for a column nothing reads.
    def forward(state):
        stack = _pvd_stack(state["overcoat_nm"], state["tin_nm"], state["stoichiometry"])
        return {"dE00": cm.delta_E_2000(master, pvd.lab(stack, angle_deg=8.0))}

    variables = [
        cap.ProcessVariable("stoichiometry", x, tol_x),
        cap.ProcessVariable("overcoat_nm", oc, tol_oc),
        cap.ProcessVariable("tin_nm", tn, tol_tn),
    ]
    return cap.monte_carlo(forward, variables, n=n), list(master)


@st.cache_data(show_spinner=False, max_entries=16)
def solve_tolerance(oc, tn, x, tol_oc, tol_tn, tol_x, de_spec, target):
    master = pvd.lab(_pvd_stack(oc, tn, x), angle_deg=8.0)

    def forward(state):
        stack = _pvd_stack(state["overcoat_nm"], state["tin_nm"], state["stoichiometry"])
        return {"dE00": cm.delta_E_2000(master, pvd.lab(stack, angle_deg=8.0))}

    variables = [
        cap.ProcessVariable("stoichiometry", x, tol_x),
        cap.ProcessVariable("overcoat_nm", oc, tol_oc),
        cap.ProcessVariable("tin_nm", tn, tol_tn),
    ]
    tightened, hist = cap.tighten_to_target(
        forward, variables, "dE00", de_spec, target, n=1000)
    return (
        pd.DataFrame([
            {"variable": a.name, "current": a.tolerance, "required": b.tolerance,
             "factor": (a.tolerance / b.tolerance) if b.tolerance else float("nan")}
            for a, b in zip(variables, tightened)]),
        pd.DataFrame(hist),
    )


@st.cache_data(show_spinner=False, max_entries=64)
def anodize_spectra(j, T, C, t, dye, dt, dT, dC, pH, temps):
    return {f"{TT:g} °C":
            list(_anodize_process(j, TT, C, t, dye, dt, dT, dC, pH).reflectance())
            for TT in temps}


@st.cache_data(show_spinner=False, max_entries=64)
def pvd_spectra(oc, tn, xs, angle):
    return {f"x = {xx:.2f}": list(pvd.reflectance(_pvd_stack(oc, tn, xx), angle_deg=angle))
            for xx in xs}


# ===========================================================================
# Presentation helpers
# ===========================================================================
def swatch(lab, caption, size=112):
    st.markdown(
        f"<div style='width:{size}px;height:{size}px;border-radius:10px;"
        f"background:{cm.hex_swatch(lab)};border:1px solid #8888'></div>"
        f"<div style='font-size:0.78rem;margin-top:.4rem;opacity:.85'>{caption}</div>",
        unsafe_allow_html=True,
    )


def spectrum_fig(curves, title):
    fig = go.Figure()
    for name, R in curves.items():
        fig.add_trace(go.Scatter(x=cm.WAVELENGTHS, y=np.asarray(R) * 100,
                                 mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="wavelength (nm)",
                      yaxis_title="reflectance (%)", height=330,
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.22))
    return fig


def capability_fig(values, usl, title, lsl=None):
    fig = go.Figure(go.Histogram(x=values, nbinsx=60))
    if usl is not None:
        fig.add_vline(x=usl, line_color="crimson", line_width=2,
                      annotation_text=f"USL {usl:g}")
    if lsl is not None:
        fig.add_vline(x=lsl, line_color="crimson", line_width=2,
                      annotation_text=f"LSL {lsl:g}")
    fig.update_layout(title=title, height=330, showlegend=False,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def pareto_fig(sens, title):
    s = sens.iloc[::-1]
    fig = go.Figure(go.Bar(x=s["contribution_pct"], y=s["variable"], orientation="h"))
    fig.update_layout(title=title, xaxis_title="contribution (%)", height=330,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ===========================================================================
# Page
# ===========================================================================
st.title("Cosmetic finish process capability")
st.caption(
    "Two finishing routes, one colour engine:  "
    "process variation → physics → R(λ) → CIELAB → "
    "ΔE₀₀ → Cpk → yield.  \n"
    f"CIE data: {cm.CMF_SOURCE}.  "
    "Full method and limitations: see the technical report in the repository."
)

route = st.sidebar.radio("Finishing route", ["Anodize + dye", "PVD thin film"])
de_spec = st.sidebar.number_input("ΔE₀₀ specification (upper)",
                                  0.2, 5.0, DE_SPEC_DEFAULT, 0.1)
# Default deliberately modest: this is the first thing a visitor waits for on
# Streamlit Community Cloud's shared CPU. 1000 samples already resolves Cpk to
# about two decimal places; the slider is there for anyone who wants tighter.
n_mc = st.sidebar.select_slider("Monte-Carlo samples",
                                [500, 1000, 2000, 4000], 1000)
target_cpk = st.sidebar.number_input("Target Cpk", 0.5, 2.5, 1.33, 0.01)
st.sidebar.divider()

# ---------------------------------------------------------------- anodizing
if route == "Anodize + dye":
    st.sidebar.subheader("Nominal process")
    c1, c2 = st.sidebar.columns(2)
    j = c1.number_input("current density (A/dm²)", 0.5, 3.0, 1.5, 0.05)
    T = c2.number_input("bath temp (°C)", 10.0, 30.0, 20.0, 0.5)
    C = c1.number_input("H₂SO₄ (g/L)", 120.0, 240.0, 180.0, 5.0)
    t = c2.number_input("anodize time (min)", 5.0, 90.0, 30.0, 1.0)
    dye_name = st.sidebar.selectbox("dye", list(an.DYES), index=1)
    dt = c1.number_input("dye time (min)", 1.0, 40.0, 12.0, 0.5)
    dT = c2.number_input("dye temp (°C)", 30.0, 75.0, 55.0, 1.0)
    dC = c1.number_input("dye conc (g/L)", 0.5, 10.0, 3.0, 0.1)
    pH = c2.number_input("dye pH", 3.0, 8.0, 5.5, 0.1)

    st.sidebar.subheader("Control tolerances (±, = 3σ)")
    tol = (
        st.sidebar.number_input("± current density", 0.0, 0.5, 0.10, 0.01),
        st.sidebar.number_input("± bath temp", 0.0, 5.0, 1.5, 0.1),
        st.sidebar.number_input("± acid conc", 0.0, 30.0, 10.0, 1.0),
        st.sidebar.number_input("± dye time", 0.0, 5.0, 1.0, 0.1),
        st.sidebar.number_input("± dye temp", 0.0, 8.0, 2.0, 0.5),
        st.sidebar.number_input("± dye conc", 0.0, 2.0, 0.3, 0.05),
        st.sidebar.number_input("± dye pH", 0.0, 1.5, 0.3, 0.05),
    )

    nominal = _anodize_process(j, T, C, t, dye_name, dt, dT, dC, pH)
    s = nominal.summary()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("oxide thickness", f"{s['thickness_um']:.2f} μm")
    k2.metric("cell voltage", f"{s['voltage_V']:.1f} V")
    k3.metric("porosity", f"{s['porosity'] * 100:.1f} %")
    k4.metric("current efficiency", f"{s['current_efficiency'] * 100:.0f} %")
    st.caption(
        f"growth {s['growth_um_min']:.3f} μm/min against back-dissolution "
        f"{s['dissolution_um_min']:.3f} μm/min · "
        f"barrier layer {s['barrier_nm']:.1f} nm"
    )

    with st.spinner("simulating..."):
        mc, master = anodize_mc(j, T, C, t, dye_name, dt, dT, dC, pH, tol, n_mc)

    left, right = st.columns([1, 3])
    with left:
        swatch(master, f"L* {master[0]:.1f}   a* {master[1]:.1f}   b* {master[2]:.1f}")
    with right:
        st.plotly_chart(
            spectrum_fig(anodize_spectra(j, T, C, t, dye_name, dt, dT, dC, pH,
                                         (T - 3, T, T + 3)),
                         "Reflectance sensitivity to bath temperature"),
            use_container_width=True)

    cpk_de = cap.cpk(mc["dE00"], None, de_spec)
    cpk_th = cap.cpk(mc["thickness_um"], THICK_LSL, THICK_USL)
    y = cap.yield_fraction(mc, {"dE00": (None, de_spec),
                                "thickness_um": (THICK_LSL, THICK_USL)})

    m1, m2, m3 = st.columns(3)
    m1.metric("colour Cpk", f"{cpk_de:.2f}", delta=f"{cpk_de - target_cpk:+.2f} vs target")
    m2.metric("thickness Cpk", f"{cpk_th:.2f}", delta=f"{cpk_th - target_cpk:+.2f} vs target")
    m3.metric("combined yield", f"{y * 100:.2f} %")

    a1, a2, a3 = st.columns(3)
    a1.plotly_chart(capability_fig(mc["dE00"], de_spec, "Colour capability"),
                    use_container_width=True)
    a2.plotly_chart(capability_fig(mc["thickness_um"], THICK_USL,
                                   "Thickness capability", THICK_LSL),
                    use_container_width=True)
    a3.plotly_chart(pareto_fig(cap.sensitivity(mc, "dE00", list(ANODIZE_TOL_ORDER)),
                               "What drives colour variation"),
                    use_container_width=True)

    st.subheader("Is the colour itself the problem?")
    st.caption(
        "Identical process window, identical tolerances, different dye. A saturated "
        "black sits deep in Beer-Lambert saturation and barely moves; pale and "
        "chromatic finishes sit on the steep part of the same exponential."
    )
    with st.spinner("comparing dyes..."):
        dyes = dye_comparison(j, T, C, t, dt, dT, dC, pH, tol,
                              min(n_mc, 1000), de_spec)
    st.dataframe(
        dyes,
        use_container_width=True, hide_index=True,
        column_config={
            "colour": st.column_config.TextColumn("hex"),
            "mean dE00": st.column_config.NumberColumn(format="%.3f"),
            "Cpk": st.column_config.NumberColumn(format="%.2f"),
            "yield %": st.column_config.NumberColumn(format="%.2f"),
        },
    )

# ---------------------------------------------------------------------- PVD
else:
    st.sidebar.subheader("Stack (outermost first)")
    overcoat = st.sidebar.number_input("SiO₂ overcoat (nm)", 0.0, 300.0, 90.0, 1.0)
    tin_nm = st.sidebar.number_input("TiN thickness (nm)", 50.0, 800.0, 300.0, 5.0)
    stoich = st.sidebar.number_input("stoichiometry x in TiNₓ",
                                     0.80, 1.20, 1.00, 0.005, format="%.3f")
    angle = st.sidebar.slider("viewing angle (°)", 0, 75, 8)

    st.sidebar.subheader("Control tolerances (±, = 3σ)")
    tol_x = st.sidebar.number_input("± stoichiometry", 0.0, 0.10, 0.04, 0.001,
                                    format="%.3f")
    tol_oc = st.sidebar.number_input("± overcoat (nm)", 0.0, 25.0, 8.0, 0.5)
    tol_tn = st.sidebar.number_input("± TiN thickness (nm)", 0.0, 60.0, 20.0, 1.0)

    stack = _pvd_stack(overcoat, tin_nm, stoich)
    st.caption(f"stack: {stack.describe()}")

    with st.spinner("simulating..."):
        mc, master = pvd_mc(overcoat, tin_nm, stoich, tol_oc, tol_tn, tol_x, n_mc)

    left, right = st.columns([1, 3])
    with left:
        swatch(master, f"L* {master[0]:.1f}   a* {master[1]:.1f}   b* {master[2]:.1f}")
        flop = pvd.angular_colour_shift(stack, (8.0, 30.0, 45.0, 60.0))
        st.caption(
            "angular flop ΔE₀₀ vs 8°<br>"
            + "<br>".join(f"{k:g}° : {v:.2f}" for k, v in flop.items()),
            unsafe_allow_html=True)
    with right:
        st.plotly_chart(
            spectrum_fig(pvd_spectra(overcoat, tin_nm,
                                     (round(stoich - 0.04, 3), stoich,
                                      round(stoich + 0.04, 3)), angle),
                         f"Reflectance vs TiNₓ stoichiometry at {angle}°"),
            use_container_width=True)

    cpk_de = cap.cpk(mc["dE00"], None, de_spec)
    y = cap.yield_fraction(mc, {"dE00": (None, de_spec)})
    m1, m2, m3 = st.columns(3)
    m1.metric("colour Cpk", f"{cpk_de:.2f}", delta=f"{cpk_de - target_cpk:+.2f} vs target")
    m2.metric("yield", f"{y * 100:.2f} %")
    m3.metric("mean ΔE₀₀", f"{mc['dE00'].mean():.3f}")

    b1, b2 = st.columns(2)
    b1.plotly_chart(capability_fig(mc["dE00"], de_spec, "Colour capability"),
                    use_container_width=True)
    b2.plotly_chart(
        pareto_fig(cap.sensitivity(mc, "dE00",
                                   ["stoichiometry", "overcoat_nm", "tin_nm"]),
                   "What drives colour variation"),
        use_container_width=True)

    st.subheader("What tolerance would actually hold the spec?")
    st.caption(
        "Repeatedly identifies the dominant contributor and tightens it, until the "
        "target capability is met."
    )
    if st.button(f"Solve for Cpk ≥ {target_cpk:.2f}", type="primary"):
        with st.spinner("tightening the dominant tolerance..."):
            table, hist = solve_tolerance(overcoat, tin_nm, stoich,
                                          tol_oc, tol_tn, tol_x, de_spec, target_cpk)
        st.dataframe(
            table, use_container_width=True, hide_index=True,
            column_config={
                "current": st.column_config.NumberColumn(format="%.4f"),
                "required": st.column_config.NumberColumn(format="%.4f"),
                "factor": st.column_config.NumberColumn("x tighter", format="%.2f"),
            })
        st.line_chart(hist.set_index("round")[["cpk"]], height=220)

st.divider()
st.caption(
    "Simulation only, no experimental validation. Optical constants are "
    "representative literature fits, not measurements of a specific supplier's "
    "coating, so trends are trustworthy but absolute L*a*b* values are not. "
    "Capability shown is short-term and within-batch."
)
