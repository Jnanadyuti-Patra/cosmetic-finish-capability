"""Interactive cosmetic-finish capability dashboard.

    streamlit run streamlit_app.py
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

st.set_page_config(page_title="Cosmetic Finish Capability", layout="wide")

DE_SPEC_DEFAULT = 1.0


def swatch(lab, caption, size=110):
    st.markdown(
        f"<div style='width:{size}px;height:{size}px;border-radius:8px;"
        f"background:{cm.hex_swatch(lab)};border:1px solid #888'></div>"
        f"<div style='font-size:0.78rem;margin-top:.35rem'>{caption}</div>",
        unsafe_allow_html=True,
    )


def spectrum_fig(curves: dict, title: str):
    fig = go.Figure()
    for name, R in curves.items():
        fig.add_trace(go.Scatter(x=cm.WAVELENGTHS, y=np.asarray(R) * 100,
                                 mode="lines", name=name))
    fig.update_layout(
        title=title, xaxis_title="wavelength (nm)", yaxis_title="reflectance (%)",
        height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"),
    )
    return fig


def capability_fig(values, usl, title, lsl=None):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=60, name="simulated parts"))
    if usl is not None:
        fig.add_vline(x=usl, line_color="crimson", line_width=2,
                      annotation_text=f"USL {usl:g}")
    if lsl is not None:
        fig.add_vline(x=lsl, line_color="crimson", line_width=2,
                      annotation_text=f"LSL {lsl:g}")
    fig.update_layout(title=title, height=340, showlegend=False,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def pareto_fig(sens: pd.DataFrame, title: str):
    s = sens.iloc[::-1]
    fig = go.Figure(go.Bar(x=s["contribution_pct"], y=s["variable"], orientation="h"))
    fig.update_layout(title=title, xaxis_title="contribution (%)", height=340,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


st.title("Cosmetic finish process capability")
st.caption(
    "Two finishing routes, one colour engine: "
    "process variation → physics → R(λ) → CIELAB → "
    "ΔE₀₀ → Cpk → yield.  "
    f"CIE data: {cm.CMF_SOURCE}."
)

route = st.sidebar.radio("Finishing route", ["Anodize + dye", "PVD thin film"])
de_spec = st.sidebar.number_input("ΔE00 specification (upper)", 0.2, 5.0,
                                  DE_SPEC_DEFAULT, 0.1)
n_mc = st.sidebar.select_slider("Monte-Carlo samples", [500, 1000, 2500, 5000], 2500)
target_cpk = st.sidebar.number_input("Target Cpk", 0.5, 2.5, 1.33, 0.01)

# ===========================================================================
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
    tol = {
        "current_density": st.sidebar.number_input("± current density", 0.0, 0.5, 0.10, 0.01),
        "bath_temp": st.sidebar.number_input("± bath temp", 0.0, 5.0, 1.5, 0.1),
        "acid_conc": st.sidebar.number_input("± acid conc", 0.0, 30.0, 10.0, 1.0),
        "dye_time": st.sidebar.number_input("± dye time", 0.0, 5.0, 1.0, 0.1),
        "dye_temp": st.sidebar.number_input("± dye temp", 0.0, 8.0, 2.0, 0.5),
        "dye_conc": st.sidebar.number_input("± dye conc", 0.0, 2.0, 0.3, 0.05),
        "dye_pH": st.sidebar.number_input("± dye pH", 0.0, 1.5, 0.3, 0.05),
    }

    nominal = an.AnodizeProcess(
        current_density_A_dm2=j, bath_temp_C=T, acid_conc_gL=C, anodize_time_min=t,
        dye=dye_name, dye_time_min=dt, dye_temp_C=dT, dye_conc_gL=dC, dye_pH=pH,
    )
    master = nominal.lab()
    s = nominal.summary()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("oxide thickness", f"{s['thickness_um']:.2f} μm")
    k2.metric("cell voltage", f"{s['voltage_V']:.1f} V")
    k3.metric("porosity", f"{s['porosity']*100:.1f} %")
    k4.metric("current efficiency", f"{s['current_efficiency']*100:.0f} %")
    st.caption(
        f"growth {s['growth_um_min']:.3f} μm/min vs back-dissolution "
        f"{s['dissolution_um_min']:.3f} μm/min · "
        f"barrier layer {s['barrier_nm']:.1f} nm"
    )

    def forward(state):
        p = an.AnodizeProcess(
            current_density_A_dm2=state.get("current_density", j),
            bath_temp_C=state.get("bath_temp", T),
            acid_conc_gL=state.get("acid_conc", C),
            anodize_time_min=t,
            dye=dye_name,
            dye_time_min=state.get("dye_time", dt),
            dye_temp_C=state.get("dye_temp", dT),
            dye_conc_gL=state.get("dye_conc", dC),
            dye_pH=state.get("dye_pH", pH),
        )
        lab = p.lab()
        return {"dE00": cm.delta_E_2000(master, lab),
                "thickness_um": p.thickness_um()}

    variables = [
        cap.ProcessVariable(k, v_nom, tol[k])
        for k, v_nom in [("current_density", j), ("bath_temp", T), ("acid_conc", C),
                         ("dye_time", dt), ("dye_temp", dT), ("dye_conc", dC),
                         ("dye_pH", pH)]
    ]
    mc = cap.monte_carlo(forward, variables, n=n_mc)
    names = [v.name for v in variables]

    left, right = st.columns([1, 3])
    with left:
        swatch(master, f"L* {master[0]:.1f}  a* {master[1]:.1f}  b* {master[2]:.1f}")
    with right:
        st.plotly_chart(
            spectrum_fig(
                {f"{TT:g} °C": an.AnodizeProcess(
                    current_density_A_dm2=j, bath_temp_C=TT, acid_conc_gL=C,
                    anodize_time_min=t, dye=dye_name, dye_time_min=dt,
                    dye_temp_C=dT, dye_conc_gL=dC, dye_pH=pH).reflectance()
                 for TT in (T - 3, T, T + 3)},
                "Reflectance sensitivity to bath temperature"),
            use_container_width=True,
        )

    cpk_de = cap.cpk(mc["dE00"], None, de_spec)
    cpk_th = cap.cpk(mc["thickness_um"], 8.0, 12.0)
    y = cap.yield_fraction(mc, {"dE00": (None, de_spec), "thickness_um": (8.0, 12.0)})

    m1, m2, m3 = st.columns(3)
    m1.metric("colour Cpk", f"{cpk_de:.2f}", delta=f"{cpk_de-target_cpk:+.2f} vs target")
    m2.metric("thickness Cpk", f"{cpk_th:.2f}", delta=f"{cpk_th-target_cpk:+.2f} vs target")
    m3.metric("combined yield", f"{y*100:.2f} %")

    a1, a2, a3 = st.columns(3)
    a1.plotly_chart(capability_fig(mc["dE00"], de_spec, "Colour capability"),
                    use_container_width=True)
    a2.plotly_chart(capability_fig(mc["thickness_um"], 12.0, "Thickness capability", 8.0),
                    use_container_width=True)
    a3.plotly_chart(pareto_fig(cap.sensitivity(mc, "dE00", names),
                               "What drives colour variation"), use_container_width=True)

    st.subheader("Is the colour itself the problem?")
    st.caption(
        "Same process window, different dye.  A saturated black sits deep in "
        "Beer-Lambert saturation and barely moves; pale and chromatic finishes "
        "sit on the steep part of the curve."
    )
    rows = []
    for d in an.DYES:
        if d == "none (clear/silver)":
            continue
        m_lab = an.AnodizeProcess(
            current_density_A_dm2=j, bath_temp_C=T, acid_conc_gL=C, anodize_time_min=t,
            dye=d, dye_time_min=dt, dye_temp_C=dT, dye_conc_gL=dC, dye_pH=pH).lab()

        def f2(state, _d=d, _m=m_lab):
            p = an.AnodizeProcess(
                current_density_A_dm2=state.get("current_density", j),
                bath_temp_C=state.get("bath_temp", T),
                acid_conc_gL=state.get("acid_conc", C), anodize_time_min=t, dye=_d,
                dye_time_min=state.get("dye_time", dt),
                dye_temp_C=state.get("dye_temp", dT),
                dye_conc_gL=state.get("dye_conc", dC),
                dye_pH=state.get("dye_pH", pH))
            return {"dE00": cm.delta_E_2000(_m, p.lab())}

        mm = cap.monte_carlo(f2, variables, n=min(n_mc, 1200))
        rows.append({"dye": d, "swatch": cm.hex_swatch(m_lab),
                     "mean dE00": mm["dE00"].mean(),
                     "Cpk": cap.cpk(mm["dE00"], None, de_spec),
                     "yield %": 100 * cap.yield_fraction(mm, {"dE00": (None, de_spec)})})
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {"mean dE00": "{:.3f}", "Cpk": "{:.2f}", "yield %": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )

# ===========================================================================
else:
    st.sidebar.subheader("Stack (outermost first)")
    preset = st.sidebar.selectbox("preset", list(pvd.PRESET_STACKS),
                                  index=len(pvd.PRESET_STACKS) - 1)
    base = pvd.PRESET_STACKS[preset]
    overcoat = st.sidebar.number_input("SiO₂ overcoat (nm)", 0.0, 300.0, 90.0, 1.0)
    tin_nm = st.sidebar.number_input("TiN thickness (nm)", 50.0, 800.0, 300.0, 5.0)
    stoich = st.sidebar.number_input("stoichiometry x in TiNₓ", 0.80, 1.20, 1.00, 0.005)
    angle = st.sidebar.slider("viewing angle (°)", 0, 75, 8)

    st.sidebar.subheader("Control tolerances (±, = 3σ)")
    tol_st = st.sidebar.number_input("± stoichiometry", 0.0, 0.10, 0.04, 0.001,
                                     format="%.3f")
    tol_oc = st.sidebar.number_input("± overcoat (nm)", 0.0, 25.0, 8.0, 0.5)
    tol_tn = st.sidebar.number_input("± TiN thickness (nm)", 0.0, 60.0, 20.0, 1.0)

    def mk(oc, tn, x):
        return pvd.Stack(
            layers=[pvd.Layer("SiO2", oc), pvd.Layer("TiN", tn, x), pvd.Layer("Ti", 80.0)],
            substrate=base.substrate)

    master_stack = mk(overcoat, tin_nm, stoich)
    master = pvd.lab(master_stack, angle_deg=8.0)
    st.caption(f"stack: {master_stack.describe()}")

    left, right = st.columns([1, 3])
    with left:
        swatch(master, f"L* {master[0]:.1f}  a* {master[1]:.1f}  b* {master[2]:.1f}")
        flop = pvd.angular_colour_shift(master_stack, (8.0, 30.0, 45.0, 60.0))
        st.caption("angular flop ΔE₀₀ vs 8°<br>"
                   + "<br>".join(f"{k:g}° : {v:.2f}" for k, v in flop.items()),
                   unsafe_allow_html=True)
    with right:
        st.plotly_chart(
            spectrum_fig({f"x = {xx:.2f}": pvd.reflectance(mk(overcoat, tin_nm, xx),
                                                           angle_deg=angle)
                          for xx in (stoich - 0.04, stoich, stoich + 0.04)},
                         f"Reflectance vs TiNₓ stoichiometry at {angle}°"),
            use_container_width=True)

    def forward(state):
        st_stack = mk(state.get("overcoat_nm", overcoat),
                      state.get("tin_nm", tin_nm),
                      state.get("stoichiometry", stoich))
        lab = pvd.lab(st_stack, angle_deg=8.0)
        return {"dE00": cm.delta_E_2000(master, lab),
                "flop45": pvd.angular_colour_shift(st_stack, (8.0, 45.0))[45.0]}

    variables = [
        cap.ProcessVariable("stoichiometry", stoich, tol_st),
        cap.ProcessVariable("overcoat_nm", overcoat, tol_oc),
        cap.ProcessVariable("tin_nm", tin_nm, tol_tn),
    ]
    mc = cap.monte_carlo(forward, variables, n=n_mc)
    names = [v.name for v in variables]

    cpk_de = cap.cpk(mc["dE00"], None, de_spec)
    y = cap.yield_fraction(mc, {"dE00": (None, de_spec)})
    m1, m2, m3 = st.columns(3)
    m1.metric("colour Cpk", f"{cpk_de:.2f}", delta=f"{cpk_de-target_cpk:+.2f} vs target")
    m2.metric("yield", f"{y*100:.2f} %")
    m3.metric("mean ΔE₀₀", f"{mc['dE00'].mean():.3f}")

    b1, b2 = st.columns(2)
    b1.plotly_chart(capability_fig(mc["dE00"], de_spec, "Colour capability"),
                    use_container_width=True)
    b2.plotly_chart(pareto_fig(cap.sensitivity(mc, "dE00", names),
                               "What drives colour variation"), use_container_width=True)

    st.subheader("What tolerance would actually hold the spec?")
    if st.button(f"Solve for Cpk ≥ {target_cpk:.2f}"):
        with st.spinner("tightening the dominant tolerance..."):
            tightened, hist = cap.tighten_to_target(
                forward, variables, "dE00", de_spec, target_cpk, n=1200)
        st.write(pd.DataFrame([
            {"variable": v0.name, "current ±": v0.tolerance,
             "required ±": v1.tolerance,
             "factor": (v0.tolerance / v1.tolerance) if v1.tolerance else float("nan")}
            for v0, v1 in zip(variables, tightened)]))
        st.line_chart(pd.DataFrame(hist).set_index("round")[["cpk"]])
