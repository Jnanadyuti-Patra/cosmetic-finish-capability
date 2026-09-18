"""Anodizing kinetics, capability statistics, and DOE machinery."""

import numpy as np
import pytest

from finishsim import anodize as an
from finishsim import capability as cap
from finishsim import colourimetry as cm
from finishsim import doe


# ---------------------------------------------------------------- anodizing
def test_nominal_recipe_hits_type_ii_thickness():
    """1.5 A/dm2 for 30 min at 20 C, 180 g/L is a ~10 um Type II film.

    This is the calibration anchor for the whole growth model.
    """
    assert an.AnodizeProcess().thickness_um() == pytest.approx(10.0, abs=1.0)


def test_thickness_grows_with_time():
    ts = [an.AnodizeProcess(anodize_time_min=t).thickness_um() for t in (10, 20, 30, 40)]
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_hotter_bath_thins_the_film():
    """Back-dissolution is Arrhenius, so a warm bath loses film."""
    hot = an.AnodizeProcess(bath_temp_C=26.0).thickness_um()
    cold = an.AnodizeProcess(bath_temp_C=16.0).thickness_um()
    assert hot < cold


def test_dissolution_roughly_doubles_per_ten_degrees():
    r20 = an.AnodizeProcess(bath_temp_C=20.0).dissolution_rate_m_s()
    r30 = an.AnodizeProcess(bath_temp_C=30.0).dissolution_rate_m_s()
    assert 1.8 < r30 / r20 < 3.0


def test_porosity_rises_with_temperature_and_acid():
    assert an.AnodizeProcess(bath_temp_C=25.0).porosity() > an.AnodizeProcess().porosity()
    assert an.AnodizeProcess(acid_conc_gL=210.0).porosity() > an.AnodizeProcess().porosity()


def test_voltage_rises_with_current_density():
    assert (an.AnodizeProcess(current_density_A_dm2=2.0).cell_voltage_V()
            > an.AnodizeProcess(current_density_A_dm2=1.0).cell_voltage_V())


def test_barrier_layer_follows_anodizing_ratio():
    """Sulfuric anodizing gives ~1.4 nm of barrier layer per volt."""
    p = an.AnodizeProcess()
    assert p.barrier_thickness_nm() / p.cell_voltage_V() == pytest.approx(1.4, abs=1e-9)


def test_undyed_film_is_light_and_neutral():
    lab = an.AnodizeProcess(dye="none (clear/silver)").lab()
    assert lab[0] > 70.0
    assert abs(lab[1]) < 5.0


def test_black_dye_is_dark():
    assert an.AnodizeProcess(dye="black").lab()[0] < 50.0


def test_longer_dye_immersion_saturates():
    """Uptake is exponential-approach, so the last minutes add little."""
    early = an.AnodizeProcess(dye_time_min=4.0).dye_loading()
    mid = an.AnodizeProcess(dye_time_min=12.0).dye_loading()
    late = an.AnodizeProcess(dye_time_min=30.0).dye_loading()
    assert early < mid < late
    assert (late - mid) < (mid - early)


def test_dye_uptake_peaks_near_ph_5_5():
    at_opt = an.AnodizeProcess(dye_pH=5.5).dye_loading()
    assert at_opt > an.AnodizeProcess(dye_pH=3.5).dye_loading()
    assert at_opt > an.AnodizeProcess(dye_pH=7.5).dye_loading()


def test_reflectance_is_physical():
    for dye in an.DYES:
        R = an.AnodizeProcess(dye=dye).reflectance()
        assert np.all(R >= 0.0) and np.all(R <= 1.0)


# --------------------------------------------------------------- capability
def test_cpk_of_a_centred_process():
    """Centred, sigma = 1, limits at +/-3 sigma  ->  Cpk = 1."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 200_000)
    assert cap.cpk(x, -3.0, 3.0) == pytest.approx(1.0, abs=0.02)


def test_cpk_one_sided():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 200_000)
    assert cap.cpk(x, None, 4.0) == pytest.approx(4.0 / 3.0, abs=0.02)


def test_offset_process_has_lower_cpk():
    rng = np.random.default_rng(2)
    centred = rng.normal(0.0, 1.0, 50_000)
    offset = rng.normal(1.0, 1.0, 50_000)
    assert cap.cpk(offset, -3.0, 3.0) < cap.cpk(centred, -3.0, 3.0)


def test_process_variable_treats_tolerance_as_three_sigma():
    v = cap.ProcessVariable("x", 10.0, tolerance=3.0)
    assert v.spread() == pytest.approx(1.0)
    s = v.sample(100_000, np.random.default_rng(3))
    assert s.std() == pytest.approx(1.0, abs=0.02)


def test_zero_tolerance_gives_a_constant():
    v = cap.ProcessVariable("x", 7.0, tolerance=0.0)
    assert np.all(v.sample(50, np.random.default_rng(4)) == 7.0)


def test_sensitivity_finds_the_dominant_input():
    """y depends on a 10x more strongly than b; the ranking must say so."""
    def forward(s):
        return {"y": 10.0 * s["a"] + 1.0 * s["b"]}

    variables = [cap.ProcessVariable("a", 0.0, 1.0), cap.ProcessVariable("b", 0.0, 1.0)]
    df = cap.monte_carlo(forward, variables, n=3000)
    sens = cap.sensitivity(df, "y", ["a", "b"])
    assert sens.iloc[0]["variable"] == "a"
    assert sens.iloc[0]["contribution_pct"] > sens.iloc[1]["contribution_pct"]


def test_yield_fraction_counts_both_specs():
    import pandas as pd

    df = pd.DataFrame({"u": [0.0, 1.0, 2.0, 3.0], "v": [0.0, 0.0, 0.0, 9.0]})
    assert cap.yield_fraction(df, {"u": (None, 2.0)}) == pytest.approx(0.75)
    assert cap.yield_fraction(df, {"u": (None, 2.0), "v": (None, 1.0)}) == pytest.approx(0.75)
    assert cap.yield_fraction(df, {"u": (None, 5.0), "v": (None, 1.0)}) == pytest.approx(0.75)


def test_tightening_improves_capability_and_reports_a_driver():
    def forward(s):
        return {"dE00": abs(5.0 * s["big"] + 0.05 * s["small"])}

    variables = [cap.ProcessVariable("big", 0.0, 0.3),
                 cap.ProcessVariable("small", 0.0, 0.3)]
    tightened, history = cap.tighten_to_target(
        forward, variables, "dE00", usl=1.0, target_cpk=1.33, n=800)
    assert history[-1]["cpk"] > history[0]["cpk"]
    assert history[0]["driver"] == "big"
    by_name = {v.name: v.tolerance for v in tightened}
    assert by_name["big"] < 0.3


def test_tightening_does_not_mutate_the_caller_variables():
    variables = [cap.ProcessVariable("a", 0.0, 0.5)]
    cap.tighten_to_target(lambda s: {"y": s["a"]}, variables, "y", 1.0, 1.33, n=300)
    assert variables[0].tolerance == 0.5


# ---------------------------------------------------------------------- DOE
def test_box_behnken_run_count():
    """2k(k-1) factor points plus centres."""
    for k in (3, 4, 5):
        d = doe.box_behnken(k, n_centre=3)
        assert d.shape == (2 * k * (k - 1) + 3, k)


def test_box_behnken_has_no_corner_points():
    """Every run leaves at least one factor at centre -- that is the point."""
    d = doe.box_behnken(4, n_centre=0)
    assert np.all((d == 0.0).sum(axis=1) >= 1)


def test_box_behnken_needs_three_factors():
    with pytest.raises(ValueError):
        doe.box_behnken(2)


def test_factor_coding_roundtrip():
    f = doe.Factor("temp", 17.0, 23.0, "C")
    assert f.decode(-1.0) == pytest.approx(17.0)
    assert f.decode(1.0) == pytest.approx(23.0)
    assert f.decode(0.0) == pytest.approx(20.0)
    assert f.code(f.decode(0.37)) == pytest.approx(0.37)


def test_response_surface_recovers_a_known_quadratic():
    factors = [doe.Factor("x1", -1.0, 1.0), doe.Factor("x2", -1.0, 1.0),
               doe.Factor("x3", -1.0, 1.0)]
    design = doe.build_design(factors, "box-behnken", n_centre=3)

    def forward(s):
        return {"y": 2.0 + 3.0 * s["x1"] - 1.5 * s["x2"] + 0.5 * s["x1"] * s["x2"]}

    res = doe.run_design(design, factors, forward)
    fit = doe.fit_response_surface(res, factors, "y")
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fit["coefficients"]["x1"] == pytest.approx(3.0, abs=1e-9)
    assert fit["coefficients"]["x2"] == pytest.approx(-1.5, abs=1e-9)
    assert fit["coefficients"]["x1*x2"] == pytest.approx(0.5, abs=1e-9)


def test_jmp_export_writes_both_files(tmp_path):
    factors = [doe.Factor("a", 0.0, 1.0), doe.Factor("b", 0.0, 1.0),
               doe.Factor("c", 0.0, 1.0)]
    design = doe.build_design(factors, "box-behnken", n_centre=2)
    res = doe.run_design(design, factors, lambda s: {"y": s["a"] + s["b"]})
    out = doe.export_jmp(res, factors, ["y"], tmp_path, "t")

    from pathlib import Path

    assert Path(out["csv"]).exists() and Path(out["jsl"]).exists()
    jsl = Path(out["jsl"]).read_text()
    assert "Fit Model" in jsl and ":a * :a" in jsl and ":a * :b" in jsl


def test_full_factorial_and_ccd_shapes():
    assert doe.full_factorial(3).shape == (8, 3)
    assert doe.central_composite(3, n_centre=2).shape == (8 + 6 + 2, 3)


# ------------------------------------------------------- end-to-end coupling
def test_temperature_excursion_shows_up_as_colour_change():
    """The whole chain must connect: bath drift -> film -> spectrum -> dE00."""
    master = an.AnodizeProcess(dye="blue").lab()
    drifted = an.AnodizeProcess(dye="blue", bath_temp_C=25.0).lab()
    assert cm.delta_E_2000(master, drifted) > 0.1
