"""Monte-Carlo propagation of process variation into cosmetic yield.

The question this module answers is the one a finishing SME is actually paid
to answer:

    Given that my bath temperature holds to +/- 1.5 C and my current density
    to +/- 0.1 A/dm2, what fraction of parts land inside the colour spec --
    and which knob do I tighten first?

Method: sample every process input from a normal distribution whose spread is
set by its stated control tolerance, push each sample through the forward
physics model, and look at the distribution of the responses.

CAPABILITY INDICES
------------------
For a two-sided spec (film thickness), the standard index is

    Cpk = min(USL - mu, mu - LSL) / (3 sigma)

For colour we use a one-sided upper spec on dE00 (there is no such thing as
"too little colour error"), so the relevant index is

    Ppk_upper = (USL - mu) / (3 sigma)

Cpk >= 1.33 is the usual consumer-electronics cosmetic gate, corresponding to
about 32 ppm outside spec if the process stays centred.

Note the honest distinction: because these samples come from a simulation run
at fixed nominal settings rather than from a time-ordered production record,
what we compute is strictly *short-term, within-batch* capability.  Real
long-term Ppk on a line will be worse, because it also carries drift between
bath changes, operator changes and rack position.

SENSITIVITY
-----------
Ranking uses standardised regression coefficients (beta weights): regress the
response on the z-scored inputs, and |beta| tells you how many standard
deviations of response you get per standard deviation of input.  That is the
right ranking for "which tolerance do I tighten", because it already folds in
both the physical gain and the current control spread.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ProcessVariable:
    """One controllable input and how tightly the line actually holds it.

    ``tolerance`` is the half-width of the control band.  By convention we
    treat that band as +/- 3 sigma, i.e. the line is assumed capable of
    holding the stated tolerance essentially all the time.  Override with
    ``sigma`` to state the spread directly.
    """

    name: str
    nominal: float
    tolerance: float = 0.0
    sigma: float | None = None
    units: str = ""

    def spread(self) -> float:
        if self.sigma is not None:
            return self.sigma
        return self.tolerance / 3.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        s = self.spread()
        if s <= 0:
            return np.full(n, self.nominal)
        return rng.normal(self.nominal, s, size=n)


def monte_carlo(
    forward: Callable[[dict], dict],
    variables: list[ProcessVariable],
    n: int = 4000,
    seed: int = 20260918,
) -> pd.DataFrame:
    """Run ``forward`` over ``n`` sampled process states.

    ``forward`` takes a dict of {variable name: value} and returns a dict of
    response name -> scalar.  Returns one tidy frame with inputs and
    responses side by side.
    """
    rng = np.random.default_rng(seed)
    samples = {v.name: v.sample(n, rng) for v in variables}
    rows = []
    for i in range(n):
        state = {k: float(samples[k][i]) for k in samples}
        rows.append({**state, **forward(state)})
    return pd.DataFrame(rows)


def cpk(values, lsl: float | None, usl: float | None) -> float:
    """Process capability index. Pass one of ``lsl``/``usl`` for a one-sided spec."""
    x = np.asarray(values, dtype=float)
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    if sd == 0:
        return float("inf")
    if lsl is None and usl is None:
        raise ValueError("need at least one specification limit")
    if lsl is None:
        return (usl - mu) / (3.0 * sd)
    if usl is None:
        return (mu - lsl) / (3.0 * sd)
    return min(usl - mu, mu - lsl) / (3.0 * sd)


def yield_fraction(df: pd.DataFrame, specs: dict[str, tuple]) -> float:
    """Fraction of simulated parts passing every spec simultaneously.

    ``specs`` maps a response column to ``(lsl, usl)``; use ``None`` for an
    unbounded side.
    """
    ok = np.ones(len(df), dtype=bool)
    for col, (lsl, usl) in specs.items():
        if lsl is not None:
            ok &= df[col].to_numpy() >= lsl
        if usl is not None:
            ok &= df[col].to_numpy() <= usl
    return float(ok.mean())


def sensitivity(df: pd.DataFrame, response: str, inputs: list[str]) -> pd.DataFrame:
    """Standardised regression coefficients of ``response`` on ``inputs``.

    Returned sorted by absolute influence, with a percentage contribution
    column suitable for a Pareto chart.
    """
    X = df[inputs].to_numpy(dtype=float)
    y = df[response].to_numpy(dtype=float)

    Xz = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    yz = (y - y.mean()) / (y.std() if y.std() else 1.0)

    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xz)), Xz]), yz, rcond=None)
    beta = beta[1:]

    out = pd.DataFrame({"variable": inputs, "beta": beta})
    out["abs_beta"] = out["beta"].abs()
    total = out["abs_beta"].sum()
    out["contribution_pct"] = 100.0 * out["abs_beta"] / (total if total else 1.0)
    return out.sort_values("abs_beta", ascending=False).reset_index(drop=True)


def capability_report(
    df: pd.DataFrame, specs: dict[str, tuple], inputs: list[str]
) -> dict:
    """Assemble the full capability picture for a simulated process window."""
    report = {"n": len(df), "overall_yield": yield_fraction(df, specs), "responses": {}}
    for col, (lsl, usl) in specs.items():
        x = df[col]
        report["responses"][col] = {
            "mean": float(x.mean()),
            "sigma": float(x.std(ddof=1)),
            "lsl": lsl,
            "usl": usl,
            "cpk": cpk(x, lsl, usl),
            "pass_rate": yield_fraction(df, {col: (lsl, usl)}),
            "sensitivity": sensitivity(df, col, inputs).to_dict("records"),
        }
    return report


def tighten_to_target(
    forward: Callable[[dict], dict],
    variables: list[ProcessVariable],
    response: str,
    usl: float,
    target_cpk: float = 1.33,
    n: int = 1500,
    max_rounds: int = 40,
    factor: float = 0.85,
) -> tuple[list[ProcessVariable], list[dict]]:
    """Greedily tighten the worst-offending tolerance until ``target_cpk`` is met.

    Each round: measure capability, find the input with the largest
    standardised influence on ``response``, shrink its tolerance by
    ``factor``, and repeat.  This is the quantitative version of "which
    control limit do I buy down first", and it returns the audit trail as
    well as the final tolerances.
    """
    variables = [ProcessVariable(**vars(v)) for v in variables]
    names = [v.name for v in variables]
    history = []

    for rnd in range(max_rounds):
        df = monte_carlo(forward, variables, n=n, seed=1000 + rnd)
        c = cpk(df[response], None, usl)
        sens = sensitivity(df, response, names)
        worst = sens.iloc[0]["variable"]
        history.append(
            {
                "round": rnd,
                "cpk": c,
                "yield": yield_fraction(df, {response: (None, usl)}),
                "driver": worst,
                "tolerances": {v.name: v.tolerance for v in variables},
            }
        )
        if c >= target_cpk:
            break
        for v in variables:
            if v.name == worst:
                v.tolerance *= factor
                if v.sigma is not None:
                    v.sigma *= factor
    return variables, history
