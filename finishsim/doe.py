"""Design of experiments, response-surface fitting, and JMP export.

Designs
-------
* **Box-Behnken** -- the default here.  For k factors it takes every pair of
  factors to all four (+/-1, +/-1) corners while the rest sit at centre, plus
  centre replicates.  It needs no axial points outside the factor ranges,
  which matters for a real bath: a central composite design would ask you to
  run the acid concentration beyond the range the line can actually hold.
  Run count is 2k(k-1) + centres, so 15 for k=3 and 27 for k=4.
* **Full factorial 2^k** -- screening, main effects and interactions only.
* **Central composite** -- offered for completeness when axial points are safe.

Model
-----
A full quadratic response surface,

    y = b0 + SUM b_i x_i + SUM b_ii x_i^2 + SUM_{i<j} b_ij x_i x_j

fitted by ordinary least squares on coded (-1..+1) factor levels.  Coding
matters: on coded levels the coefficients are directly comparable, so the
biggest |b| really is the biggest effect.

Export
------
:func:`export_jmp` writes the design plus responses as CSV *and* emits a JSL
script that builds the Fit Model platform with the same quadratic terms.  Open
the .jsl in JMP, run it, and you get the RSM analysis, the prediction profiler
and the contour profiler without re-specifying anything by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Factor:
    """One DOE factor with its real-world low/high levels."""

    name: str
    low: float
    high: float
    units: str = ""

    def decode(self, coded: float) -> float:
        """Coded (-1..+1) -> engineering units."""
        mid, half = 0.5 * (self.high + self.low), 0.5 * (self.high - self.low)
        return mid + half * coded

    def code(self, real: float) -> float:
        """Engineering units -> coded (-1..+1)."""
        mid, half = 0.5 * (self.high + self.low), 0.5 * (self.high - self.low)
        return (real - mid) / half if half else 0.0


def box_behnken(k: int, n_centre: int = 3) -> np.ndarray:
    """Box-Behnken design matrix in coded units; shape (2k(k-1) + n_centre, k)."""
    if k < 3:
        raise ValueError("Box-Behnken needs at least 3 factors")
    rows = []
    for i, j in combinations(range(k), 2):
        for a, b in product((-1.0, 1.0), repeat=2):
            row = np.zeros(k)
            row[i], row[j] = a, b
            rows.append(row)
    rows.extend(np.zeros(k) for _ in range(n_centre))
    return np.array(rows)


def full_factorial(k: int, n_centre: int = 0) -> np.ndarray:
    """2^k factorial in coded units."""
    rows = [np.array(c, dtype=float) for c in product((-1.0, 1.0), repeat=k)]
    rows.extend(np.zeros(k) for _ in range(n_centre))
    return np.array(rows)


def central_composite(k: int, n_centre: int = 3, alpha: float | None = None) -> np.ndarray:
    """Rotatable central composite design in coded units."""
    if alpha is None:
        alpha = (2.0**k) ** 0.25
    rows = list(full_factorial(k))
    for i in range(k):
        for s in (-alpha, alpha):
            row = np.zeros(k)
            row[i] = s
            rows.append(row)
    rows.extend(np.zeros(k) for _ in range(n_centre))
    return np.array(rows)


def build_design(
    factors: list[Factor], design: str = "box-behnken", n_centre: int = 3
) -> pd.DataFrame:
    """Coded design matrix plus decoded engineering-unit columns."""
    k = len(factors)
    coded = {
        "box-behnken": box_behnken,
        "factorial": full_factorial,
        "ccd": central_composite,
    }[design](k, n_centre)

    df = pd.DataFrame(coded, columns=[f"{f.name}_coded" for f in factors])
    for i, f in enumerate(factors):
        df[f.name] = [f.decode(c) for c in coded[:, i]]
    df.insert(0, "run", np.arange(1, len(df) + 1))
    return df


def run_design(
    design_df: pd.DataFrame, factors: list[Factor], forward: Callable[[dict], dict]
) -> pd.DataFrame:
    """Evaluate the forward model at every design point."""
    out = []
    for _, row in design_df.iterrows():
        state = {f.name: float(row[f.name]) for f in factors}
        out.append(forward(state))
    return pd.concat([design_df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def fit_response_surface(
    df: pd.DataFrame, factors: list[Factor], response: str
) -> dict:
    """Fit a full quadratic model on coded levels; return coefficients and fit stats."""
    names = [f.name for f in factors]
    X_coded = np.column_stack([df[f"{n}_coded"].to_numpy(dtype=float) for n in names])
    y = df[response].to_numpy(dtype=float)

    cols, labels = [np.ones(len(df))], ["intercept"]
    for i, n in enumerate(names):
        cols.append(X_coded[:, i])
        labels.append(n)
    for i, n in enumerate(names):
        cols.append(X_coded[:, i] ** 2)
        labels.append(f"{n}^2")
    for (i, a), (j, b) in combinations(list(enumerate(names)), 2):
        cols.append(X_coded[:, i] * X_coded[:, j])
        labels.append(f"{a}*{b}")

    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    ss_res, ss_tot = float(resid @ resid), float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    dof = len(y) - X.shape[1]
    r2_adj = 1.0 - (1.0 - r2) * (len(y) - 1) / dof if dof > 0 else float("nan")

    return {
        "response": response,
        "terms": labels,
        "coefficients": dict(zip(labels, beta.tolist())),
        "r2": r2,
        "r2_adj": r2_adj,
        "rmse": float(np.sqrt(ss_res / dof)) if dof > 0 else float("nan"),
        "predicted": pred,
        "residuals": resid,
    }


def effect_table(fit: dict) -> pd.DataFrame:
    """Coefficients ranked by magnitude, excluding the intercept."""
    items = [(k, v) for k, v in fit["coefficients"].items() if k != "intercept"]
    out = pd.DataFrame(items, columns=["term", "coefficient"])
    out["abs"] = out["coefficient"].abs()
    return out.sort_values("abs", ascending=False).reset_index(drop=True)


def export_jmp(
    df: pd.DataFrame, factors: list[Factor], responses: list[str], out_dir, stem="doe"
) -> dict:
    """Write a JMP-ready CSV and a JSL script that fits the RSM directly."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    jsl_path = out_dir / f"{stem}.jsl"

    keep = [f.name for f in factors] + [r for r in responses if r in df.columns]
    df[keep].to_csv(csv_path, index=False)

    names = [f.name for f in factors]
    effects = []
    for n in names:
        effects.append(f'\t\t:{n},')
    for n in names:
        effects.append(f'\t\t:{n} * :{n},')
    for a, b in combinations(names, 2):
        effects.append(f'\t\t:{a} * :{b},')
    effects_block = "\n".join(effects).rstrip(",")

    y_block = ",\n".join(f'\t\t:{r}' for r in responses if r in df.columns)

    jsl = f"""// Auto-generated by finishsim.doe.export_jmp
// Open this in JMP and run it (Ctrl+R) to build the response-surface analysis.

dt = Open( "{csv_path.name}" );

Fit Model(
\tY(
{y_block}
\t),
\tEffects(
{effects_block}
\t),
\tPersonality( "Standard Least Squares" ),
\tEmphasis( "Effect Screening" ),
\tRun(
\t\tProfiler( 1, Confidence Intervals( 1 ), Desirability Functions( 1 ) ),
\t\tContour Profiler( 1 )
\t)
);
"""
    jsl_path.write_text(jsl, encoding="utf-8")
    return {"csv": str(csv_path), "jsl": str(jsl_path), "runs": len(df)}
