# Contributing

## Setup

```bash
python -m pip install -e ".[all]"
```

On Windows, `setup_windows.bat` does the same thing inside a virtual environment.

## Before opening a pull request

```bash
ruff check finishsim tests scripts
python -m pytest
python scripts/run_study.py
```

All three must pass. CI runs exactly these on Python 3.10, 3.11 and 3.12.

## What the tests are protecting

The suite is not decoration. Three checks in particular encode hard-won bugs:

- `test_reflectance_is_physical` catches the n+ik versus n-ik sign-convention
  error. Feeding optical-constant-convention indices into the Macleod
  characteristic matrix silently turns absorbing layers into gain media and
  drives reflectance above 1. It is a quiet failure: the code runs, the plots
  look plausible, and every colour is wrong.
- `test_ciede2000_reference_pairs` checks all 13 Sharma, Wu & Dalal (2005)
  pairs. They exist specifically to exercise the hue-wraparound and rotation
  terms that naive CIEDE2000 implementations get wrong.
- `test_nominal_recipe_hits_type_ii_thickness` is the calibration anchor for
  the whole anodizing model. If a change moves the nominal recipe off
  approximately 10 um, the kinetics no longer describe a real Type II bath.

## Adding a material

Drop a CSV with columns `wavelength_nm,n,k` into `data/`, named after the
material. `finishsim.materials.index` prefers it over the built-in model with
no code change. See `data/README.md`.

## Style

Ruff config lives in `pyproject.toml`. Physics functions carry the governing
equation in the docstring; keep that convention, because the equation is the
part a reviewer needs to check.
