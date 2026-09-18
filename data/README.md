# Measured optical constants

Drop a CSV here named after the material, with a header row:

```
wavelength_nm,n,k
380,1.85,1.22
385,1.86,1.24
...
```

`finishsim.materials.index` resolves a file here **before** the built-in
Drude-Lorentz or Sellmeier model, so adding measured data needs no code
change. `materials.source_of(name)` reports which path was used, which is what
you cite in a report.

## Why you would do this

The shipped models use representative literature parameters for the material
*class*. They reproduce the right colour family and the right dispersion
shape, which is enough for a sensitivity study: the conclusions are about how
strongly colour responds to process variation, and that is governed by stack
geometry and the shape of n(lambda).

They are not a measurement of a specific supplier's coating. For absolute
colour prediction against a real colour master, use measured data.

## Where to get it

- refractiveindex.info exports directly in this format.
- Spectroscopic ellipsometry on your own witness coupons is better still,
  because it captures your deposition conditions rather than someone else's.

## Note on stoichiometry

A tabulated file describes one fixed composition. Requesting a non-unity
`stoichiometry` forces the Drude-Lorentz path, since a fixed table cannot
represent an off-stoichiometric film. To study stoichiometry with measured
data, supply one CSV per composition and name them distinctly.
