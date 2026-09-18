"""finishsim -- cosmetic finish process-capability simulator.

Two finishing routes, one colour engine, one capability question:

    process variation -> physics -> R(lambda) -> CIELAB -> dE00 -> Cpk -> yield

Modules
-------
``colourimetry``  spectral -> CIE XYZ -> CIELAB -> CIEDE2000
``materials``     n(lambda) + i k(lambda) for substrates and coatings
``pvd``           thin-film stack reflectance by the transfer-matrix method
``anodize``       Type II sulfuric anodizing kinetics + dye uptake
``capability``    Monte-Carlo variation, Cpk, yield, sensitivity ranking
``doe``           Box-Behnken / factorial designs, response surfaces, JMP export
"""

from . import anodize, capability, colourimetry, doe, materials, pvd

__version__ = "1.0.0"

__all__ = [
    "anodize",
    "capability",
    "colourimetry",
    "doe",
    "materials",
    "pvd",
    "__version__",
]
