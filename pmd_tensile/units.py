"""Annotated types for node inputs and outputs: units and generic QUDT quantity kinds."""

from __future__ import annotations

import numpy as np
from semantikon import SemantikonURI, u

# ---------------------------------------------------------------------------
# Annotated types (generic QUDT quantity kinds only)
# ---------------------------------------------------------------------------

QUDT_QUANTITY_KIND = "http://qudt.org/vocab/quantitykind/"


QK_FORCE = SemantikonURI(QUDT_QUANTITY_KIND + "Force")

QK_STRESS = SemantikonURI(QUDT_QUANTITY_KIND + "Stress")

QK_STRAIN = SemantikonURI(QUDT_QUANTITY_KIND + "Strain")

QK_MODULUS = SemantikonURI(QUDT_QUANTITY_KIND + "ModulusOfElasticity")

QK_AREA = SemantikonURI(QUDT_QUANTITY_KIND + "Area")

QK_LENGTH = SemantikonURI(QUDT_QUANTITY_KIND + "Length")

QK_TIME = SemantikonURI(QUDT_QUANTITY_KIND + "Time")


QUANTITY_KIND_IRIS: tuple[str, ...] = tuple(
    str(iri) for iri in (QK_FORCE, QK_STRESS, QK_STRAIN, QK_MODULUS, QK_AREA, QK_LENGTH, QK_TIME)
)
"""Every quantity-kind IRI used in annotations (for review and tests)."""


ForceArray = u(np.ndarray, units="N", uri=QK_FORCE)

StressArray = u(np.ndarray, units="MPa", uri=QK_STRESS)

StrainArray = u(np.ndarray, units="Unitless", uri=QK_STRAIN)

CrossheadArray = u(np.ndarray, units="mm", uri=QK_LENGTH)

TimeArray = u(np.ndarray, units="s", uri=QK_TIME)


ForceN = u(float, units="N", uri=QK_FORCE)

StressMPa = u(float, units="MPa", uri=QK_STRESS)

ModulusGPa = u(float, units="GPa", uri=QK_MODULUS)

StrainRatio = u(float, units="Unitless", uri=QK_STRAIN)

OptionalStrainRatio = u(float | None, units="Unitless", uri=QK_STRAIN)

StrainPercent = u(float, units="percent", uri=QK_STRAIN)

OptionalStrainPercent = u(float | None, units="percent", uri=QK_STRAIN)

AreaMM2 = u(float, units="mm**2", uri=QK_AREA)

OptionalAreaMM2 = u(float | None, units="mm**2", uri=QK_AREA)

LengthMM = u(float, units="mm", uri=QK_LENGTH)

OptionalLengthMM = u(float | None, units="mm", uri=QK_LENGTH)

OptionalCrossheadArray = u(np.ndarray | None, units="mm", uri=QK_LENGTH)
