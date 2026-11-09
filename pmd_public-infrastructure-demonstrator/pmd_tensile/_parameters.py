"""The analysis parameters as a JSON-able workflow input (decision D4).

The method parameters are the frozen dataclass
:class:`pmd_tensile.analysis.AnalysisParameters`. Workflow inputs must be
JSON-able (convention IC3), so the workflows take the parameters as a plain
dictionary of field values instead; the node
:func:`pmd_tensile.workflows.analysis_parameters` converts it with
:func:`parameters_from_values`.

The dictionary form:

- keys are ``AnalysisParameters`` field names; a missing key keeps the method
  default, so ``None`` or ``{}`` means :data:`DEFAULT_PARAMETERS`;
- numbers are JSON numbers; the tuple fields are JSON arrays
  (``window_grid_lo``/``window_grid_hi``: numbers; ``linearity_windows``:
  ``[lo, hi]`` pairs).

The checks are on form, not on the method: unknown keys, wrong types,
non-finite or negative numbers, integers below 1, ``f_lo >= f_hi``, an empty
grid and a linearity window with ``lo >= hi`` are rejected with a
``TypeError`` or ``ValueError`` that names the field.

The module is private, so that sim_atlas does not catalogue these plain
functions or the default instance (an instance has no sim_atlas parser);
notebooks reach them through :mod:`pmd_tensile.demo`.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Mapping, Sequence

from pmd_tensile.analysis import AnalysisParameters

DEFAULT_PARAMETERS = AnalysisParameters()
"""The method defaults (method-spec v2.6), equal to ``AnalysisParameters()``."""

_FIELDS = {field.name: field for field in dataclasses.fields(AnalysisParameters)}
_GRID_FIELDS = ("window_grid_lo", "window_grid_hi")
_PAIR_FIELDS = ("linearity_windows",)


def parameter_values(params: AnalysisParameters | None = None) -> dict:
    """The JSON-able dictionary of all field values (tuples as lists).

    ``None`` gives the values of :data:`DEFAULT_PARAMETERS`. The result
    round-trips: ``parameters_from_values(parameter_values(p)) == p``.
    """
    params = DEFAULT_PARAMETERS if params is None else params
    if not isinstance(params, AnalysisParameters):
        raise TypeError(f"expected AnalysisParameters or None, got {type(params).__name__}")
    return {name: _json_value(getattr(params, name)) for name in _FIELDS}


def parameters_from_values(values: Mapping | None = None) -> AnalysisParameters:
    """Validate the dictionary form and build :class:`AnalysisParameters`.

    Missing keys keep the method defaults; ``None`` gives the defaults.
    Raises ``TypeError`` for a non-dictionary or a value of the wrong type and
    ``ValueError`` for unknown keys or values out of range.
    """
    if values is None:
        return DEFAULT_PARAMETERS
    if not isinstance(values, Mapping):
        raise TypeError(
            "analysis parameters must be a dictionary of AnalysisParameters field values "
            f"or None, got {type(values).__name__}"
        )
    unknown = sorted(str(key) for key in values if key not in _FIELDS)
    if unknown:
        raise ValueError(f"unknown analysis parameters {unknown}; known: {sorted(_FIELDS)}")
    converted = {name: _convert(name, value) for name, value in values.items()}
    params = dataclasses.replace(DEFAULT_PARAMETERS, **converted)
    if not params.f_lo < params.f_hi:
        raise ValueError(f"analysis parameters need f_lo < f_hi, got {params.f_lo} and {params.f_hi}")
    return params


def _json_value(value):
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"analysis parameter {name!r} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"analysis parameter {name!r} must be finite and non-negative, got {value!r}")
    return number


def _numbers(name: str, value) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise TypeError(f"analysis parameter {name!r} must be a non-empty list of numbers, got {value!r}")
    return tuple(_number(name, item) for item in value)


def _convert(name: str, value):
    default = getattr(DEFAULT_PARAMETERS, name)
    if name in _GRID_FIELDS:
        return _numbers(name, value)
    if name in _PAIR_FIELDS:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
            raise TypeError(f"analysis parameter {name!r} must be a non-empty list of [lo, hi] pairs, got {value!r}")
        pairs = []
        for pair in value:
            lo_hi = _numbers(name, pair)
            if len(lo_hi) != 2 or not lo_hi[0] < lo_hi[1]:
                raise ValueError(f"analysis parameter {name!r} needs [lo, hi] pairs with lo < hi, got {pair!r}")
            pairs.append(lo_hi)
        return tuple(pairs)
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"analysis parameter {name!r} must be an integer, got {value!r}")
        if value < 1:
            raise ValueError(f"analysis parameter {name!r} must be at least 1, got {value!r}")
        return int(value)
    return _number(name, value)
