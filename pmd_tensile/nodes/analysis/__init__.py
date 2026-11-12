"""Ontology-agnostic analysis of one tensile test (method version 3).

The nodes take neutral arrays and values (force in N, strain as a ratio,
crosshead in mm, S0 in mm²) and know no ontology. In the order of a run:

- ``extensometer``: :func:`check_time_order`, :func:`count_dropped_samples`
  and :func:`extensometer_validity` (the samples with valid strain);
- ``elastic_line``: :func:`provisional_elastic_line` (highest force, a
  provisional line and the noise level), then, after the yield search,
  :func:`elastic_line` (E and the strain origin), :func:`proof_strength`
  (Rp0.2) and :func:`window_sensitivity`;
- ``yielding``: :func:`yield_points` (yield type, ReH, ReL);
- ``maximum_force``: :func:`maximum_force` (Fm, Rm) and
  :func:`elongation_at_maximum_force` (Agt, Ag);
- ``geometry``: :func:`s0_scatter`, :func:`elongation_after_fracture` (A),
  :func:`reduction_of_area` (Z);
- ``comparison``: :func:`compare_with_reported`, :func:`summarize_by_group`.

Every property is defined by this method and inspired by ISO 6892-1; nothing
here claims conformity with the standard. Undefined values are NaN, flags are
strings ``"<property>:<flag>"``. Notebook ``02-tensile-method`` of the
repository describes the method in depth and verifies it on synthetic curves.
"""
