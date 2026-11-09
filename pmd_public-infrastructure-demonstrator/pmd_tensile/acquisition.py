"""Acquisition nodes (ontology-aware): graph content -> neutral contract.

These nodes, together with ``pmd_tensile.exploration``, are the only
workflow nodes that know an ontology. Their outputs are the ontology-free
dataclasses of ``pmd_tensile.contract``, in contract units. Every unit is
converted through an explicit table; an unknown or mismatched unit raises.

The plain-function forms (``build_specimen_record``,
``build_reported_properties``, ``download_curve``) are defined in
``pmd_tensile._acquisition``; this module holds only nodes, so that sim_atlas
catalogues nothing else.
"""

from __future__ import annotations

import flowrep as fr

from pmd_tensile._acquisition import build_reported_properties, build_specimen_record, download_curve
from pmd_tensile.contract import Curve, ReportedProperties, SpecimenRecord


@fr.atomic
def get_specimen_record(
    profile_id: str, source: dict, specimen_id: str, curve_locator: dict | None = None
) -> SpecimenRecord:
    """Neutral record of one specimen: geometry, orientation, notes, curve link.

    A semantic profile reads the curve link from the graph's curve tables.
    ``curve_locator`` is a visibly non-semantic URL template with column
    meanings (``profiles.base.CurveLocator``). A profile with
    ``accepts_curve_locator`` (``pmdco3-tto``) also accepts one, which then
    replaces the graph's curve tables; this is needed for the original TTO
    graph, which has none, and a run records it as an input in its knowledge
    graph. Other semantic profiles (``pmdco2-tto``) reject it; a profile
    without semantic curve links uses it instead of its own locator.
    """
    record = build_specimen_record(profile_id, source, specimen_id, curve_locator)
    return record


@fr.atomic
def get_reported_properties(profile_id: str, source: dict, specimen_id: str) -> ReportedProperties:
    """Lab-reported values of one specimen, for comparison only."""
    reported = build_reported_properties(profile_id, source, specimen_id)
    return reported


@fr.atomic
def fetch_curve(record: SpecimenRecord) -> Curve:
    """Download (in memory, token-free) and parse the specimen's raw curve."""
    curve = download_curve(record)
    return curve
