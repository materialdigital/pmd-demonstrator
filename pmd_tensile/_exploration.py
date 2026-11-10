"""Plain-function forms behind the exploration nodes (ontology-aware).

:mod:`pmd_tensile.exploration` holds the nodes. The functions here are the
conformance check without failing, and the specimen table; notebooks reach
them through :mod:`pmd_tensile.demo`. Inputs are JSON-able: a profile ID (see
``pmd_tensile.profiles``) and a source dict (see ``pmd_tensile._sources``).
"""

from __future__ import annotations

import re

import pandas as pd

from pmd_tensile import _sources as sources
from pmd_tensile.profiles import get_profile


class ConformanceError(RuntimeError):
    """The source does not conform to the selected ontology profile."""


def conformance_results(profile_id: str, source: dict) -> dict[str, dict[str, bool]]:
    """Evaluate every conformance ASK of the profile on every part of the source.

    Returns ``{ask name: {part: answer}}``. Does not raise on a mismatch.
    """
    profile = get_profile(profile_id)
    return {
        name: sources.ask(source, query) for name, query in profile.conformance_asks.items()
    }


def is_conformant(profile_id: str, source: dict) -> bool:
    """True if every ASK holds in every part (file or dataset) of the source."""
    results = conformance_results(profile_id, source)
    return all(all(answers.values()) for answers in results.values())


def specimen_table(profile_id: str, source: dict) -> pd.DataFrame:
    """One row per specimen: ``specimen_id``, ``orientation``, ``process``.

    Raises if a specimen ID occurs in more than one test, a test has more than
    one specimen ID, or a test has no or more than one orientation.
    """
    profile = get_profile(profile_id)
    rows = sources.select(source, profile.specimens.query, profile.specimens.columns)
    if rows.empty:
        raise LookupError(f"no specimens found with profile {profile_id!r}")
    for process, group in rows.groupby("process", sort=False):
        ids = sorted(set(group["specimen_id"]))
        if len(ids) != 1:
            raise ValueError(f"test {process} has several specimen IDs: {ids}")
    records = []
    for specimen_id, group in rows.groupby("specimen_id", sort=False):
        processes = sorted(set(group["process"]))
        if len(processes) != 1:
            raise ValueError(f"specimen {specimen_id!r} occurs in several tests: {processes}")
        orientations = sorted({o for o in group["orientation"] if o})
        if len(orientations) != 1:
            raise ValueError(
                f"specimen {specimen_id!r} needs exactly one orientation, found {orientations}"
            )
        records.append(
            {"specimen_id": specimen_id, "orientation": orientations[0], "process": processes[0]}
        )
    table = pd.DataFrame(records, dtype=object)
    order = sorted(range(len(table)), key=lambda i: _natural_key(table["specimen_id"][i]))
    return table.iloc[order].reset_index(drop=True)


def _natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]
