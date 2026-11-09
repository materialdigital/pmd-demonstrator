"""Exploration nodes (ontology-aware): conformance check and specimen listing.

Inputs are JSON-able: a profile ID (see ``pmd_tensile.profiles``) and a
source dict (see ``pmd_tensile._sources``). The plain functions behind the
nodes (``conformance_results``, ``is_conformant``, ``specimen_table``) and
:class:`ConformanceError` are defined in ``pmd_tensile._exploration``; this
module holds only nodes, so that sim_atlas catalogues nothing else.
"""

from __future__ import annotations

import flowrep as fr

from pmd_tensile._exploration import ConformanceError, conformance_results, specimen_table


@fr.atomic
def check_conformance(profile_id: str, source: dict) -> str:
    """Fail early unless the source conforms to the profile; passes the ID on.

    Every conformance ASK of the profile must hold in every part of the
    source. Returns ``profile_id`` so that downstream nodes can depend on the
    check.
    """
    results = conformance_results(profile_id, source)
    failed = {
        name: sorted(part for part, answer in answers.items() if not answer)
        for name, answers in results.items()
        if not all(answers.values())
    }
    if failed:
        raise ConformanceError(
            f"source does not conform to profile {profile_id!r}; failed ASKs: {failed}"
        )
    checked_profile_id = profile_id
    return checked_profile_id


@fr.atomic
def list_specimens(profile_id: str, source: dict) -> list[str]:
    """Specimen IDs (test-piece names) described in the source, naturally sorted."""
    specimen_ids = list(specimen_table(profile_id, source)["specimen_id"])
    return specimen_ids
