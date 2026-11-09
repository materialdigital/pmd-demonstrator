"""Registry of ontology profiles, keyed by profile ID (private helpers).

The functions are re-exported by :mod:`pmd_tensile.profiles`, which also
registers the built-in profiles; use them from there. They live in this
top-level private module because sim_atlas collects by module-name prefix:
defined anywhere under ``pmd_tensile.profiles``, they would be catalogued as
workflow nodes when ``pmd_tensile.profiles`` is collected in module mode.

This module imports no profile module, so that ``pmd_tensile.profiles`` can
import it without an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmd_tensile.profiles.base import OntologyProfile

_REGISTRY: dict[str, OntologyProfile] = {}


def register_profile(profile: OntologyProfile) -> None:
    """Register a profile under its ID; an ID can be registered only once."""
    if profile.profile_id in _REGISTRY:
        raise ValueError(f"profile {profile.profile_id!r} is already registered")
    _REGISTRY[profile.profile_id] = profile


def get_profile(profile_id: str) -> OntologyProfile:
    """Return the registered profile for ``profile_id``."""
    try:
        return _REGISTRY[profile_id]
    except KeyError:
        raise KeyError(
            f"unknown ontology profile {profile_id!r}; available: {available_profiles()}"
        ) from None


def available_profiles() -> list[str]:
    """IDs of all registered profiles, sorted."""
    return sorted(_REGISTRY)
