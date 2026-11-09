"""Registry of ontology profiles, keyed by profile ID.

Adding an ontology means adding a profile module and registering it here;
the exploration and acquisition nodes stay unchanged.
"""

from __future__ import annotations

from pmd_tensile.profiles import pmdco2_tto, pmdco3_tto
from pmd_tensile.profiles.base import CurveLocator, OntologyProfile, SelectQuery

__all__ = [
    "CurveLocator",
    "OntologyProfile",
    "SelectQuery",
    "available_profiles",
    "get_profile",
    "register_profile",
]

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


register_profile(pmdco2_tto.PROFILE)
register_profile(pmdco3_tto.PROFILE)
