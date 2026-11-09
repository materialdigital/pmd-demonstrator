"""Registry of ontology profiles, keyed by profile ID.

Adding an ontology means adding a profile module and registering it here;
the exploration and acquisition nodes stay unchanged.

The registry functions are defined in the private module
``pmd_tensile._profile_registry`` and re-exported here, so that sim_atlas
does not catalogue them as workflow nodes when it collects this package.
"""

from __future__ import annotations

from pmd_tensile._profile_registry import available_profiles, get_profile, register_profile
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

register_profile(pmdco2_tto.PROFILE)
register_profile(pmdco3_tto.PROFILE)
