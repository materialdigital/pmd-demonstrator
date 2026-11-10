"""Profile registry: re-registering the same object is a no-op (finding CR2-1).

The registry lives in ``pmd_tensile._profile_registry``, which reloading
``pmd_tensile.profiles`` does not reset; ``profiles/__init__`` then registers
its profile objects a second time. That must work, while a *different* object
under a registered ID must still be refused.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

import pmd_tensile
from pmd_tensile import _profile_registry, profiles

PACKAGE_PARENT = Path(pmd_tensile.__path__[0]).parent

_RELOAD_CHECK = r"""
import importlib
import json
import socket
import sys


def offline(*args, **kwargs):
    raise OSError("the reload check must stay offline")


socket.socket.connect = offline
socket.getaddrinfo = offline
sys.path.insert(0, sys.argv[1])

import pmd_tensile.profiles as profiles

before = {pid: id(profiles.get_profile(pid)) for pid in profiles.available_profiles()}
importlib.reload(profiles)
importlib.reload(profiles)
after = {pid: id(profiles.get_profile(pid)) for pid in profiles.available_profiles()}
print(json.dumps({"file": profiles.__file__, "before": before, "after": after}))
"""


def test_reloading_profiles_in_a_fresh_interpreter_succeeds():
    # -I: no user site, no PYTHON* variables, no script directory on sys.path;
    # -B: write no bytecode. The checkout is put first on sys.path explicitly.
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _RELOAD_CHECK, str(PACKAGE_PARENT)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(report["file"]).resolve().is_relative_to(PACKAGE_PARENT.resolve())
    assert sorted(report["before"]) == ["pmdco2-tto", "pmdco3-tto"]
    assert report["after"] == report["before"]  # the profile modules were not reloaded


def test_registering_the_same_object_again_is_a_no_op():
    profile = profiles.get_profile("pmdco2-tto")
    profiles.register_profile(profile)
    assert profiles.get_profile("pmdco2-tto") is profile
    assert profiles.available_profiles() == ["pmdco2-tto", "pmdco3-tto"]


def test_a_different_object_under_a_registered_id_is_refused():
    profile = profiles.get_profile("pmdco2-tto")
    copy = dataclasses.replace(profile)
    assert copy is not profile
    with pytest.raises(ValueError, match="'pmdco2-tto' is already registered"):
        _profile_registry.register_profile(copy)
    assert profiles.get_profile("pmdco2-tto") is profile
