"""The two local copies of the pmdco2 graphs give the same analysis inputs and results.

The offline tests use ``downloaded/pmdco2_tto_example_*.ttl``, copies of the
graphs hosted on Ontodocker. The older ``datasets/pmdco2_tto_example_*.ttl``
files, which the tests used before, describe the same tests but serialize some
nodes as pseudo-IRIs ``<_:tNNN>``, so the two are not isomorphic. Here both
sources must give equal specimen tables, records and reported values for all
ten specimens, and identical result tables on identical (stubbed) curve bytes.
"""

from __future__ import annotations

import pytest
from conftest import PMDCO2_DATASETS_FILES, PMDCO2_ORIENTATION_FILES
from workflow_helpers import PMDCO2, curve_bytes_for, run_with_stubbed_curves

from pmd_tensile import profiles
from pmd_tensile.nodes import acquisition

pytestmark = pytest.mark.slow

DOWNLOADED = {"kind": "turtle", "locations": [str(path) for path in PMDCO2_ORIENTATION_FILES.values()]}
DATASETS = {"kind": "turtle", "locations": [str(path) for path in PMDCO2_DATASETS_FILES.values()]}
RUN_SPECIMENS = ("Zx1", "Zy1", "Zd2")


def test_the_tests_use_the_downloaded_copies():
    assert all(path.parent.name == "downloaded" and path.is_file() for path in PMDCO2_ORIENTATION_FILES.values())
    assert all(path.parent.name == "datasets" and path.is_file() for path in PMDCO2_DATASETS_FILES.values())


def test_both_copies_conform_and_list_the_same_specimens():
    assert profiles.is_conformant(PMDCO2, DOWNLOADED)
    assert profiles.is_conformant(PMDCO2, DATASETS)
    downloaded = profiles.specimen_table(PMDCO2, DOWNLOADED)
    datasets = profiles.specimen_table(PMDCO2, DATASETS)
    assert len(downloaded) == 10
    assert downloaded.to_dict("records") == datasets.to_dict("records")


def test_both_copies_give_equal_records_and_reported_values():
    ids = list(profiles.specimen_table(PMDCO2, DOWNLOADED)["specimen_id"])
    for sid in ids:
        assert acquisition.get_specimen_record(PMDCO2, DOWNLOADED, sid) == acquisition.get_specimen_record(
            PMDCO2, DATASETS, sid
        ), sid
        assert acquisition.get_reported_properties(
            PMDCO2, DOWNLOADED, sid
        ) == acquisition.get_reported_properties(PMDCO2, DATASETS, sid), sid


@pytest.mark.filterwarnings("ignore:The hashes produced for directed graphs:UserWarning")
def test_both_copies_give_identical_results():
    curves = curve_bytes_for(RUN_SPECIMENS, source=DOWNLOADED)
    downloaded, fetched_downloaded = run_with_stubbed_curves(curves, PMDCO2, DOWNLOADED, specimen_ids=RUN_SPECIMENS)
    datasets, fetched_datasets = run_with_stubbed_curves(curves, PMDCO2, DATASETS, specimen_ids=RUN_SPECIMENS)
    assert sorted(fetched_downloaded) == sorted(fetched_datasets)  # fetch order varies between runs
    for name in ("results", "orientation_summary", "comparison"):
        new, old = getattr(downloaded, name), getattr(datasets, name)
        assert list(new.columns) == list(old.columns), name
        for column in new.columns:
            assert [repr(v) for v in new[column]] == [repr(v) for v in old[column]], (name, column)
