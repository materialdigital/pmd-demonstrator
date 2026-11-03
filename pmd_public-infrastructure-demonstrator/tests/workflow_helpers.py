"""Shared test helpers (no tests): the pmdco3 structural fixture and stubbed curves.

The pmdco3 builders write a structural fixture: it has the PMDco3/TTO modelling
pattern, with made-up numbers. The curve helpers serve synthetic CSV bytes in
the layout of the S355 curve files (``;`` delimiter, ``,`` decimal mark, a
header row and a unit row). The stresses are method-spec §14 (a) with noise,
scaled to each specimen's S0; their numbers are made up, not lab data.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest
from conftest import PMDCO2_ORIENTATION_FILES

import synthetic as sy
from tensile_workflow import acquisition, sources, workflows

PMDCO2 = "pmdco2-tto"
PMDCO3 = "pmdco3-tto"
PMDCO2_SOURCE = {"kind": "turtle", "locations": [str(p) for p in PMDCO2_ORIENTATION_FILES.values()]}
NOISE_SEEDS = {"Zx1": 101, "Zy1": 102, "Zd2": 103, "Zx2": 104}

# ---------------------------------------------------------------------------
# pmdco3 structural fixture
# ---------------------------------------------------------------------------

PMDCO3_PREFIXES = """\
@prefix ex:    <https://example.org/fixture/> .
@prefix obo:   <http://purl.obolibrary.org/obo/> .
@prefix pmdco: <https://w3id.org/pmd/co/> .
@prefix tto:   <https://w3id.org/pmd/tto/> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix unit:  <http://qudt.org/vocab/unit/> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
"""


def pmdco3_measured(
    test: str,
    name: str,
    cls: str,
    bearer: str,
    value: str,
    unit: str | None,
    relation: str = "obo:RO_0000086",
) -> str:
    """A quality, its measurement datum (output of the test) and its value spec.

    ``relation`` links the bearer to the quality (default: has quality).
    """
    unit_line = f"    obo:IAO_0000039 {unit} ;\n" if unit else ""
    return f"""
ex:{test}_{name} a {cls} .
{bearer} {relation} ex:{test}_{name} .
ex:{test} obo:OBI_0000299 ex:{test}_{name}_datum .
ex:{test}_{name}_datum a obo:IAO_0000109 ;
    obo:IAO_0000136 ex:{test}_{name} ;
    obo:OBI_0001938 ex:{test}_{name}_value .
ex:{test}_{name}_value a obo:OBI_0001931 ;
{unit_line}    obo:OBI_0001927 ex:{test}_{name} ;
    pmdco:PMD_0000006 "{value}"^^xsd:float .
"""


def pmdco3_test(
    test: str, piece: str, orientation: str, s0_unit: str = "unit:MilliM2", switch: str = "3.8"
) -> str:
    """The fixture block of one tensile test."""
    tp, broken, material = f"ex:{piece}", f"ex:{piece}_after", f"ex:{piece}_material"
    extensometer, load_cell = f"ex:{test}_extensometer", f"ex:{test}_load_cell"
    text = f"""
ex:{test} a pmdco:PMD_0000974 ;
    obo:OBI_0000293 {tp} ;
    obo:OBI_0000299 {broken} .
{tp} a obo:BFO_0000030 ;
    pmdco:PMD_0020005 {material} .
ex:{piece}_id obo:IAO_0000219 {tp} ;
    rdfs:label "Test piece identifier"^^xsd:string ;
    obo:OBI_0001938 ex:{piece}_id_value .
ex:{piece}_id_value pmdco:PMD_0000006 "{piece}"^^xsd:string .
ex:{piece}_material_id obo:IAO_0000219 {material} ;
    obo:OBI_0001938 ex:{piece}_material_id_value .
ex:{piece}_material_id_value pmdco:PMD_0000006 "S355"^^xsd:string .
{material} obo:RO_0000086 ex:{piece}_rolling_direction .
ex:{piece}_rolling_direction a obo:BFO_0000019 ;
    obo:OBI_0001938 ex:{piece}_rolling_direction_value .
ex:{piece}_rolling_direction_value pmdco:PMD_0000006 "{orientation}"^^xsd:string .
ex:{test}_identifier obo:IAO_0000219 ex:{test} ;
    rdfs:label "Process identifier"^^xsd:string ;
    obo:OBI_0001938 ex:{test}_identifier_value .
ex:{test}_identifier_value pmdco:PMD_0000006 "process-{test}"^^xsd:string .
ex:{test}_note obo:IAO_0000219 ex:{test} ;
    rdfs:label "process note"^^xsd:string ;
    obo:OBI_0001938 ex:{test}_note_value .
ex:{test}_note_value pmdco:PMD_0000006 "note of {piece}"^^xsd:string .
"""
    text += pmdco3_measured(test, "S0", "tto:TTO_0000026", tp, "100.2", s0_unit)
    text += pmdco3_measured(test, "s1", "tto:TTO_0000012", tp, "100.0", "unit:MilliM2")
    text += pmdco3_measured(test, "s2", "tto:TTO_0000012", tp, "100.3", "unit:MilliM2")
    text += pmdco3_measured(test, "s3", "tto:TTO_0000012", tp, "100.3", "unit:MilliM2")
    text += pmdco3_measured(test, "L0", "tto:TTO_0000028", tp, "60.0", "unit:MilliM")
    text += pmdco3_measured(test, "ext_gauge", "tto:TTO_0000028", extensometer, "50.0", "unit:MilliM")
    text += pmdco3_measured(test, "Lu", "tto:TTO_0000018", broken, "80.5", "unit:MilliM")
    text += pmdco3_measured(test, "Su", "tto:TTO_0000024", broken, "45.0", "unit:MilliM2")
    text += pmdco3_measured(test, "E", "pmdco:PMD_0000618", material, "205.0", "unit:GigaPa")
    text += pmdco3_measured(test, "Rp02", "tto:TTO_0000047", material, "NaN", "unit:MegaPa")
    text += pmdco3_measured(test, "ReH", "tto:TTO_0000059", material, "400.0", "unit:MegaPa")
    text += pmdco3_measured(test, "ReL", "tto:TTO_0000022", material, "380.0", "unit:MegaPa")
    text += pmdco3_measured(test, "Rm", "tto:TTO_0000053", material, "510.0", "unit:MegaPa")
    text += pmdco3_measured(test, "Fm", "tto:TTO_0000023", material, "51.1", "unit:KiloN")
    text += pmdco3_measured(test, "load_range", "tto:TTO_0000023", load_cell, "100.0", "unit:KiloN")
    text += pmdco3_measured(test, "A", "tto:TTO_0000033", material, "34.0", "unit:PERCENT")
    text += pmdco3_measured(test, "Z", "tto:TTO_0000038", material, "55.0", "unit:PERCENT")
    text += pmdco3_measured(
        test, "switch", "tto:TTO_0000058", f"ex:{test}", switch, "unit:PERCENT",
        relation="pmdco:PMD_0000009",  # has process attribute
    )
    return text


def _set_fixture_value(text: str, test: str, name: str, value) -> str:
    """Replace one value literal of the fixture, ``ex:<test>_<name>_value``."""
    literal = "NaN" if value is None else repr(float(value))
    pattern = re.compile(
        rf'(ex:{test}_{name}_value a obo:OBI_0001931 ;.*?pmdco:PMD_0000006 ")[^"]*(")', re.DOTALL
    )
    text, count = pattern.subn(rf"\g<1>{literal}\g<2>", text)
    assert count == 1, (test, name)
    return text


def pmdco3_twin_of_pmdco2(specimen_ids, tmp_path) -> dict:
    """A pmdco3 structural graph describing the given pmdco2 specimens with their values."""
    text = PMDCO3_PREFIXES
    for i, sid in enumerate(specimen_ids, start=1):
        record = acquisition.build_specimen_record(PMDCO2, PMDCO2_SOURCE, sid)
        test = f"test_{i}"
        switch = round(100.0 * record.rate_switch_strain, 10)  # percent, as in the graphs
        block = pmdco3_test(test, sid, record.orientation, switch=repr(switch))
        s1, s2, s3 = record.cross_section_measurements_mm2
        values = {
            "S0": record.s0_mm2, "s1": s1, "s2": s2, "s3": s3,
            "L0": record.l0_mm, "Lu": record.lu_mm, "Su": record.su_mm2,
        }
        for name, value in values.items():
            block = _set_fixture_value(block, test, name, value)
        text += block
    path = tmp_path / "pmdco3_twin.ttl"
    path.write_text(text, encoding="utf-8")
    return {"kind": "turtle", "locations": [str(path)]}


# ---------------------------------------------------------------------------
# Synthetic curve bytes and the stubbed download
# ---------------------------------------------------------------------------


def synthetic_csv(s0_mm2: float, seed: int) -> bytes:
    """§14 (a) with noise, as CSV bytes in the S355 layout, force scaled to ``s0_mm2``."""
    curve = sy.noisy(sy.reference_a(), seed)
    force_kn = curve.force_n / curve.s0_mm2 * s0_mm2 / 1000.0
    time_s = np.arange(len(curve)) * 0.1
    lines = ["Time;Crosshead separation;Load;Extensometer elongation", "(sec);(mm);(kN);(%)"]
    for row in zip(time_s, curve.crosshead_mm, force_kn, curve.strain * 100.0, strict=True):
        lines.append(";".join(f"{value:.10g}".replace(".", ",") for value in row))
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def curve_bytes_for(specimen_ids, profile_id: str = PMDCO2, source: dict = PMDCO2_SOURCE) -> dict:
    """Synthetic CSV bytes per specimen ID, force scaled to the record's S0."""
    return {
        sid: synthetic_csv(acquisition.build_specimen_record(profile_id, source, sid).s0_mm2, NOISE_SEEDS[sid])
        for sid in specimen_ids
    }


def stub_fetch(curves: dict):
    """A ``sources.fetch_bytes`` stand-in: serves the bytes by the URL's file name ``<ID>.csv``."""
    fetched = []

    def fetch(url: str) -> tuple[bytes, str]:
        fetched.append(url)
        data = curves[url.rsplit("/", 1)[-1].removesuffix(".csv")]
        return data, hashlib.sha256(data).hexdigest()

    fetch.fetched = fetched
    return fetch


def run_with_stubbed_curves(curves: dict, profile_id: str, source: dict, **kwargs):
    """``run_tensile_analysis`` with the curve download stubbed; returns (analysis, fetched URLs)."""
    fetch = stub_fetch(curves)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sources, "fetch_bytes", fetch)
        result = workflows.run_tensile_analysis(profile_id, source, **kwargs)
    return result, list(fetch.fetched)
