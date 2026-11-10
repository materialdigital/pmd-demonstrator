"""Ontology-agnostic tensile-test analysis (method-defined, ISO 6892-1-inspired).

Implements ``.notes/tensile-workflow/method-spec.md`` (APPROVED v2.6) behind the
public interface fixed in ``.notes/tensile-workflow/analysis-api.md``. Every
property is **"method-defined, ISO 6892-1-inspired"**; nothing here claims
conformity with ISO 6892-1 or its Annex G. E is the **"regression slope of the
elastic part (mE-type), not Annex-G-conformant"**, not a material modulus.

Units (fixed): force N, strain as a ratio (not percent), crosshead and lengths
mm, areas mm**2, stresses MPa, E GPa, Ag/Agt/A/Z percent. Values that are not
defined are ``float("nan")``, never ``None``. Flags are strings
``"<property>:<flag>"`` (e.g. ``"a:missing_input"``).

Module boundary (acceptance criterion 2): this module imports no rdflib,
praeco, SPARQL helper or profile, and no ``pmd_tensile`` module other
than ``contract``. It sees no ontology IRIs; its only IRIs are the generic
QUDT quantity kinds below.

Annotations (decision D4, conventions IC1/IC2)
----------------------------------------------
Step nodes carry ``semantikon.u`` annotations with units (``"N"``,
``"Unitless"`` for ratio strain, ``"percent"``, ``"MPa"``, ``"GPa"``,
``"mm"``, ``"mm**2"``, ``"s"``; never ``"dimensionless"``) and QUDT quantity
kinds. semantikon needs ``uri=`` to be an ``rdflib.URIRef``; because this
module may not import rdflib, the IRIs are built with ``semantikon.SemantikonURI``,
semantikon's public ``URIRef`` subclass. The quantity-kind IRIs are hard-coded
(semantikon's bundled QUDT file has no quantity kinds, so they cannot be
validated there) and are listed for review:

- ``http://qudt.org/vocab/quantitykind/Force``
- ``http://qudt.org/vocab/quantitykind/Stress``
- ``http://qudt.org/vocab/quantitykind/Strain`` (ratio and percent strains)
- ``http://qudt.org/vocab/quantitykind/ModulusOfElasticity``
- ``http://qudt.org/vocab/quantitykind/Area``
- ``http://qudt.org/vocab/quantitykind/Length``
- ``http://qudt.org/vocab/quantitykind/Time``

Z (reduction of area), relative uncertainties, counts, indices, flags and
result dataclasses carry no quantity kind.

Step nodes (flowrep ``@fr.atomic``; each is also a plain, directly callable function)
-------------------------------------------------------------------------------------
Curve analysis, in the M5 bootstrap order. ``analyse_curve`` (a plain
function in the private module ``pmd_tensile._analysis``, so that sim_atlas
does not catalogue it as a node) is exactly this composition, so a workflow
wiring the same nodes gives the same numbers.

1. :func:`count_dropped_samples` (M1): checks lengths; non-finite counts per channel.
2. :func:`extensometer_validity` (M1, M2, M3): ``k_ext`` and the end checks.
3. :func:`bootstrap_fit` (M5 steps 1-2): Fm from force alone; provisional
   E0, b0; the noise level s_n, fixed from here on.
4. :func:`yield_analysis` (M5 step 3; M7, M8, M9): yield type, ReH, ReL.
5. :func:`final_elastic_fit` (M4, M5 step 4-5, M6, M16): E, strain origin.
6. :func:`proof_strength` (M10): Rp0.2.
7. :func:`maximum_force` (M11): Fm, Rm.
8. :func:`elongation_at_maximum_force` (M12): Agt, Ag.
9. :func:`window_sensitivity` (M15): E-window ranges, linearity.
10. :func:`assemble_curve_analysis`: collects 1-9 into :class:`CurveAnalysis`.

Other nodes: :func:`check_time_order` (M1 time order; the curve analysis has
no time input by the fixed API), :func:`engineering_stress` (display),
:func:`elastic_fit`, :func:`s0_scatter`, :func:`elongation_after_fracture`,
:func:`reduction_of_area`, :func:`compare_with_reported`,
:func:`summarize_by_group`.

Implementation notes (no change of the method)
----------------------------------------------
- **Force space.** All curve decisions (windows, drops, bands, crossings)
  are made on the force array; S0 only converts results into stresses. With
  every threshold relative, this makes S0 scaling (VP-4) exact: no sample
  selection can flip through rounding of F/S0. The fit slope dF/de is E*S0.
- **Rm in M5/M7.** Before the yield type is known, "Rm" (provisional
  reference 0.5*Rm, and the M7 search condition sigma > 0.5*Rm) is the
  maximum over all finite-force samples. M11's final Fm may differ only
  under discontinuous yielding without work hardening.
- **First loading branch.** For a reference stress sigma_ref, the samples
  before sigma first reaches sigma_ref (with valid strain, M2).
- **Provisional fit** needs at least 3 points (for a residual MAD); the M4
  minimum of 10 applies to the final fit and the M15 fits.
- **Yield section (M7-M9)** uses the bootstrap E0, b0 and s_n throughout, so
  the final fit cannot feed back into the yield type, ReH or ReL.
- **Rp0.2 search** starts at the first valid sample at or above the lower
  bound of the E window it uses, so a nonlinear toe below the window cannot
  produce a spurious crossing.
- **Ag band mid-point** is (min e + max e)/2 over the band; the half-width is
  (max e - min e)/2.
- **near_rate_switch** compares the raw strain channel at the evaluation
  point of ReH, ReL, Rp0.2 and the extensometer end with ``rate_switch_strain``
  (distance <= ``rate_switch_proximity``).

M3 extensometer end (concrete rule)
-----------------------------------
Windows are consecutive runs of samples over which the crosshead travels
``ext_window_mm`` (|c[end] - c[start]| >= 0.1 mm; consecutive windows share
their boundary sample; samples with non-finite crosshead never close a
window, so holds do not advance it). A trailing remainder of less than one
window is merged into the last window; with no complete window, M3 is not
applied (flag ``extensometer:no_complete_window``). A window is *invalid* if its strain has a non-finite value, an exact
zero, a single-step drop larger than ``ext_jump``, or
r = (e[end] - e[start]) / ((c[end] - c[start]) / L0) < ``ext_ratio_min``.
``k_ext`` is the start of the earliest window from which every window to the
end of the record is invalid; per M1 it is also at most the index of the
first non-finite strain after a finite one. Missing crosshead or L0 skips the
window rule with flag ``extensometer:missing_input``.

M9 ReL (concrete rule; option (A))
----------------------------------
Y is the yield range (i_p, i_h) on finite-force samples, where i_h is the
first sample after i_p with sigma >= ReH. If the stress never returns to ReH
(method-spec v2.2, M11 amendment, band reading), the remaining record ends at
the last sample with F > 0.5*Rm, the M7 search threshold with the provisional
Rm, so the terminal force drop of fracture or unloading is excluded (v2.5,
R8-1). Every result from this branch has the flag ``fm:no_return_to_reh``
(v2.6; see the known limitations below). The rule below is
first applied to this remainder (i_p, end); it gives the plateau level P, the
plateau minimum (candidate, step 4) and the plateau band
[candidate, 2P - candidate]. Work hardening begins only when the stress,
after having been inside the band at or after the recovery i_r, rises above
it by more than the M7 drop threshold d at ReH (v2.3 margin:
sigma > 2P - candidate + d, because a symmetric serrated plateau's own maxima
lie on 2P - candidate); then i_h = i_m is the highest sample from that rise on, Fm = F(i_m), and
the rule is applied again over (i_p, i_h). If it never rises above the band,
Fm and Rm are NaN with ``fm:no_work_hardening`` and ``rel:no_work_hardening``,
ReL is the result over the remainder, and the yield range runs to its end (so
Rp0.2 gets ``rp02:in_yield_plateau``).

1. ``rel_incl_transient`` = min sigma over Y.
2. Plateau level P = median of the samples in Y with
   sigma <= (rel_incl_transient + ReH)/2. The bound keeps the work-hardening
   rise towards ReH out of the estimate.
3. Transient: starting at i_p, the first sample with sigma < P - d begins it
   (d = max(drop_rel*ReH, drop_noise_factor*s_n), the M7 drop threshold at
   ReH; this hysteresis stops noise on the falling flank from ending the
   transient early). The first later sample with sigma >= P ends it
   (completed recovery to the plateau level) at i_r. If no sample is below
   P - d, there is no transient (i_r = first sample of Y). If the curve never
   recovers to P, there is no plateau (go to 6).
4. Candidate ReL = min sigma over [i_r, i_h).
5. Plateau extent: the band [candidate, P + (P - candidate)] is the plateau
   band (it widens with the observed serration depth). The extent is
   e_p(i_last) - e_p(i_r), with i_last the last sample in [i_r, i_h) inside
   the band (strain-valid samples only). If this in-band run reaches the last
   strain-valid sample while later finite-force samples are still in the band,
   the plateau extends beyond ``k_ext``: ReL is NaN with
   ``rel:extensometer_invalid`` (M2, R8-2). If the extent is >=
   ``plateau_min_extent``, ReL = candidate.
6. Otherwise ReL = the transient minimum (min sigma over (i_p, i_r), or over
   Y without a transient or recovery) with flag ``rel:single_minimum``.

Reference cases (method-spec M9): (i) plateau 380 +- 1 after a 372 dip: P is
about 380, recovery ends on the rising flank, ReL = 379. (ii) serrations
+- 3: P about 380, band [377, 383], ReL = 377. (iii) monotone rise after the
dip: the candidate is the recovery sample itself, the band collapses to
[P, P], the extent is about 0, so ReL = 372 with ``single_minimum``.
(iv) no upper peak: M7 gives continuous yielding, ReL is NaN with
``no_upper_yield``. (v) noise: P is a median, so recovery is found at the
same place; the noisy ReL is low by the noise extreme over the plateau (the
min/max noise bias, A5).

Known method limitations (method-spec v2.6; documented, not corrected)
----------------------------------------------------------------------
- **R8-1, gradual tail in the M11 no-return branch.** The v2.5 truncation
  handles abrupt drops only. A gradual force decline after the plateau that
  stays above 0.5*Rm still corrupts the branch: necking to 0.8*Rm gives ReL
  316 instead of 379 MPa on the synthetic case, and Fm NaN. The flag
  ``fm:no_return_to_reh`` marks every result from this branch, with or without
  such a tail, as unreliable for records with a gradual tail. S355 is expected
  never to enter it. Follow-up F9 is a better rule.
- **R8-7, M7 under noise.** A noisy plateau without an upper peak (M9 case
  (iv)) can be classified as discontinuous, which reports a spurious ReH: 65 %
  of 1000 seeds at 0.5 MPa noise on the synthetic case. The opt-in statistical
  test ``test_r8_7_...`` records the rate. Follow-up F7 is a prominence
  criterion.
- **R8-8, M9 on a smoothly rising plateau.** P is a median, so the recovery
  falls mid-plateau and the lower half counts as transient. Noise-free, the
  band collapses as in case (iii), so ReL is the transient minimum with
  ``rel:single_minimum``. With noise, ReL is about P minus the noise extreme.
  Neither is the plateau start. ReL is not compared with lab values. Follow-up
  F8 is a reference case with a declared expectation.
"""
