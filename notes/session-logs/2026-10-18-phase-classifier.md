# Phase-classifier implementation, 2026-10-18

Source design: `new_arch.md` supplied by the user. No external scientific
validation claims are added by this implementation.

## Task ledger

- PHASE-1 — done — Codex: inspect original Fe geometry and O-parent logic.
  Findings: ideal angle multiplicities and image-corrected vectors are correct;
  O parents are the reverse of the same Fe→O graph. Sharing categories do not
  identify phases. Original LAMMPS export loses cell tilt.
- PHASE-2 — done — Codex: shared Fe geometry in `count.py`, O-first classifier
  in `phase.py`, pipeline/export integration in `main.py`, configuration in
  `phase_config.json`, and reference fitting in `fit_phase_references.py`.
- PHASE-3 — done — Codex: 14 behavior/integration tests in
  `tests/test_phases.py`; independent supplied-structure CLI smoke checks in
  `analysis_results/phase_validation/`.
- PHASE-4 — done — Codex: usage, scoring equations, interpretation, diagnostics,
  neighbor comparison, and scientific limitations documented in `README.md`.

## Checks that ran

- `.venv/bin/python -m unittest discover -s tests -v`: 14 tests passed.
- Supplied Fe3O4 reference: 3000 Fe and 4000 O, all magnetite-like; hard Fe
  geometry counts 1000 tetrahedral, 2000 octahedral, 0 other.
- Supplied FeO reference: 500 Fe and 500 O, all wustite-like; hard Fe geometry
  counts 0 tetrahedral, 500 octahedral, 0 other.
- Representative `input_frames/100_NPT_allatom/reduction.0.poscar`:
  1068/3600 Fe and 1566/4800 O magnetite-like; 2532/3600 Fe and 3234/4800 O
  other; zero wustite-like with heuristic defaults. This is classifier output,
  not ground-truth phase composition. Other O fraction = 67.375%.
- Reference fitting CLI ran on supplied pure structures, producing
  `analysis_results/phase_validation/reference_model.json`. That serialized
  model labels separately generated ideal FeO (8 atoms) and magnetite
  (56 atoms) correctly in a smoke check; results saved in
  `fitted_model_smoke.json`.
- Tests exercise distinct periodic-image bonds, angle/length geometry,
  permutation/translation invariance, rotated triclinic dump round trips,
  soft scores, cutoff sensitivity, unknown/metal/dilute environments, small
  coordinate perturbations, a vacancy, a tetrahedral interstitial, mixed-domain
  interior/boundary behavior, disorder rejection, reference model settings,
  parallel processing, and separate geometry/phase summary totals.

## Interpretation and remaining scientific work

The example model is fitted on pure reference structures, not calibrated
against finite-temperature data. The initial trajectory frame has a large
unknown fraction, so matching thermal/strain/defect references and independent
validation are necessary before quantitative trajectory phase interpretation.
Small-displacement tests do not establish thermal robustness. No full
trajectory rerun was performed; original trajectory exports were preserved.

The implemented second-shell descriptors measure graph context, not exact
spinel graph matching or atomic-template fitting. Optional affine template
fitting, automatic RDF cutoff selection, and temporal hysteresis were not
implemented. POSCAR site indices cannot safely serve as persistent atom IDs
through reordering/removal; temporal analysis needs source IDs. Interfaces are
candidates distinguished within the requested `other` category, not asserted
physical interfaces. All phase scores are heuristic similarities, not
calibrated probabilities or oxidation-state assignments.
