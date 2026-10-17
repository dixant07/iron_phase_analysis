analysis_results/ ==> directory contains analysis results on input frames
input_frames/ ==> directory contains input frames
visualization/ ==> directory contains ovito visualization states

main.py ==> I/O logic
count.py ==> voronoi tessellation and angle rms logic

## Fe coordination counts versus frame

Run `.venv/bin/python fe_coordination.py` to analyze frames 0 through 1001
(inclusive) in `analysis_results/output_ovito/`. The script counts Fe atoms
(type 1) using the exported `coordination_number`, which is the number of
oxygen neighbors selected by CrystalNN in `main.py` and `count.py`.
Counts include distorted environments regardless of their `fe_type` class.

Results are written to `analysis_results/fe_coordination/`:
- `fe_coordination_counts.csv`: frame, timestep, status, total Fe, and one
  count column for each observed coordination number (including zero).
- `fe_coordination_vs_frame.png` and `.pdf`: counts versus frame number.
- `missing_frames.txt`: requested frame numbers with no dump file.
- `fe_cn4_cn6_percentages.csv` and `fe_cn4_cn6_percentages_vs_frame.png` / `.pdf`:
  CN4 and CN6 percentages, using `100 * count(CN) / (count(CN4) + count(CN6))`.
  All other coordination numbers are excluded from this denominator. Missing
  frames or frames without either CN4 or CN6 have blank percentages and plot gaps.

Missing frames have blank CSV counts and plot gaps; an absent coordination
number in an available frame has count zero. Use `--strict` to fail when any
requested file is missing. Customize the range and paths with `--from-frame`,
`--to-frame`, `--input-dir`, and `--output-dir`. Run with `--help` for details.

## Local phase classification

`main.py` now implements the O-centered architecture in `new_arch.md` through
`phase.py`. Run it on POSCAR coordinates, rather than trying to infer phases
from the old polyhedron totals or from the CN4/CN6 percentage plot:

```bash
.venv/bin/python main.py 0 1001 \
  --frames-dir input_frames/100_NPT_allatom \
  --output-dir analysis_results/phase_analysis \
  --phase-config phase_config.json --workers 4
```

Both `.dump` and `.extxyz` now include `phase_id`:

- `0`: other (rejected/unknown, inconsistent, or mixed/interface candidate).
- `1`: magnetite-like local environment.
- `2`: wustite-like local environment.

`phase_reason` separates accepted sites (`0`), insufficient/unfamiliar
structure (`1`), ambiguous/interface candidates (`2`), and inconsistent
neighborhoods (`3`). An interface candidate is a structural interpretation,
not proof of an interface. These are local structural labels, not oxidation
states, thermodynamic phase fractions, or calibrated probabilities.

### What was correct and what changed

The original sorted angle-RMS test has the correct ideal angle multiplicities:
tetrahedral 6 × 109.47 degrees; octahedral 12 × 90 and 3 × 180 degrees.
CrystalNN's neighbor `site.coords` already contains its periodic-image offset,
so subtracting the central coordinates was correct. Reversing the selected
Fe→O edges to obtain oxygen parent lists was also correct and is retained.
Oxygen is **not** queried independently with CrystalNN; Fe and O use the same
reciprocal bipartite graph. The original `parent_type_id` describes only which
Fe site types occur among an oxygen's parents, not their phase or multiplicity.
Octahedral Fe alone cannot distinguish the two phases.

The updated pipeline:

1. Builds the periodic Fe–O graph with the original CrystalNN shell selection
   followed by O filtering. Distinct periodic images remain distinct bonds;
   repeated base IDs in a small periodic cell are valid.
2. Uses one shared Fe geometry function in `count.py` and `main.py`. It keeps
   CN4/CN6 angle criteria and additionally rejects excessive normalized bond
   length distortion (`std(length)/mean(length)`). Other CNs remain unknown
   at this stage. Invalid/zero-length vectors now fail explicitly.
3. Computes continuous tetra/octa/unknown site similarities. For the matching
   CN, geometry similarity is `exp(-0.5*((angle_rms/angle_limit)^2 +
   (bond_distortion/bond_limit)^2))`; unknown receives the remainder.
4. Sums these soft scores at each O to form `(tet, oct, unknown)` fingerprints.
   The ideal centers are `(1,3,0)` and `(0,6,0)`.
5. Measures physical-sphere Fe/O ratios at two radii (6 and 9 angstrom by
   default), including the central O. Different periodic images in the sphere
   represent different physical atoms. Small spheres are rejected when they
   lack enough samples. Finite-shell ratios can differ substantially from
   bulk stoichiometry even for ideal crystals; use matching references.
6. Builds graph context through O→Fe→O and then collects the distinct Fe
   indices connected to that oxygen neighborhood. Tetra/unknown fractions
   measure second-shell context. A lone tetrahedral interstitial cannot pass
   the magnetite contextual-support requirement on its own.
7. Computes absolute phase mismatch and relative similarity. Applies one
   simultaneous, spatially and fingerprint-weighted smoothing pass. Unknown
   context remains in the support denominator; smoothing cannot rescue an
   absolutely rejected site. Low confidence, unfamiliar fingerprints, too many
   unknown parents, sparse composition sampling, and poor neighborhood support
   return `other` rather than forcing a binary assignment.
8. Transfers phase labels to Fe only when at least 60% of **all** its bonded
   oxygen atoms support one phase. Fe with fewer than three O bonds stays
   other. Thus defective CN5 Fe can acquire a contextual phase without being
   relabeled as an octahedral site.

The LAMMPS writer also now preserves tilted cells: it rotates the lattice and
positions together to restricted triclinic form and writes tilt factors and
correct bounding limits. Previously it used lattice lengths as an orthogonal
box, corrupting the exported geometry of tilted inputs.

### Diagnostics, configuration, and summaries

The JSON file `phase_config.json` contains all phase/neighbor defaults.
Partial JSON overrides are also accepted. The existing 30-degree angle limits
remain configurable via `--tetra-limit` and `--octa-limit`; they are inherited
starting values, not validated universal thermal tolerances.

To compare neighbor schemes, set `neighbor_method` to `cutoff` and choose
`bond_cutoff` in angstrom. The cutoff graph uses pymatgen's periodic image
search, including triclinic cells. Counts at cutoff ± `cutoff_delta` define
`neighbor_stability = 1/(1 + CN_high - CN_low)`, reducing geometry similarity
for sensitive shells. This is a **user-selected cutoff**, not an automatic RDF
minimum finder. With CrystalNN, `neighbor_stability=-1` explicitly means this
cutoff sensitivity was not evaluated. Neighbor-finder failures are reported
as frame failures, not converted into zero coordination.

Exports retain the old property names and sharing IDs, and append bond
length distortion, soft site/parent scores, phase scores, support, absolute
distances, composition ratios/sample sizes, and second-shell fractions.
Distances and composition descriptors are O-centered (`-1` on Fe).
`phase_confidence` is the selected label's relative similarity, zero when
rejected. The raw two-way phase scores can express a numerical preference
even for a rejected site; always consult `phase_id` and `phase_reason`.

The summary CSV keeps `octahedral_fe`, `tetrahedral_fe`, and `other_fe` as
**geometry counts**. New `phase_magnetite_fe`, `phase_wustite_fe`,
`phase_other_fe` and the corresponding `_o` columns contain phase counts.
`pct_phase_*` uses all atoms of the corresponding species as the denominator,
including other; it is separate from the earlier CN4/CN6 normalization.
`phase_metadata.json` records configuration, angle limits, IDs, input range,
and the fitted model if supplied. Frame calculation errors produce a nonzero
exit status after saving successfully processed frames.

### Fit reference distributions

Without a reference model the classifier uses explicit ideal feature centers
and heuristic scales in `phase.py`. Fit temperature/potential/strain-matched
homogeneous reference structures to replace these with regularized Gaussian
(Mahalanobis) descriptor distances:

```bash
.venv/bin/python fit_phase_references.py \
  --magnetite input_frames/fe3o4/reduction.0.poscar \
  --wustite input_frames/feo/reduction.0.poscar \
  --phase-config phase_config.json \
  --output analysis_results/phase_reference_model.json

.venv/bin/python main.py 0 1001 \
  --frames-dir input_frames/100_NPT_allatom \
  --output-dir analysis_results/phase_analysis_fitted \
  --phase-config phase_config.json \
  --reference-model analysis_results/phase_reference_model.json --workers 4
```

Repeat `--magnetite`/`--wustite` for multiple structures. Model metadata must
match the inference configuration and angle limits. Fits use standardized
features with a 0.25 diagonal covariance regularizer and absolute rejection
limits (99th training distance percentile × 1.25, floored by `max_distance`).
Training coverage is **not** held-out accuracy or probability calibration;
use separate thermal, strain, defect, and interface samples to tune thresholds
and measure confusion matrices/unknown fractions. The fitted model under
`analysis_results/phase_validation/` is a demonstration fitted to the two
provided pure references, not a temperature-calibrated production model.

### Validation and remaining scientific scope

Run `.venv/bin/python -m unittest discover -s tests -v`. The suite checks ideal
FeO/Fe3O4, geometry and graph invariants, small random displacements, a vacancy,
a tetrahedral interstitial, separated phase domains with a joined interface,
a disordered cell, oxygen-free Fe, unknown rejection, permutation/translation
invariance, cutoff sensitivity, reference-model loading, parallel CLI output,
and triclinic LAMMPS geometry round trips.

`analysis_results/phase_validation/` holds separate smoke-test outputs from
the supplied pure structures and a representative trajectory frame. Existing
trajectory analysis files are not automatically regenerated by the code update.

The topology descriptors measure local graph context; they do not perform
full spinel/rock-salt graph isomorphism or species-matched affine atomic-template
fitting. Those more expensive optional checks in `new_arch.md` remain future
extensions. Automatic RDF-based cutoff selection and temporal hysteresis are
also not applied. In particular, POSCAR row numbers are not guaranteed stable
atom IDs when atoms are reordered or removed; temporal filtering would require
persistent source IDs to avoid mixing different atoms. Small random
perturbations in tests are not equilibrated finite-temperature validation.
