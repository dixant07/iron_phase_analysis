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
