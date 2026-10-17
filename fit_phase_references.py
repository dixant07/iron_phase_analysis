#!/usr/bin/env python3
"""Fit O-centered phase descriptor distributions from labeled structures.

Example (use additional held-out thermal/defect references for validation):
  .venv/bin/python fit_phase_references.py \
      --magnetite input_frames/fe3o4/reduction.0.poscar \
      --wustite input_frames/feo/reduction.0.poscar \
      --output analysis_results/phase_reference_model.json

Repeat --magnetite / --wustite for multiple independent structures. Labels
must describe homogeneous reference domains; do not label an entire mixed
trajectory as a pure phase. This fits similarity distances, not probabilities.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from count import tetra_limit, octa_limit
from main import calculate_frame_properties
from phase import PhaseConfig, fit_reference_model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--magnetite", action="append", required=True, type=Path)
    parser.add_argument("--wustite", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("analysis_results/phase_reference_model.json"))
    parser.add_argument("--phase-config", type=Path)
    parser.add_argument("--tetra-limit", type=float, default=tetra_limit)
    parser.add_argument("--octa-limit", type=float, default=octa_limit)
    args = parser.parse_args()
    try:
        config = PhaseConfig.load(args.phase_config)
        samples, sources = {}, {}
        for phase, paths in (("magnetite", args.magnetite), ("wustite", args.wustite)):
            features, sources[phase] = [], []
            for path in paths:
                structure = Structure.from_file(path)
                props = calculate_frame_properties(structure, args.tetra_limit, args.octa_limit,
                                                   phase_config=config)
                oxygen = np.array([s.specie.symbol == "O" for s in structure])
                eligible = (oxygen & (props["coordination_number"] >= 3) &
                            (props["coordination_number"] <= 7) &
                            (props["composition_sample_sizes"].min(axis=1) >= config.min_composition_atoms))
                values = props["phase_features"][eligible]
                if not len(values):
                    raise ValueError(f"No eligible O environments in {path}")
                features.append(values)
                sources[phase].append(str(path.resolve()))
                print(f"{phase}: {path}: {len(values)} O-centered descriptors", flush=True)
            samples[phase] = np.vstack(features)
        model = fit_reference_model(samples, config, args.tetra_limit, args.octa_limit)
        model["sources"] = sources
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(model, indent=2) + "\n")
        print(f"Saved {args.output}. Validate on independent references before scientific use.")
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
