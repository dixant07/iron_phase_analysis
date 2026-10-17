#!/usr/bin/env python3
"""
main.py - Main Pipeline for Trajectory Polyhedra Analysis & OVITO Visualization

This script manages I/O and all calculations needed for visualization outputs:
  - Takes from_frame and to_frame inputs (via CLI flags, positional arguments, or interactive prompt).
  - Shares Fe geometry and periodic neighbor filtering with `count.py`.
  - Classifies O-centered environments and transfers supported phase labels to Fe.
  - Calculates per-atom visualization properties (fe_type, parent_fe_id array, coordination numbers, angle RMS).
  - Includes all parent Fe IDs (tetrahedral, octahedral, and other Fe parents) in parent_fe_id.
  - Exports per-frame OVITO visualization files (.extxyz, .dump).
  - Exports a summary CSV file with polyhedral counts for each frame.
"""

import os
import sys
import argparse
import warnings
import json
from dataclasses import asdict
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN

# Suppress pymatgen warnings when run
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

# Import the core geometric algorithm and constants from count.py
from count import (
    fe_geometry,
    oxygen_neighbors,
    tetra_limit as DEFAULT_TETRA_LIMIT,
    octa_limit as DEFAULT_OCTA_LIMIT,
)

from phase import (PhaseConfig, PHASE_NAMES, REASONS, FEATURE_NAMES,
                   classify_phases, load_reference_model)


def calculate_frame_properties(
    structure: Structure,
    t_limit: float = DEFAULT_TETRA_LIMIT,
    o_limit: float = DEFAULT_OCTA_LIMIT,
    crystal_nn: Optional[CrystalNN] = None,
    phase_config: Optional[PhaseConfig] = None,
    reference_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one periodic bipartite graph, Fe geometry, then O-first phases.

    Original property names and IDs are retained. parent_fe_id and O CN count
    periodic bonds, so a small cell may repeat a base site ID for different images.
    O parent_type_id denotes sharing categories, independently of phase_id.
    """
    config = phase_config or PhaseConfig()
    if not all(np.isfinite(x) and x > 0 for x in (t_limit, o_limit)):
        raise ValueError("Angle limits must be finite and positive")
    symbols = [site.specie.symbol for site in structure]
    if set(symbols) - {"Fe", "O"}:
        raise ValueError("The phase pipeline supports Fe/O structures only")
    if crystal_nn is None:
        crystal_nn = CrystalNN()
    n = len(structure)
    fe_indices = [i for i, symbol in enumerate(symbols) if symbol == "Fe"]
    o_indices = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    fe_type = np.full(n, -1, dtype=int)
    cn = np.zeros(n, dtype=int)
    rms = np.full(n, -1.)
    distortion = np.full(n, -1.)
    soft = np.zeros((n, 3))
    stability = np.full(n, -1.)  # -1: cutoff sensitivity not evaluated
    fe_to_o = {i: [] for i in fe_indices}
    o_to_fe = {i: [] for i in o_indices}
    shell_rows = defaultdict(list)
    if config.neighbor_method == "cutoff" and o_indices and fe_indices:
        centers, neighbors, images, distances = structure.get_neighbor_list(
            config.bond_cutoff + config.cutoff_delta,
            sites=[structure[i] for i in fe_indices])
        for center, neighbor, image, distance in zip(centers, neighbors, images, distances):
            if symbols[neighbor] == "O":
                shell_rows[fe_indices[center]].append((int(neighbor), image, distance))
    for i in fe_indices:
        if config.neighbor_method == "crystalnn":
            neighbors = oxygen_neighbors(structure, i, crystal_nn) if o_indices else []
            vectors = [nb["site"].coords - structure[i].coords for nb in neighbors]
            targets = [int(nb["site_index"]) for nb in neighbors]
        else:
            rows = shell_rows[i]
            selected = [(j, image) for j, image, d in rows if d <= config.bond_cutoff]
            targets = [j for j, _ in selected]
            vectors = [structure.lattice.get_cartesian_coords(
                structure[j].frac_coords + image - structure[i].frac_coords)
                for j, image in selected]
            low = sum(d <= config.bond_cutoff - config.cutoff_delta for _, _, d in rows)
            high = len(rows)
            stability[i] = 1. / (1 + high - low)
        ft, value, dr, scores = fe_geometry(vectors, t_limit, o_limit, config.bond_limit)
        if stability[i] >= 0:
            scores[:2] *= stability[i]
            scores[2] = 1 - scores[:2].sum()
        fe_type[i], cn[i], rms[i], distortion[i], soft[i] = ft, len(targets), value, dr, scores
        fe_to_o[i] = targets
        for o in targets:
            o_to_fe[o].append(i)

    n_oct, n_tet, n_other = (np.zeros(n, dtype=int) for _ in range(3))
    parent_type = np.full(n, -1, dtype=int)
    max_parents = max(6, max((len(v) for v in o_to_fe.values()), default=0))
    parents = np.zeros((n, max_parents), dtype=int)
    for i in fe_indices:
        parents[i, 0] = i + 1
    # Existing sharing category IDs, retained for saved OVITO visualizations.
    sharing = {(True, False, False): 0, (False, True, False): 1,
               (False, False, True): 2, (True, True, False): 3,
               (True, False, True): 4, (False, True, True): 5,
               (True, True, True): 6}
    soft_parents = np.zeros((n, 3))
    for o, indices in o_to_fe.items():
        cn[o] = len(indices)
        parents[o, :len(indices)] = np.asarray(indices, dtype=int) + 1
        labels = fe_type[indices]
        n_oct[o], n_tet[o], n_other[o] = (np.count_nonzero(labels == k) for k in (0, 1, 2))
        parent_type[o] = sharing.get((n_oct[o] > 0, n_tet[o] > 0, n_other[o] > 0), -1)
        if indices:
            soft_parents[o] = soft[indices].sum(axis=0)
    props = {
        "fe_type": fe_type, "parent_fe_id": parents, "coordination_number": cn,
        "angle_rms": rms, "bond_distortion": distortion, "fe_site_scores": soft,
        "fe_score_tet": soft[:, 0], "fe_score_oct": soft[:, 1], "fe_score_unknown": soft[:, 2],
        "neighbor_stability": stability,
        "soft_tet_parents": soft_parents[:, 0], "soft_oct_parents": soft_parents[:, 1],
        "soft_unknown_parents": soft_parents[:, 2],
        "parent_type_id": parent_type, "n_oct_parents": n_oct,
        "n_tet_parents": n_tet, "n_other_parents": n_other,
        "octahedral_count": int(np.count_nonzero(fe_type == 0)),
        "tetrahedral_count": int(np.count_nonzero(fe_type == 1)),
        "other_count": int(np.count_nonzero(fe_type == 2)),
        "total_fe": len(fe_indices), "total_o": len(o_indices),
        "octa_rms_list": rms[fe_type == 0].tolist(), "tetra_rms_list": rms[fe_type == 1].tolist(),
    }
    props.update(classify_phases(structure, props, fe_to_o, o_to_fe, config, reference_model))
    props["composition_atoms_inner"] = props["composition_sample_sizes"][:, 0]
    props["composition_atoms_outer"] = props["composition_sample_sizes"][:, 1]
    return props


# Scalar diagnostics appended to both export formats with identical names.
EXTRA_PROPERTIES = {
    "bond_distortion": "R", "fe_score_tet": "R", "fe_score_oct": "R",
    "fe_score_unknown": "R", "neighbor_stability": "R",
    "soft_tet_parents": "R", "soft_oct_parents": "R", "soft_unknown_parents": "R",
    "phase_id": "I", "phase_reason": "I", "phase_score_magnetite": "R",
    "phase_score_wustite": "R", "phase_confidence": "R", "phase_support": "R",
    "phase_distance_magnetite": "R", "phase_distance_wustite": "R",
    "local_fe_o_ratio_inner": "R", "local_fe_o_ratio_outer": "R",
    "second_shell_tet_fraction": "R", "second_shell_unknown_fraction": "R",
    "composition_atoms_inner": "I", "composition_atoms_outer": "I",
}


def extra_values(props, index):
    return " ".join(str(int(props[key][index])) if kind == "I" else f"{props[key][index]:.8f}"
                    for key, kind in EXTRA_PROPERTIES.items())


def write_extxyz(filepath: str, structure: Structure, props: Dict[str, Any]) -> None:
    """
    Write structure and per-atom properties in Extended XYZ format (.extxyz).
    parent_fe_id is written as an array property with dimension matching the number of parent slots.
    Natively readable by OVITO with automatic property type assignment.
    """
    n_atoms = len(structure)
    lat = structure.lattice.matrix.flatten()
    lat_str = " ".join(f"{x:.8f}" for x in lat)

    parent_fe_id = props["parent_fe_id"]
    dim_p = parent_fe_id.shape[1]

    prop_header = (
        f"Properties=species:S:1:pos:R:3:fe_type:I:1:parent_fe_id:I:{dim_p}:"
        "coordination_number:I:1:angle_rms:R:1:parent_type_id:I:1:"
        "n_oct_parents:I:1:n_tet_parents:I:1:n_other_parents:I:1"
    )
    prop_header += "".join(f":{key}:{kind}:1" for key, kind in EXTRA_PROPERTIES.items())
    pbc = " ".join("T" if value else "F" for value in structure.lattice.pbc)
    header = f'Lattice="{lat_str}" {prop_header} pbc="{pbc}"' 

    fe_type = props["fe_type"]
    cn = props["coordination_number"]
    angle_rms_val = props["angle_rms"]
    parent_type_id = props["parent_type_id"]
    n_oct = props["n_oct_parents"]
    n_tet = props["n_tet_parents"]
    n_oth = props["n_other_parents"]

    with open(filepath, "w") as f:
        f.write(f"{n_atoms}\n{header}\n")
        for i, atom in enumerate(structure):
            sym = atom.specie.symbol
            x, y, z = atom.coords
            p_str = " ".join(f"{pid:6d}" for pid in parent_fe_id[i])
            f.write(
                f"{sym:<2} {x:12.6f} {y:12.6f} {z:12.6f} "
                f"{fe_type[i]:3d} {p_str} "
                f"{cn[i]:3d} {angle_rms_val[i]:10.4f} {parent_type_id[i]:3d} "
                f"{n_oct[i]:3d} {n_tet[i]:3d} {n_oth[i]:3d} {extra_values(props, i)}\n"
            )


def write_lammps_dump(filepath: str, structure: Structure, frame_num: int,
                      props: Dict[str, Any]) -> None:
    """
    Write structure and per-atom properties in LAMMPS dump format (.dump).
    parent_fe_id is written with bracket notation parent_fe_id[1]..parent_fe_id[M],
    which OVITO automatically groups into a single vector/array property.
    """
    n_atoms = len(structure)
    lat = structure.lattice
    # Rotate to a restricted triclinic LAMMPS cell without changing geometry.
    a_vec, b_vec, c_vec = lat.matrix
    lx = np.linalg.norm(a_vec)
    e1 = a_vec / lx
    xy, xz = np.dot(b_vec, e1), np.dot(c_vec, e1)
    b_perp = b_vec - xy * e1
    ly = np.linalg.norm(b_perp)
    e2 = b_perp / ly
    yz = np.dot(c_vec, e2)
    lz = np.dot(c_vec, np.cross(e1, e2))
    if lz <= 0:
        raise ValueError("LAMMPS export requires a right-handed lattice")
    restricted = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
    positions = structure.frac_coords @ restricted
    tilted = not np.allclose([xy, xz, yz], 0, atol=1e-10)
    boundary = " ".join("pp" if value else "ff" for value in lat.pbc)

    parent_fe_id = props["parent_fe_id"]
    dim_p = parent_fe_id.shape[1]
    parent_cols = " ".join(f"parent_fe_id[{k+1}]" for k in range(dim_p))

    fe_type = props["fe_type"]
    cn = props["coordination_number"]
    angle_rms_val = props["angle_rms"]
    parent_type_id = props["parent_type_id"]
    n_oct = props["n_oct_parents"]
    n_tet = props["n_tet_parents"]
    n_oth = props["n_other_parents"]

    with open(filepath, "w") as f:
        f.write(f"ITEM: TIMESTEP\n{frame_num}\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{n_atoms}\n")
        if tilted:
            f.write(f"ITEM: BOX BOUNDS xy xz yz {boundary}\n")
            xlo = min(0., xy, xz, xy+xz)
            xhi = lx + max(0., xy, xz, xy+xz)
            ylo, yhi = min(0., yz), ly + max(0., yz)
            f.write(f"{xlo:.10f} {xhi:.10f} {xy:.10f}\n")
            f.write(f"{ylo:.10f} {yhi:.10f} {xz:.10f}\n")
            f.write(f"0.0000000000 {lz:.10f} {yz:.10f}\n")
        else:
            f.write(f"ITEM: BOX BOUNDS {boundary}\n")
            f.write(f"0.000000 {lx:.10f}\n0.000000 {ly:.10f}\n0.000000 {lz:.10f}\n")
        f.write(
            f"ITEM: ATOMS id type x y z fe_type {parent_cols} "
            "coordination_number angle_rms parent_type_id "
            "n_oct_parents n_tet_parents n_other_parents " + " ".join(EXTRA_PROPERTIES) + "\n"
        )
        for i, atom in enumerate(structure):
            atom_type = 1 if atom.specie.symbol == "Fe" else 2
            x, y, z = positions[i]
            p_str = " ".join(f"{pid:6d}" for pid in parent_fe_id[i])
            f.write(
                f"{i + 1:6d} {atom_type:2d} {x:12.6f} {y:12.6f} {z:12.6f} "
                f"{fe_type[i]:3d} {p_str} "
                f"{cn[i]:3d} {angle_rms_val[i]:10.4f} {parent_type_id[i]:3d} "
                f"{n_oct[i]:3d} {n_tet[i]:3d} {n_oth[i]:3d} {extra_values(props, i)}\n"
            )


def process_single_frame(
    frame_num: int,
    frames_dir: str,
    output_dir: str,
    export_format: str = "both",
    t_limit: float = DEFAULT_TETRA_LIMIT,
    o_limit: float = DEFAULT_OCTA_LIMIT,
    phase_config: Optional[PhaseConfig] = None,
    reference_model: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Process a single frame:
      - Reads POSCAR file.
      - Calculates visualization properties using angle_rms from count.py.
      - Writes OVITO files (.extxyz, .dump).
      - Returns summary dict for the CSV file.
    """
    filename = f"reduction.{frame_num}.poscar"
    filepath = os.path.join(frames_dir, filename)

    if not os.path.exists(filepath):
        print(f"[Warning] Frame {frame_num} not found at {filepath}. Skipping.", file=sys.stderr)
        return None

    structure = Structure.from_file(filepath)
    props = calculate_frame_properties(structure, t_limit=t_limit, o_limit=o_limit,
                                       phase_config=phase_config, reference_model=reference_model)

    os.makedirs(output_dir, exist_ok=True)

    if export_format in ("extxyz", "both"):
        extxyz_path = os.path.join(output_dir, f"reduction.{frame_num}.extxyz")
        write_extxyz(extxyz_path, structure, props)

    if export_format in ("dump", "both"):
        dump_path = os.path.join(output_dir, f"reduction.{frame_num}.dump")
        write_lammps_dump(dump_path, structure, frame_num, props)

    oct_count = props["octahedral_count"]
    tet_count = props["tetrahedral_count"]
    other_count = props["other_count"]
    fe_total = props["total_fe"]
    o_total = props["total_o"]

    summary = {
        "frame": frame_num,
        "octahedral_fe": oct_count,
        "tetrahedral_fe": tet_count,
        "other_fe": other_count,
        "total_fe": fe_total,
        "total_o": o_total,
        "pct_octahedral": round((oct_count / fe_total * 100.0) if fe_total else 0.0, 2),
        "pct_tetrahedral": round((tet_count / fe_total * 100.0) if fe_total else 0.0, 2),
        "pct_other": round((other_count / fe_total * 100.0) if fe_total else 0.0, 2),
        "mean_oct_rms": round(float(np.mean(props["octa_rms_list"])), 3) if props["octa_rms_list"] else None,
        "mean_tet_rms": round(float(np.mean(props["tetra_rms_list"])), 3) if props["tetra_rms_list"] else None,
    }

    for species in ("Fe", "O"):
        mask = np.array([site.specie.symbol == species for site in structure])
        total = int(mask.sum())
        for phase_id, name in PHASE_NAMES.items():
            value = int(np.count_nonzero(props["phase_id"][mask] == phase_id))
            summary[f"phase_{name}_{species.lower()}"] = value
            summary[f"pct_phase_{name}_{species.lower()}"] = 100. * value / total if total else None
    summary["interface_or_ambiguous_atoms"] = int(np.count_nonzero(props["phase_reason"] == 2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Main Pipeline: Run polyhedra analysis on reduction frames and export OVITO visualization & CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pos_from_frame", nargs="?", type=int, default=None, help="Starting frame (positional)")
    parser.add_argument("pos_to_frame", nargs="?", type=int, default=None, help="Ending frame (positional)")
    parser.add_argument("--from-frame", "-f", type=int, default=None, help="Starting frame number")
    parser.add_argument("--to-frame", "-t", type=int, default=None, help="Ending frame number")
    parser.add_argument("--frames-dir", "-d", type=str, default="reduction_frames", help="Directory containing reduction.<frame>.poscar files")
    parser.add_argument("--output-dir", "-o", type=str, default="output_ovito", help="Output directory for OVITO files and CSV")
    parser.add_argument("--format", choices=["extxyz", "dump", "both"], default="both", help="Export format for OVITO")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of parallel worker processes (default: CPU count)")
    parser.add_argument("--tetra-limit", type=float, default=DEFAULT_TETRA_LIMIT, help="RMS cutoff for tetrahedral geometry (°)")
    parser.add_argument("--octa-limit", type=float, default=DEFAULT_OCTA_LIMIT, help="RMS cutoff for octahedral geometry (°)")
    parser.add_argument("--csv-filename", type=str, default="polyhedra_summary.csv", help="Name of summary CSV file")

    parser.add_argument("--phase-config", help="JSON PhaseConfig overrides; see phase_config.json")
    parser.add_argument("--reference-model", help="Fitted phase model from fit_phase_references.py")
    args = parser.parse_args()
    try:
        phase_config = PhaseConfig.load(args.phase_config)
        reference_model = (load_reference_model(args.reference_model, phase_config,
                           args.tetra_limit, args.octa_limit) if args.reference_model else None)
        if not all(np.isfinite(x) and x > 0 for x in (args.tetra_limit, args.octa_limit)):
            raise ValueError("Angle RMS cutoffs must be positive and finite")
        if args.workers is not None and args.workers < 1:
            raise ValueError("workers must be >= 1")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    # Determine from_frame and to_frame (CLI arguments or interactive prompt)
    from_frame = args.from_frame if args.from_frame is not None else args.pos_from_frame
    to_frame = args.to_frame if args.to_frame is not None else args.pos_to_frame

    if from_frame is None:
        try:
            from_frame = int(input("Enter starting frame number: ").strip())
        except (ValueError, EOFError):
            print("Invalid input. Please provide an integer frame number.", file=sys.stderr)
            sys.exit(1)

    if to_frame is None:
        try:
            to_frame = int(input("Enter ending frame number: ").strip())
        except (ValueError, EOFError):
            print("Invalid input. Please provide an integer frame number.", file=sys.stderr)
            sys.exit(1)

    if from_frame > to_frame:
        print(f"Error: Starting frame ({from_frame}) cannot be greater than ending frame ({to_frame}).", file=sys.stderr)
        sys.exit(1)

    frames = list(range(from_frame, to_frame + 1))
    num_frames = len(frames)
    max_cpus = os.cpu_count() or 1
    num_workers = min(args.workers or max_cpus, num_frames)

    print("=" * 65)
    print("   POLYHEDRA ANALYSIS & OVITO VISUALIZATION PIPELINE")
    print("=" * 65)
    print(f"Frames to process : {from_frame} to {to_frame} ({num_frames} total frames)")
    print(f"Geometry engine   : count.py (angle_rms)")
    print(f"Input directory   : {args.frames_dir}")
    print(f"Output directory  : {args.output_dir}")
    print(f"Export format     : {args.format}")
    print(f"Parallel workers  : {num_workers} (available CPUs: {max_cpus})")
    print(f"Angle RMS cutoffs : Tetrahedral <= {args.tetra_limit}°, Octahedral <= {args.octa_limit}°")
    print("=" * 65)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "phase_metadata.json"), "w") as stream:
        json.dump({"phase_ids": PHASE_NAMES, "phase_reasons": REASONS,
                   "feature_names": FEATURE_NAMES, "phase_config": asdict(phase_config),
                   "tetra_limit": args.tetra_limit, "octa_limit": args.octa_limit,
                   "reference_model": reference_model,
                   "phase_scores": "relative similarity, not calibrated probabilities",
                   "atom_ids": "1-based POSCAR site indices; not guaranteed persistent between frames",
                   "frames_dir": os.path.abspath(args.frames_dir),
                   "requested_frames": [from_frame, to_frame]},
                  stream, indent=2)
    results = []
    failures = 0
    if num_workers > 1:
        try:
            import multiprocessing as mp
            mp.set_start_method("fork", force=True)
        except Exception:
            pass

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_frame = {
                executor.submit(
                    process_single_frame,
                    f,
                    args.frames_dir,
                    args.output_dir,
                    args.format,
                    args.tetra_limit,
                    args.octa_limit,
                    phase_config,
                    reference_model,
                ): f
                for f in frames
            }
            for future in as_completed(future_to_frame):
                frame_idx = future_to_frame[future]
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                        print(
                            f"[Frame {frame_idx:4d}] "
                            f"Oct: {res['octahedral_fe']:4d} ({res['pct_octahedral']:5.1f}%) | "
                            f"Tet: {res['tetrahedral_fe']:4d} ({res['pct_tetrahedral']:5.1f}%) | "
                            f"Other: {res['other_fe']:4d} ({res['pct_other']:5.1f}%)"
                        )
                except Exception as e:
                    failures += 1
                    print(f"[Error] Frame {frame_idx} failed: {e}", file=sys.stderr)
    else:
        for f in frames:
            try:
                res = process_single_frame(
                    f,
                    args.frames_dir,
                    args.output_dir,
                    args.format,
                    args.tetra_limit,
                    args.octa_limit,
                    phase_config,
                    reference_model,
                )
                if res:
                    results.append(res)
                    print(
                        f"[Frame {f:4d}] "
                        f"Oct: {res['octahedral_fe']:4d} ({res['pct_octahedral']:5.1f}%) | "
                        f"Tet: {res['tetrahedral_fe']:4d} ({res['pct_tetrahedral']:5.1f}%) | "
                        f"Other: {res['other_fe']:4d} ({res['pct_other']:5.1f}%)"
                    )
            except Exception as e:
                failures += 1
                print(f"[Error] Frame {f} failed: {e}", file=sys.stderr)

    if not results:
        print("\nNo frames were processed successfully.", file=sys.stderr)
        sys.exit(1)

    # Sort results by frame index
    results.sort(key=lambda r: r["frame"])
    df = pd.DataFrame(results)

    # Save summary CSV
    csv_path = os.path.join(args.output_dir, args.csv_filename)
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 65)
    print("   PROCESSING COMPLETED")
    print("=" * 65)
    print(f"Summary CSV saved to: {csv_path}")
    print(f"OVITO trajectory files saved in: {args.output_dir}/")
    print("\nSummary Preview:")
    print(df.to_string(index=False))
    if failures:
        print(f"{failures} frame(s) failed; summary contains successful frames only.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
