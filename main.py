#!/usr/bin/env python3
"""
main.py - Main Pipeline for Trajectory Polyhedra Analysis & OVITO Visualization

This script manages I/O and all calculations needed for visualization outputs:
  - Takes from_frame and to_frame inputs (via CLI flags, positional arguments, or interactive prompt).
  - Imports core geometry and reference functions from `count.py`:
      `angle_rms`, `tetrahedral_ideal`, `octahedral_ideal`, `tetra_limit`, `octa_limit`.
  - Calculates per-atom visualization properties (fe_type, parent_fe_id array, coordination numbers, angle RMS).
  - Includes all parent Fe IDs (tetrahedral, octahedral, and other Fe parents) in parent_fe_id.
  - Exports per-frame OVITO visualization files (.extxyz, .dump).
  - Exports a summary CSV file with polyhedral counts for each frame.
"""

import os
import sys
import argparse
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN

# Suppress pymatgen warnings when run
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

# Import the core geometric algorithm and constants from count.py
from count import (
    angle_rms,
    tetrahedral_ideal,
    octahedral_ideal,
    tetra_limit as DEFAULT_TETRA_LIMIT,
    octa_limit as DEFAULT_OCTA_LIMIT,
)


def calculate_frame_properties(
    structure: Structure,
    t_limit: float = DEFAULT_TETRA_LIMIT,
    o_limit: float = DEFAULT_OCTA_LIMIT,
    crystal_nn: Optional[CrystalNN] = None,
) -> Dict[str, Any]:
    """
    Perform all per-atom calculations needed for OVITO visualization and summary CSV:
      1. Classifies Fe atoms into Oct (0), Tet (1), or Other (2) using angle_rms from count.py.
      2. Assembles the parent_fe_id array containing all parent Fe IDs for every atom
         (including tetrahedral, octahedral, and other Fe parents).
      3. Computes parent sharing counts (n_oct_parents, n_tet_parents, n_other_parents) and parent_type_id.
    """
    if crystal_nn is None:
        crystal_nn = CrystalNN()

    n_atoms = len(structure)

    fe_type = np.full(n_atoms, -1, dtype=int)
    coordination_number = np.zeros(n_atoms, dtype=int)
    angle_rms_val = np.full(n_atoms, -1.0, dtype=float)

    n_oct_parents = np.zeros(n_atoms, dtype=int)
    n_tet_parents = np.zeros(n_atoms, dtype=int)
    n_other_parents = np.zeros(n_atoms, dtype=int)
    parent_type_id = np.full(n_atoms, -1, dtype=int)

    # Oxygen parent Fe mapping: o_site_index -> list of 1-based parent Fe IDs (Oct, Tet, and Other)
    oxygen_parents_map = defaultdict(list)

    oct_count = 0
    tet_count = 0
    other_count = 0
    total_fe = 0
    total_o = 0

    tetra_rms_list: List[float] = []
    octa_rms_list: List[float] = []

    for i, atom in enumerate(structure):
        if atom.specie.symbol != "Fe":
            continue

        total_fe += 1
        fe_id = i + 1  # 1-based Fe atom ID

        neighbors = crystal_nn.get_nn_info(structure, i)
        oxygen_neighbors = [
            nb for nb in neighbors
            if nb["site"].specie.symbol == "O"
        ]

        cn = len(oxygen_neighbors)
        coordination_number[i] = cn

        if cn == 4:
            vectors = np.array([
                nb["site"].coords - atom.coords
                for nb in oxygen_neighbors
            ])
            rms = angle_rms(vectors, tetrahedral_ideal)
            angle_rms_val[i] = rms
            if rms <= t_limit:
                ft = 1  # Tetrahedral
                tet_count += 1
                tetra_rms_list.append(rms)
            else:
                ft = 2  # Distorted 4-coord -> Other
                other_count += 1
        elif cn == 6:
            vectors = np.array([
                nb["site"].coords - atom.coords
                for nb in oxygen_neighbors
            ])
            rms = angle_rms(vectors, octahedral_ideal)
            angle_rms_val[i] = rms
            if rms <= o_limit:
                ft = 0  # Octahedral
                oct_count += 1
                octa_rms_list.append(rms)
            else:
                ft = 2  # Distorted 6-coord -> Other
                other_count += 1
        else:
            ft = 2  # Other CN (nor tetra neither octa)
            other_count += 1

        fe_type[i] = ft

        # Every oxygen neighbor records this Fe atom as a parent (Oct, Tet, or Other)
        for nb in oxygen_neighbors:
            o_idx = nb["site_index"]
            coordination_number[o_idx] += 1
            oxygen_parents_map[o_idx].append(fe_id)
            if ft == 0:
                n_oct_parents[o_idx] += 1
            elif ft == 1:
                n_tet_parents[o_idx] += 1
            else:
                n_other_parents[o_idx] += 1

    # Dimension of parent_fe_id array (at least 6 components)
    max_parents = 6
    if oxygen_parents_map:
        max_parents = max(6, max(len(p) for p in oxygen_parents_map.values()))

    parent_fe_id = np.zeros((n_atoms, max_parents), dtype=int)

    # Fe atoms: first component is its own Fe ID
    for i, atom in enumerate(structure):
        if atom.specie.symbol == "Fe":
            parent_fe_id[i, 0] = i + 1

    # Oxygen atoms: fill all parent Fe IDs (Oct, Tet, and Other)
    for i, atom in enumerate(structure):
        if atom.specie.symbol != "O":
            continue

        total_o += 1
        parents = oxygen_parents_map[i]
        if parents:
            parent_fe_id[i, :len(parents)] = parents[:max_parents]

        # Categorical sharing classification for Oxygen
        has_oct = n_oct_parents[i] > 0
        has_tet = n_tet_parents[i] > 0
        has_oth = n_other_parents[i] > 0

        if has_oct and not has_tet and not has_oth:
            parent_type_id[i] = 0  # Octahedral only
        elif has_tet and not has_oct and not has_oth:
            parent_type_id[i] = 1  # Tetrahedral only
        elif has_oth and not has_oct and not has_tet:
            parent_type_id[i] = 2  # Other only
        elif has_oct and has_tet and not has_oth:
            parent_type_id[i] = 3  # Shared Oct + Tet
        elif has_oct and has_oth and not has_tet:
            parent_type_id[i] = 4  # Shared Oct + Other
        elif has_tet and has_oth and not has_oct:
            parent_type_id[i] = 5  # Shared Tet + Other
        elif has_oct and has_tet and has_oth:
            parent_type_id[i] = 6  # Shared Oct + Tet + Other
        else:
            parent_type_id[i] = -1  # Unbound / Free Oxygen

    return {
        "fe_type": fe_type,
        "parent_fe_id": parent_fe_id,
        "coordination_number": coordination_number,
        "angle_rms": angle_rms_val,
        "parent_type_id": parent_type_id,
        "n_oct_parents": n_oct_parents,
        "n_tet_parents": n_tet_parents,
        "n_other_parents": n_other_parents,
        "octahedral_count": oct_count,
        "tetrahedral_count": tet_count,
        "other_count": other_count,
        "total_fe": total_fe,
        "total_o": total_o,
        "octa_rms_list": octa_rms_list,
        "tetra_rms_list": tetra_rms_list,
    }


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
    header = f'Lattice="{lat_str}" {prop_header} pbc="T T T"'

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
                f"{n_oct[i]:3d} {n_tet[i]:3d} {n_oth[i]:3d}\n"
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
    a, b, c = lat.a, lat.b, lat.c

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
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"0.000000 {a:.6f}\n")
        f.write(f"0.000000 {b:.6f}\n")
        f.write(f"0.000000 {c:.6f}\n")
        f.write(
            f"ITEM: ATOMS id type x y z fe_type {parent_cols} "
            "coordination_number angle_rms parent_type_id "
            "n_oct_parents n_tet_parents n_other_parents\n"
        )
        for i, atom in enumerate(structure):
            atom_type = 1 if atom.specie.symbol == "Fe" else 2
            x, y, z = atom.coords
            p_str = " ".join(f"{pid:6d}" for pid in parent_fe_id[i])
            f.write(
                f"{i + 1:6d} {atom_type:2d} {x:12.6f} {y:12.6f} {z:12.6f} "
                f"{fe_type[i]:3d} {p_str} "
                f"{cn[i]:3d} {angle_rms_val[i]:10.4f} {parent_type_id[i]:3d} "
                f"{n_oct[i]:3d} {n_tet[i]:3d} {n_oth[i]:3d}\n"
            )


def process_single_frame(
    frame_num: int,
    frames_dir: str,
    output_dir: str,
    export_format: str = "both",
    t_limit: float = DEFAULT_TETRA_LIMIT,
    o_limit: float = DEFAULT_OCTA_LIMIT,
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
    props = calculate_frame_properties(structure, t_limit=t_limit, o_limit=o_limit)

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

    return {
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

    args = parser.parse_args()

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

    results = []
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


if __name__ == "__main__":
    main()
