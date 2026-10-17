#!/usr/bin/env python3
"""
count.py - Polyhedra Classification Algorithm & Coordination Analysis

Implements the Angle RMS geometry check to evaluate deviations from ideal
tetrahedral (109.47°) and octahedral (90° / 180°) coordination geometries
for central Fe atoms with Oxygen neighbors.
"""

from typing import List, Tuple
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN

# Ideal reference angles (in degrees)
tetrahedral_ideal = np.full(6, 109.4712206)
octahedral_ideal = np.array([90.0] * 12 + [180.0] * 3)

# Angle RMS threshold limits (in degrees)
tetra_limit = 30.0
octa_limit = 30.0


def angle_rms(vectors: np.ndarray, ideal_angles: np.ndarray) -> float:
    """
    Return RMS deviation of all pair angles from the sorted ideal angles.

    Parameters:
        vectors: (N, 3) displacement vectors from central atom to neighbors.
        ideal_angles: 1D array of ideal pair inter-bond angles in degrees.

    Returns:
        float: RMS angular deviation in degrees.
    """
    vectors = np.asarray(vectors, dtype=float)
    ideal_angles = np.asarray(ideal_angles, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) < 2:
        raise ValueError("Expected at least two 3D bond vectors")
    if ideal_angles.shape != (len(vectors) * (len(vectors) - 1) // 2,):
        raise ValueError("Reference angle count does not match the bond vectors")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) <= 1e-12):
        raise ValueError("Bond vectors must be finite and nonzero")
    unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    dots = np.clip(unit @ unit.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots[np.triu_indices(len(unit), k=1)]))
    return float(np.sqrt(np.mean((np.sort(angles) - np.sort(ideal_angles)) ** 2)))


def fe_geometry(vectors, t_limit=tetra_limit, o_limit=octa_limit,
                bond_limit=0.20):
    """Shared hard label and soft geometry scores for a Fe–O shell.

    Scores are similarities, not calibrated probabilities. CN other than 4/6
    remains unknown. Periodic-image-corrected vectors must be supplied.
    """
    if not all(np.isfinite(value) and value > 0 for value in (t_limit, o_limit, bond_limit)):
        raise ValueError("Geometry limits must be finite and positive")
    vectors = np.asarray(vectors, dtype=float)
    if not vectors.size:
        vectors = vectors.reshape(0, 3)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("Expected an (N, 3) array of Fe–O bond vectors")
    lengths = np.linalg.norm(vectors, axis=1)
    if len(lengths) and (not np.isfinite(lengths).all() or np.any(lengths <= 1e-12)):
        raise ValueError("Fe–O bonds must be finite and nonzero")
    distortion = float(np.std(lengths) / np.mean(lengths)) if len(lengths) else -1.0
    rms, label = -1.0, 2
    scores = np.array([0.0, 0.0, 1.0])  # tetra, octa, unknown
    if len(vectors) in (4, 6):
        is_tet = len(vectors) == 4
        limit = t_limit if is_tet else o_limit
        rms = angle_rms(vectors, tetrahedral_ideal if is_tet else octahedral_ideal)
        similarity = float(np.exp(-0.5 * ((rms / limit)**2 + (distortion / bond_limit)**2)))
        scores[0 if is_tet else 1] = similarity
        scores[2] = 1.0 - similarity
        if rms <= limit and distortion <= bond_limit:
            label = 1 if is_tet else 0
    return label, rms, distortion, scores


def oxygen_neighbors(structure, index, crystal_nn):
    """CrystalNN shell filtered to O, preserving distinct periodic images.

    CrystalNN selects the shell before chemical filtering, as in the original
    implementation. This is a shared Fe→O graph, not independent O-centered
    CrystalNN queries. Callers skip this query for oxygen-free structures.
    """
    neighbors = crystal_nn.get_nn_info(structure, index)
    result, seen = [], set()
    for nb in neighbors:
        if nb["site"].specie.symbol != "O":
            continue
        key = (int(nb["site_index"]), tuple(int(x) for x in nb["image"]))
        if key not in seen:
            seen.add(key)
            result.append(nb)
    return result


def count_polyhedra(
    structure: Structure,
    t_limit: float = tetra_limit,
    o_limit: float = octa_limit,
    bond_limit: float = 0.20,
) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """
    Count tetrahedral, octahedral, and other Fe polyhedra in a structure.

    Returns:
        tetrahedral: list of site indices for tetrahedral Fe atoms
        octahedral: list of site indices for octahedral Fe atoms
        other: list of (index, coordination_number) for other Fe sites
    """
    crystal_nn = CrystalNN()

    tetrahedral = []
    octahedral = []
    other = []
    has_oxygen = any(site.specie.symbol == "O" for site in structure)

    for i, atom in enumerate(structure):
        if atom.specie.symbol != "Fe":
            continue

        # Find the nearest neighbors of this Fe atom.
        neighbors = oxygen_neighbors(structure, i, crystal_nn) if has_oxygen else []
        vectors = [nb["site"].coords - atom.coords for nb in neighbors]
        label, _, _, _ = fe_geometry(vectors, t_limit, o_limit, bond_limit)
        if label == 1:
            tetrahedral.append(i)
        elif label == 0:
            octahedral.append(i)
        else:
            other.append((i, len(neighbors)))

    return tetrahedral, octahedral, other


if __name__ == "__main__":
    # Change this to your downloaded file.
    # Examples: "POSCAR", "CONTCAR", "magnetite.cif"
    filename = "POSCAR"

    structure = Structure.from_file(filename)

    print(f"Formula: {structure.composition.reduced_formula}")
    print(f"Number of atoms in cell: {len(structure)}")

    tetrahedral, octahedral, other = count_polyhedra(structure, tetra_limit, octa_limit)

    print("\n========== RESULTS ==========")
    print(f"Tetrahedral FeO4 polyhedra: {len(tetrahedral)}")
    print(f"Octahedral FeO6 polyhedra:  {len(octahedral)}")
    print(f"Other Fe coordination:      {len(other)}")

    print("\nTetrahedral Fe atom indices (1-based):")
    print([i + 1 for i in tetrahedral])

    print("\nOctahedral Fe atom indices (1-based):")
    print([i + 1 for i in octahedral])

    if other:
        print("\nFe sites requiring inspection (index, O coordination):")
        print([(i + 1, cn) for i, cn in other])

    print("\nTetrahedral Fe fractional coordinates:")
    for i in tetrahedral:
        print(f"Fe {i+1}: {structure[i].frac_coords}")

    print("\nOctahedral Fe fractional coordinates:")
    for i in octahedral:
        print(f"Fe {i+1}: {structure[i].frac_coords}")
