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
    unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    dots = np.clip(unit @ unit.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots[np.triu_indices(len(unit), k=1)]))
    return float(np.sqrt(np.mean((np.sort(angles) - np.sort(ideal_angles)) ** 2)))


def count_polyhedra(
    structure: Structure,
    t_limit: float = tetra_limit,
    o_limit: float = octa_limit,
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

    for i, atom in enumerate(structure):
        if atom.specie.symbol != "Fe":
            continue

        # Find the nearest neighbors of this Fe atom.
        neighbors = crystal_nn.get_nn_info(structure, i)

        # Count only oxygen neighbors, not other iron atoms.
        oxygen_neighbors = [
            neighbor
            for neighbor in neighbors
            if neighbor["site"].specie.symbol == "O"
        ]

        coordination_number = len(oxygen_neighbors)

        if coordination_number == 4:
            # Displacement vectors from central Fe atom to each oxygen neighbor
            vectors = np.array([
                neighbor["site"].coords - atom.coords
                for neighbor in oxygen_neighbors
            ])
            tetra_rms = angle_rms(vectors, tetrahedral_ideal)
            if tetra_rms <= t_limit:
                tetrahedral.append(i)
            else:
                other.append((i, coordination_number))
        elif coordination_number == 6:
            # Displacement vectors from central Fe atom to each oxygen neighbor
            vectors = np.array([
                neighbor["site"].coords - atom.coords
                for neighbor in oxygen_neighbors
            ])
            octa_rms = angle_rms(vectors, octahedral_ideal)
            if octa_rms <= o_limit:
                octahedral.append(i)
            else:
                other.append((i, coordination_number))
        else:
            other.append((i, coordination_number))

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
