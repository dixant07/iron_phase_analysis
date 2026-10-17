"""Scientific behavior checks; run .venv/bin/python -m unittest discover -s tests -v."""

import json
import csv
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.lammps.outputs import LammpsDump

from count import angle_rms, count_polyhedra, fe_geometry, tetrahedral_ideal, octahedral_ideal
from fe_coordination import read_coordination_counts
from main import calculate_frame_properties, write_extxyz, write_lammps_dump
from phase import PhaseConfig, fit_reference_model, load_reference_model


def ideal_feo():
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.3),
                                     ["Fe", "O"], [[0, 0, 0], [.5, 0, 0]])


def ideal_magnetite():
    # Origin choice 2 is essential for these Wyckoff coordinates.
    return Structure.from_spacegroup("Fd-3m:2", Lattice.cubic(8.4),
                                     ["Fe", "Fe", "O"],
                                     [[.125]*3, [.5]*3, [.255]*3])


class PhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feo, cls.mag = ideal_feo(), ideal_magnetite()
        cls.feo_props = calculate_frame_properties(cls.feo)
        cls.mag_props = calculate_frame_properties(cls.mag)

    def test_angle_geometry_and_invalid_vectors(self):
        tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        octa = np.vstack((np.eye(3), -np.eye(3)))
        self.assertLess(angle_rms(tetra, tetrahedral_ideal), 1e-6)
        self.assertLess(angle_rms(octa, octahedral_ideal), 1e-6)
        self.assertEqual(fe_geometry(tetra)[0], 1)
        self.assertEqual(fe_geometry(octa)[0], 0)
        self.assertEqual(fe_geometry(octa[:-1])[0], 2)
        distorted = tetra.astype(float)
        distorted[0] *= 3
        self.assertEqual(fe_geometry(distorted)[0], 2)
        with self.assertRaises(ValueError):
            angle_rms(np.zeros((4, 3)), tetrahedral_ideal)

    def test_ideal_bulk_phases_and_reciprocal_bonds(self):
        for structure, props, phase in ((self.feo, self.feo_props, 2), (self.mag, self.mag_props, 1)):
            self.assertTrue(np.all(props["phase_id"] == phase))
            fe = np.array([s.specie.symbol == "Fe" for s in structure])
            oxygen = ~fe
            self.assertEqual(props["coordination_number"][fe].sum(),
                             props["coordination_number"][oxygen].sum())
            np.testing.assert_array_equal(props["coordination_number"][oxygen],
                (props["n_tet_parents"] + props["n_oct_parents"] + props["n_other_parents"])[oxygen])
            np.testing.assert_allclose(props["fe_site_scores"][fe].sum(axis=1), 1.)
            for o in np.flatnonzero(oxygen):
                parents = props["parent_fe_id"][o]
                parents = parents[parents > 0] - 1
                self.assertTrue(fe[parents].all())
                self.assertEqual(len(parents), props["coordination_number"][o])
        self.assertEqual(self.mag_props["tetrahedral_count"], 8)
        self.assertEqual(self.mag_props["octahedral_count"], 16)
        # Primitive FeO has repeated base Fe IDs in distinct periodic images.
        self.assertTrue(np.all(self.feo_props["coordination_number"] == 6))

    def test_count_py_and_pipeline_agree(self):
        tetra, octa, other = count_polyhedra(self.mag)
        self.assertEqual(tetra, np.flatnonzero(self.mag_props["fe_type"] == 1).tolist())
        self.assertEqual(octa, np.flatnonzero(self.mag_props["fe_type"] == 0).tolist())
        self.assertEqual(other, [])

    def test_rotated_triclinic_export_preserves_geometry(self):
        base = np.array([[4.3, 0, 0], [.3, 4.3, 0], [-.2, .4, 4.3]])
        angle = .47
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        s = Structure(Lattice(base @ rotation), self.feo.species, self.feo.frac_coords)
        props = calculate_frame_properties(s)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"tilted.dump"
            write_lammps_dump(str(path), s, 12, props)
            parsed = LammpsDump.from_str(path.read_text())
            lattice = parsed.box.to_lattice()
            np.testing.assert_allclose(lattice.metric_tensor, s.lattice.metric_tensor, atol=1e-7)
            positions = parsed.data[["x", "y", "z"]].to_numpy()
            np.testing.assert_allclose(lattice.get_fractional_coords(positions), s.frac_coords, atol=1e-6)
            timestep, counts = read_coordination_counts(path)
            self.assertEqual(timestep, 12)
            self.assertEqual(sum(counts.values()), 4)
            self.assertIn("phase_id", parsed.data)
            xyz = Path(directory)/"test.extxyz"
            write_extxyz(str(xyz), s, props)
            lines = xyz.read_text().splitlines()
            spec = lines[1].split("Properties=")[1].split()[0].split(":")
            self.assertIn("phase_id", spec)
            self.assertTrue(all(len(line.split()) == sum(int(spec[k]) for k in range(2, len(spec), 3))
                                for line in lines[2:]))

    def test_permutation_and_translation_invariance(self):
        rng = np.random.default_rng(19)
        order = rng.permutation(len(self.mag))
        shuffled = Structure(self.mag.lattice, [self.mag[i].specie for i in order],
                             self.mag.frac_coords[order] + [.17, -.23, .11])
        props = calculate_frame_properties(shuffled)
        np.testing.assert_array_equal(props["fe_type"], self.mag_props["fe_type"][order])
        np.testing.assert_array_equal(props["phase_id"], self.mag_props["phase_id"][order])

    def test_cutoff_alternative_and_stability(self):
        stable = calculate_frame_properties(self.feo, phase_config=PhaseConfig(neighbor_method="cutoff", bond_cutoff=2.6))
        self.assertTrue(np.all(stable["phase_id"] == 2))
        unstable = calculate_frame_properties(self.feo, phase_config=PhaseConfig(neighbor_method="cutoff", bond_cutoff=2.15))
        self.assertTrue(np.all(unstable["neighbor_stability"][:4] < 1))
        self.assertGreater(unstable["fe_score_unknown"][:4].mean(), stable["fe_score_unknown"][:4].mean())

    def test_small_thermal_displacements(self):
        s = self.mag.copy()
        rng = np.random.default_rng(55)
        for i in range(len(s)):
            s.translate_sites([i], rng.normal(0, .025, 3), frac_coords=False)
        props = calculate_frame_properties(s)
        self.assertGreater(np.mean(props["phase_id"] == 1), .9)
        self.assertFalse(np.any(props["phase_id"] == 2))

    def test_vacancy_and_tetra_interstitial_do_not_create_magnetite(self):
        s = self.feo * (2, 2, 2)
        s.remove_sites([0])
        props = calculate_frame_properties(s)
        self.assertFalse(np.any(props["phase_id"] == 1))
        self.assertGreater(np.mean(props["phase_id"] == 2), .5)
        # Tetrahedral interstitial in an otherwise rock-salt host.
        s = self.feo * (2, 2, 2)
        s.append("Fe", [.125, .125, .125])
        props = calculate_frame_properties(s)
        self.assertFalse(np.any(props["phase_id"] == 1))

    def test_empty_oxygen_shell_and_metal_are_rejected(self):
        metal = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        props = calculate_frame_properties(metal)
        self.assertEqual(props["phase_id"].tolist(), [0])
        self.assertEqual(props["coordination_number"].tolist(), [0])
        oxygen_only = Structure(Lattice.cubic(10), ["O"], [[0, 0, 0]])
        props = calculate_frame_properties(oxygen_only)
        self.assertEqual(props["phase_id"].tolist(), [0])
        self.assertEqual(props["phase_reason"].tolist(), [1])
        self.assertEqual(props["phase_confidence"].tolist(), [0.])
        distant = Structure(Lattice.cubic(30), ["Fe", "O"], [[0, 0, 0], [.5, .5, .5]])
        props = calculate_frame_properties(distant, phase_config=PhaseConfig(neighbor_method="cutoff"))
        self.assertEqual(props["phase_id"].tolist(), [0, 0])

    def test_mixed_domains_keep_bulk_and_reject_boundary_sites(self):
        mag = self.mag * (3, 1, 1)
        feo = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.2),
                                       ["Fe", "O"], [[0, 0, 0], [.5, 0, 0]]) * (6, 2, 2)
        s = Structure(Lattice.orthorhombic(50.4, 8.4, 8.4), mag.species + feo.species,
                      np.vstack([mag.cart_coords, feo.cart_coords + [25.2, 0, 0]]),
                      coords_are_cartesian=True)
        props = calculate_frame_properties(s, phase_config=PhaseConfig(neighbor_method="cutoff", bond_cutoff=2.6))
        x = s.cart_coords[:, 0]
        self.assertTrue(np.all(props["phase_id"][(x > 8) & (x < 17)] == 1))
        self.assertTrue(np.all(props["phase_id"][(x > 33) & (x < 42)] == 2))
        boundary = (x < 3) | (abs(x - 25.2) < 3) | (x > 47.4)
        self.assertGreater(np.mean(props["phase_id"][boundary] == 0), .5)

    def test_disordered_cell_rejected(self):
        rng = np.random.default_rng(11)
        positions = []
        while len(positions) < 56:
            candidate = rng.random(3) * 12
            if all(np.linalg.norm((candidate-old+6) % 12 - 6) > 1.5 for old in positions):
                positions.append(candidate)
        s = Structure(Lattice.cubic(12), ["Fe"]*24 + ["O"]*32,
                      positions, coords_are_cartesian=True)
        props = calculate_frame_properties(s, phase_config=PhaseConfig(neighbor_method="cutoff", bond_cutoff=2.6))
        self.assertGreater(np.mean(props["phase_id"] == 0), .9)

    def test_cli_parallel_export_and_separate_geometry_phase_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vacancy = self.feo * (2, 2, 2)
            vacancy.remove_sites([0])
            vacancy.to(filename=str(root/"reduction.0.poscar"), fmt="poscar")
            self.mag.to(filename=str(root/"reduction.1.poscar"), fmt="poscar")
            script = Path(__file__).resolve().parents[1]/"main.py"
            result = subprocess.run([sys.executable, str(script), "0", "1", "--workers", "2",
                                     "--frames-dir", str(root), "--output-dir", str(root/"output")],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (root/"output/polyhedra_summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["frame"]) for row in rows], [0, 1])
            for row in rows:
                self.assertEqual(sum(int(row[k]) for k in ("octahedral_fe", "tetrahedral_fe", "other_fe")),
                                 int(row["total_fe"]))
                for symbol in ("fe", "o"):
                    self.assertEqual(sum(int(row[f"phase_{phase}_{symbol}"])
                                         for phase in ("other", "magnetite", "wustite")),
                                     int(row[f"total_{symbol}"]))
            _, _, expected_other = count_polyhedra(vacancy)
            self.assertEqual(int(rows[0]["other_fe"]), len(expected_other))
            self.assertTrue((root/"output/phase_metadata.json").is_file())

    def test_reference_fit_roundtrip_and_setting_mismatch(self):
        samples = {"magnetite": self.mag_props["phase_features"][24:],
                   "wustite": self.feo_props["phase_features"][4:]}
        model = fit_reference_model(samples, PhaseConfig(), 30., 30.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.json"
            path.write_text(json.dumps(model))
            loaded = load_reference_model(path, PhaseConfig(), 30., 30.)
            props = calculate_frame_properties(self.mag, reference_model=loaded)
            self.assertTrue(np.all(props["phase_id"] == 1))
            with self.assertRaises(ValueError):
                load_reference_model(path, replace(PhaseConfig(), composition_radii=(7., 10.)), 30., 30.)

    def test_invalid_settings_and_unsupported_species(self):
        with self.assertRaises(ValueError):
            PhaseConfig(composition_radii=(9., 6.))
        with self.assertRaises(ValueError):
            PhaseConfig(bond_limit=float("nan"))
        with self.assertRaises(ValueError):
            calculate_frame_properties(Structure(Lattice.cubic(4), ["C"], [[0, 0, 0]]))


if __name__ == "__main__":
    unittest.main()
