"""O-centered Fe–O phase classification following new_arch.md.

IDs: 0=other, 1=magnetite-like, 2=wustite-like. A separate reason distinguishes
unknown/rejected sites from mixed/interface candidates. Scores are similarities,
not calibrated phase probabilities. No oxidation state is inferred.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PHASE_NAMES = {0: "other", 1: "magnetite", 2: "wustite"}
REASONS = {0: "accepted", 1: "insufficient_or_unfamiliar_environment",
           2: "ambiguous_or_interface", 3: "inconsistent_neighborhood"}
FEATURE_NAMES = ("soft_tet", "soft_oct", "soft_unknown", "mean_bond_distortion",
                 "fe_o_ratio_inner", "fe_o_ratio_outer", "second_shell_tet_fraction",
                 "second_shell_unknown_fraction")
FEATURE_SCALES = np.array([0.75, 1.25, 0.75, 0.15, 0.20, 0.15, 0.15, 0.20])
IDEAL_FEATURES = np.array([[1., 3., 0., 0., .75, .75, 1/3, 0.],
                           [0., 6., 0., 0., 1., 1., 0., 0.]])


@dataclass(frozen=True)
class PhaseConfig:
    composition_radii: tuple = (6.0, 9.0)
    min_composition_atoms: int = 24
    min_confidence: float = 0.80
    max_distance: float = 16.0
    max_unknown_fraction: float = 0.50
    min_spatial_support: float = 0.55
    smoothing_strength: float = 0.25
    fingerprint_sigma: float = 1.5
    spatial_sigma: float = 4.0
    fe_min_support: float = 0.60
    bond_limit: float = 0.20
    # Optional fixed-cutoff alternative to the original Voronoi/CrystalNN shell.
    neighbor_method: str = "crystalnn"
    bond_cutoff: float = 2.8
    cutoff_delta: float = 0.10

    def __post_init__(self):
        if len(self.composition_radii) != 2 or not (0 < self.composition_radii[0] < self.composition_radii[1]):
            raise ValueError("composition_radii must contain two increasing positive radii")
        for name in ("min_confidence", "fe_min_support"):
            if not 0.5 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0.5, 1]")
        for name in ("max_unknown_fraction", "min_spatial_support"):
            if not 0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not 0 <= self.smoothing_strength <= 1:
            raise ValueError("smoothing_strength must lie in [0, 1]")
        if min(self.max_distance, self.fingerprint_sigma, self.spatial_sigma,
               self.bond_limit, self.bond_cutoff, self.cutoff_delta) <= 0:
            raise ValueError("Distance scales and geometry limits must be positive")
        if self.cutoff_delta >= self.bond_cutoff or self.min_composition_atoms < 1:
            raise ValueError("Invalid cutoff_delta or min_composition_atoms")
        if self.neighbor_method not in ("crystalnn", "cutoff"):
            raise ValueError("neighbor_method must be crystalnn or cutoff")
        if not all(np.isfinite(float(value)) for value in asdict(self).values()
                   if isinstance(value, (int, float))):
            raise ValueError("Configuration values must be finite")
        if not np.isfinite(self.composition_radii).all():
            raise ValueError("Composition radii must be finite")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())) if path else cls()


def local_descriptors(structure, props, fe_to_o, o_to_fe, config):
    """Physical spherical composition + graph second-shell topology.

    Sphere neighbors include the center and distinct periodic images, counted
    once per (site index, image). This also works for small periodic cells:
    different images represent different atoms in the physical sphere.
    Graph topology uses distinct unit-cell site indices and includes the center.
    """
    n = len(structure)
    oxygen = np.array([i for i, s in enumerate(structure) if s.specie.symbol == "O"], dtype=int)
    features = np.full((n, len(FEATURE_NAMES)), np.nan)
    comp_sizes = np.zeros((n, 2), dtype=int)
    contexts = {}
    soft = props["fe_site_scores"]
    for o in oxygen:
        parents = o_to_fe[o]
        if parents:
            # Count periodic bonds in the fingerprint, including repeated base IDs.
            features[o, :3] = soft[parents].sum(axis=0)
            features[o, 3] = np.mean(props["bond_distortion"][parents])
        else:
            features[o, :4] = 0.
        context_o = {o}
        for fe in parents:
            context_o.update(fe_to_o[fe])
        contexts[o] = sorted(context_o)
        context_fe = {fe for k in context_o for fe in o_to_fe[k]}
        if context_fe:
            mean = soft[sorted(context_fe)].mean(axis=0)
            features[o, 6:] = mean[[0, 2]]
        else:
            features[o, 6:] = [0., 1.]

    # Query only oxygen centers, avoiding a large all-atom neighbor-list allocation.
    if len(oxygen):
        center, neighbor, images, distance = structure.get_neighbor_list(
            config.composition_radii[-1], sites=[structure[i] for i in oxygen])
        symbols = np.array([s.specie.symbol for s in structure])
        for k, radius in enumerate(config.composition_radii):
            within = distance <= radius + 1e-8
            # pymatgen emits unique (center, neighbor, image) contacts. Central O
            # is excluded by the API, then added explicitly to the denominator.
            fe_counts = np.bincount(center[within & (symbols[neighbor] == "Fe")], minlength=len(oxygen))
            o_counts = 1 + np.bincount(center[within & (symbols[neighbor] == "O")], minlength=len(oxygen))
            sizes = fe_counts + o_counts
            comp_sizes[oxygen, k] = sizes
            features[oxygen, 4+k] = fe_counts / o_counts
    return oxygen, features, comp_sizes, contexts


def load_reference_model(path, config, t_limit, o_limit):
    model = json.loads(Path(path).read_text())
    if model.get("version") != 1 or model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Unsupported reference model or descriptor schema")
    # Never silently apply fitted descriptors using a different neighbor graph.
    expected = {"phase_config": json.loads(json.dumps(asdict(config))),
                "tetra_limit": t_limit, "octa_limit": o_limit}
    if model.get("descriptor_settings") != expected:
        raise ValueError("Reference model descriptor settings differ from the current configuration/angle limits")
    for name in ("magnetite", "wustite"):
        item = model["classes"][name]
        mean = np.asarray(item["mean"], dtype=float)
        inverse = np.asarray(item["inverse_covariance"], dtype=float)
        if mean.shape != (len(FEATURE_NAMES),) or inverse.shape != (len(FEATURE_NAMES),)*2:
            raise ValueError("Invalid reference model array dimensions")
        if not np.isfinite(mean).all() or not np.isfinite(inverse).all():
            raise ValueError("Non-finite reference model parameters")
        if not np.allclose(inverse, inverse.T) or np.linalg.eigvalsh(inverse).min() <= 0:
            raise ValueError("Reference precision must be positive definite")
        if not np.isfinite(item["rejection_distance"]) or item["rejection_distance"] <= 0:
            raise ValueError("Invalid reference rejection distance")
    return model


def phase_distances(features, model=None):
    """Return two absolute mismatch distances and their rejection limits."""
    if model is None:
        delta = (features[:, None, :] - IDEAL_FEATURES[None, :, :]) / FEATURE_SCALES
        return np.sum(delta**2, axis=2), None
    distances, thresholds = [], []
    for name in ("magnetite", "wustite"):
        ref = model["classes"][name]
        delta = (features - np.asarray(ref["mean"])) / FEATURE_SCALES
        distances.append(np.einsum("ni,ij,nj->n", delta, ref["inverse_covariance"], delta))
        thresholds.append(ref["rejection_distance"])
    return np.array(distances).T, np.array(thresholds)


def classify_phases(structure, props, fe_to_o, o_to_fe, config=None, model=None):
    """Score O first, reject unfamiliar sites, then transfer supported labels to Fe.

    Spatial averaging is one simultaneous edge-aware pass over graph-connected
    oxygen sites. It cannot turn rejected sites into accepted sites. Fe needs a
    majority of *all* bonded O, including unknown O, supporting a single phase.
    """
    config = config or PhaseConfig()
    n = len(structure)
    oxygen, features, comp_sizes, contexts = local_descriptors(
        structure, props, fe_to_o, o_to_fe, config)
    phase_id = np.zeros(n, dtype=int)
    reason = np.ones(n, dtype=int)
    score = np.zeros((n, 2))
    distance_out = np.full((n, 2), -1.)
    support = np.zeros(n)
    fractional_coords = structure.frac_coords
    if len(oxygen):
        distances, limits = phase_distances(features[oxygen], model)
        limits = np.full(2, config.max_distance) if limits is None else limits
        distance_out[oxygen] = distances
        logits = -0.5 * distances
        logits -= logits.max(axis=1, keepdims=True)
        raw = np.exp(logits)
        raw /= raw.sum(axis=1, keepdims=True)
        cn = features[oxygen, :3].sum(axis=1)
        valid = ((cn >= 3) & (cn <= 7) &
                 (features[oxygen, 2] / np.maximum(cn, 1) <= config.max_unknown_fraction) &
                 (comp_sizes[oxygen].min(axis=1) >= config.min_composition_atoms) &
                 np.isfinite(features[oxygen]).all(axis=1))
        within = distances <= limits[None, :]
        valid &= within.any(axis=1)
        valid_by_site = np.zeros(n, dtype=bool)
        valid_by_site[oxygen] = valid
        raw_by_site = np.zeros((n, 2))
        raw_by_site[oxygen] = raw
        score[oxygen] = raw
        for row, o in enumerate(oxygen):
            if not valid[row]:
                continue
            ctx = np.array([k for k in contexts[o] if valid_by_site[k]], dtype=int)
            delta = features[ctx, :3] - features[o, :3]
            distances_o = structure.lattice.get_all_distances(
                [fractional_coords[o]], fractional_coords[ctx])[0]
            weights = np.exp(-np.sum(delta**2, axis=1)/(2*config.fingerprint_sigma**2)
                             -distances_o**2/(2*config.spatial_sigma**2))
            averaged = np.average(raw_by_site[ctx], axis=0, weights=weights)
            score[o] = (1-config.smoothing_strength)*raw[row] + config.smoothing_strength*averaged
            candidate = int(np.argmax(score[o]))
            # Absolute rejection applies to the chosen class, even after smoothing.
            if not within[row, candidate]:
                continue
            if score[o, candidate] < config.min_confidence:
                reason[o] = 2
                continue
            # Unknown neighbors remain in the denominator and cannot be erased.
            support[o] = float(np.mean([
                valid_by_site[k] and raw_by_site[k, candidate] >= config.min_confidence
                for k in contexts[o]]))
            if support[o] < config.min_spatial_support:
                reason[o] = 3
                continue
            # A lone tetrahedral Fe cannot create magnetite in an octahedral host.
            if candidate == 0 and (features[o, 0] < .25 or features[o, 6] < .10):
                reason[o] = 3
                continue
            phase_id[o], reason[o] = candidate + 1, 0

    for fe, neighbors in fe_to_o.items():
        if len(neighbors) < 3:
            continue
        ids = np.asarray(neighbors, dtype=int)
        votes = np.array([np.mean(phase_id[ids] == label) for label in (1, 2)])
        score[fe] = np.mean(score[ids], axis=0)
        candidate = int(np.argmax(votes))
        support[fe] = votes[candidate]
        if votes[candidate] >= config.fe_min_support:
            phase_id[fe], reason[fe] = candidate + 1, 0
        elif np.any(votes > 0):
            reason[fe] = 2

    return {"phase_id": phase_id, "phase_reason": reason,
            "phase_score_magnetite": score[:, 0], "phase_score_wustite": score[:, 1],
            "phase_confidence": np.where(
                phase_id > 0, score[np.arange(n), np.maximum(phase_id-1, 0)], 0.),
            "phase_support": support,
            "phase_distance_magnetite": distance_out[:, 0],
            "phase_distance_wustite": distance_out[:, 1],
            "phase_features": features, "composition_sample_sizes": comp_sizes,
            "local_fe_o_ratio_inner": np.nan_to_num(features[:, 4], nan=-1.),
            "local_fe_o_ratio_outer": np.nan_to_num(features[:, 5], nan=-1.),
            "second_shell_tet_fraction": np.nan_to_num(features[:, 6], nan=-1.),
            "second_shell_unknown_fraction": np.nan_to_num(features[:, 7], nan=-1.)}


def fit_reference_model(samples, config, t_limit, o_limit):
    """Fit regularized Gaussian descriptor distances, not calibrated probabilities.

    samples maps phase name to pooled O-centered feature matrices. Rejection
    limits cover the 99th training percentile with a floor for ideal references.
    Separate held-out thermal/defect/interface data are needed for validation.
    """
    classes = {}
    for name in ("magnetite", "wustite"):
        values = np.asarray(samples[name], dtype=float)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or len(values) < 2:
            raise ValueError(f"Need at least two oxygen descriptors for {name}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite reference descriptors for {name}")
        mean = values.mean(axis=0)
        scaled = (values - mean) / FEATURE_SCALES
        covariance = np.cov(scaled, rowvar=False) + np.eye(len(FEATURE_NAMES)) * 0.25
        inverse = np.linalg.inv(covariance)
        distances = np.einsum("ni,ij,nj->n", scaled, inverse, scaled)
        classes[name] = {"mean": mean.tolist(), "inverse_covariance": inverse.tolist(),
                         "rejection_distance": max(float(np.quantile(distances, .99))*1.25,
                                                   config.max_distance),
                         "n_oxygen_samples": len(values)}
    return {"version": 1, "feature_names": list(FEATURE_NAMES), "classes": classes,
            "descriptor_settings": {"phase_config": asdict(config),
                                    "tetra_limit": t_limit, "octa_limit": o_limit},
            "score_interpretation": "relative similarity; not calibrated probability"}
