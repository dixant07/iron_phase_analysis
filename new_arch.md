# Robust local phase classification of Fe₃O₄ (magnetite) and FeO (wüstite)

Your two approaches are physically well motivated. In fact, I would combine them into a hierarchical, graph-based local structure classifier, supplemented by bond-angle analysis, local chemical composition, and spatial consistency.

The most important distinction is:

Identifying tetrahedral and octahedral Fe coordination is not the same as identifying magnetite and wüstite phases.

An Fe atom in an octahedral environment can belong to either phase. Even tetrahedrally coordinated Fe is not conclusive evidence of magnetite, because nonstoichiometric wüstite can contain tetrahedral Fe interstitials associated with vacancy clusters. This has been observed in atomistic studies of Fe₁₋ₓO.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.nature.com\&sz=32)

npj Computational Materials

+1



Consequently, I recommend using three levels of information:

Level 1

Fe-centered geometry

Identify tetrahedral, octahedral, distorted, and unclassified Fe sites using Fe–O coordination and bond geometry.

Level 2

O-centered connectivity

Determine how many neighboring Fe atoms are tetrahedral versus octahedral.

Level 3

Spatial and chemical consistency

Combine multiple neighboring atoms, local Fe/O ratio, crystal topology, and temporal stability to assign a phase with confidence.

## 1. Crystallographic basis

[File:Magnetite crystal structure (Fleet 1986, 633D) polyhedra model crystallographic standard alignmend.png - Wikimedia Commons](https://images.openai.com/static-rsc-4/zvtgMCsYDA1WyHPmKCGXaRQR9gBsaHP-wsdm1O1iJ-OKbxt5W_IYExx-EMvGDq_ddweYum6EiTISNEgzUEc6gduKvJWHbQwYalLM76WhJW99GmKK_HOCZ3p1i5weToqcbW_pX8H32NPoeTerGqTcXG7i5Cu-5IBv8TCah8gwaJQ?purpose=inline)

Magnetite — Fe₃O₄

Inverse spinel structure, space group Fd-3m (ideal cubic phase)

[Materials Project](https://images.openai.com/static-rsc-4/jAqhHzfp6lxwkCUyKvAzifePfJvj9jhN6oFlq7XuHXJYHgJhFHHqkN2DCMTTq9EHfSz3zAZLCs5NYcXg8AZ1IfBqTUx7GI5DMimgz9pV75LRh6nyONzoSRSehb3xIXmDCzXIJREzkUHctSuE8hHpXDKrb5OpAIZTe8Txk-_itm0?purpose=inline)

Wüstite — FeO

Rock-salt structure, space group Fm-3m (ideal cubic phase)

For ideal, defect-free structures:

| Property                     | Magnetite Fe₃O₄ | Wüstite FeO |
| ---------------------------- | --------------- | ----------- |
| Fe tetrahedral coordination  | 4 O             | Absent      |
| Fe octahedral coordination   | 6 O             | 6 O         |
| O–Fe coordination            | 4 Fe            | 6 Fe        |
| O's tetrahedral Fe neighbors | 1               | 0           |
| O's octahedral Fe neighbors  | 3               | 6           |
| Fe(tet):Fe(oct)              | 1:2             | 0:1         |
| Fe:O                         | 3:4             | 1:1         |

These are ideal reference signatures, not universal hard constraints for defective or thermally fluctuating environments.

An important structural observation is that both phases have approximately close-packed oxygen frameworks. Therefore, classifying the oxygen sublattice alone is insufficient; the Fe occupancy of tetrahedral and octahedral sites and its spatial ordering matter.

## 2. Stage I — Detect Fe–O neighbors

Given atomic positions \\(\mathbf r_i\\), construct a bipartite graph:

\\[ G=(V\_{\mathrm{Fe}},V\_{\mathrm O},E) \\]

An edge \\((i,j)\in E\\) means Fe atom \\(i\\) is coordinated to O atom \\(j\\).

With periodic boundaries, calculate

\\[ d\_{ij}=\left\\|\mathbf r_j-\mathbf r_i+\mathbf L\mathbf n\_{ij}^{\*}\right\\| \\]

where \\(\mathbf L\\) is the simulation-cell matrix and the integer image vector \\(\mathbf n\_{ij}^{\*}\\) minimizes the distance.

For orthorhombic boxes, this is the familiar minimum-image convention. For general triclinic cells, use a correct closest-periodic-image search rather than assuming componentwise rounding always finds it.

### Why the cutoff is crucial

Suppose you use:

\\[ N\_{\mathrm{Fe-O}}(i)=\sum\_{j\in O}\mathbf 1(d\_{ij}\<r_c) \\]

A fixed cutoff may classify a thermally distorted octahedron with one elongated bond as five-coordinate, even though the underlying local topology is still octahedral.

I recommend comparing two neighbor-selection schemes:

1. Adaptive shell cutoff: Determine the first Fe–O coordination-shell boundary from the partial radial distribution function \\(g\_{\mathrm{FeO}}(r)\\), optionally adapting it to local density or strain.
2. Voronoi-based neighbors: Use relative Voronoi solid-angle weights and Fe–O chemical filtering to select meaningful contacts.

The `CrystalNN` algorithm in pymatgen is one available implementation of Voronoi-based coordination with additional geometric and chemical weighting.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://pymatgen.org\&sz=32)

pymatgen 2026.9.23 documentation



For reliable classification, the neighbor graph should ideally be stable over a reasonable interval of cutoffs. A site whose neighbor count changes drastically with a small cutoff adjustment should receive lower confidence.

## 3. Stage II — Classify Fe as tetrahedral or octahedral

For each Fe \\(i\\), compute its Fe–O neighbor vectors:

\\[ \mathbf v\_{ij}=\mathbf r_j-\mathbf r_i \\]

using periodic-image-corrected displacement vectors.

The angle between two bonds is

\\[ \theta\_{jik}= \cos^{-1} \left( \frac{\mathbf v\_{ij}\cdot\mathbf v\_{ik}} {\\|\mathbf v\_{ij}\\|\\|\mathbf v\_{ik}\\|} \right) \\]

The expected angle distributions are:

Tetrahedral

4 Fe–O bonds

Octahedral

6 Fe–O bonds

| Ideal bond angles    | 109.47°     | 90°, 180°          |
| -------------------- | ----------- | ------------------ |
| Angle pairs          | 6           | 15                 |
| Angle multiplicities | 6 × 109.47° | 12 × 90°, 3 × 180° |

Define sorted measured bond angles \\(\theta^{(s)}\_k\\). The reference angles are

\\[ \Theta\_{\mathrm{tet}}= (109.47^\circ,\ldots,109.47^\circ) \\]

and

\\[ \Theta\_{\mathrm{oct}}= (\underbrace{90^\circ,\ldots,90^\circ}\_{12}, \underbrace{180^\circ,\ldots,180^\circ}\_{3}) \\]

Then, for a correctly coordinated candidate,

\\[ \boxed{ R\_{\mathrm{angle}}^{(m)}(i)= \sqrt{\frac{1}{M_m} \sum\_{k=1}^{M_m} \left(\theta_k^{(s)}-\Theta\_{m,k}\right)^2} } \\]

where \\(m\\) is tetrahedral or octahedral.

Sorting minimizes the squared mismatch between the two angle multisets, but it does not establish that all angles correspond to a geometrically consistent polyhedron. For that reason, I would add a bond-length metric and, preferably, a full geometric template fit.

For example,

\\[ D_r(i)= \frac{ \sqrt{\frac{1}{N_i}\sum_j(d\_{ij}-\bar d_i)^2} }{\bar d_i} \\]

measures normalized Fe–O bond-length distortion.

A preliminary classification could be:

| Condition                                    | Preliminary Fe type |
| -------------------------------------------- | ------------------- |
| \\(N_O=4\\), low tetrahedral geometric error | Tetrahedral         |
| \\(N_O=6\\), low octahedral geometric error  | Octahedral          |
| Other coordination or ambiguous geometry     | Distorted / unknown |

Do not force a five-coordinate Fe to be octahedral or a three-coordinate Fe to be tetrahedral. Later stages can use neighborhood context to determine whether these are defective members of a particular phase.

For thermal MD trajectories, I would favor continuous tetrahedral and octahedral scores over purely discrete coordination labels.

## 4. Stage III — Classify the oxygen environment

This is the strongest part of your second approach.

For each oxygen atom \\(j\\), let

\\[ \mathcal N\_{\mathrm{Fe}}(j) =\\{i:(i,j)\in E\\} \\]

be its directly coordinated Fe neighbors.

If the Fe sites have been classified as tetrahedral, octahedral, or unknown, calculate

\\[ n_T(j)=\sum\_{i\in\mathcal N\_{\mathrm{Fe}}(j)} \mathbf 1(c_i=T) \\]

\\[ n_O(j)=\sum\_{i\in\mathcal N\_{\mathrm{Fe}}(j)} \mathbf 1(c_i=O) \\]

\\[ n_U(j)=\sum\_{i\in\mathcal N\_{\mathrm{Fe}}(j)} \mathbf 1(c_i=U) \\]

Here \\(T,O,U\\) are Fe-site types, not oxidation states.

The oxygen environment vector is

\\[ \boxed{ \mathbf x_j= (n_T(j),n_O(j),n_U(j)) } \\]

For ideal structures:

\\[ \mathbf x_j^{\mathrm{mag}}=(1,3,0) \\]

\\[ \mathbf x_j^{\mathrm{wus}}=(0,6,0) \\]

This creates a natural local phase signature.

### Example

Consider five O atoms in a mixed structure:

| Oxygen | Tetra Fe | Octa Fe | Unknown Fe | Interpretation                |
| ------ | -------- | ------- | ---------- | ----------------------------- |
| O1     | 1        | 3       | 0          | Magnetite-like                |
| O2     | 0        | 6       | 0          | Wüstite-like                  |
| O3     | 1        | 2       | 0          | Defective magnetite candidate |
| O4     | 0        | 5       | 0          | Defective wüstite candidate   |
| O5     | 1        | 5       | 0          | Ambiguous/interface candidate |

For O3 and O4, the missing Fe neighbor is consistent with a possible vacancy but does not prove that one exists. Thermal disorder, an incorrect cutoff, and phase boundaries may produce similar patterns.

Likewise, O5 must not be called wüstite merely because it has six Fe neighbors, or magnetite merely because it has a tetrahedral Fe neighbor.

### Turn exact patterns into probabilistic scores

Instead of requiring exact equality to \\((1,3,0)\\) or \\((0,6,0)\\), define deviations.

A simple version is

\\[ D_M(j)= w_T[n_T(j)-1]^2+ w_O[n_O(j)-3]^2+ w_U n_U(j)^2 \\]

\\[ D_W(j)= w_T[n_T(j)]^2+ w_O[n_O(j)-6]^2+ w_U n_U(j)^2 \\]

Then obtain normalized scores:

\\[ \boxed{ P_M(j)= \frac{\exp[-D_M(j)/(2s^2)]} {\exp[-D_M(j)/(2s^2)]+\exp[-D_W(j)/(2s^2)]} } \\]

\\[ P_W(j)=1-P_M(j) \\]

Here \\(s\\) is a calibration parameter controlling how quickly the scores decrease with deviation.

These are heuristic scores, not calibrated statistical probabilities, unless the model is fitted and validated using labeled reference structures.

One limitation is that a zero-Fe-neighbor oxygen can still be assigned a numerical preference between the two phases. Therefore, these scores must be supplemented by an out-of-distribution check and an explicit unknown class.

### An important improvement: use soft Fe-site assignments

At elevated temperature, it is more robust to compute

\\[ p_i^T,\qquad p_i^O,\qquad p_i^U \\]

for each Fe, rather than immediately deciding whether it is tetrahedral or octahedral.

Then define the expected O-centered coordination:

\\[ \widetilde n_T(j)= \sum\_{i\in\mathcal N\_{\mathrm{Fe}}(j)}p_i^T \\]

\\[ \widetilde n_O(j)= \sum\_{i\in\mathcal N\_{\mathrm{Fe}}(j)}p_i^O \\]

This makes the algorithm less sensitive to marginal angular distortions.

For example, a neighboring Fe with a tetrahedral confidence of 0.85 contributes 0.85 to \\(\widetilde n_T\\), instead of abruptly contributing either zero or one.

## 5. Stage IV — Add a local composition and topology check

O-centered coordination is powerful, but a vacancy-rich region of wüstite may locally resemble defective magnetite.

Two additional descriptors substantially improve discrimination.

### 5.1 Local Fe/O ratio

For an oxygen-centered spherical neighborhood of radius \\(R\\),

\\[ \rho_j(R)= \frac{N\_{\mathrm{Fe}}(j,R)} {N\_{\mathrm O}(j,R)} \\]

where each atom in the physical neighborhood is counted once.

Expected bulk values:

\\[ \rho_M=\frac34=0.75 \\]

\\[ \rho_W=1 \\]

But do not use 0.75 and 1.0 as exact constraints in a small neighborhood.

For example, counting the four Fe neighbors of one oxygen and only the central oxygen gives 4, not 0.75. Local composition only becomes meaningful when the sampling region contains several coordination polyhedra, and boundary-related counting errors become relatively small.

For typical bulk-like neighborhoods, a multiscale estimate is better:

\\[ \boldsymbol\rho_j= \left[ \rho_j(R_1), \rho_j(R_2), \rho_j(R_3) \right] \\]

A strong algorithm examines how the ratio changes with sampling radius rather than trusting one radius.

At an interface, this also reveals whether the oxygen lies within a bulk domain or a region of mixed composition.

### 5.2 Fe-site connectivity

This is potentially more important than local stoichiometry.

Represent Fe atoms by nodes with type \\(T\\) or \\(O\\). Connect two Fe atoms if they share an oxygen neighbor.

You now have an Fe–O–Fe connectivity graph.

Magnetite has a characteristic ordered arrangement of tetrahedral and octahedral Fe sublattices. Wüstite instead has an underlying octahedrally coordinated rock-salt Fe sublattice, potentially containing vacancies and tetrahedral interstitial defects.

This means two neighborhoods can contain the same numbers of tetrahedral and octahedral Fe yet have different topologies.

For example, an isolated tetrahedral interstitial within FeO should not automatically be interpreted as the beginning of magnetite.

A robust classifier should ask:

- Are tetrahedral Fe atoms part of a spatially extended spinel-like arrangement?
- Do the surrounding octahedral Fe atoms exhibit the corresponding connectivity?
- Are the oxygen environments consistent with the same phase over multiple coordination shells?
- Does the Fe–O network align with a magnetite or wüstite reference topology?

For particularly demanding simulations, fit the local atomic neighborhood to ideal reference structures using translation, rotation, and a modest allowed affine deformation.

A reference-template residual could be written as

\\[ D\_{\mathrm{template}}^P(i)= \min\_{\mathbf R,\mathbf F,\pi,\mathbf t} \frac{1}{N} \sum\_{k=1}^{N} \left\\| \mathbf r_k- (\mathbf t+\mathbf R\mathbf F\mathbf r\_{\pi(k)}^{P,\mathrm{ref}}) \right\\|^2 \\]

where \\(P\in\\{M,W\\}\\), \\(\pi\\) matches atoms of the same species, \\(\mathbf R\\) is a rotation, \\(\mathbf F\\) is a constrained deformation, and \\(\mathbf t\\) is a translation.

Species-preserving correspondence is essential. A mismatch penalty should also be applied for missing or additional atoms rather than silently omitting them from the fit.

This option is computationally more expensive, but it provides an independent test of the crystal structure beyond coordination numbers.

## 6. Stage V — Spatial consistency and interface identification

Suppose a small pocket contains one Fe atom with tetrahedral coordination, but all surrounding oxygen atoms look wüstite-like.

A classifier that uses only that Fe might label it magnetite. A spatial classifier would recognize that the neighboring lattice is overwhelmingly wüstite and mark the tetrahedral Fe as a possible interstitial or defect.

For oxygen \\(j\\), define a neighborhood \\(\mathcal B_R(j)\\) and locally average the initial phase scores:

\\[ \overline P_M(j)= \frac{ \sum\_{k\in\mathcal B_R(j)} K(d\_{jk})P_M(k) }{ \sum\_{k\in\mathcal B_R(j)}K(d\_{jk}) } \\]

where a Gaussian kernel is one possibility:

\\[ K(d)=\exp\left(-\frac{d^2}{2\sigma_R^2}\right) \\]

This smooths out isolated geometric misclassifications.

However, smoothing has a potential drawback: it can wash out physically real interfaces.

I would therefore perform edge-aware smoothing where sites connected through similar local environments interact more strongly than sites with significantly different coordination fingerprints.

For example,

\\[ K\_{jk}= \exp\left(-\frac{d\_{jk}^2}{2\sigma_R^2}\right) \exp\left(-\frac{\\|\mathbf f_j-\mathbf f_k\\|^2} {2\sigma_f^2}\right) \\]

where \\(\mathbf f\\) contains coordination and topology descriptors.

A more formal version uses a conditional random field or graph-labeling model:

\\[ E(\mathbf z)= \sum_j U_j(z_j) + \lambda\sum\_{(j,k)\in E\_{\mathrm{OO}}} w\_{jk}\mathbf1(z_j\neq z_k) \\]

Here \\(U_j\\) is the local mismatch cost for assigning phase \\(z_j\\), and the second term discourages isolated label fluctuations while permitting boundaries.

This is especially useful when magnetite and wüstite coexist.

The final classes should be:

Magnetite

Wüstite

Interface / mixed

Defect / unknown

An explicit uncertain class is necessary. Otherwise, the algorithm will incorrectly force every atom into one of the two crystalline phases, including amorphous or severely reduced regions.

## 7. Recommended complete algorithm

For an Fe–O molecular dynamics simulation, I recommend the following workflow.

Atomic coordinates + atom types + simulation box

LAMMPS dump or relaxed structure

Build periodic Fe–O neighbor graph

Adaptive cutoff / Voronoi / shell-gap analysis

Classify Fe coordination

Tetrahedral / octahedral / distorted / unknown

Calculate O-centered structural fingerprints

Tet-Fe counts, oct-Fe counts, Fe–O bond geometry

Evaluate second-shell and multiscale information

Topology, Fe/O ratio, spatial ordering

Compute phase scores and uncertainty

Reference-calibrated magnetite and wüstite similarity

Final per-atom classification

Magnetite / Wüstite / Interface / Unknown

### Pseudocode

```
def classify_local_phases(atoms, cell, references):

    # 1. Find periodic Fe-O neighbors
    graph = build_fe_o_graph(
        atoms,
        cell,
        adaptive_cutoff=True
    )

    # 2. Fe-centered descriptors
    for fe in atoms.fe:

        neighbors = graph.oxygen_neighbors(fe)

        fe.features = {
            "coordination": len(neighbors),
            "tet_score": tetrahedral_score(fe, neighbors),
            "oct_score": octahedral_score(fe, neighbors),
            "bond_distortion": bond_distortion(fe, neighbors)
        }

        fe.site_probabilities = classify_fe_site(
            fe.features, references
        )

    # 3. O-centered descriptors
    for oxygen in atoms.oxygen:

        neighbors = graph.iron_neighbors(oxygen)

        oxygen.features = {
            "fe_coordination": len(neighbors),

            "tet_count": sum(
                fe.site_probabilities["tet"]
                for fe in neighbors
            ),

            "oct_count": sum(
                fe.site_probabilities["oct"]
                for fe in neighbors
            ),

            "unknown_count": sum(
                fe.site_probabilities["unknown"]
                for fe in neighbors
            )
        }

    # 4. Multiscale structural features
    for oxygen in atoms.oxygen:

        oxygen.features["stoichiometry"] = (
            local_fe_o_ratios(oxygen, atoms, cell)
        )

        oxygen.features["topology"] = (
            local_graph_fingerprint(oxygen, graph)
        )

    # 5. Classify against reference distributions
    for oxygen in atoms.oxygen:

        oxygen.phase_scores = (
            reference_phase_scores(
                oxygen.features,
                references
            )
        )

    # 6. Spatial consistency
    labels = spatial_phase_assignment(
        atoms.oxygen,
        graph
    )

    # 7. Reject ambiguous or unfamiliar environments
    labels = apply_confidence_and_unknown_rules(
        labels,
        atoms.oxygen
    )

    return labels
```

The functions here describe the architecture, not a fully implemented Python library. The reference scoring, topology fingerprints, and threshold calibration still need to be defined.

## 8. How to select tolerances robustly

I would strongly avoid choosing arbitrary values such as 10° for angle RMS and expecting them to work at all temperatures.

Instead, create labeled reference datasets:

1. Perfect magnetite and wüstite crystals.
2. Thermally equilibrated crystals at relevant simulation temperatures.
3. Strained and distorted structures.
4. Crystals containing Fe vacancies, oxygen vacancies, and Fe interstitials.
5. Mixed-phase structures with known interfaces.

For each descriptor \\(f\\), measure its distribution in each reference class:

\\[ p(f\mid M),\qquad p(f\mid W) \\]

For multiple descriptors, an initial statistical model is a multivariate Gaussian or a fitted classifier on their combined feature vector:

\\[ \mathbf f_j= \begin{bmatrix} \widetilde n_T\\\ \widetilde n_O\\\ \widetilde n_U\\\ D_r\\\ \rho(R_1)\\\ \rho(R_2)\\\ D\_{\mathrm{template}}^M\\\ D\_{\mathrm{template}}^W \end{bmatrix} \\]

One possible metric is a Mahalanobis distance:

\\[ D_P^2= (\mathbf f-\boldsymbol\mu_P)^T (\boldsymbol\Sigma_P+\epsilon I)^{-1} (\mathbf f-\boldsymbol\mu_P) \\]

where \\(\boldsymbol\mu_P\\) and \\(\boldsymbol\Sigma_P\\) are the class mean and covariance estimated from references.

This naturally accounts for correlations between descriptors, such as coordination number and bond-angle RMS.

A full implementation should include a rejection threshold for points too far from both reference distributions.

### Illustrative thresholds to calibrate

| Parameter             | Illustrative starting point                                       |
| --------------------- | ----------------------------------------------------------------- |
| Tetrahedral angle RMS | 8–15°<br>Temperature-sensitive                                    |
| Octahedral angle RMS  | 8–15°<br>Temperature-sensitive                                    |
| High phase confidence | 0.90<br>Only after probability calibration                        |
| Unknown / rejection   | Outside reference distribution<br>Do not force a phase assignment |

These numbers are initial trial settings, not experimentally established universal thresholds. Ideally, tune them on data generated with the same interatomic potential, temperatures, and pressure conditions as your production trajectories.

## 9. Special considerations for Fe–O reduction simulations

There are three situations that are especially important.

First: Nonstoichiometric wüstite. Fe₁₋ₓO commonly contains iron vacancies and may contain tetrahedral Fe interstitials. Thus a small number of tetrahedral Fe atoms does not demonstrate a magnetite phase. Defect clustering is a well-established aspect of wüstite chemistry.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.nature.com\&sz=32)

npj Computational Materials

+1



Second: Phase interfaces. An O atom at the Fe₃O₄/FeO interface may have neither ideal coordination signature. It should be marked as interfacial unless multi-shell evidence supports a clear assignment.

Third: Highly reduced or disordered regions. Fe atoms with very few O neighbors may belong to metallic Fe, a surface, a pore, or a disordered reduction region. A binary Fe₃O₄/FeO classifier cannot distinguish those states. For a complete iron-oxide reduction study, consider extending it to Fe metal and noncrystalline material using Fe–Fe coordination and additional structural descriptors.

For MD trajectories, additionally use temporal analysis. A phase assignment that persists over many saved frames is more convincing than one appearing for a single timestep. However, avoid smoothing away genuine short-lived transformations.

## 10. How I would validate the algorithm

I would test the classifier before applying it to a real phase-transition trajectory.

| Test structure                        | Expected behavior                                          |
| ------------------------------------- | ---------------------------------------------------------- |
| Perfect Fe₃O₄                         | Nearly all bulk oxygen classified magnetite                |
| Perfect FeO                           | Nearly all bulk oxygen classified wüstite                  |
| Thermal Fe₃O₄                         | High magnetite fraction, limited fluctuating unknown sites |
| Thermal FeO                           | High wüstite fraction                                      |
| Fe-deficient FeO                      | Mostly wüstite, with defects identified                    |
| FeO with tetrahedral Fe interstitials | Avoid false magnetite labels                               |
| Fe₃O₄/FeO coherent interface          | Two bulk domains and an interfacial region                 |
| Amorphous Fe–O                        | High unknown fraction                                      |
| Metallic Fe                           | Not mislabeled as an oxide phase                           |

Also test sensitivity to neighbor cutoffs, local averaging radii, simulation temperature, strain, and vacancy concentration.

For each known phase, calculate precision, recall, confusion matrices, and the fraction classified as unknown. For mixed interfaces, examine classification accuracy as a function of distance from the interface.

## Final recommendation

For your Fe–O atomistic simulations, I would build the classifier around the oxygen-centered Fe-site composition, because it directly distinguishes the ideal environments:

\\[ \boxed{\text{Magnetite: }(n_T,n_O)=(1,3)} \\]

\\[ \boxed{\text{Wüstite: }(n_T,n_O)=(0,6)} \\]

But I would not stop there. The stronger implementation is:

\\[ \boxed{ \text{Fe geometry} + \text{O coordination} + \text{multi-shell topology} + \text{spatial consistency} } \\]

with explicit rejection of ambiguous structures.

If computational cost is important, start with Fe geometry and O-centered coordination, then apply the more expensive topology or reference-template checks only to uncertain atoms and interfaces. This gives a practical balance between accuracy and computational efficiency.

For the Fe₃O₄ → FeO transformation, I would prefer classifying oxygen-centered local environments first, then assigning each Fe atom a local phase label based on its surrounding oxygen environments and structural connectivity. This avoids treating every octahedral Fe as intrinsically magnetite or wüstite and gives an interpretable picture of how the oxygen-centered network changes during reduction.
