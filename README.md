# BRMSD

**Bayesian RMSD alignment, smoothing, clustering, and domain identification for molecular dynamics trajectories.**

BRMSD implements a Bayesian framework for structure-based analysis of MD trajectories. Unlike standard RMSD, which weights all atoms equally, BRMSD simultaneously optimises per-atom weights and structural quantities (average structure, cluster centers, smoothed frames) so that rigid, well-aligned atoms automatically receive high weight and flexible or disordered regions are down-weighted. The result is a self-consistent, Bayes-optimal description of structural variability.

---

## Features

- **Alignment** — find the Bayes-optimal average structure and optimal per-atom weights for a trajectory
- **Smoothing** — replace each frame with a temporally-windowed average, suppressing thermal noise while preserving slow conformational changes
- **Clustering** — soft K-means with per-cluster per-atom weights
- **Domain identification** — identify K rigid domains by sequential peeling (default) or simultaneous optimisation with a mutual-overlap penalty
- **Focused alignment** — bias the weight distribution toward a user-specified domain to obtain a domain-centric average structure
- All inner loops use batched SVD (one LAPACK call per alignment step) and joblib parallelism where applicable
- Reads any trajectory format supported by MDAnalysis (XTC, TRR, DCD, …)

> **If you use BRMSD in your research, please cite:**
> Mukherjee, S.; Hummer, G. Bayesian Learning of Distance Metrics Beyond RMSD for Biomolecule Alignment, Clustering, and Domain Identification.  J. Chem. Theory Comput. 2026, 22, 10475–10487. DOI: [https://doi.org/10.1021/acs.jctc.6c01395](10.1021/acs.jctc.6c01395)

<details>
<summary>BibTeX</summary>

```bibtex
@article{Mukherjee2026brmsd,
    author = {Mukherjee, Saumyak and Hummer, Gerhard},
    title = {Bayesian Learning of Distance Metrics Beyond RMSD for Biomolecule Alignment, Clustering, and Domain Identification},
    journal = {Journal of Chemical Theory and Computation},
    volume = {22},
    number = {19},
    pages = {10475-10487},
    year = {2026},
    month = {10},
    issn = {1549-9618},
    doi = {10.1021/acs.jctc.6c01395},
    url = {https://doi.org/10.1021/acs.jctc.6c01395},
    eprint = {https://pubs.acs.org/jctcce/article-pdf/22/19/10475/68722013/acs.jctc.6c01395.pdf},
}
```

</details>
---
## Installation

We recommend installing BRMSD in a dedicated conda environment to avoid dependency conflicts.

### Step 1 — create and activate a conda environment

```bash
conda create -n BRMSD python=3.11
conda activate BRMSD
```

Any Python version from 3.8 to 3.12 is supported. Python 3.11 is recommended for best compatibility with MDAnalysis.

### Step 2 — install BRMSD

#### From source

```bash
git clone https://github.com/bio-phys/BRMSD.git
cd BRMSD
pip install .
```

#### Development install (editable, for contributors)

```bash
git clone https://github.com/bio-phys/BRMSD.git
cd BRMSD
pip install -e .
```

> **Note:** BRMSD is not yet available on PyPI. Install from source as shown above.

After installation the `BRMSD` command is available in your PATH whenever the environment is active.

### Verify installation

```bash
BRMSD --help
```

---

## Requirements

| Package | Minimum version |
|---------|----------------|
| Python | 3.8 |
| numpy | 1.21 |
| MDAnalysis | 2.0 |
| scipy | 1.7 |
| joblib | 1.1 |
| tqdm | 4.60 |

All dependencies are installed automatically by pip.

---

## Quick start

```bash
# Align trajectory and find average structure
BRMSD align -f traj.xtc -s topol.tpr -sel "protein" -sg 0.5

# Smooth trajectory (suppress thermal noise)
BRMSD smooth -f traj.xtc -s topol.tpr -sel "protein" -W 10 -sg 0.5 -j 8

# Cluster into 3 states (soft K-means)
BRMSD cluster -f traj.xtc -s topol.tpr -sel "protein" -A 3 -sg 0.1 -tau 2.0

# Identify rigid domains (sequential peeling, 3 domains by default)
BRMSD domain -f traj.xtc -s topol.tpr -sel "protein" -sg 0.05

# Write per-frame rotation vectors for post-analysis / dimensionality reduction
BRMSD align -f traj.xtc -s topol.tpr -sel "protein" -rot rotations.dat -rfmt rotvec
BRMSD domain -f traj.xtc -s topol.tpr -sel "protein" -rot rot.dat -rfmt quaternion
# → rot_domain1.dat, rot_domain2.dat, rot_domain3.dat

# Focused alignment biased toward a specific domain
BRMSD focus -f traj.xtc -s topol.tpr -sel "protein" -dom "resid 1:30" -sg 0.5 -mr 1.0
```

---

## Theory

All methods minimise a Bayesian free energy (the negative log-posterior) of the form

```
G = (structural fit term) + θ · Σ_α w_α ln(w_α / W_α)
```

where `w_α` are per-atom weights, `W_α` is a prior (uniform by default), and `σ` (Å) controls the sharpness of the weight distribution. Internally **θ = M·σ²**, where M is the number of structures. (For smoothing the windowed deviation is already a per-structure mean, so θ = σ²; clustering uses a single constant θ = M·σ².) Only the M factor appears — it is what makes the converged weights independent of sample size in the weight-update rule; the atom count N does not enter, so σ has the same meaning regardless of selection/pool size. The weight-update exponent reduces to ⟨MSF_α⟩/σ²:

- **Small σ** — weights sharpen onto the most rigid atoms only
- **Large σ** — weights stay near-uniform (approaches standard RMSD)

The structural fit term and the variable being optimised differ between subcommands (average structure, smoothed frame, cluster center). In all cases the optimal rotation is computed via the weighted Kabsch algorithm, vectorised over all frames using batched SVD.

---

## Subcommands

### `BRMSD align` — average structure and per-atom weights

Finds the self-consistent average structure and per-atom weights `w_α` that minimise the Bayesian free energy over the whole trajectory.

```bash
BRMSD align \
  -f traj.xtc -s topol.tpr \
  -sel "protein" \
  -ax xyz \
  -sg 0.5 -max 100 -tol 1e-3 \
  -b 0 -e 100000 -dt 20 \
  -o aligned.xtc \
  -p average.pdb \
  -rmsd rmsd.dat \
  -rmsf rmsf.dat \
  -wt weights.dat \
  -g BRMSD_align.log
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-sg` / `--sigma` | `0.5` | Position-fluctuation parameter σ (Å); θ = M·σ² internally. Decrease toward 0 to sharpen weights onto the rigid core; increase for standard RMSD alignment |
| `-max` / `--maxiter` | `100` | Maximum fixed-point iterations |
| `-tol` / `--tol` | `1e-3` | Convergence tolerance (structure and weight change) |
| `-sel` / `--selection` | `protein and not name H*` | MDAnalysis atom selection |
| `-ax` / `--axes` | `xyz` | Alignment axes: `xyz`, `xy`, `xz`, `yz`, `x`, … |
| `-b` / `-e` / `-dt` | `0` / last / `1.0` | Start time (ps), end time (ps), stride (ps) |
| `-o` | `None` | Aligned trajectory (optional) |
| `-rot` / `--outrotation` | `None` | Output file for per-frame rotation data (optional) |
| `-rfmt` / `--rotation-fmt` | `rotvec` | Rotation representation: `matrix` (9 cols, row-major) \| `quaternion` (4 cols, w x y z) \| `euler` (3 cols, radians) \| `rotvec` (3 cols, axis×φ) |
| `-eseq` / `--euler-seq` | `ZYX` | Euler angle convention, e.g. `ZYX`, `ZXZ`, `XYZ` (only for `--rotation-fmt euler`) |

**Outputs:**

| File | Contents |
|------|----------|
| `average.pdb` | Converged average structure (BRMSD-weighted) |
| `rmsd.dat` | Per-frame BRMSD to average, one block per iteration |
| `rmsf.dat` | Per-atom RMSF after alignment, one block per iteration |
| `weights.dat` | Converged per-atom weights `w_α` (sum to 1) |
| `aligned.xtc` | All frames rotated onto the average structure (optional) |
| `rotations.dat` | Per-frame rotation data in chosen format (optional; see `-rot`) |

---

### `BRMSD smooth` — trajectory smoothing

Replaces each frame `j` with a weighted average of its temporal neighbours within a window of half-width `W`. Thermal rattling (fast fluctuations) averages out; slow conformational changes survive because they are coherent across the window.

```bash
BRMSD smooth \
  -f traj.xtc -s topol.tpr \
  -sel "protein" \
  -W 10 -win triangular \
  -sg 0.5 -max 100 -tol 1e-3 \
  -j 8 \
  -b 0 -e 100000 -dt 20 \
  -o smoothed.xtc \
  -rmsd rmsd_smooth.dat \
  -rmsf rmsf_smooth.dat \
  -wt weights_smooth.dat \
  -g BRMSD_smooth.log
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-W` / `--half-width` | `5` | Window half-width in frames. Total window = 2W+1 frames |
| `-win` / `--window-type` | `triangular` | Window shape: `triangular` (hat function) or `uniform` (box) |
| `-sg` / `--sigma` | `0.5` | Alignment width σ (Å); θ = σ² internally (smoothing; M_eff=1) |
| `-j` / `--n-jobs` | `-1` | Parallel workers (-1 = all cores). Each frame is an independent job |

**Choosing W:** W should span the timescale of fast fluctuations but be much shorter than the slowest conformational transition you wish to preserve. As a starting point, set `W` to cover ~1–5% of the total trajectory length.

**Outputs:**

| File | Contents |
|------|----------|
| `smoothed.xtc` | Smoothed trajectory; each frame replaced by the windowed average |
| `rmsd_smooth.dat` | Per-frame residual BRMSD within the window after smoothing |
| `rmsf_smooth.dat` | Per-atom RMSF of the smoothed trajectory |
| `weights_smooth.dat` | Mean per-atom weights `w_α` averaged over all windows |

---

### `BRMSD cluster` — soft K-means clustering

Simultaneously optimises `A` cluster centers, per-cluster per-atom weights, and soft frame assignments (responsibilities `q(a|i)`). `A` is fixed by the user; screen it with `BRMSD sweep -m cluster --mode K` (Calinski-Harabasz / Davies-Bouldin / free-energy profile), and find σ and τ with `--mode sigma` and `--mode tau`.

```bash
BRMSD cluster \
  -f traj.xtc -s topol.tpr \
  -sel "protein" \
  -A 3 \
  -sg 0.1 -tau 2.0 \
  -max 100 -tol 1e-3 \
  -nr 5 -j 8 \
  -b 0 -e 100000 -dt 20 \
  -p cluster_centers.pdb \
  -rmsd rmsd_cluster.dat \
  -rmsf rmsf_cluster.dat \
  -wt weights_cluster.dat \
  -q responsibilities.dat \
  -g BRMSD_cluster.log
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-A` / `--n-clusters` | `2` | Number of clusters K |
| `-sg` / `--sigma` | `0.1` | Position-fluctuation parameter σ (Å); θ = M·σ² internally (single constant). Small σ sharpens onto the rigid core; large σ → near-uniform (standard RMSD-like) |
| `-tau` / `--tau` | `2.0` | Membership entropy penalty τ (Å²). Set ≈ inter-state MSD. Too large → uniform (degenerate) responsibilities; too small → hard assignments |
| `-nr` / `--n-restarts` | `1` | Independent random restarts; the lowest-G solution is kept |
| `-j` / `--n-jobs` | `-1` | Parallel workers across restarts (-1 = all cores) |

**Choosing τ:** Set τ ≈ the squared RMSD between distinct conformational states. If responsibilities are all ~1/A (degenerate clusters), τ is too large. If responsibilities are all 0 or 1 (collapsed to hard assignment), decrease τ.

**Outputs:**

| File | Contents |
|------|----------|
| `cluster_centers.pdb` | Multi-model PDB, one MODEL per cluster center. REMARK lines report per-cluster membership statistics |
| `rmsd_cluster.dat` | Per-frame BRMSD to each cluster center after alignment |
| `rmsf_cluster.dat` | Per-atom RMSF within each cluster (computed after alignment) |
| `weights_cluster.dat` | Per-cluster per-atom weights `w_{a,α}` (sum to 1 per cluster) |
| `responsibilities.dat` | `q(a|i)` — soft frame-to-cluster assignment. Columns: time(ps), q(cluster_1), q(cluster_2), … Sum to 1 across clusters per row |

---

### `BRMSD domain` — rigid domain identification

Identifies K rigid domains using one of two modes selected with `--peeling-mode`:

- **Sequential (default):** runs K independent K=1 optimisations on a shrinking atom pool. At each step the atoms claimed by the current domain (weight > 0.5 × max) are removed from the pool before the next domain is found. θ = M·σ² is the same constant at each step (pool-size independent), so σ has a single meaning across all peeling steps. No overlap penalty is needed (μ=0 enforced).
- **Simultaneous:** optimises K weight vectors jointly with an overlap-penalty term (μ/2) Σ_{j<k} (w_j·w_k)² that enforces non-overlapping domains. Multiple restarts are supported; the solution with the lowest free energy is returned.

```bash
# Sequential peeling (default)
BRMSD domain \
  -f traj.xtc -s topol.tpr \
  -sel "protein" \
  -sg 0.05 -nd 3 -nr 5 \
  -max 100 -tol 1e-3 \
  -b 0 -e 100000 -dt 20 \
  -p domain_rounds.pdb \
  -wt weights_domain.dat \
  -d domains.dat \
  -g BRMSD_domain.log

# Simultaneous K-domain with overlap penalty
BRMSD domain \
  -f traj.xtc -s topol.tpr \
  -sel "protein" \
  -pm simultaneous \
  -sg 0.05 -mu 1.0 -nd 3 -nr 5 \
  -max 100 -tol 1e-3 \
  -p domain_rounds.pdb \
  -wt weights_domain.dat \
  -d domains.dat
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-sg` / `--sigma` | `0.05` | Position-fluctuation parameter σ (Å); θ = M·σ² internally (same for sequential peeling and simultaneous, pool-size independent). Small σ sharpens onto the rigid core |
| `-pm` / `--peeling-mode` | `sequential` | `sequential`: iterative peeling with shrinking atom pool; `simultaneous`: joint K-domain with overlap penalty |
| `-nd` / `--n-domains` | `3` | Number of domains (peeling steps or simultaneous K) |
| `-nr` / `--n-restarts` | `5` | Independent restarts per step; best G is kept |
| `-mu` / `--mu` | `0.0` | Overlap penalty μ (Å⁻²); used only in simultaneous mode. Large μ enforces non-overlapping domains |
| `-mt` / `--mask-thr` | `0.5` | Fraction of max weight for domain mask: atom claimed if w > mask_thr × max(w) |
| `-max` / `--maxiter` | `100` | Max fixed-point iterations per restart |
| `-tol` / `--tol` | `1e-3` | Convergence tolerance |
| `-rot` / `--outrotation` | `None` | Base filename for per-domain rotation output; domain index appended automatically (e.g. `rot.dat` → `rot_domain1.dat`, `rot_domain2.dat`, …) |
| `-rfmt` / `--rotation-fmt` | `rotvec` | Rotation representation: `matrix` \| `quaternion` \| `euler` \| `rotvec` |
| `-eseq` / `--euler-seq` | `ZYX` | Euler angle convention (only for `--rotation-fmt euler`) |

**Outputs:**

| File | Contents |
|------|----------|
| `domain_rounds.pdb` | Multi-model PDB, one MODEL per domain + a final MODEL with the rigidity map (max weight across all K domains). B-factor column encodes w_α × N (uniform=1.0, rigid>1, flexible≈0). Load in PyMOL with `spectrum b, blue_white_red` |
| `weights_domain.dat` | Per-domain per-atom weights w_{k,α} (sum to 1 per domain) — primary result |
| `weights_domain_priors.dat` | Uniform prior W_α = 1/N |
| `weights_domain_G.dat` | Free-energy convergence history (G, G_fit, G_ent, G_ovl, Δstruct, Δweights per iteration) |
| `domains.dat` | Binary domain assignment per atom: 1 = weight > mask_thr × max(w) for that domain, 0 = below |
| `rot_domain1.dat` … | Per-frame rotation data for each domain in chosen format (optional; see `-rot`) |

**Interpreting the results:** In sequential mode each domain is the rigid core identified from the atoms not yet claimed by a previous domain. In simultaneous mode, distinct domains concentrate on non-overlapping atomic subsets enforced by the overlap penalty. Increase μ if two simultaneous domains are nearly identical.

---

### `BRMSD focus` — focused alignment

Runs the BRMSD alignment with an additional concentration penalty applied to atoms in a user-specified domain D. The modified free energy is:

```
G = Σ_i MSD(s, x_i; w) + θ Σ_α w_α ln(w_α / W_α) + μ Σ_{α∈D} w_α ln(n w_α)
```

where n = |D| and μ = mu_ratio × θ. The extra term biases the weight distribution so that atoms in D receive higher weight relative to the rest of the protein. Setting `--mu-ratio 0` recovers standard `BRMSD align`.

The weight update bifurcates: atoms outside D follow the standard `BRMSD align` update; atoms inside D use an effective penalty θ + μ with a modified prefactor W_α^{θ/(θ+μ)} / n^{μ/(θ+μ)}.

```bash
BRMSD focus \
  -f traj.xtc -s topol.tpr \
  -sel "protein and name CA" \
  -dom "resid 1:30" \
  -sg 0.5 -mr 1.0 \
  -max 100 -tol 1e-3 \
  -b 0 -e 100000 -dt 20 \
  -o focused_aligned.xtc \
  -p average_focus.pdb \
  -rmsd rmsd_focus.dat \
  -rmsf rmsf_focus.dat \
  -wt weights_focus.dat \
  -g BRMSD_focus.log
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-dom` / `--domain-sel` | (required) | MDAnalysis selection for focus domain D (e.g. `"resid 1:30"`) |
| `-sg` / `--sigma` | `0.5` | Position-fluctuation parameter σ (Å); θ = M·σ² internally |
| `-mr` / `--mu-ratio` | `1.0` | Domain penalty μ/θ (dimensionless). 0 → standard alignment; larger values concentrate weight more strongly inside D |
| `-max` / `--maxiter` | `100` | Max fixed-point iterations |
| `-tol` / `--tol` | `1e-3` | Convergence tolerance |

**Outputs:**

| File | Contents |
|------|----------|
| `average_focus.pdb` | Domain-focused average structure |
| `rmsd_focus.dat` | Per-frame BRMSD per iteration |
| `rmsf_focus.dat` | Per-atom RMSF per iteration |
| `weights_focus.dat` | Converged per-atom weights (sum to 1); atoms in D carry elevated weight |

---

### `BRMSD sweep` — parameter scan for any module

Scans one parameter of any BRMSD module over a range of values and writes a CSV of quality diagnostics, so you can choose an operating point (σ, τ, K, μ, window width, or mask threshold) instead of guessing.

```bash
# Screen the number of clusters K (Calinski-Harabasz / Davies-Bouldin / free energy)
BRMSD sweep -m cluster --mode K -f traj.xtc -s topol.tpr -sel "protein" \
  --range 2 8 7 --sigma 4.0 --tau 2.0 -o cluster_K_sweep.csv

# Find the alignment σ (log-spaced scan)
BRMSD sweep -m align --mode sigma -f traj.xtc -s topol.tpr -sel "protein" \
  --range 0.1 10 20 --log-space -o align_sigma_sweep.csv

# Scan the smoothing window width
BRMSD sweep -m smooth --mode window -f traj.xtc -s topol.tpr -sel "protein" \
  --list 5 10 20 40 --sigma 0.5 -o smooth_window_sweep.csv
```

**Key options:**

| Flag | Description |
|------|-------------|
| `-m` / `--module` | Module to sweep: `align` \| `cluster` \| `focus` \| `domain` \| `smooth` |
| `--mode` | Parameter to sweep: `sigma` \| `tau` \| `K` \| `mu` \| `mask_thr` \| `window` (must be valid for the chosen module) |
| `--range MIN MAX N` | Sweep N values from MIN to MAX |
| `--list VAL …` | Explicit list of values (alternative to `--range`) |
| `--log-space` | Log-spaced values with `--range` (recommended for σ) |
| `--sigma` / `--tau` / `-K` / … | Fixed value for any parameter that is *not* being swept |
| `-dom` / `--domain-sel` | Focus domain (required for `focus` / `mu` sweeps) |
| `-o` / `--out` | Output CSV [`{module}_{mode}_sweep.csv`] |

The output CSV has the swept parameter in the first column and the module's quality metrics in the rest. Integer modes (`K`, `window`, and `mask_thr` steps) are rounded and de-duplicated.

---

## Common options across all subcommands

| Flag | Description |
|------|-------------|
| `-f` / `--intrajectory` | Input trajectory (XTC, TRR, DCD, …) |
| `-s` / `--topology` | Topology file (TPR, GRO, PDB, PSF, …) |
| `-sel` / `--selection` | MDAnalysis atom selection string (default: `protein and not name H*`) |
| `-ax` / `--axes` | Alignment axes: `xyz` (default), `xy`, `xz`, `yz`, `x`, `y`, or `z` |
| `-b` / `--begin` | Start time in ps (default: 0) |
| `-e` / `--end` | End time in ps (default: last frame) |
| `-dt` / `--skiptime` | Time stride in ps (default: 1.0; uses every frame with default trajectory dt) |
| `-g` / `--log` | Log file |

---

## Parameter guide

### Rigidity threshold σ

| Use case | σ |
|----------|---|
| Standard RMSD-like alignment (uniform weights) | large σ (≳ a few Å) |
| Concentrate on the rigid core (alignment, domain ID) | small σ |
| Clustering a conformational transition (broad weights) | large σ |

σ → ∞ recovers standard RMSD (uniform weights); decreasing σ progressively concentrates weight onto the rigid core. The operating point is system-dependent — choose it from the `n_eff` / quality diagnostics, not a fixed default. The benchmark values, for reference: alignment/smoothing/focus σ ≈ 0.43 Å (SND3), clustering σ = 4 Å and domain detection σ = 3 Å (ADK). Note that clustering of a transition needs a *broad* σ: concentrating weight on the rigid core erases the mobile-domain signal that separates the states and collapses the clusters.

### Membership penalty τ (clustering)

τ controls how soft the cluster assignments are. Set τ ≈ the squared RMSD between distinct conformational states. If responsibilities are all ~1/A (degenerate clusters), τ is too large. If responsibilities collapse to 0 or 1 (hard assignment), decrease τ.

### Rotation output formats

Four representations of the per-frame rotation matrix R are available via `-rfmt`:

| Format | Columns | Description |
|--------|---------|-------------|
| `rotvec` | 3 | Rotation vector axis×φ (radians); \|ω\| = rotation angle φ. Recommended for PCA / dimensionality reduction — minimal, no redundancy |
| `quaternion` | 4 | Unit quaternion (w x y z). Compact, singularity-free |
| `euler` | 3 | Euler angles in radians; convention set by `-eseq` (default ZYX = yaw-pitch-roll) |
| `matrix` | 9 | Full 3×3 rotation matrix, row-major flattened |

All formats are plain-text and loadable with `np.loadtxt`. Column 0 is time in ps (or frame index if time is unavailable).

### Domain concentration penalty μ (focused alignment)

`--mu-ratio` sets μ/θ (dimensionless). A value of 1.0 (default) applies a penalty equal in strength to the alignment entropy term. Larger values concentrate weight more strongly inside the focus domain at the expense of the rest of the structure.

### Smoothing window W

W should cover the timescale of fast thermal fluctuations but not the timescale of conformational transitions. For a 1 ns trajectory with 1 ps frames, W=50–200 is a reasonable starting range.

---

## Usage as a Python library

All analysis functions are importable directly:

```python
import numpy as np
import MDAnalysis as mda
from BRMSD import (load_frames, BRMSD_iterate, BRMSD_smooth,
                    soft_kmeans, identify_domains, identify_domains_sequential,
                    BRMSD_focus_iterate,
                    compute_rotations_per_frame, write_rotation_data)

# Load trajectory
u = mda.Universe("topol.tpr", "traj.xtc")
sel = u.select_atoms("protein")
all_pos, times = load_frames(u, sel.indices, list(range(500)))

axes = [0, 1, 2]   # xyz
N    = all_pos.shape[1]

# All entry points take sigma (Å) directly; theta = M·sigma² is computed
# internally (sigma² for smoothing, where the windowed deviation is already a
# per-structure mean; M·sigma² for clustering). Small sigma concentrates weight
# on the rigid core; large sigma → uniform (RMSD).

# Alignment  (sigma=0.01 Å)
avg, weights, history = BRMSD_iterate(all_pos, axes, sigma=0.01,
                                       maxiter=100, tol=1e-3)

# Smoothing  (same sigma meaning as alignment)
smoothed, w_frames, rmsds = BRMSD_smooth(
    all_pos, axes, sigma=0.01, maxiter=100, tol=1e-3,
    half_width=10, window_type='triangular', n_jobs=4)

# Soft K-means clustering  (sigma=0.1 Å, tau=2.0 Å²)
centers, w_ca, q_ai, history, free_energy, dev_ai = soft_kmeans(
    all_pos, n_clusters=3, axes_idx=axes, sigma=0.1, tau=2.0,
    maxiter=100, tol=1e-3)

# Domain identification — sequential peeling (default)
results = identify_domains_sequential(all_pos, axes, sigma=0.12,
                                      maxiter=100, tol=1e-3, n_domains=3)

# Domain identification — simultaneous with overlap penalty
results_sim = identify_domains(all_pos, axes, sigma=0.12, mu=1.0,
                               maxiter=100, tol=1e-3, n_domains=3)

# Focused alignment — concentrate weight on domain D (atom indices 0–29)
domain_mask = np.zeros(N, dtype=bool)
domain_mask[:30] = True
avg_f, w_f, history_f = BRMSD_focus_iterate(all_pos, axes, 0.01, 1.0,
                                             domain_mask, maxiter=100, tol=1e-3)

# Per-frame rotation matrices at convergence (M, 3, 3)
R_arr = compute_rotations_per_frame(all_pos, avg, weights, axes)

# Write rotation data — four formats
write_rotation_data(R_arr, times, 'rotations_rotvec.dat',   fmt='rotvec')
write_rotation_data(R_arr, times, 'rotations_quat.dat',     fmt='quaternion')
write_rotation_data(R_arr, times, 'rotations_euler.dat',    fmt='euler', euler_seq='ZYX')
write_rotation_data(R_arr, times, 'rotations_matrix.dat',   fmt='matrix')

# Load and run PCA on rotation vectors
from sklearn.decomposition import PCA
rotvec = np.loadtxt('rotations_rotvec.dat', comments='#')[:, 1:]  # drop time col
pca = PCA(n_components=2).fit_transform(rotvec)
```

---

## Output file formats

All tabular outputs are plain-text whitespace-delimited files with `#`-prefixed comment lines, readable directly by NumPy, pandas, or xmgrace:

```python
import numpy as np
data = np.loadtxt("rmsd.dat", comments="#")
# data[:, 0] = time (ps)
# data[:, 1] = RMSD (Å)
```

The multi-model PDB (`cluster_centers.pdb`) can be loaded in PyMOL or VMD:

```
# PyMOL
load cluster_centers.pdb

# VMD (command line)
vmd -pdb cluster_centers.pdb
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
