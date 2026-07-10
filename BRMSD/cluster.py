"""
BRMSD cluster  —  soft K-means conformational clustering with per-cluster atom weights

SYNOPSIS

    BRMSD cluster -s TOPOLOGY -f TRAJECTORY -A K [options]

DESCRIPTION

    Partitions a trajectory into K conformational clusters by jointly
    optimising K cluster centers {s_a}, K per-cluster atom weight vectors
    {w_a}, and per-frame soft responsibilities q(a|i) ∈ [0,1].  Each
    cluster carries its own weight vector, so the BRMSD metric adapts to
    the internal rigidity of each conformational state.  The joint free
    energy is:

        G = Σ_a Σ_i q(a|i) Σ_α w_{a,α} |s_{a,α} − R_{ia} r_{i,α}|²
            + θ   Σ_a Σ_α w_{a,α} ln w_{a,α}
            + τ   Σ_i Σ_a q(a|i)  ln q(a|i)

    with θ = M·σ² and τ > 0 (Å²).  Large τ produces soft, diffuse
    responsibilities; small τ recovers hard K-means.  Multiple restarts
    from random seeds (--n-restarts) guard against local minima; the run
    with the lowest G is kept.

    K is fixed by the user.  Use "BRMSD sweep -m cluster --mode K" to
    screen K values via the Calinski-Harabasz and Davies-Bouldin indices
    and the free-energy profile.  Find σ_op with "--mode sigma" and
    τ_op with "--mode tau".

OUTPUT

    cluster_centers.pdb     Multi-model PDB, one MODEL per cluster center;
                            B-factor = w·N (rigid > 1, uniform = 1).
    weights_cluster.dat     Per-cluster per-atom weights w_{a,α}.
    rmsf_cluster.dat        Per-atom RMSF within each cluster.
    rmsd_cluster.dat        Per-frame RMSD to each cluster center.
    responsibilities.dat    Soft responsibilities q(a|i) per frame.
    labels.dat              Hard cluster label (argmax_a q) per frame.
    populations.dat         Cluster population summary.
    BRMSD_cluster.log       Run log.

EXAMPLES

    Soft K-means with K=2:

        BRMSD cluster -s topol.tpr -f prod.xtc -A 2 \\
                      -sg 4.0 -tau 10.0 -sel "protein and not name H*"

    Screen K from 2 to 6 at known σ and τ:

        BRMSD sweep -m cluster --mode K --range 2 6 5 \\
                    --sigma 4.0 --tau 10.0 \\
                    -s topol.tpr -f prod.xtc -sel "protein and not name H*"

THEORY

    Fixed-point updates (simultaneous over all K clusters):

        q^(n+1)(a|i) ∝ exp( −Σ_α w_{a,α}^(n) |s_{a,α}^(n) − R_{ia}^(n) r_{i,α}|² / τ )

        w_{a,α}^(n+1) ∝ exp( −Σ_i q^(n+1)(a|i) |s_{a,α}^(n) − R_{ia}^(n) r_{i,α}|² / θ )

        s_{a,α}^(n+1) = Σ_i q^(n+1)(a|i) R_{ia}^(n+1) r_{i,α}  /  Σ_i q^(n+1)(a|i)

    with θ = M·σ².  K is fixed; G is non-convex, so multiple restarts are used.
"""

import logging
import numpy as np
from joblib import Parallel, delayed
from .core import align_frame, batch_align_frames


# Cluster quality metrics

def pairwise_center_rmsd(centers, axes_idx):
    """
    Compute the (A, A) pairwise RMSD matrix between cluster centers.

    Each pair is optimally superimposed (unweighted Kabsch) before computing
    RMSD, so the matrix is independent of the global reference frame.
    """
    A = len(centers)
    mat = np.zeros((A, A))
    d   = len(axes_idx)
    for a in range(A):
        for b in range(a + 1, A):
            ca = centers[a][:, axes_idx] - centers[a][:, axes_idx].mean(axis=0)
            cb = centers[b][:, axes_idx] - centers[b][:, axes_idx].mean(axis=0)
            H  = ca.T @ cb
            U, _, Vht = np.linalg.svd(H)
            D = np.eye(d)
            D[-1, -1] = np.linalg.det(U @ Vht)
            ca_rot = ca @ (U @ D @ Vht)
            rmsd   = float(np.sqrt(np.mean(np.sum((ca_rot - cb) ** 2, axis=1))))
            mat[a, b] = mat[b, a] = rmsd
    return mat


def cluster_quality_metrics(rmsd_ak, q_ai, center_rmsd_mat):
    """
    Calinski-Harabasz (CH) and Davies-Bouldin (DB) quality indices
    for soft clustering.

    Parameters
    ----------
    rmsd_ak          : (A, M) per-cluster per-frame RMSD to center
    q_ai             : (A, M) soft responsibilities
    center_rmsd_mat  : (A, A) pairwise center-to-center RMSD

    Returns
    -------
    ch  : float  Calinski-Harabasz index (higher = better separated)
    db  : float  Davies-Bouldin index (lower = more compact)
    s_a : (A,)   per-cluster soft-weighted mean within-RMSD (linear; for DB)
    """
    A, M = q_ai.shape
    n_eff = q_ai.sum(axis=1)                             # cluster populations (A,)

    # per-cluster soft-weighted mean within-RMSD (linear; used by DB + reporting)
    s_a = np.array([
        np.average(rmsd_ak[a], weights=q_ai[a]) if n_eff[a] > 1e-300 else 0.0
        for a in range(A)
    ])

    # Calinski-Harabasz: ratio of between- to within-cluster SQUARED scatter.
    # Within scatter  W = Σ_a Σ_i q(a|i) d(x_i, c_a)²   (soft-weighted SS).
    # Between scatter B = Σ_a n_a |c_a − c_global|², with the squared
    # center-to-global distances recovered from the pairwise center-RMSD
    # matrix by the weighted double-centering identity (c_global = the
    # population-weighted mean of the cluster centers).
    if A > 1:
        W = float(np.sum(q_ai * rmsd_ak ** 2))
        p   = n_eff / n_eff.sum()
        D2  = center_rmsd_mat ** 2
        center_var = 0.5 * float(p @ D2 @ p)
        dist_sq = np.maximum(D2 @ p - center_var, 0.0)   # |c_a − c_global|²
        B = float(np.dot(n_eff, dist_sq))
        ch = float((B * (M - A)) / (W * (A - 1))) if W > 1e-300 else np.nan
    else:
        ch = np.nan

    db_terms = []
    for a in range(A):
        best = 0.0
        for b in range(A):
            if a == b:
                continue
            d_ab = center_rmsd_mat[a, b]
            if d_ab > 1e-10:
                best = max(best, (s_a[a] + s_a[b]) / d_ab)
        db_terms.append(best)
    db = float(np.mean(db_terms)) if A > 1 else np.nan

    return ch, db, s_a


def compute_free_energy(dev_ai, q_ai, w_ca, sigma, tau, N, M):
    """
    Evaluate the soft K-means free energy G at converged iterates.

    dev_ai : (A, M)  weighted MSD per (cluster, frame)
    q_ai   : (A, M)  responsibilities
    w_ca   : (A, N)  per-cluster atom weights
    sigma  : float   position-fluctuation parameter (Å); the atom-weight
                     entropy penalty is the single constant theta = M*sigma**2
                     (constant in q, so alternating updates descend G
                     monotonically).
    tau    : float   responsibility entropy penalty
    N, M   : number of atoms, frames

    Returns G (scalar).
    """
    # MSD term: sum_a sum_i q(a|i) * dev(a, i)
    G_msd = float(np.einsum('am,am->', q_ai, dev_ai))

    # atom-weight entropy: theta * sum_a sum_alpha w ln(w)  [pure Shannon],
    # theta = M*sigma**2 constant (does not depend on the optimized q).
    theta = M * sigma ** 2
    safe_w = np.where(w_ca > 1e-300, w_ca, 1.0)
    G_w = float(theta * np.sum(w_ca * np.log(safe_w)))

    # responsibility entropy: tau * sum_i sum_a q(a|i) ln(q(a|i) * M)
    safe_q = np.where(q_ai > 1e-300, q_ai, 1.0)
    G_q = float(tau * np.sum(q_ai * np.log(safe_q * M)))

    return G_msd + G_w + G_q


# Soft K-means clustering

def soft_kmeans(all_pos, n_clusters, axes_idx, sigma, tau, maxiter, tol,
                prior_w=None, init_centers=None,
                n_restarts=1, n_jobs=1):
    """
    Soft K-means clustering with BRMSD metric.

    Parameters
    ----------
    all_pos      : (M, N, 3)
    n_clusters   : int  A, number of clusters
    axes_idx     : list of axis indices
    sigma        : float  position-fluctuation parameter (Å).  The atom-weight
                   entropy uses a single constant penalty theta = M*sigma**2
                   (M = total frames), so G is a fixed objective and the
                   alternating q/w/center updates each minimise G exactly →
                   monotone descent.  The effective concentration of cluster a
                   then scales with its population fraction n_a/M (a more-sampled
                   state concentrates more); the result is invariant to
                   proportional resampling.
    tau          : float  responsibility entropy penalty (Å²)
    maxiter      : int
    tol          : float
    prior_w      : (N,)   W_alpha [default 1/N]
    init_centers : (A, N, 3) initial cluster centers [default: random draw]
    n_restarts   : int    number of independent random restarts; best G is kept
    n_jobs       : int    parallel workers for restarts (joblib, prefer='threads')

    Returns
    -------
    centers     : (A, N, 3)  cluster center structures
    w_ca        : (A, N)     cluster-specific per-atom weights
    q_ai        : (A, M)     responsibilities q(a|i)
    history     : list of dict  per-iteration diagnostics
    free_energy : float  converged free energy G
    dev_ai_final: (A, M)  converged weighted MSD per (cluster, frame)
    """
    M, N, _ = all_pos.shape
    A = n_clusters

    # theta constant in q → G is a fixed objective with monotone descent.
    theta = M * sigma ** 2

    if prior_w is None:
        prior_w = np.ones(N) / N

    # multi-restart: run n_restarts independent initialisations
    if n_restarts > 1 and init_centers is None:
        def _one_restart(seed):
            rng_r = np.random.default_rng(seed=seed)
            ctrs  = all_pos[rng_r.choice(M, size=A, replace=False)].copy().astype(float)
            return soft_kmeans(all_pos, n_clusters, axes_idx, sigma, tau,
                               maxiter, tol, prior_w=prior_w,
                               init_centers=ctrs, n_restarts=1, n_jobs=1)
        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_one_restart)(42 + r) for r in range(n_restarts))
        best = min(results, key=lambda x: x[4])  # x[4] = free_energy
        logging.info(f"  Best restart G = {best[4]:.6f}  "
                     f"(over {n_restarts} restarts)")
        return best

    # initialise cluster centers (random)
    # Draw A frames at random without replacement.  For reproducibility each
    # restart is seeded deterministically; the caller controls the seed via
    # init_centers (provide explicit (A, N, 3) centers to override).
    if init_centers is not None:
        centers = init_centers.copy().astype(float)
    else:
        rng = np.random.default_rng(seed=42)
        init_idx = rng.choice(M, size=A, replace=False)
        centers = all_pos[init_idx].copy().astype(float)

    # initialise per-cluster atom weights and responsibilities
    w_ca = np.tile(prior_w, (A, 1))       # (A, N)
    # uniform initial responsibilities
    q_ai = np.ones((A, M)) / A            # (A, M)

    history = []

    for it in range(1, maxiter + 1):

        # vectorised: for each cluster, align all M frames at once
        dev_ai   = np.zeros((A, M))     # weighted MSD per (cluster, frame)
        atom_dev = np.zeros((A, M, N))  # per-atom sq-dev per (cluster, frame)

        for a in range(A):
            aln_a, _ = batch_align_frames(all_pos, centers[a], w_ca[a], axes_idx)
            diff_a   = centers[a][:, axes_idx][np.newaxis] - aln_a   # (M,N,d)
            pa       = np.einsum('mnd,mnd->mn', diff_a, diff_a)       # (M,N)
            atom_dev[a] = pa                                           # (M,N)
            dev_ai[a]   = pa @ w_ca[a]                                 # (M,)

        # responsibility update
        log_q = -dev_ai / tau                            # (A, M)
        log_q -= log_q.max(axis=0, keepdims=True)        # per-frame stabilisation
        new_q  = np.exp(log_q)
        new_q /= new_q.sum(axis=0, keepdims=True)        # normalise over clusters

        # atom-weight update
        # Single constant entropy penalty theta = M*sigma**2 (independent of
        # the optimized q) → exact minimiser of the fixed free energy G.
        q_atom_dev = np.einsum('am,amn->an', new_q, atom_dev)   # (A, N)
        log_w = -q_atom_dev / theta
        log_w -= log_w.max(axis=1, keepdims=True)
        new_w  = np.exp(log_w)                                   # (A, N)  pure entropy: no prior
        w_sums = new_w.sum(axis=1, keepdims=True)
        w_sums = np.where(w_sums > 1e-300, w_sums, 1.0)
        new_w /= w_sums                                          # normalise over atoms

        # center update — re-align with updated w
        new_centers = centers.copy()
        for a in range(A):
            q_sum = new_q[a].sum()
            if q_sum < 1e-300:
                continue
            aln_a, _ = batch_align_frames(all_pos, centers[a], new_w[a], axes_idx)
            new_centers[a][:, axes_idx] = (
                np.einsum('m,mnd->nd', new_q[a], aln_a) / q_sum)

        # convergence check
        ds = max(np.linalg.norm(new_centers[a] - centers[a]) for a in range(A))
        dw = max(np.linalg.norm(new_w[a]       - w_ca[a])    for a in range(A))
        dq = np.linalg.norm(new_q - q_ai)

        logging.debug(f"  iter {it:4d}:  Δcenter={ds:.2e}  Δweights={dw:.2e}  "
                      f"Δrespons={dq:.2e}")

        history.append({'iter': it, 'ds': ds, 'dw': dw, 'dq': dq,
                        'q_ai': new_q.copy(), 'w_ca': new_w.copy()})

        centers, w_ca, q_ai = new_centers, new_w, new_q

        if ds < tol and dw < tol and dq < tol:
            logging.info(f"  Converged at iteration {it}  "
                         f"(Δcenter={ds:.2e}, Δweights={dw:.2e}, Δrespons={dq:.2e}).")
            break

    # Recompute final dev_ai with converged iterates for free energy / metrics
    final_dev_ai = np.zeros((A, M))
    for a in range(A):
        aln_a, _ = batch_align_frames(all_pos, centers[a], w_ca[a], axes_idx)
        diff_a   = centers[a][:, axes_idx][np.newaxis] - aln_a
        pa       = np.einsum('mnd,mnd->mn', diff_a, diff_a)
        final_dev_ai[a] = pa @ w_ca[a]

    N = all_pos.shape[1]
    free_energy = compute_free_energy(final_dev_ai, q_ai, w_ca, sigma, tau, N, M)
    logging.debug(f"  Converged free energy G = {free_energy:.6f}")

    return centers, w_ca, q_ai, history, free_energy, final_dev_ai


# CLI entry point

def run(args):
    import sys
    import MDAnalysis as mda
    from .core import (setup_logging, backup_file, parse_axes,
                       load_frames, write_cluster_centers_pdb,
                       write_rmsd, write_rmsf, write_weights)

    # logging + backups
    output_files = [args.outpdb, args.outrmsd, args.outrmsf, args.outweights,
                    args.outrespons, args.outlabels, args.outpop, args.log]
    for f in output_files:
        backup_file(f)
    setup_logging(args.log, verbose=args.verbose)
    logging.info("=== BRMSD cluster: soft K-means ===")

    # load trajectory
    u   = mda.Universe(args.topology, args.intrajectory)
    dt  = u.trajectory.dt
    start = int(args.begin / dt)
    end   = int(u.trajectory.n_frames) if args.end is None else int(args.end / dt)
    skip  = max(1, int(args.skiptime / dt))
    frames = list(range(start, end, skip))
    M = len(frames)
    logging.info(f"Frames {start}–{end-1}, stride {skip}, total {M}")

    axes = parse_axes(args.axes)
    if not axes:
        logging.error("No valid axes.")
        sys.exit(1)

    sel     = u.select_atoms(args.selection)
    sel_idx = sel.indices
    if len(sel_idx) == 0:
        logging.error("Atom selection returned no atoms.")
        sys.exit(1)
    logging.info(f"Selection '{args.selection}' → {len(sel_idx)} atoms")
    logging.info(f"sigma={args.sigma}  (theta = M*sigma² internally)  "
                 f"tau={args.tau}  maxiter={args.maxiter}  tol={args.tol}")

    all_pos, times = load_frames(u, sel_idx, frames)

    # soft K-means
    A = args.n_clusters
    logging.info(f"n_clusters={A}  n_restarts={args.n_restarts}")
    centers, w_ca, q_ai, history, free_energy, dev_ai = soft_kmeans(
        all_pos, A, axes, args.sigma, args.tau,
        args.maxiter, args.tol,
        n_restarts=args.n_restarts, n_jobs=args.n_jobs)

    w_all   = list(w_ca)
    hard    = np.argmax(q_ai, axis=0)   # (M,)
    repr_frames = [int(np.argmax(q_ai[a])) for a in range(A)]

    # per-frame unweighted RMSD to each cluster center (written to rmsd_cluster.dat)
    rmsd_ak = np.zeros((A, M))
    for k_idx in range(M):
        for a, (ctr, w_a) in enumerate(zip(centers, w_all)):
            aln_k, _, _, _ = align_frame(all_pos[k_idx], ctr, w_a, axes)
            sq = np.sum((aln_k - ctr[:, axes]) ** 2, axis=1)
            rmsd_ak[a, k_idx] = np.sqrt(np.mean(sq))

    # quality metrics use the weighted per-(cluster, frame) RMSD (sqrt of the
    # converged weighted MSD), matching the definition used by BRMSD sweep so
    # CH/DB scores agree between the two entry points at identical parameters.
    rmsd_ak_w    = np.sqrt(np.maximum(dev_ai, 0))
    ctr_rmsd_mat = pairwise_center_rmsd(list(centers), axes)
    ch, db, s_a  = cluster_quality_metrics(rmsd_ak_w, q_ai, ctr_rmsd_mat)

    # degeneracy check
    if A > 1:
        max_sep = float(ctr_rmsd_mat.max())
        mean_within = float(s_a.mean())
        if max_sep < 0.5 * mean_within:
            logging.warning(
                f"DEGENERATE CLUSTERING: all {A} cluster centers are nearly "
                f"identical (max pairwise RMSD = {max_sep:.4f} Å, "
                f"mean within-cluster RMSD = {mean_within:.4f} Å). "
                f"This happens when (a) the trajectory contains only one "
                f"conformational state, or (b) tau={args.tau:.3g} Å² is too "
                f"large — try tau ≈ 10–100 × within-cluster RMSD² "
                f"(~{mean_within**2 * 10:.0f}–{mean_within**2 * 100:.0f} Å²). "
                f"Use 'BRMSD sweep -m cluster --mode K' to find the optimal "
                f"number of clusters.")

    logging.info("=== Cluster quality metrics ===")
    logging.info(f"  Calinski-Harabasz score : {ch:.4f}  (higher = better)")
    logging.info(f"  Davies-Bouldin score    : {db:.4f}  (lower  = better)")
    if free_energy is not None:
        logging.info(f"  Free energy G           : {free_energy:.6f}")
    # compact summary of pairwise separations
    if A > 1:
        off_diag = ctr_rmsd_mat[np.triu_indices(A, k=1)]
        logging.info(f"  Center separations (Å)  : "
                     f"min={off_diag.min():.3f}  "
                     f"mean={off_diag.mean():.3f}  "
                     f"max={off_diag.max():.3f}")
    # full matrix at debug level only
    logging.debug("  Pairwise center RMSD matrix (Å):")
    for a in range(A):
        row_str = "  ".join(f"{ctr_rmsd_mat[a, b]:.3f}" for b in range(A))
        logging.debug(f"    cluster_{a+1}:  {row_str}")

    # RMSF per cluster
    rmsf_data = {}
    for a, (ctr, w_a) in enumerate(zip(centers, w_all)):
        aln_a, _ = batch_align_frames(all_pos, ctr, w_a, axes)
        sq = np.einsum('mnd,mnd->mn',
                       aln_a - ctr[:, axes][None],
                       aln_a - ctr[:, axes][None])
        rmsf_data[f"cluster_{a+1}"] = np.sqrt(
            np.average(sq, axis=0, weights=q_ai[a]))

    # build PDB labels
    pdb_labels = []
    for a in range(A):
        n_hard = int((hard == a).sum())
        repr_t = times[repr_frames[a]]
        lbl = (f"Model {a+1}: hard_assigned={n_hard}/{M}  "
               f"soft_pop={q_ai[a].mean():.3f}  "
               f"repr_frame={repr_frames[a]}  (t={repr_t:.1f} ps)  "
               f"within_rmsd={s_a[a]:.3f} Å")
        pdb_labels.append(lbl)
    N = len(sel_idx)
    write_cluster_centers_pdb(u, sel_idx, list(centers), args.outpdb,
                              labels=pdb_labels,
                              bfactors=[w * N for w in w_all])

    # write standard outputs
    rmsd_data = []
    for k_idx in range(M):
        for a in range(A):
            rmsd_data.append((f"cluster_{a+1}", k_idx, rmsd_ak[a, k_idx]))

    weight_data  = {f"cluster_{a+1}": w_all[a]  for a in range(A)}
    respons_data = {f"cluster_{a+1}": q_ai[a]   for a in range(A)}

    write_rmsd(rmsd_data, args.outrmsd, times,
               header="Per-frame unweighted RMSD to each cluster center")
    write_rmsf(rmsf_data, sel_idx, args.outrmsf,
               header="Per-atom RMSF within each cluster")
    write_weights(weight_data, sel_idx, args.outweights,
                  header="Per-cluster per-atom BRMSD weights w_{a,alpha}")

    # write responsibilities (xmgrace: one dataset per cluster)
    with open(args.outrespons, 'w') as f:
        f.write("# Per-frame soft cluster responsibilities\n")
        f.write("# xmgrace: xmgrace responsibilities.dat\n")
        for a in range(A):
            f.write(f"# q(cluster_{a+1})\n")
            f.write("# time(ps)   responsibility\n")
            for k_idx in range(M):
                f.write(f"{times[k_idx]:.6f}\t{q_ai[a][k_idx]:.6f}\n")
            f.write("&\n")
    logging.info(f"Wrote responsibilities → {args.outrespons}")

    # write hard labels
    with open(args.outlabels, 'w') as f:
        f.write("# Hard cluster label per frame (argmax of soft responsibilities)\n")
        f.write("# xmgrace: xmgrace labels.dat\n")
        f.write("# time(ps)   hard_label\n")
        for k_idx in range(M):
            f.write(f"{times[k_idx]:.6f}\t{hard[k_idx] + 1}\n")
        f.write("&\n")
    logging.info(f"Wrote hard labels → {args.outlabels}")

    # write population summary
    with open(args.outpop, 'w') as f:
        f.write("# Cluster population summary\n")
        f.write(f"# n_clusters={A}  n_frames={M}\n")
        if not np.isnan(ch):
            f.write(f"# Calinski-Harabasz={ch:.4f}  Davies-Bouldin={db:.4f}\n")
        if free_energy is not None:
            f.write(f"# free_energy={free_energy:.6f}\n")
        f.write("# cluster  soft_pop  hard_pop  hard_count  "
                "repr_frame  repr_time_ps  within_rmsd_A\n")
        for a in range(A):
            n_hard = int((hard == a).sum())
            f.write(f"cluster_{a+1}  "
                    f"{q_ai[a].mean():.4f}  "
                    f"{n_hard / M:.4f}  "
                    f"{n_hard}  "
                    f"{repr_frames[a]}  "
                    f"{times[repr_frames[a]]:.2f}  "
                    f"{s_a[a]:.4f}\n")
    logging.info(f"Wrote population summary → {args.outpop}")

    # summary
    logging.info("=== Cluster summary ===")
    for a in range(A):
        n_hard = int((hard == a).sum())
        logging.debug(f"  Cluster {a+1}: hard_pop={n_hard}/{M} ({100*n_hard/M:.1f}%)  "
                      f"soft_pop={q_ai[a].mean():.3f}  "
                      f"repr_frame={repr_frames[a]} (t={times[repr_frames[a]]:.1f} ps)  "
                      f"within_rmsd={s_a[a]:.3f} Å")
    logging.info("Done.")


# argparse sub-parser definition

def add_subparser(subparsers):
    from .core import CompactHelpFormatter
    p = subparsers.add_parser(
        'cluster',
        help='BRMSD soft K-means conformational clustering',
        description=__doc__,
        formatter_class=CompactHelpFormatter)

    io = p.add_argument_group('I/O')
    io.add_argument('-f', '--intrajectory', required=True,  metavar=None,
                    help='Input trajectory')
    io.add_argument('-s', '--topology',     required=True,  metavar=None,
                    help='Topology file')
    io.add_argument('-p',   '--outpdb',    default='cluster_centers.pdb',   metavar=None,
                    help='Multi-model PDB of cluster centers [cluster_centers.pdb]')
    io.add_argument('-rmsd', '--outrmsd',   default='rmsd_cluster.dat',     metavar=None,
                    help='Per-frame RMSD to each cluster center [rmsd_cluster.dat]')
    io.add_argument('-rmsf', '--outrmsf',   default='rmsf_cluster.dat',     metavar=None,
                    help='Per-atom RMSF per cluster [rmsf_cluster.dat]')
    io.add_argument('-wt',   '--outweights',default='weights_cluster.dat',  metavar=None,
                    help='Per-cluster atom weights [weights_cluster.dat]')
    io.add_argument('-q',    '--outrespons',default='responsibilities.dat', metavar=None,
                    help='Soft responsibilities q(a|i) per frame [responsibilities.dat]')
    io.add_argument('-l',    '--outlabels', default='labels.dat',           metavar=None,
                    help='Hard cluster label per frame [labels.dat]')
    io.add_argument('-pop', '--outpop',     default='populations.dat',      metavar=None,
                    help='Cluster population summary [populations.dat]')
    io.add_argument('-g',    '--log',       default='BRMSD_cluster.log',   metavar=None,
                    help='Log file [BRMSD_cluster.log]')

    sel = p.add_argument_group('Selection & axes')
    sel.add_argument('-sel','--selection', default='protein and not name H*', metavar=None,
                     help="Atom selection [protein and not name H*]")
    sel.add_argument('-ax', '--axes',      default='xyz', metavar=None,
                     help="Alignment axes [xyz]")

    opt = p.add_argument_group('Algorithm')
    opt.add_argument('-A',   '--n-clusters', type=int,   default=2,    metavar=None,
                     dest='n_clusters',
                     help='Number of clusters K [2]')
    opt.add_argument('-sg',  '--sigma',    type=float, default=0.1,    metavar=None,
                     help='Position-fluctuation parameter σ (Å); θ = M·σ² internally. Small σ concentrates weight on the rigid core; large σ → uniform weights (standard RMSD). [0.1]')
    opt.add_argument('-tau', '--tau',      type=float, default=2.0,    metavar=None,
                     help='Membership entropy penalty τ (Å²) — set ≈ inter-state MSD; '
                          'tau >> MSD gives uniform (degenerate) assignments [2.0]')
    opt.add_argument('-max', '--maxiter',  type=int,   default=100,    metavar=None,
                     help='Max iterations [100]')
    opt.add_argument('-tol', '--tol',      type=float, default=1e-3,   metavar=None,
                     help='Convergence tolerance [1e-3]')
    opt.add_argument('-nr',  '--n-restarts', type=int,  default=1,     metavar=None,
                     dest='n_restarts',
                     help='Independent random restarts; the lowest-G solution is kept [1]')
    opt.add_argument('-j',   '--n-jobs',   type=int,   default=-1,     metavar=None,
                     dest='n_jobs',
                     help='Parallel workers across restarts (-1 = all cores) [-1]')
    opt.add_argument('-v',  '--verbose',   action='store_true', default=False,
                     help='Print per-iteration convergence and full RMSD matrix to screen [off]')

    traj = p.add_argument_group('Trajectory range')
    traj.add_argument('-b', '--begin',    type=float, default=0.0,  metavar=None,
                      help='Start time (ps) [0.0]')
    traj.add_argument('-e', '--end',      type=float, default=None, metavar=None,
                      help='End time (ps) [last frame]')
    traj.add_argument('-dt','--skiptime', type=float, default=1.0,  metavar=None,
                      help='Time stride (ps) [1.0]')

    p.set_defaults(func=run)
