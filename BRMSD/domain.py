"""
BRMSD domain  —  rigid domain identification by sequential peeling or simultaneous fit

SYNOPSIS

    BRMSD domain -s TOPOLOGY -f TRAJECTORY [options]

DESCRIPTION

    Identifies K rigid domains in a trajectory by partitioning per-atom
    BRMSD weight vectors across domains.  Two strategies are available
    (--peeling-mode):

      sequential   Peel K domains iteratively.  At each step, one BRMSD
                   alignment is run on the current atom pool; atoms whose
                   weight exceeds mask_thr × max(w) are claimed for that
                   domain and removed before the next step.  θ = M·σ² is
                   rescaled to the reduced pool size at each step.
                   [default]

      simultaneous Optimise all K weight vectors jointly with an overlap
                   penalty (μ > 0) that enforces mutual non-overlap:

                       G = Σ_k Σ_i MSD(s_k, x_i; w_k)
                           + θ Σ_k Σ_α w_{k,α} ln(w_{k,α} / W_α)
                           + (μ/2) Σ_{j<k} (w_j · w_k)²

    G is non-convex; multiple restarts (--n-restarts) are used and the
    solution with the lowest G is retained.  Use "BRMSD sweep -m domain
    --mode sigma" to find σ_op and "--mode K" to screen the number of
    domains.  "--mode mask_thr" sweeps the assignment threshold without
    re-optimising.

OUTPUT

    domain_rounds.pdb      Multi-model PDB: one MODEL per domain + final
                           rigidity map.  B-factor = w/w_max × 100
                           (rigid = 100, flexible ≈ 0).  Load in PyMOL:
                           "load domain_rounds.pdb; spectrum b, blue_white_red".
    weights_domain.dat     Per-domain per-atom weights.
    domains.dat            Hard domain assignment per atom (argmax over domains).
    BRMSD_domain.log       Run log.
    [rot_domain{k}.dat]    Per-domain rotation time series (optional, -rot).

EXAMPLES

    Sequential peeling into 3 domains:

        BRMSD domain -s topol.tpr -f prod.xtc -sg 3.0 -nd 3 \\
                     -sel "protein and not name H*"

    Simultaneous fit with overlap penalty:

        BRMSD domain -s topol.tpr -f prod.xtc -sg 3.0 -nd 3 \\
                     -pm simultaneous -mu 1.0 \\
                     -sel "protein and not name H*"

THEORY

    Sequential peeling: each step is a standard BRMSD alignment on the
    remaining atom pool.  After convergence, atoms with w_α > mask_thr·max(w)
    are claimed and removed.  θ is rescaled to the pool size at each step
    so σ has the same physical meaning throughout.

    Simultaneous mode fixed-point update (Jacobi, all K domains updated
    from the previous iteration's values):

        w_{k,α}^(n+1) ∝ W_α exp( −dev_{k,α}/θ  −  μ Σ_{j≠k}(w_j·w_k) w_{j,α} )

        s_{k,α}^(n+1) = M⁻¹ Σ_i R_{ki}^(n+1) r_{i,α}

    where dev_{k,α} = Σ_i |s_{k,α} − R_{ki} r_{i,α}|².  Large μ enforces
    non-overlapping domains; μ = 0 recovers K independent BRMSD alignments.
    Restart 0 uses K evenly-spaced frames; subsequent restarts draw K frames
    randomly (seeded by restart index for reproducibility).
"""

import logging
import numpy as np
from joblib import Parallel, delayed
from .core import batch_align_frames


# Free-energy evaluation

def _compute_G(dev_sums, w, prior, theta, mu):
    """
    Evaluate the free energy G given current state.

    Uses quantities already in memory — no extra alignment calls needed.

    Parameters
    ----------
    dev_sums : list of K (N,) arrays   Σ_i |s_{k,α} − R_{ki} r_{i,α}|²
    w        : (K, N) array            current weight vectors
    prior    : (N,) array              uniform prior (1/N)
    theta    : float                   entropy penalty
    mu       : float                   overlap penalty

    Returns
    -------
    G        : float   free energy value
    G_fit    : float   structural-fit contribution
    G_ent    : float   entropy-penalty contribution
    G_ovl    : float   overlap-penalty contribution
    """
    K, N = w.shape

    # Term 1: structural fit  Σ_k Σ_α w_{k,α} dev_{k,α}
    G_fit = sum(float(np.dot(w[k], dev_sums[k])) for k in range(K))

    # Term 2: entropy penalty  θ Σ_k Σ_α w_{k,α} ln(w_{k,α} / W_α)
    # W_α = 1/N  →  ln(w/W) = ln(w·N)
    G_ent = 0.0
    for k in range(K):
        wk  = w[k]
        pos = wk > 0
        G_ent += float(np.dot(wk[pos], np.log(wk[pos] * N)))
    G_ent *= theta

    # Term 3: overlap penalty  (μ/2) Σ_{j<k} (w_j · w_k)²
    G_ovl = 0.0
    for j in range(K):
        for k in range(j + 1, K):
            G_ovl += float(np.dot(w[j], w[k])) ** 2
    G_ovl *= mu / 2.0

    return G_fit + G_ent + G_ovl, G_fit, G_ent, G_ovl


# Single fixed-point run from a given initialisation

def _single_run(all_pos, axes_idx, theta, mu, maxiter, tol, prior, avg_init,
                w_init=None):
    """
    Run the fixed-point iteration from a given set of starting averages.

    Parameters
    ----------
    all_pos   : (M, N, 3)
    axes_idx  : list of axis indices
    theta, mu : floats
    maxiter   : int
    tol       : float
    prior     : (N,) uniform prior
    avg_init  : (K, N, 3) starting averages

    Returns
    -------
    avg     : (K, N, 3)  converged averages
    w       : (K, N)     converged weights
    G_final : float      free energy of converged solution
    history : list of dicts with keys 'iter', 'G', 'G_fit', 'G_ent', 'G_ovl', 'ds', 'dw'
    converged : bool
    """
    K = avg_init.shape[0]
    N = len(prior)

    avg = avg_init.copy()
    # Seed the weight vectors: uniform by default, or from a warm start (e.g. the
    # sequential-peeling solutions) so the K domains begin in distinct basins
    # rather than all collapsing onto the globally most rigid region.
    w   = w_init.copy() if w_init is not None else np.tile(prior, (K, 1))
    history = []
    ds = dw = np.inf

    for it in range(1, maxiter + 1):

        # Step 1: align every domain; collect dev_sums (used for G and weight update)
        dev_sums = []
        for k in range(K):
            _, dev_k = batch_align_frames(all_pos, avg[k], w[k], axes_idx)
            dev_sums.append(dev_k)

        # Compute G with current (pre-update) state — zero extra alignment cost
        G, G_fit, G_ent, G_ovl = _compute_G(dev_sums, w, prior, theta, mu)

        # Step 2: update all weight vectors simultaneously (Jacobi)
        new_w = np.zeros((K, N))
        for k in range(K):
            overlap = np.zeros(N)
            for j in range(K):
                if j != k:
                    overlap += np.dot(w[j], w[k]) * w[j]
            overlap *= mu

            exponent  = -dev_sums[k] / theta - overlap
            exponent -= exponent.max()
            nwk = prior * np.exp(exponent)
            s   = nwk.sum()
            new_w[k] = nwk / s if s > 1e-300 else prior.copy()

        # Step 3: update averages using freshly normalised weights
        new_avg = avg.copy()
        for k in range(K):
            al_k, _ = batch_align_frames(all_pos, avg[k], new_w[k], axes_idx)
            new_avg[k][:, axes_idx] = al_k.mean(axis=0)

        ds = max(np.linalg.norm(new_avg[k] - avg[k]) for k in range(K))
        dw = max(np.linalg.norm(new_w[k]   - w[k])   for k in range(K))

        logging.debug(f"    iter {it:4d}: G={G:.6g}  "
                      f"(fit={G_fit:.4g}  ent={G_ent:.4g}  ovl={G_ovl:.4g})  "
                      f"Δstruct={ds:.2e}  Δweights={dw:.2e}")

        history.append({'iter': it, 'G': G,
                        'G_fit': G_fit, 'G_ent': G_ent, 'G_ovl': G_ovl,
                        'ds': ds, 'dw': dw})

        avg, w = new_avg, new_w

        if ds < tol and dw < tol:
            break

    converged = (ds < tol and dw < tol)

    # Compute final G with the converged state (one extra alignment pass)
    dev_sums_final = []
    for k in range(K):
        _, dev_k = batch_align_frames(all_pos, avg[k], w[k], axes_idx)
        dev_sums_final.append(dev_k)
    G_final, *_ = _compute_G(dev_sums_final, w, prior, theta, mu)

    return avg, w, G_final, history, converged


# Public entry point: multi-restart domain identification

def identify_domains(all_pos, axes_idx, sigma, mu, maxiter, tol,
                     n_domains=5, n_restarts=1, mask_thr=0.5, n_jobs=1,
                     seed_w=None, seed_avg=None):
    """
    Simultaneously identify K rigid domains with an overlap-penalty free energy.

    Runs up to n_restarts independent fixed-point iterations from different
    starting frames and returns the solution with the lowest final G.

    Parameters
    ----------
    all_pos    : (M, N, 3)  per-frame positions of selected atoms
    axes_idx   : list       axis indices, e.g. [0, 1, 2]
    sigma      : float      position-fluctuation parameter (Å); entropy penalty
                            theta = M*sigma**2 (Å²), computed internally from
                            the shape of all_pos (N = pool size at this call).
    mu         : float      overlap penalty (Å⁻²)
    maxiter    : int        maximum iterations per restart
    tol        : float      convergence tolerance
    n_domains  : int        number of domains K
    n_restarts : int        number of independent restarts (default 1)
    mask_thr   : float      fraction of max weight for domain mask: atom included
                            if w > mask_thr * max(w)  (default 0.5)
    n_jobs     : int        parallel workers for restarts; prefer='threads'
                            (numpy releases GIL — no data copy overhead).
                            -1 = all available cores  (default 1)

    Returns
    -------
    results : list of dict, one per domain (k = 1 … K), with keys:
        'round'      : domain index (1-based)
        'avg'        : (N, 3) average structure
        'w'          : (N,)   per-atom weights
        'prior'      : (N,)   uniform prior (1/N)
        'max_w'      : (N,)   max weight across all K domains
        'domain_mask': (N,) bool  atoms with w > mask_thr * max(w)
        'history'    : list of per-iteration dicts
                       {'iter', 'G', 'G_fit', 'G_ent', 'G_ovl', 'ds', 'dw'}
        'G_final'    : float  free energy of the returned solution
    """
    M, N, _ = all_pos.shape
    K        = n_domains
    prior    = np.ones(N) / N

    # Entropy penalty: theta = M*sigma**2.  dev_sums are raw sums over the M
    # frames, so the exponent ~ <MSF>/sigma**2 is independent of M (sample size)
    # and of the pool size N — sigma has the same meaning on a full set or a
    # sub-pool (sequential peeling).
    theta = M * sigma ** 2

    def _run_restart(restart):
        if restart == 0:
            init_idx = np.round(np.linspace(0, M - 1, K)).astype(int)
        else:
            rng      = np.random.default_rng(seed=restart)
            init_idx = rng.choice(M, size=K, replace=False)
        avg_init = np.array([all_pos[f].copy() for f in init_idx])
        avg, w, G_final, history, converged = _single_run(
            all_pos, axes_idx, theta, mu, maxiter, tol, prior, avg_init)
        logging.info(f"  Restart {restart+1}/{n_restarts}: "
                     f"{'converged' if converged else 'NOT converged'} "
                     f"in {history[-1]['iter']} iters  G={G_final:.6g}")
        return avg, w, G_final, history

    if seed_w is not None and seed_avg is not None:
        # Warm-started single run (e.g. refinement from sequential-peeling seeds):
        # each domain begins in its own basin, so the joint optimum is reachable.
        K = seed_w.shape[0]
        avg, w, G_final, history, converged = _single_run(
            all_pos, axes_idx, theta, mu, maxiter, tol, prior,
            seed_avg.copy(), w_init=seed_w.copy())
        logging.info(f"  Refinement (warm start, K={K}, mu={mu:.4g}): "
                     f"{'converged' if converged else 'NOT converged'} in "
                     f"{history[-1]['iter']} iters  G={G_final:.6g}")
        run_results = [(avg, w, G_final, history)]
    elif n_restarts == 1 or n_jobs == 1:
        run_results = [_run_restart(r) for r in range(n_restarts)]
    else:
        run_results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_run_restart)(r) for r in range(n_restarts))

    best = min(run_results, key=lambda x: x[2])
    best_avg, best_w, best_G, best_history = best
    if n_restarts > 1:
        logging.info(f"  Best G across {n_restarts} restarts: {best_G:.6g}")

    # build results from the best solution
    avg, w = best_avg, best_w
    max_w_all = np.max(w, axis=0)

    results = []
    for k in range(K):
        domain_mask = w[k] > mask_thr * w[k].max()
        logging.info(f"  Domain {k + 1}: {domain_mask.sum()}/{N} atoms above "
                     f"{mask_thr:.2f}*max (w_max={w[k].max():.3e}  w_mean={w[k].mean():.3e})")
        results.append({
            'round':       k + 1,
            'avg':         avg[k].copy(),
            'w':           w[k].copy(),
            'prior':       prior.copy(),
            'max_w':       max_w_all.copy(),
            'domain_mask': domain_mask.copy(),
            'history':     best_history,
            'G_final':     best_G,
        })

    for j in range(K):
        for k in range(j + 1, K):
            ov = float(np.dot(w[j], w[k]))
            logging.info(f"  Domain {j+1}·Domain {k+1} overlap (w_j·w_k): {ov:.6f}")

    return results


# Sequential peeling: K independent K=1 runs with shrinking atom pools

def identify_domains_sequential(all_pos, axes_idx, sigma, maxiter, tol,
                                n_domains=3, n_restarts=1, mask_thr=0.5,
                                n_jobs=1, refine=False, mu_refine=1.0):
    """
    Identify K rigid domains by sequential peeling.

    Each step runs a single K=1, mu=0 BRMSD optimization on the current atom
    pool.  Atoms whose weight exceeds mask_thr * max(w) are claimed for that
    domain and removed before the next step.  theta = M*sigma**2 is the same
    constant inside identify_domains at every step, so the meaning of sigma is
    independent of pool size and trajectory length.

    Parameters
    ----------
    all_pos    : (M, N, 3)  per-frame positions of selected atoms
    axes_idx   : list       axis indices
    sigma      : float      position-fluctuation parameter (Å); theta =
                            M*sigma**2 (same every step)
    maxiter    : int        maximum iterations per step
    tol        : float      convergence tolerance
    n_domains  : int        number of peeling steps
    n_restarts : int        independent restarts per step; best G is kept
    mask_thr   : float      fraction of max weight for claiming atoms (default 0.5)

    Returns
    -------
    results : list of dict, one per step, with the same keys as identify_domains
              plus 'pool_indices' : (N_k,) int  indices of the pool at this step.
              All weight and average arrays are in the full N-atom space (zeros
              outside the pool at the time of this step).
    """
    M, N, _ = all_pos.shape
    remaining = np.arange(N, dtype=int)
    results = []

    for step in range(1, n_domains + 1):
        N_k = len(remaining)
        if N_k == 0:
            logging.warning(f"  Step {step}: pool exhausted; "
                            f"stopping at {step - 1} domain(s).")
            break

        sub_pos = all_pos[:, remaining, :]   # (M, N_k, 3)

        logging.info(f"  Step {step}/{n_domains}: pool={N_k} atoms  "
                     f"sigma={sigma}  (theta = M*sigma² internally)  "
                     f"({n_restarts} restart(s))")

        # K=1, mu=0 run — reuses the multi-restart engine.  identify_domains
        # computes theta = M*sigma**2 internally (pool-size independent).
        step_res = identify_domains(
            sub_pos, axes_idx, sigma, mu=0.0, maxiter=maxiter, tol=tol,
            n_domains=1, n_restarts=n_restarts, mask_thr=mask_thr,
            n_jobs=n_jobs)

        r = step_res[0]

        # map N_k-space arrays back to the full N-atom space
        w_full = np.zeros(N)
        w_full[remaining] = r['w']
        avg_full = np.zeros((N, 3))
        avg_full[remaining] = r['avg']
        prior_full = np.zeros(N)
        prior_full[remaining] = r['prior']
        max_w_full = np.zeros(N)
        max_w_full[remaining] = r['max_w']
        mask_full  = np.zeros(N, dtype=bool)
        mask_full[remaining] = r['domain_mask']

        logging.info(f"  Step {step}: {r['domain_mask'].sum()}/{N_k} atoms claimed "
                     f"(mask_thr={mask_thr})  G_final={r['G_final']:.6g}")

        results.append({
            'round':         step,
            'avg':           avg_full,
            'w':             w_full,
            'prior':         prior_full,
            'max_w':         max_w_full,
            'domain_mask':   mask_full,
            'pool_indices':  remaining.copy(),
            'history':       r['history'],
            'G_final':       r['G_final'],
        })

        remaining = remaining[~r['domain_mask']]
        logging.info(f"  Remaining pool after step {step}: {len(remaining)} atoms")

    # Optional joint refinement
    # Use the peeling solutions as warm-start seeds for a single simultaneous
    # optimization over the FULL atom set (overlap-penalty functional).  Each
    # domain starts in its own basin, so the collapse-onto-CORE failure mode of
    # cold-started simultaneous detection is avoided; boundary/hinge atoms can
    # reassign and all domains are re-optimized self-consistently against one
    # free energy.  Refinement is local — it improves peeling, not rescues it.
    if refine and len(results) >= 2:
        K = len(results)
        theta  = M * sigma ** 2
        mu_abs = mu_refine * theta
        w_seed   = np.array([r['w']   for r in results])   # (K, N) full-atom space
        avg_seed = np.array([r['avg'] for r in results])   # (K, N, 3)
        logging.info(f"  === Joint refinement: simultaneous K={K} from peeling "
                     f"seeds  (mu/theta={mu_refine}) ===")
        refined = identify_domains(
            all_pos, axes_idx, sigma, mu_abs, maxiter, tol,
            n_domains=K, n_restarts=1, mask_thr=mask_thr, n_jobs=n_jobs,
            seed_w=w_seed, seed_avg=avg_seed)

        # Collapse guard
        # If the overlap penalty is too weak, refined domains can drift back onto
        # the globally most-rigid region (partial collapse), giving fewer than K
        # distinct domains — a result worse than peeling.  Detect this via the
        # argmax (hard) partition of the refined weights: each domain must own a
        # non-trivial set of atoms.  If not, keep the peeling result and warn.
        w_ref  = np.array([rr['w'] for rr in refined])     # (K, N)
        hard   = np.argmax(w_ref, axis=0)
        counts = np.bincount(hard, minlength=K)
        live   = int((counts >= max(3, int(0.01 * N))).sum())
        if live < K:
            logging.warning(f"  Refinement collapsed ({live}/{K} distinct domains, "
                            f"counts={counts.tolist()}); keeping peeling result. "
                            f"Try a larger --mu-refine.")
            return results
        logging.info(f"  Refinement accepted: {live}/{K} distinct domains "
                     f"(argmax counts={counts.tolist()}).")
        for k, rr in enumerate(refined):
            rr['round']        = k + 1
            rr['pool_indices'] = np.arange(N)          # refinement spans all atoms
            rr['peeling_w']    = results[k]['w']       # keep greedy result for comparison
        return refined

    return results


# CLI entry point

def run(args):
    import sys
    import os
    import MDAnalysis as mda
    from .core import (setup_logging, backup_file, parse_axes,
                       load_frames, write_domain_rounds_pdb,
                       write_weights, compute_rotations_per_frame,
                       write_rotation_data)

    base, ext = os.path.splitext(args.outweights)
    priors_fname = base + '_priors' + (ext if ext else '.dat')
    outputs = [args.outrounds, args.outweights, priors_fname,
               args.outdomains, args.log]
    if args.outrotation:
        rot_base, rot_ext = os.path.splitext(args.outrotation)
        rot_ext = rot_ext if rot_ext else '.dat'
    for f in outputs:
        backup_file(f)
    setup_logging(args.log, verbose=args.verbose)
    logging.info("=== BRMSD domain identification ===")

    u     = mda.Universe(args.topology, args.intrajectory)
    dt    = u.trajectory.dt
    start = int(args.begin / dt)
    end   = int(u.trajectory.n_frames) if args.end is None else int(args.end / dt)
    skip  = max(1, int(args.skiptime / dt))
    frames = list(range(start, end, skip))
    logging.info(f"Frames {start}–{end-1}, stride {skip}, total {len(frames)}")

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

    logging.info(f"sigma={args.sigma}  n_domains={args.n_domains}  "
                 f"mode={args.peeling_mode}  n_restarts={args.n_restarts}  "
                 f"mask_thr={args.mask_thr}")
    if args.peeling_mode == 'simultaneous':
        logging.info(f"(theta = M*sigma² computed internally)  mu={args.mu}")

    all_pos, times = load_frames(u, sel_idx, frames)

    # run domain identification
    if args.peeling_mode == 'sequential':
        results = identify_domains_sequential(
            all_pos, axes, args.sigma, args.maxiter, args.tol,
            n_domains=args.n_domains, n_restarts=args.n_restarts,
            mask_thr=args.mask_thr, n_jobs=args.n_jobs,
            refine=args.refine, mu_refine=args.mu_refine)
    else:
        results = identify_domains(
            all_pos, axes, args.sigma, args.mu, args.maxiter, args.tol,
            n_domains=args.n_domains, n_restarts=args.n_restarts,
            mask_thr=args.mask_thr, n_jobs=args.n_jobs)

    if results:
        logging.info(f"G_final (step 1) = {results[0]['G_final']:.6g}")

    # write multi-model PDB
    write_domain_rounds_pdb(u, sel_idx, results, args.outrounds)

    # write per-atom weights
    weight_data = {f"domain_{r['round']}": r['w']     for r in results}
    prior_data  = {f"domain_{r['round']}": r['prior'] for r in results}
    write_weights(weight_data, sel_idx, args.outweights,
                  header="Per-atom BRMSD weights w_alpha per domain")
    write_weights(prior_data,  sel_idx, priors_fname,
                  header="Per-atom prior W_alpha per domain (uniform = 1/N)")

    # write G convergence history
    g_fname = base + '_G' + (ext if ext else '.dat')
    backup_file(g_fname)
    with open(g_fname, 'w') as f:
        f.write("# BRMSD domain: free-energy convergence history (best restart)\n")
        f.write("# iter   G          G_fit      G_ent      G_ovl      "
                "ds         dw\n")
        for h in results[0]['history']:
            f.write(f"{h['iter']:6d}  {h['G']:.6e}  {h['G_fit']:.6e}  "
                    f"{h['G_ent']:.6e}  {h['G_ovl']:.6e}  "
                    f"{h['ds']:.6e}  {h['dw']:.6e}\n")
    logging.info(f"Wrote G convergence history → {g_fname}")

    # write domain assignment file
    with open(args.outdomains, 'w') as f:
        f.write("# Rigid domain assignments\n")
        f.write("# atom_index  " +
                "  ".join(f"domain_{r['round']}" for r in results) + "\n")
        for j, atom in enumerate(sel_idx):
            cols = [str(atom)] + [str(int(r['domain_mask'][j])) for r in results]
            f.write("  ".join(cols) + "\n")
    logging.info(f"Wrote domain assignments → {args.outdomains}")

    # write per-domain rotation data
    if args.outrotation:
        for r in results:
            k = r['round']
            outfile = f"{rot_base}_domain{k}{rot_ext}"
            backup_file(outfile)
            R_arr = compute_rotations_per_frame(
                all_pos, r['avg'], r['w'], axes)
            write_rotation_data(R_arr, times, outfile,
                                fmt=args.rotation_fmt,
                                euler_seq=args.euler_seq)

    logging.info("Done.")


# argparse sub-parser definition

def add_subparser(subparsers):
    from .core import CompactHelpFormatter
    p = subparsers.add_parser(
        'domain',
        help='Rigid domain identification: sequential peeling (default) or simultaneous',
        description=__doc__,
        formatter_class=CompactHelpFormatter)

    io = p.add_argument_group('I/O')
    io.add_argument('-f', '--intrajectory', required=True, metavar=None,
                    help='Input trajectory')
    io.add_argument('-s', '--topology',     required=True, metavar=None,
                    help='Topology file')
    io.add_argument('-p',  '--outrounds',   default='domain_rounds.pdb',  metavar=None,
                    help='Multi-model PDB: one MODEL per domain + rigidity map '
                         '[domain_rounds.pdb]')
    io.add_argument('-wt', '--outweights',  default='weights_domain.dat', metavar=None,
                    help='Per-domain per-atom weights [weights_domain.dat]')
    io.add_argument('-d',  '--outdomains',  default='domains.dat',        metavar=None,
                    help='Domain assignment per atom [domains.dat]')
    io.add_argument('-g',  '--log',         default='BRMSD_domain.log',  metavar=None,
                    help='Log file [BRMSD_domain.log]')
    io.add_argument('-rot', '--outrotation', default=None, metavar=None,
                    help='Base filename for per-domain rotation output; domain index '
                         'is appended, e.g. rot.dat → rot_domain1.dat … [None]')
    io.add_argument('-rfmt', '--rotation-fmt',
                    choices=['matrix', 'quaternion', 'euler', 'rotvec'],
                    default='rotvec', dest='rotation_fmt', metavar=None,
                    help='Rotation representation: matrix | quaternion | euler | rotvec '
                         '[rotvec]')
    io.add_argument('-eseq', '--euler-seq', default='ZYX', dest='euler_seq',
                    metavar=None,
                    help='Euler angle convention (used only with --rotation-fmt euler) '
                         '[ZYX]')

    sel = p.add_argument_group('Selection & axes')
    sel.add_argument('-sel', '--selection', default='protein and not name H*',
                     metavar=None, help="Atom selection [protein and not name H*]")
    sel.add_argument('-ax',  '--axes',      default='xyz', metavar=None,
                     help="Alignment axes [xyz]")

    opt = p.add_argument_group('Algorithm')
    opt.add_argument('-sg',  '--sigma',      type=float, default=0.05, metavar=None,
                     help='Position-fluctuation parameter σ (Å); θ = M·σ² internally '
                          '(same for sequential peeling and simultaneous). Small σ concentrates weight. [0.05]')
    opt.add_argument('-mu',  '--mu',         type=float, default=0.0,  metavar=None,
                     help='Overlap penalty μ (Å⁻²); penalises (w_j·w_k)² between '
                          'domains. Used only in simultaneous mode. '
                          'μ=0 gives K independent BRMSD runs. [0.0]')
    opt.add_argument('-nd',  '--n-domains',  type=int,   default=3,    metavar=None,
                     dest='n_domains',
                     help='Number of domains (peeling steps or simultaneous K) [3]')
    opt.add_argument('-nr',  '--n-restarts', type=int,   default=5,    metavar=None,
                     dest='n_restarts',
                     help='Independent restarts per step/run; best G is kept. [5]')
    opt.add_argument('--refine', dest='refine', action='store_true', default=False,
                     help='(sequential) Opt in to a simultaneous joint refinement '
                          'seeded by the peeling solutions after peeling. Off by '
                          'default — refinement helps only when domains are of '
                          'comparable rigidity; with a dominant rigid core it can '
                          'degrade the weaker domains. [default: off]')
    opt.add_argument('-mr', '--mu-refine', type=float, default=1.0, metavar=None,
                     dest='mu_refine',
                     help='Overlap penalty for the refinement step as a ratio μ/θ '
                          '(μ = mu_refine·θ, θ=M·σ²). [1.0]')
    opt.add_argument('-pm',  '--peeling-mode',
                     choices=['sequential', 'simultaneous'], default='sequential',
                     dest='peeling_mode', metavar=None,
                     help="'sequential': peel K domains iteratively with shrinking atom "
                          "pools (theta = M*sigma² held fixed per step, mu=0 enforced); "
                          "'simultaneous': joint K-domain with overlap penalty mu. "
                          "[sequential]")
    opt.add_argument('-mt',  '--mask-thr',   type=float, default=0.5,  metavar=None,
                     dest='mask_thr',
                     help='Fraction of max weight for domain mask: atom claimed if '
                          'w > mask_thr*max(w). [0.5]')
    opt.add_argument('-max', '--maxiter',    type=int,   default=100,  metavar=None,
                     help='Max iterations per restart [100]')
    opt.add_argument('-tol', '--tol',        type=float, default=1e-3, metavar=None,
                     help='Convergence tolerance [1e-3]')
    opt.add_argument('-j',   '--n-jobs',     type=int,   default=1,    metavar=None,
                     dest='n_jobs',
                     help='Parallel workers across restarts (-1 = all cores) [1]')
    opt.add_argument('-v',   '--verbose',    action='store_true', default=False,
                     help='Print per-iteration detail to screen [off]')

    traj = p.add_argument_group('Trajectory range')
    traj.add_argument('-b',  '--begin',    type=float, default=0.0,  metavar=None,
                      help='Start time (ps) [0.0]')
    traj.add_argument('-e',  '--end',      type=float, default=None, metavar=None,
                      help='End time (ps) [last frame]')
    traj.add_argument('-dt', '--skiptime', type=float, default=1.0,  metavar=None,
                      help='Time stride (ps) [1.0]')

    p.set_defaults(func=run)
