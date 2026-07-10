"""
BRMSD smooth  —  trajectory smoothing with per-frame entropy-optimal weights

SYNOPSIS

    BRMSD smooth -s TOPOLOGY -f TRAJECTORY [options]

DESCRIPTION

    Suppresses high-frequency thermal noise in an MD trajectory while
    preserving slow conformational dynamics, by computing for each frame j
    a locally averaged structure s_j.  The global average in the BRMSD
    alignment free energy is replaced by a windowed average centred on j:

        G_j = Σ_i p(i|j) MSD(s_j, x_i; w)  +  θ Σ_α w_α ln(w_α / W_α)

    Two window shapes are supported (--window-type):

        triangular  p(i|j) ∝ max(0, 1 − |i−j| / (W+1))   [default]
        uniform     p(i|j) = 1/(2W+1)  for |i−j| ≤ W

    Because the window weights are normalised (Σ_i p(i|j) = 1), the
    effective frame count is M = 1 and θ = σ² (not M·σ²).  Each frame j
    is solved independently, which makes the full trajectory smoothing
    embarrassingly parallel over frames.

    W (--half-width) controls the temporal smoothing range; σ (--sigma)
    controls the spatial weight concentration.  Use "BRMSD sweep -m smooth
    --mode sigma" and "--mode window" to find operating values.

OUTPUT

    smoothed.xtc           Smoothed trajectory.
    rmsd_smooth.dat        Per-frame BRMSD between smoothed and raw frame.
    rmsf_smooth.dat        Per-atom RMSF of the smoothed trajectory.
    weights_smooth.dat     Mean per-atom weights averaged over all frames.
    BRMSD_smooth.log       Run log.

EXAMPLES

    Smooth with triangular window:

        BRMSD smooth -s topol.tpr -f prod.xtc -sg 0.5 -W 10 \\
                     -sel "protein and not name H*"

    Use uniform window, wider half-width:

        BRMSD smooth -s topol.tpr -f prod.xtc -sg 0.5 -W 20 \\
                     -win uniform -sel "protein and not name H*"

THEORY

    Fixed-point updates for frame j (s_j⁽¹⁾ = x_j, w⁽¹⁾ = 1/N):

        w_α^(n+1) ∝ W_α exp( −Σ_i p(i|j) |s_{j,α}^(n) − R_i^(n) r_{i,α}|² / θ )

        s_{j,α}^(n+1) = Σ_i p(i|j) R_i^(n) r_{i,α}

    with θ = σ² (window weights are already normalised, effective M = 1).
    The structure update is a p-weighted mean over aligned frames, unlike
    the plain 1/M average in BRMSD align.  Each frame j is solved
    independently; the full trajectory is parallelised over frames with
    joblib (prefer='threads' to share the large all_pos array without copying).
"""

import logging
import numpy as np
from joblib import Parallel, delayed
from .core import batch_align_frames


# Window weight functions

def triangular_window(n_frames, j, half_width):
    """
    Triangular (hat) window centred on frame j with half-width W.
    p(i|j) ∝ max(0, 1 - |i-j|/(W+1))
    """
    p = np.maximum(0.0, 1.0 - np.abs(np.arange(n_frames) - j) / (half_width + 1))
    s = p.sum()
    return p / s if s > 0 else np.ones(n_frames) / n_frames


def uniform_window(n_frames, j, half_width):
    """
    Uniform (box) window: all frames within half_width of j get equal weight.
    """
    p = np.zeros(n_frames)
    lo = max(0, j - half_width)
    hi = min(n_frames - 1, j + half_width)
    p[lo:hi + 1] = 1.0
    s = p.sum()
    return p / s if s > 0 else np.ones(n_frames) / n_frames


WINDOW_FUNCTIONS = {
    'triangular': triangular_window,
    'uniform':    uniform_window,
}


# Core fixed-point iteration for a single target frame j

def smooth_single_frame(all_pos, j, p_ij, axes_idx, theta, maxiter, tol, prior):
    """
    Run the smoothing fixed-point iteration for target frame j.

    Parameters
    ----------
    all_pos   : (M, N, 3)
    j         : int  index of the target frame
    p_ij      : (M,) normalised window weights for this target frame
    axes_idx  : list of axis indices
    theta     : float  entropy penalty (Å²)
    maxiter   : int
    tol       : float
    prior     : (N,) W_alpha

    Returns
    -------
    s_j  : (N, 3)  smoothed average structure for frame j
    w_j  : (N,)    per-atom weights for frame j
    rmsd : float   final BRMSD for this smoothed frame
    """
    # Restrict to the window: only frames with non-zero p(i|j).
    # For half-width W, this is at most 2W+1 frames instead of all M frames,
    # cutting work from O(M) to O(W) per iteration — critical for large M.
    nz      = np.where(p_ij > 0)[0]           # indices of windowed frames
    sub_pos = all_pos[nz]                      # (W, N, 3)
    sub_p   = p_ij[nz]                         # (W,)  already normalised

    # Local index of target frame j within the window (j always has p>0)
    j_local = int(np.searchsorted(nz, j))

    # Initialise at frame j itself
    s_j = all_pos[j].copy()
    w_j = prior.copy()

    for it in range(1, maxiter + 1):

        # vectorised: align windowed frames to s_j in one batched SVD
        aligned, _ = batch_align_frames(sub_pos, s_j, w_j, axes_idx)
        # aligned: (W, N, d)

        # p(i|j)-weighted per-atom squared deviation
        diff     = s_j[:, axes_idx][np.newaxis] - aligned   # (W, N, d)
        atom_sq  = np.einsum('mnd,mnd->mn', diff, diff)     # (W, N)
        dev_sum  = sub_p @ atom_sq                           # (N,)  Σᵢ p·|s-aln|²

        # weight update
        exponent = -dev_sum / theta
        exponent -= exponent.max()
        new_w = prior * np.exp(exponent)
        w_sum = new_w.sum()
        new_w = new_w / w_sum if w_sum > 1e-300 else prior.copy()

        # smoothed structure update: s_{j,alpha} = Σᵢ p(i|j) R_i r_i
        new_s_j = s_j.copy()
        new_s_j[:, axes_idx] = np.einsum('m,mnd->nd', sub_p, aligned)

        ds = np.linalg.norm(new_s_j - s_j)
        dw = np.linalg.norm(new_w    - w_j)
        s_j, w_j = new_s_j, new_w

        if ds < tol and dw < tol:
            break

    # Recompute alignment with the final s_j and w_j so that the RMSD is
    # consistent with the converged iterate (not the second-to-last one).
    # This also handles the maxiter=0 edge case where `aligned` is never set.
    aligned, _ = batch_align_frames(sub_pos, s_j, w_j, axes_idx)

    # final weighted BRMSD for this frame
    sq_dev = np.einsum('mnd,mnd->mn',
                       s_j[:, axes_idx][np.newaxis] - aligned,
                       s_j[:, axes_idx][np.newaxis] - aligned)  # (W, N)
    rmsd = float(np.sqrt(np.dot(w_j, sq_dev[j_local])))

    return s_j, w_j, rmsd


# Full smoothing over all frames

def BRMSD_smooth(all_pos, axes_idx, sigma, maxiter, tol,
                  half_width=5, window_type='triangular', prior=None,
                  n_jobs=-1):
    """
    Smooth every frame in the trajectory.

    Each frame j is an independent fixed-point problem → parallelised with
    joblib across all available CPU cores (n_jobs=-1).
    The inner Kabsch loop is further vectorised via batch_align_frames.

    Parameters
    ----------
    all_pos     : (M, N, 3)
    axes_idx    : list of axis indices
    sigma       : float  position-fluctuation parameter (Å).  The per-frame
                  window weights p(i|j) are normalised (Σ_i p = 1), so the
                  deviation sum is already a per-structure mean (effective
                  M = 1) and the entropy penalty is theta = sigma**2 (Å²).
    maxiter     : int
    tol         : float
    half_width  : int    window half-width W  (frames on each side)
    window_type : 'triangular' or 'uniform'
    prior       : (N,) W_alpha, defaults to 1/N
    n_jobs      : int  number of parallel workers (-1 = all cores)

    Returns
    -------
    smoothed    : (M, N, 3)  smoothed structures for every frame
    weights     : (M, N)     per-frame per-atom weights
    rmsds       : (M,)       per-frame BRMSD after smoothing
    """
    M, N, _ = all_pos.shape
    if prior is None:
        prior = np.ones(N) / N

    # Window weights are normalised (Σ_i p(i|j) = 1) → the deviation sum is a
    # mean, so effective M = 1 and theta = sigma**2 (M = 1, no N factor).
    theta = sigma ** 2

    win_fn = WINDOW_FUNCTIONS.get(window_type, triangular_window)

    # Pre-compute all window weight vectors (cheap, do outside parallel loop)
    p_ij_all = [win_fn(M, j, half_width) for j in range(M)]

    logging.info(f"  Smoothing {M} frames using {n_jobs} workers …")

    # Each frame j is independent → embarrassingly parallel
    # prefer='threads': numpy releases the GIL during BLAS/LAPACK calls,
    # so threads share memory without copying the large all_pos array.
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(smooth_single_frame)(
            all_pos, j, p_ij_all[j], axes_idx, theta, maxiter, tol, prior)
        for j in range(M)
    )

    smoothed = np.array([r[0] for r in results])   # (M, N, 3)
    weights  = np.array([r[1] for r in results])   # (M, N)
    rmsds    = np.array([r[2] for r in results])   # (M,)

    logging.info(f"  Smoothing done. Mean BRMSD = {rmsds.mean():.4f} Å")
    return smoothed, weights, rmsds


# CLI entry point

def run(args):
    import sys
    import MDAnalysis as mda
    from .core import (setup_logging, backup_file, parse_axes,
                       load_frames, write_smoothed_trajectory,
                       write_rmsd, write_rmsf, write_weights)

    for f in [args.outtrajectory, args.outrmsd, args.outrmsf,
              args.outweights, args.log]:
        backup_file(f)
    setup_logging(args.log, verbose=args.verbose)
    logging.info("=== BRMSD smooth ===")

    u   = mda.Universe(args.topology, args.intrajectory)
    dt  = u.trajectory.dt
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
    logging.info(f"Window: {args.window_type}, half-width W={args.half_width}")
    logging.info(f"sigma={args.sigma}  (theta = sigma² computed internally)  "
                 f"maxiter={args.maxiter}  tol={args.tol}")

    all_pos, times = load_frames(u, sel_idx, frames)

    smoothed, weights, rmsds = BRMSD_smooth(
        all_pos, axes, args.sigma, args.maxiter, args.tol,
        half_width=args.half_width, window_type=args.window_type,
        n_jobs=args.n_jobs)

    # build output dicts
    rmsd_data   = [("smoothed", k, rmsds[k]) for k in range(len(frames))]

    # RMSF of smoothed trajectory: align smoothed frames to their mean first
    # so global translation/rotation is removed before computing fluctuations.
    N_sel = len(sel_idx)
    mean_smooth = smoothed.mean(axis=0)                     # (N, 3)
    w_uniform   = np.ones(N_sel) / N_sel
    aligned_s, _ = batch_align_frames(smoothed, mean_smooth, w_uniform, axes)
    rmsf_data   = {"smoothed": np.sqrt(np.mean(
                       np.sum((aligned_s - mean_smooth[:, axes][np.newaxis]) ** 2,
                              axis=2), axis=0))}

    weight_data = {"mean": weights.mean(axis=0)}

    # write smoothed trajectory
    # Build full-atom smoothed positions: replace sel_idx atoms, keep others
    full_smoothed = []
    for k, fr in enumerate(frames):
        u.trajectory[fr]
        pos = u.atoms.positions.copy()
        pos[sel_idx] = smoothed[k]
        full_smoothed.append(pos)

    if args.outtrajectory:
        write_smoothed_trajectory(u, full_smoothed, args.outtrajectory, frames)
    write_rmsd(rmsd_data, args.outrmsd, times,
               header="BRMSD smoothed trajectory: per-frame BRMSD")
    write_rmsf(rmsf_data, sel_idx, args.outrmsf,
               header="RMSF of smoothed trajectory")
    write_weights(weight_data, sel_idx, args.outweights,
                  header="Mean per-atom weights from smoothing")
    logging.info("Done.")


# argparse sub-parser definition

def add_subparser(subparsers):
    from .core import CompactHelpFormatter
    p = subparsers.add_parser(
        'smooth',
        help='Smooth a trajectory using BRMSD window weighting',
        description=__doc__,
        formatter_class=CompactHelpFormatter)

    io = p.add_argument_group('I/O')
    io.add_argument('-f', '--intrajectory', required=True, metavar=None,
                    help='Input trajectory')
    io.add_argument('-s', '--topology',     required=True, metavar=None,
                    help='Topology file')
    io.add_argument('-o', '--outtrajectory', default='smoothed.xtc', metavar=None,
                    help='Smoothed trajectory output [smoothed.xtc]')
    io.add_argument('-rmsd','--outrmsd',  default='rmsd_smooth.dat',   metavar=None,
                    help='Per-frame BRMSD [rmsd_smooth.dat]')
    io.add_argument('-rmsf','--outrmsf',  default='rmsf_smooth.dat',   metavar=None,
                    help='Per-atom RMSF [rmsf_smooth.dat]')
    io.add_argument('-wt', '--outweights', default='weights_smooth.dat', metavar=None,
                    help='Per-atom weights [weights_smooth.dat]')
    io.add_argument('-g',  '--log', default='BRMSD_smooth.log', metavar=None,
                    help='Log file [BRMSD_smooth.log]')

    sel = p.add_argument_group('Selection & axes')
    sel.add_argument('-sel','--selection', default='protein and not name H*', metavar=None,
                     help="Atom selection [protein and not name H*]")
    sel.add_argument('-ax', '--axes',      default='xyz', metavar=None,
                     help="Alignment axes [xyz]")

    opt = p.add_argument_group('Algorithm')
    opt.add_argument('-sg',  '--sigma',       type=float, default=0.5,          metavar=None,
                     help='Position-fluctuation parameter σ (Å); θ = σ² internally. Small σ concentrates weight on the rigid core; large σ → uniform weights (standard RMSD). [0.5]')
    opt.add_argument('-max', '--maxiter',     type=int,   default=100,          metavar=None,
                     help='Max iterations per frame [100]')
    opt.add_argument('-tol', '--tol',         type=float, default=1e-3,         metavar=None,
                     help='Convergence tolerance [1e-3]')
    opt.add_argument('-W',   '--half-width',  type=int,   default=5,            metavar=None,
                     dest='half_width',
                     help='Window half-width in frames [5]')
    opt.add_argument('-win', '--window-type', default='triangular',             metavar=None,
                     dest='window_type', choices=['triangular', 'uniform'],
                     help='Window shape: triangular | uniform [triangular]')
    opt.add_argument('-j',   '--n-jobs',      type=int,   default=-1,           metavar=None,
                     dest='n_jobs',
                     help='Parallel workers for frame loop (-1 = all cores) [-1]')
    opt.add_argument('-v',   '--verbose',     action='store_true', default=False,
                     help='Print per-frame convergence detail to screen [off]')

    traj = p.add_argument_group('Trajectory range')
    traj.add_argument('-b', '--begin',    type=float, default=0.0,  metavar=None,
                      help='Start time (ps) [0.0]')
    traj.add_argument('-e', '--end',      type=float, default=None, metavar=None,
                      help='End time (ps) [last frame]')
    traj.add_argument('-dt','--skiptime', type=float, default=1.0,  metavar=None,
                      help='Time stride (ps) [1.0]')

    p.set_defaults(func=run)
