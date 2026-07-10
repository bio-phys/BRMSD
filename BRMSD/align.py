"""
BRMSD align  —  average structure and entropy-optimal per-atom weights

SYNOPSIS

    BRMSD align -s TOPOLOGY -f TRAJECTORY [options]

DESCRIPTION

    Computes the Bayesian RMSD (BRMSD) average structure and per-atom weights
    for a trajectory by minimising the maximum-entropy free energy:

        G = Σ_i min_R Σ_α w_α |s_α - R r_{i,α}|² + θ Σ_α w_α ln(w_α / W_α)

    with θ = M·σ² and uniform prior W_α = 1/N.  Alternating minimisation
    yields the Kabsch-superposed average structure s and weights w that are
    self-consistent: w suppresses flexible atoms, and s is the weighted mean
    of all aligned frames.  Convergence is guaranteed because each step
    decreases G monotonically from s⁽¹⁾ = x₁, w⁽¹⁾ = 1/N.

    σ (Å) is the single user-facing parameter.  Small σ concentrates weight
    on the rigid structural core; large σ recovers classical equal-weight
    RMSD.  Use "BRMSD sweep -m align --mode sigma" to find σ_op.

OUTPUT

    average.pdb        Average structure; B-factor = w·N (rigid > 1, uniform = 1).
    weights.dat        Per-atom BRMSD weights w_α.
    rmsf.dat           Per-atom RMSF across all frames.
    rmsd.dat           Per-frame BRMSD at each iteration.
    BRMSD_align.log    Run log with convergence detail.
    [aligned.xtc]      Aligned trajectory (optional, -o).
    [rotations.dat]    Per-frame rotation matrices/quaternions (optional, -rot).

EXAMPLES

    Standard alignment:

        BRMSD align -s topol.tpr -f prod.xtc -sg 0.43 \\
                    -sel "protein and not name H*"

    Write aligned trajectory and rotation data:

        BRMSD align -s topol.tpr -f prod.xtc -sg 0.43 \\
                    -o aligned.xtc -rot rotations.dat

THEORY

    Free energy:

        G = Σ_i min_R Σ_α w_α |s_α - R r_{i,α}|² + θ Σ_α w_α ln(w_α / W_α)

    Fixed-point updates:

        w_α^(n+1) ∝ W_α exp( −Σ_i |s_α^(n) − R_i^(n) r_{i,α}|² / θ )

        s_α^(n+1) = M⁻¹ Σ_i R_i^(n+1) r_{i,α}

    with θ = M·σ².  Small σ concentrates weight on the rigid core;
    large σ gives uniform weights (classical RMSD).
"""

import logging
import numpy as np
from .core import batch_align_frames


def BRMSD_iterate(all_pos, axes_idx, sigma, maxiter, tol, prior=None):
    """
    Fixed-point iteration for BRMSD alignment.

    all_pos  : (M, N, 3) per-frame positions
    axes_idx : alignment axis indices, e.g. [0,1,2]
    sigma    : position-fluctuation parameter (Angstrom); theta = M*sigma**2
    prior    : (N,) reference weights W_alpha, default uniform 1/N

    Returns the converged average (N,3), weights (N,), and a per-iteration
    history (keys: iter, ds, dw, rmsd_frames, rmsf).
    """
    M, N, _ = all_pos.shape
    d = len(axes_idx)
    theta = M * sigma ** 2          # M factor makes converged weights M-independent

    if prior is None:
        prior = np.ones(N) / N
    avg = all_pos[0].copy()         # s^(1) = x_1
    w   = prior.copy()              # w^(1) = W_alpha

    history = []
    for it in range(1, maxiter + 1):
        aligned, dev_sum = batch_align_frames(all_pos, avg, w, axes_idx)

        # weight update (max-subtracted for numerical stability)
        exponent = -dev_sum / theta
        exponent -= exponent.max()
        new_w = prior * np.exp(exponent)
        w_sum = new_w.sum()
        new_w = new_w / w_sum if w_sum > 1e-300 else prior.copy()

        # average structure update
        new_avg = avg.copy()
        new_avg[:, axes_idx] = aligned.mean(axis=0)

        sq_dev = np.sum((aligned - avg[:, axes_idx][None]) ** 2, axis=2)  # (M, N)
        rmsd_frames = np.sqrt(np.sum(w[None, :] * sq_dev, axis=1))         # (M,)
        rmsf = np.sqrt(np.mean(sq_dev, axis=0))                            # (N,)

        ds = np.linalg.norm(new_avg - avg)
        dw = np.linalg.norm(new_w   - w)
        logging.debug(f"  iter {it:4d}:  Δstruct={ds:.2e}  Δweights={dw:.2e}  "
                      f"mean-RMSD={rmsd_frames.mean():.4f} Å")

        history.append({'iter': it, 'ds': ds, 'dw': dw,
                        'rmsd_frames': rmsd_frames.copy(),
                        'rmsf': rmsf.copy()})

        avg, w = new_avg, new_w

        if ds < tol and dw < tol:
            logging.info(f"  Converged at iteration {it}  "
                         f"(Δstruct={ds:.2e}, Δweights={dw:.2e}).")
            break

    return avg, w, history



def run(args):
    """
    Entry point called by cli.py for the 'align' subcommand.
    """
    import sys
    import MDAnalysis as mda
    from .core import (setup_logging, backup_file, parse_axes,
                       load_frames, write_average_structure,
                       write_aligned_trajectory, write_rmsd,
                       write_rmsf, write_weights,
                       compute_rotations_per_frame, write_rotation_data)

    for f in [args.outtrajectory, args.outrmsd, args.outrmsf,
              args.outweights, args.outpdb, args.log, args.outrotation]:
        backup_file(f)
    setup_logging(args.log, verbose=args.verbose)
    logging.info("=== BRMSD align ===")

    u   = mda.Universe(args.topology, args.intrajectory)
    dt  = u.trajectory.dt
    start = int(args.begin / dt)
    end   = int(u.trajectory.n_frames) if args.end is None else int(args.end / dt)
    skip  = max(1, int(args.skiptime / dt))
    frames = list(range(start, end, skip))
    logging.info(f"Frames {start}–{end-1}, stride {skip}, total {len(frames)}")

    axes = parse_axes(args.axes)
    if not axes:
        logging.error("No valid axes; use xyz, xy, xz, z, etc.")
        sys.exit(1)

    sel     = u.select_atoms(args.selection)
    sel_idx = sel.indices
    if len(sel_idx) == 0:
        logging.error("Atom selection returned no atoms.")
        sys.exit(1)
    logging.info(f"Selection '{args.selection}' → {len(sel_idx)} atoms")
    logging.info(f"Axes: {args.axes} → indices {axes}")
    logging.info(f"sigma={args.sigma}  (theta = M*sigma² computed internally)  "
                 f"maxiter={args.maxiter}  tol={args.tol}")

    all_pos, times = load_frames(u, sel_idx, frames)

    avg, w, history = BRMSD_iterate(
        all_pos, axes, args.sigma, args.maxiter, args.tol)

    rmsd_data = []
    for h in history:
        it = h['iter']
        for k, fr in enumerate(frames):
            rmsd_data.append((f"Iteration {it}", k, h['rmsd_frames'][k]))

    rmsf_data    = {f"Iteration {h['iter']}": h['rmsf']    for h in history}
    weight_data  = {"final": w}

    write_average_structure(u, sel_idx, avg, args.outpdb, bfactors=w * len(sel_idx))
    write_rmsd(rmsd_data, args.outrmsd, times,
               header="BRMSD per frame per iteration")
    write_rmsf(rmsf_data, sel_idx, args.outrmsf,
               header="RMSF per atom per iteration")
    write_weights(weight_data, sel_idx, args.outweights,
                  header="Per-atom BRMSD weights w_alpha")
    if args.outtrajectory:
        write_aligned_trajectory(u, sel_idx, avg, w, axes,
                                 args.outtrajectory, frames)
    if args.outrotation:
        R_arr = compute_rotations_per_frame(all_pos, avg, w, axes)
        write_rotation_data(R_arr, times, args.outrotation,
                            fmt=args.rotation_fmt,
                            euler_seq=args.euler_seq)
    logging.info("Done.")



def add_subparser(subparsers):
    from .core import CompactHelpFormatter
    p = subparsers.add_parser(
        'align',
        help='Compute BRMSD average structure and per-atom weights',
        description=__doc__,
        formatter_class=CompactHelpFormatter)

    io = p.add_argument_group('I/O')
    io.add_argument('-f', '--intrajectory', required=True, metavar=None,
                    help='Input trajectory (XTC/TRR/DCD)')
    io.add_argument('-s', '--topology',     required=True, metavar=None,
                    help='Topology (TPR/GRO/PDB/PSF)')
    io.add_argument('-o', '--outtrajectory', default=None, metavar=None,
                    help='Aligned trajectory output [None]')
    io.add_argument('-p', '--outpdb',   default='average.pdb',  metavar=None,
                    help='Average structure PDB [average.pdb]')
    io.add_argument('-rmsd','--outrmsd',  default='rmsd.dat',   metavar=None,
                    help='Per-frame RMSD output [rmsd.dat]')
    io.add_argument('-rmsf','--outrmsf',  default='rmsf.dat',   metavar=None,
                    help='Per-atom RMSF output [rmsf.dat]')
    io.add_argument('-wt', '--outweights', default='weights.dat', metavar=None,
                    help='Per-atom weights output [weights.dat]')
    io.add_argument('-g', '--log', default='BRMSD_align.log',  metavar=None,
                    help='Log file [BRMSD_align.log]')
    io.add_argument('-rot', '--outrotation', default=None, metavar=None,
                    help='Output file for per-frame rotation data [None]')
    io.add_argument('-rfmt', '--rotation-fmt',
                    choices=['matrix', 'quaternion', 'euler', 'rotvec'],
                    default='rotvec', dest='rotation_fmt', metavar=None,
                    help='Rotation representation: matrix (9 cols) | quaternion (4 cols) '
                         '| euler (3 cols, see --euler-seq) | rotvec (3 cols, axis×φ) '
                         '[rotvec]')
    io.add_argument('-eseq', '--euler-seq', default='ZYX', dest='euler_seq',
                    metavar=None,
                    help='Euler angle convention, e.g. ZYX, ZXZ, XYZ (used only with '
                         '--rotation-fmt euler) [ZYX]')

    sel = p.add_argument_group('Selection & axes')
    sel.add_argument('-sel','--selection', default='protein and not name H*', metavar=None,
                     help="Atom selection [protein and not name H*]")
    sel.add_argument('-ax', '--axes',      default='xyz', metavar=None,
                     help="Alignment axes: xyz, xy, z … [xyz]")

    opt = p.add_argument_group('Algorithm')
    opt.add_argument('-sg', '--sigma',   type=float, default=0.5, metavar=None,
                     help='Position-fluctuation parameter σ (Å); θ = M·σ² internally. Small σ concentrates weight on the rigid core; large σ → uniform weights (standard RMSD). [0.5]')
    opt.add_argument('-max','--maxiter', type=int,   default=100, metavar=None,
                     help='Max iterations [100]')
    opt.add_argument('-tol','--tol',     type=float, default=1e-3, metavar=None,
                     help='Convergence tolerance [1e-3]')
    opt.add_argument('-v', '--verbose',  action='store_true', default=False,
                     help='Print per-iteration convergence detail to screen [off]')

    traj = p.add_argument_group('Trajectory range')
    traj.add_argument('-b', '--begin',    type=float, default=0.0,  metavar=None,
                      help='Start time (ps) [0.0]')
    traj.add_argument('-e', '--end',      type=float, default=None, metavar=None,
                      help='End time (ps) [last frame]')
    traj.add_argument('-dt','--skiptime', type=float, default=1.0,  metavar=None,
                      help='Time stride (ps) [1.0]')

    p.set_defaults(func=run)
