"""
BRMSD focus  —  focused alignment biased toward a specific domain

SYNOPSIS

    BRMSD focus -s TOPOLOGY -f TRAJECTORY -dom DOMAIN_SEL [options]

DESCRIPTION

    Extends BRMSD alignment with a concentration penalty that biases atom
    weights toward a user-specified domain D while still superposing the full
    structure.  The free energy augments the standard alignment functional
    (see "BRMSD align") with a term that raises the effective weight of atoms
    inside D relative to the surrounding scaffold:

        G = Σ_i min_R Σ_α w_α |s_α - R r_{i,α}|²
            + θ  Σ_α       w_α ln(w_α / W_α)
            + μ  Σ_{α∈D}   w_α ln(n w_α)

    with θ = M·σ², μ = μ_ratio·θ, n = |D|, and W_α = 1/N.

    The dimensionless ratio μ/θ (--mu-ratio) controls the degree of focus.
    At μ_ratio = 0 the result equals standard BRMSD align.  As μ_ratio grows,
    atoms in D receive progressively more weight regardless of their MSF.
    Use "BRMSD sweep -m focus --mode mu" to find the operating μ_ratio.

OUTPUT

    average_focus.pdb    Average structure; B-factor = w·N.
    weights_focus.dat    Per-atom BRMSD weights.
    rmsf_focus.dat       Per-atom RMSF.
    rmsd_focus.dat       Per-frame BRMSD at each iteration.
    BRMSD_focus.log      Run log.
    [aligned.xtc]        Aligned trajectory (optional, -o).

EXAMPLES

    Focus alignment on a transmembrane domain:

        BRMSD focus -s topol.tpr -f prod.xtc -sg 0.43 -mr 0.5 \\
                    -dom "resid 10:80" -sel "protein and not name H*"

THEORY

    Fixed-point updates (s⁽¹⁾ = x₁, w⁽¹⁾ = 1/N):

        α ∉ D:  w_α ∝ (1/N) exp( −d_α / θ )

        α ∈ D:  w_α ∝ (1/N)^{θ/(θ+μ)} · n^{−μ/(θ+μ)} · exp( −d_α / (θ+μ) )

        s_α^(n+1) = M⁻¹ Σ_i R_i^(n) r_{i,α}

    where d_α = Σ_i |s_α^(n) − R_i^(n) r_{i,α}|².  Atoms outside D follow
    the standard BRMSD update; atoms inside D experience a reduced effective
    concentration parameter θ+μ and an amplified prior that raises their
    weight relative to atoms outside D.
"""

import logging
import numpy as np
from .core import batch_align_frames


# Core fixed-point iteration  (no I/O; works on pre-loaded numpy arrays)

def BRMSD_focus_iterate(all_pos, axes_idx, sigma, mu_ratio, domain_mask,
                         maxiter, tol):
    """
    Fixed-point iteration for focused BRMSD alignment.

    Parameters
    ----------
    all_pos     : (M, N, 3)  per-frame positions of selected atoms
    axes_idx    : list of axis indices, e.g. [0,1,2]
    sigma       : float  position-fluctuation parameter (Å); base entropy
                  penalty theta = M*sigma**2 (Å²).
    mu_ratio    : float  domain concentration penalty as a fraction of theta;
                  the absolute penalty is mu = mu_ratio * theta.
    domain_mask : (N,) bool  True for atoms in focus domain D
    maxiter     : int    maximum iterations
    tol         : float  convergence tolerance

    Returns
    -------
    avg      : (N, 3)  converged average structure (full 3-D)
    w        : (N,)    converged per-atom weights
    history  : list of dict with per-iteration diagnostics
                 keys: 'iter', 'ds', 'dw', 'rmsd_frames', 'rmsf'
    """
    M, N, _ = all_pos.shape
    n = int(domain_mask.sum())  # |D|
    theta     = M * sigma ** 2       # base entropy penalty
    mu        = mu_ratio * theta     # absolute overlap/concentration penalty
    theta_eff = theta + mu           # effective penalty for atoms in D

    # Precompute log-scale prefactors (before subtracting the stability offset).
    # For α ∉ D: log_pf = -log(N)
    # For α ∈ D: log_pf = -(θ/(θ+μ)) log N  −  (μ/(θ+μ)) log n
    log_pf = np.full(N, -np.log(N))
    if n > 0 and mu > 0:
        log_pf[domain_mask] = (-(theta / theta_eff) * np.log(N)
                               - (mu   / theta_eff) * np.log(n))

    prior = np.ones(N) / N
    avg   = all_pos[0].copy()   # s^(1) = x_1
    w     = prior.copy()        # w^(1) = W_alpha

    history = []

    for it in range(1, maxiter + 1):

        aligned, dev_sum = batch_align_frames(all_pos, avg, w, axes_idx)

        # weight update (two branches)
        log_w = log_pf.copy()
        out_mask = ~domain_mask
        log_w[out_mask] -= dev_sum[out_mask] / theta
        if n > 0 and mu > 0:
            log_w[domain_mask] -= dev_sum[domain_mask] / theta_eff
        else:
            log_w[domain_mask] -= dev_sum[domain_mask] / theta

        log_w -= log_w.max()    # numerical stability
        new_w  = np.exp(log_w)
        w_sum  = new_w.sum()
        new_w  = new_w / w_sum if w_sum > 1e-300 else prior.copy()

        # average structure update (same as standard BRMSD align)
        new_avg = avg.copy()
        new_avg[:, axes_idx] = aligned.mean(axis=0)

        # per-frame RMSD (weighted)
        sq_dev      = np.sum((aligned - avg[:, axes_idx][None]) ** 2, axis=2)
        rmsd_frames = np.sqrt(np.sum(w[None, :] * sq_dev, axis=1))
        rmsf        = np.sqrt(np.mean(sq_dev, axis=0))

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


# CLI entry point

def run(args):
    """
    Entry point called by cli.py for the 'focus' subcommand.
    """
    import sys
    import numpy as np
    import MDAnalysis as mda
    from .core import (setup_logging, backup_file, parse_axes,
                       load_frames, write_average_structure,
                       write_aligned_trajectory, write_rmsd,
                       write_rmsf, write_weights)

    for f in [args.outtrajectory, args.outrmsd, args.outrmsf,
              args.outweights, args.outpdb, args.log]:
        backup_file(f)
    setup_logging(args.log, verbose=args.verbose)
    logging.info("=== BRMSD focus ===")

    u  = mda.Universe(args.topology, args.intrajectory)
    dt = u.trajectory.dt
    start  = int(args.begin / dt)
    end    = int(u.trajectory.n_frames) if args.end is None else int(args.end / dt)
    skip   = max(1, int(args.skiptime / dt))
    frames = list(range(start, end, skip))
    logging.info(f"Frames {start}–{end-1}, stride {skip}, total {len(frames)}")

    axes = parse_axes(args.axes)
    if not axes:
        logging.error("No valid axes; use xyz, xy, xz, z, etc.")
        sys.exit(1)

    # atom selections
    sel = u.select_atoms(args.selection)
    if len(sel) == 0:
        logging.error("Atom selection returned no atoms.")
        sys.exit(1)
    logging.info(f"Selection '{args.selection}' → {len(sel)} atoms")

    dom = u.select_atoms(args.domain_sel)
    if len(dom) == 0:
        logging.error("Domain selection returned no atoms.")
        sys.exit(1)

    # Build domain mask: True for atoms in D that also appear in sel
    sel_global  = set(sel.indices)
    dom_in_sel  = np.array([idx in sel_global for idx in dom.indices])
    n_dom_valid = dom_in_sel.sum()
    if n_dom_valid == 0:
        logging.error("Domain selection has no overlap with the main selection.")
        sys.exit(1)
    logging.info(f"Domain '{args.domain_sel}' → {len(dom)} atoms total, "
                 f"{n_dom_valid} in selection")

    dom_global_set = set(dom.indices[dom_in_sel])
    domain_mask = np.array([idx in dom_global_set for idx in sel.indices])

    axes_idx = axes
    logging.info(f"sigma={args.sigma}  (theta = M*sigma² computed internally)  "
                 f"mu_ratio={args.mu_ratio}  |D|={domain_mask.sum()}")
    logging.info(f"Axes: {args.axes} → indices {axes_idx}  "
                 f"maxiter={args.maxiter}  tol={args.tol}")

    all_pos, times = load_frames(u, sel.indices, frames)

    # run focused iteration
    avg, w, history = BRMSD_focus_iterate(
        all_pos, axes_idx, args.sigma, args.mu_ratio, domain_mask,
        args.maxiter, args.tol)

    # collect output data
    rmsd_data = []
    for h in history:
        it = h['iter']
        for k, fr in enumerate(frames):
            rmsd_data.append((f"Iteration {it}", k, h['rmsd_frames'][k]))

    rmsf_data   = {f"Iteration {h['iter']}": h['rmsf']   for h in history}
    weight_data = {"final": w}

    # write outputs
    write_average_structure(u, sel.indices, avg, args.outpdb, bfactors=w * len(sel.indices))
    write_rmsd(rmsd_data, args.outrmsd, times,
               header="BRMSD focus per-frame RMSD per iteration")
    write_rmsf(rmsf_data, sel.indices, args.outrmsf,
               header="RMSF per atom per iteration")
    write_weights(weight_data, sel.indices, args.outweights,
                  header="Per-atom focused BRMSD weights w_alpha")
    if args.outtrajectory:
        write_aligned_trajectory(u, sel.indices, avg, w, axes_idx,
                                 args.outtrajectory, frames)
    logging.info("Done.")


# argparse sub-parser definition  (called from cli.py)

def add_subparser(subparsers):
    from .core import CompactHelpFormatter
    p = subparsers.add_parser(
        'focus',
        help='Focused BRMSD alignment biased toward a specific domain',
        description=__doc__,
        formatter_class=CompactHelpFormatter)

    io = p.add_argument_group('I/O')
    io.add_argument('-f', '--intrajectory', required=True, metavar=None,
                    help='Input trajectory (XTC/TRR/DCD)')
    io.add_argument('-s', '--topology',     required=True, metavar=None,
                    help='Topology (TPR/GRO/PDB/PSF)')
    io.add_argument('-o', '--outtrajectory', default=None, metavar=None,
                    help='Aligned trajectory output [None]')
    io.add_argument('-p', '--outpdb',    default='average_focus.pdb',  metavar=None,
                    help='Average structure PDB [average_focus.pdb]')
    io.add_argument('-rmsd','--outrmsd', default='rmsd_focus.dat',     metavar=None,
                    help='Per-frame RMSD output [rmsd_focus.dat]')
    io.add_argument('-rmsf','--outrmsf', default='rmsf_focus.dat',     metavar=None,
                    help='Per-atom RMSF output [rmsf_focus.dat]')
    io.add_argument('-wt', '--outweights', default='weights_focus.dat', metavar=None,
                    help='Per-atom weights output [weights_focus.dat]')
    io.add_argument('-g', '--log', default='BRMSD_focus.log',         metavar=None,
                    help='Log file [BRMSD_focus.log]')

    sel = p.add_argument_group('Selection & axes')
    sel.add_argument('-sel','--selection', default='protein and not name H*', metavar=None,
                     help="Full atom selection [protein and not name H*]")
    sel.add_argument('-dom','--domain-sel', required=True, metavar=None,
                     help='MDAnalysis selection for focus domain D (e.g. "resid 1:30")')
    sel.add_argument('-ax', '--axes',       default='xyz', metavar=None,
                     help="Alignment axes: xyz, xy, z … [xyz]")

    opt = p.add_argument_group('Algorithm')
    opt.add_argument('-sg', '--sigma',    type=float, default=0.5,  metavar=None,
                     help='Position-fluctuation parameter σ (Å); θ = M·σ² internally. Small σ concentrates weight; large σ → uniform. [0.5]')
    opt.add_argument('-mr', '--mu-ratio', type=float, default=1.0,  metavar=None,
                     help='Domain penalty μ/θ (dimensionless). '
                          'μ = mu_ratio × θ concentrates weight inside domain D. '
                          '0 → standard BRMSD align; larger values → stronger focus. [1.0]')
    opt.add_argument('-max','--maxiter',  type=int,   default=100,  metavar=None,
                     help='Max iterations [100]')
    opt.add_argument('-tol','--tol',      type=float, default=1e-3, metavar=None,
                     help='Convergence tolerance [1e-3]')
    opt.add_argument('-v', '--verbose',   action='store_true', default=False,
                     help='Print per-iteration convergence detail to screen [off]')

    traj = p.add_argument_group('Trajectory range')
    traj.add_argument('-b', '--begin',    type=float, default=0.0,  metavar=None,
                      help='Start time (ps) [0.0]')
    traj.add_argument('-e', '--end',      type=float, default=None, metavar=None,
                      help='End time (ps) [last frame]')
    traj.add_argument('-dt','--skiptime', type=float, default=1.0,  metavar=None,
                      help='Time stride (ps) [1.0]')

    p.set_defaults(func=run)
