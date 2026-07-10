"""
BRMSD sweep  —  parameter scan for any BRMSD module

SYNOPSIS

    BRMSD sweep -m MODULE --mode MODE [options]

DESCRIPTION

    Runs a single-parameter sweep over the specified BRMSD module and writes
    a CSV summary of diagnostic quantities as a function of the swept value.
    Use the output to identify optimal parameter values before a full run.

    Modules and their sweep modes
    ─────────────────────────────
    align   sigma      Sweep σ (Å).  Reports effective atom count (n_eff =
                       1/Σw²), weight entropy, and rigid-core size (n_rigid).
                       Pick σ_op as the smallest σ with n_eff ≥ 0.2·N.
                       Use --log-space.

    cluster sigma      Sweep σ at fixed K and τ.  Reports Calinski-Harabasz
                       (CH) and Davies-Bouldin (DB) scores, free energy G,
                       and within/between-cluster RMSD.  Peak CH → σ_op.

    cluster tau        Sweep τ at fixed σ and K.  Reports CH, DB, G, and
                       within-cluster RMSD.  Minimise within-RMSD for τ_op.

    cluster K          Sweep number of clusters K at fixed σ and τ.  Reports
                       CH, DB, G, and within-cluster RMSD.  Elbow in G or
                       peak CH identifies optimal K.

    focus   mu         Sweep μ_ratio at fixed σ.  Reports domain vs complement
                       weight sums.  Pick μ_ratio where domain weight saturates.

    domain  sigma      Sweep σ at fixed K.  Reports domain sizes, total rigid
                       atoms, and convergence iterations.  Pick σ where domain
                       sizes stabilise.  Use --log-space.

    domain  K          Sweep number of domains K at fixed σ.  Reports domain
                       sizes and total rigid atoms per K.

    domain  mask_thr   Sweep the domain-assignment threshold at fixed σ and K.
                       Reports per-domain atom counts without re-optimising.

    smooth  sigma      Sweep σ (Å).  Reports mean per-frame BRMSD and mean
                       weight entropy.  Pick σ where BRMSD flattens.
                       Use --log-space.

    smooth  window     Sweep window half-width W (frames) at fixed σ.  Reports
                       mean per-frame BRMSD.  Pick W at the plateau.

    Sweep values are given with --range MIN MAX N (N evenly spaced points;
    add --log-space for logarithmic spacing) or --list val1 val2 ... for an
    explicit set.  For integer parameters (K, window), steps are always
    integer regardless of spacing.

OUTPUT

    A CSV file (default: {module}_{mode}_sweep.csv) with one row per sweep
    point.  The first column is the swept parameter; remaining columns are the
    module-specific diagnostic quantities.  Domain sizes are encoded as
    pipe-separated integers (e.g. "120|98|54") in the domain_sizes column.

    When --plot is given, a PNG figure with one panel per numeric column is
    saved alongside the CSV (same stem, .png extension).

EXAMPLES

    Find σ_op for alignment:

        BRMSD sweep -m align --mode sigma --range 0.1 10.0 20 --log-space \\
                    -s topol.tpr -f prod.xtc -sel "protein and not name H*"

    Find K for clustering (σ and τ already known):

        BRMSD sweep -m cluster --mode K --range 2 8 7 \\
                    --sigma 4.0 --tau 10.0 \\
                    -s topol.tpr -f prod.xtc -sel "protein and not name H*"

    Find μ_ratio for focused alignment:

        BRMSD sweep -m focus --mode mu --range 0.0 0.9 10 \\
                    --sigma 0.43 -dom "resid 1:50" \\
                    -s topol.tpr -f prod.xtc -sel "protein and not name H*"

    Sweep window half-width for smoothing:

        BRMSD sweep -m smooth --mode window --range 2 20 19 \\
                    --sigma 0.5 \\
                    -s topol.tpr -f prod.xtc -sel "protein and not name H*"

    Sweep domain-assignment threshold (no re-optimisation):

        BRMSD sweep -m domain --mode mask_thr --range 0.1 0.9 17 \\
                    --sigma 3.0 --n-domains 3 \\
                    -s topol.tpr -f prod.xtc -sel "protein and not name H*"
"""

import csv
import logging
import os
import sys
import numpy as np


_INTEGER_MODES = {('cluster', 'K'), ('domain', 'K'), ('smooth', 'window')}

_VALID_MODES = {
    'align':   ['sigma'],
    'cluster': ['sigma', 'tau', 'K'],
    'focus':   ['mu'],
    'domain':  ['sigma', 'K', 'mask_thr'],
    'smooth':  ['sigma', 'window'],
}


# ─── sweep value parser ───────────────────────────────────────────────────────

def _parse_sweep_values(args):
    integer_mode = (args.module, args.mode) in _INTEGER_MODES
    if args.list:
        if integer_mode:
            return [int(float(v)) for v in args.list]
        return [float(v) for v in args.list]
    if not args.range:
        logging.error("Specify sweep values with --range MIN MAX N or --list val1 val2 ...")
        sys.exit(1)
    if len(args.range) != 3:
        logging.error("--range requires exactly 3 values: MIN MAX N")
        sys.exit(1)
    mn = float(args.range[0])
    mx = float(args.range[1])
    n  = int(float(args.range[2]))
    if integer_mode:
        vals = [int(round(v)) for v in np.linspace(mn, mx, n)]
        unique = []
        for v in vals:
            if v not in unique:
                unique.append(v)
        return unique
    if args.log_space:
        if mn <= 0:
            logging.error("--log-space requires MIN > 0")
            sys.exit(1)
        return list(np.logspace(np.log10(mn), np.log10(mx), n))
    return list(np.linspace(mn, mx, n))


# ─── output helpers ───────────────────────────────────────────────────────────

def _fmt(v):
    if np.isnan(v):
        return 'nan'
    return round(float(v), 4)


def _require(args, attr_list, mode_label):
    missing = [a for a in attr_list if getattr(args, a, None) is None]
    if missing:
        flags = ', '.join('--' + a.replace('_', '-') for a in missing)
        logging.error(f"{mode_label} requires: {flags}")
        sys.exit(1)


def _write_csv(rows, outfile):
    with open(outfile, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Wrote sweep CSV → {outfile}")


def _plot_sweep(rows, x_key, outfile):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available — skipping plot.")
        return

    x_vals = [r[x_key] for r in rows]
    y_keys = [k for k in rows[0] if k != x_key]

    numeric_keys = []
    for yk in y_keys:
        try:
            [float(r[yk]) for r in rows]
            numeric_keys.append(yk)
        except (ValueError, TypeError):
            pass

    if not numeric_keys:
        logging.warning("No numeric columns to plot.")
        return

    n = len(numeric_keys)
    fig, axes = plt.subplots(n, 1, figsize=(7, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, yk in zip(axes, numeric_keys):
        y_vals = [float(r[yk]) for r in rows]
        ax.plot(x_vals, y_vals, 'o-', linewidth=1.5, markersize=5)
        ax.set_ylabel(yk, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    axes[-1].set_xlabel(x_key, fontsize=10)
    fig.tight_layout(pad=1.5)
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Wrote sweep plot → {outfile}")


# ─── common trajectory loader ─────────────────────────────────────────────────

def _load_common(args):
    import MDAnalysis as mda
    from .core import parse_axes, load_frames, setup_logging

    setup_logging(args.log, verbose=args.verbose)
    logging.info("=== BRMSD sweep ===")
    logging.info(f"module={args.module}  mode={args.mode}")

    u  = mda.Universe(args.topology, args.intrajectory)
    dt = u.trajectory.dt
    start  = int(args.begin / dt)
    end    = int(u.trajectory.n_frames) if args.end is None else int(args.end / dt)
    skip   = max(1, int(args.skiptime / dt))
    frames = list(range(start, end, skip))

    logging.info(f"Frames {start}–{end - 1}, stride {skip}, total {len(frames)}")

    axes = parse_axes(args.axes)
    if not axes:
        logging.error("No valid axes specified.")
        sys.exit(1)

    sel = u.select_atoms(args.selection)
    if len(sel) == 0:
        logging.error("Atom selection returned no atoms.")
        sys.exit(1)
    logging.info(f"Selection '{args.selection}' → {len(sel)} atoms")

    all_pos, _ = load_frames(u, sel.indices, frames)
    return u, sel, all_pos, axes


# ─── per-module sweep functions ───────────────────────────────────────────────

def _sweep_align_sigma(all_pos, axes, args):
    from .align import BRMSD_iterate
    values = _parse_sweep_values(args)
    rows   = []
    for sigma in values:
        logging.info(f"  sigma={sigma:.4g}")
        _, w, history = BRMSD_iterate(all_pos, axes, sigma, args.maxiter, args.tol)
        n_iter    = len(history)
        mean_rmsd = float(history[-1]['rmsd_frames'].mean())
        eps       = 1e-300
        entropy   = float(-np.sum(w * np.log(w + eps)))
        n_eff     = 1.0 / float(np.dot(w, w))
        n_rigid   = int((w > w.mean()).sum())
        rows.append({
            'sigma':          round(sigma, 6),
            'n_iter':         n_iter,
            'mean_rmsd':      round(mean_rmsd, 4),
            'n_eff':          round(n_eff, 2),
            'weight_entropy': round(entropy, 6),
            'n_rigid':        n_rigid,
            'w_max':          round(float(w.max()), 6),
            'w_min':          round(float(w.min()), 6),
        })
    return rows


def _sweep_cluster_sigma(all_pos, axes, args):
    from .cluster import soft_kmeans, pairwise_center_rmsd, cluster_quality_metrics
    _require(args, ['tau', 'n_clusters'], 'cluster/sigma')
    values = _parse_sweep_values(args)
    K   = args.n_clusters
    tau = args.tau
    rows = []
    for sigma in values:
        logging.info(f"  sigma={sigma:.4g}  K={K}  tau={tau}")
        centers, w_ca, q_ai, _, G, dev_ai = soft_kmeans(
            all_pos, K, axes, sigma, tau, args.maxiter, args.tol,
            n_restarts=args.n_restarts, n_jobs=args.n_jobs)
        rmsd_ak  = np.sqrt(np.maximum(dev_ai, 0))
        ctr_rmsd = pairwise_center_rmsd(list(centers), axes)
        ch, db, s_a = cluster_quality_metrics(rmsd_ak, q_ai, ctr_rmsd)
        min_ctr  = float(ctr_rmsd[ctr_rmsd > 0].min()) if K > 1 else float('nan')
        rows.append({
            'sigma':            round(sigma, 6),
            'free_energy':      round(float(G), 6),
            'CH_score':         _fmt(ch),
            'DB_score':         _fmt(db),
            'mean_within_rmsd': round(float(s_a.mean()), 4),
            'max_within_rmsd':  round(float(s_a.max()),  4),
            'min_center_rmsd':  _fmt(min_ctr),
        })
    return rows


def _sweep_cluster_tau(all_pos, axes, args):
    from .cluster import soft_kmeans, pairwise_center_rmsd, cluster_quality_metrics
    _require(args, ['sigma', 'n_clusters'], 'cluster/tau')
    values = _parse_sweep_values(args)
    K     = args.n_clusters
    sigma = args.sigma
    rows  = []
    for tau in values:
        logging.info(f"  tau={tau:.4g}  K={K}  sigma={sigma}")
        centers, w_ca, q_ai, _, G, dev_ai = soft_kmeans(
            all_pos, K, axes, sigma, tau, args.maxiter, args.tol,
            n_restarts=args.n_restarts, n_jobs=args.n_jobs)
        rmsd_ak  = np.sqrt(np.maximum(dev_ai, 0))
        ctr_rmsd = pairwise_center_rmsd(list(centers), axes)
        ch, db, s_a = cluster_quality_metrics(rmsd_ak, q_ai, ctr_rmsd)
        min_ctr  = float(ctr_rmsd[ctr_rmsd > 0].min()) if K > 1 else float('nan')
        rows.append({
            'tau':              round(tau, 6),
            'free_energy':      round(float(G), 6),
            'CH_score':         _fmt(ch),
            'DB_score':         _fmt(db),
            'mean_within_rmsd': round(float(s_a.mean()), 4),
            'max_within_rmsd':  round(float(s_a.max()),  4),
            'min_center_rmsd':  _fmt(min_ctr),
        })
    return rows


def _sweep_cluster_K(all_pos, axes, args):
    from .cluster import soft_kmeans, pairwise_center_rmsd, cluster_quality_metrics
    _require(args, ['sigma', 'tau'], 'cluster/K')
    values = _parse_sweep_values(args)
    sigma  = args.sigma
    tau    = args.tau
    rows   = []
    for K in values:
        logging.info(f"  K={K}  sigma={sigma}  tau={tau}")
        centers, w_ca, q_ai, _, G, dev_ai = soft_kmeans(
            all_pos, K, axes, sigma, tau, args.maxiter, args.tol,
            n_restarts=args.n_restarts, n_jobs=args.n_jobs)
        rmsd_ak  = np.sqrt(np.maximum(dev_ai, 0))
        ctr_rmsd = pairwise_center_rmsd(list(centers), axes)
        ch, db, s_a = cluster_quality_metrics(rmsd_ak, q_ai, ctr_rmsd)
        min_ctr  = float(ctr_rmsd[ctr_rmsd > 0].min()) if K > 1 else float('nan')
        rows.append({
            'K':                K,
            'free_energy':      round(float(G), 6),
            'CH_score':         _fmt(ch),
            'DB_score':         _fmt(db),
            'mean_within_rmsd': round(float(s_a.mean()), 4),
            'max_within_rmsd':  round(float(s_a.max()),  4),
            'min_center_rmsd':  _fmt(min_ctr),
        })
    return rows


def _sweep_focus_mu(all_pos, axes, domain_mask, args):
    from .focus import BRMSD_focus_iterate
    values = _parse_sweep_values(args)
    sigma  = args.sigma
    rows   = []
    for mu_ratio in values:
        logging.info(f"  mu_ratio={mu_ratio:.4g}  sigma={sigma}")
        _, w, history = BRMSD_focus_iterate(
            all_pos, axes, sigma, mu_ratio, domain_mask, args.maxiter, args.tol)
        n_iter       = len(history)
        domain_w_sum = float(w[domain_mask].sum())
        compl_w_sum  = float(w[~domain_mask].sum())
        mean_rmsd    = float(history[-1]['rmsd_frames'].mean())
        rows.append({
            'mu_ratio':          round(mu_ratio, 6),
            'domain_weight_sum': round(domain_w_sum, 6),
            'compl_weight_sum':  round(compl_w_sum,  6),
            'n_domain_atoms':    int(domain_mask.sum()),
            'mean_rmsd':         round(mean_rmsd, 4),
            'n_iter':            n_iter,
        })
    return rows


def _sweep_domain_sigma(all_pos, axes, args):
    from .domain import identify_domains, identify_domains_sequential
    _require(args, ['n_domains'], 'domain/sigma')
    values = _parse_sweep_values(args)
    K      = args.n_domains
    rows   = []
    for sigma in values:
        logging.info(f"  sigma={sigma:.4g}  K={K}")
        if args.peeling_mode == 'sequential':
            results = identify_domains_sequential(
                all_pos, axes, sigma, args.maxiter, args.tol,
                n_domains=K, n_restarts=args.n_restarts,
                mask_thr=args.mask_thr, n_jobs=args.n_jobs)
        else:
            results = identify_domains(
                all_pos, axes, sigma, args.mu, args.maxiter, args.tol,
                n_domains=K, n_restarts=args.n_restarts,
                mask_thr=args.mask_thr, n_jobs=args.n_jobs)
        sizes     = [int(r['domain_mask'].sum()) for r in results]
        n_rigid   = sum(sizes)
        conv_iter = max(r['history'][-1]['iter'] for r in results)
        rows.append({
            'sigma':         round(sigma, 6),
            'domain_sizes':  '|'.join(str(s) for s in sizes),
            'n_rigid_total': n_rigid,
            'conv_iters':    conv_iter,
        })
    return rows


def _sweep_domain_K(all_pos, axes, args):
    from .domain import identify_domains, identify_domains_sequential
    _require(args, ['sigma'], 'domain/K')
    values = _parse_sweep_values(args)
    sigma  = args.sigma
    rows   = []
    for K in values:
        logging.info(f"  K={K}  sigma={sigma}")
        if args.peeling_mode == 'sequential':
            results = identify_domains_sequential(
                all_pos, axes, sigma, args.maxiter, args.tol,
                n_domains=K, n_restarts=args.n_restarts,
                mask_thr=args.mask_thr, n_jobs=args.n_jobs)
        else:
            results = identify_domains(
                all_pos, axes, sigma, args.mu, args.maxiter, args.tol,
                n_domains=K, n_restarts=args.n_restarts,
                mask_thr=args.mask_thr, n_jobs=args.n_jobs)
        sizes     = [int(r['domain_mask'].sum()) for r in results]
        n_rigid   = sum(sizes)
        conv_iter = max(r['history'][-1]['iter'] for r in results)
        rows.append({
            'K':             K,
            'domain_sizes':  '|'.join(str(s) for s in sizes),
            'n_rigid_total': n_rigid,
            'conv_iters':    conv_iter,
        })
    return rows


def _sweep_domain_mask_thr(all_pos, axes, args):
    from .domain import identify_domains, identify_domains_sequential
    _require(args, ['sigma', 'n_domains'], 'domain/mask_thr')
    values = _parse_sweep_values(args)
    sigma  = args.sigma
    K      = args.n_domains
    logging.info(f"  Running domain optimisation once at sigma={sigma}, K={K} ...")
    if args.peeling_mode == 'sequential':
        results_base = identify_domains_sequential(
            all_pos, axes, sigma, args.maxiter, args.tol,
            n_domains=K, n_restarts=args.n_restarts, mask_thr=0.5,
            n_jobs=args.n_jobs)
    else:
        results_base = identify_domains(
            all_pos, axes, sigma, args.mu, args.maxiter, args.tol,
            n_domains=K, n_restarts=args.n_restarts, mask_thr=0.5,
            n_jobs=args.n_jobs)
    weights = [r['w'] for r in results_base]
    rows    = []
    for thr in values:
        sizes   = [int((w > thr * w.max()).sum()) for w in weights]
        n_rigid = sum(sizes)
        rows.append({
            'mask_thr':      round(thr, 6),
            'domain_sizes':  '|'.join(str(s) for s in sizes),
            'n_rigid_total': n_rigid,
        })
    return rows


def _sweep_smooth_sigma(all_pos, axes, args):
    from .smooth import BRMSD_smooth
    values = _parse_sweep_values(args)
    W      = args.half_width
    rows   = []
    for sigma in values:
        logging.info(f"  sigma={sigma:.4g}  W={W}")
        _, weights, rmsds = BRMSD_smooth(
            all_pos, axes, sigma, args.maxiter, args.tol,
            half_width=W, window_type=args.window_type, n_jobs=args.n_jobs)
        mean_rmsd = float(rmsds.mean())
        eps       = 1e-300
        w_mean    = weights.mean(axis=0)
        entropy   = float(-np.sum(w_mean * np.log(w_mean + eps)))
        rows.append({
            'sigma':          round(sigma, 6),
            'mean_rmsd':      round(mean_rmsd, 4),
            'weight_entropy': round(entropy, 6),
        })
    return rows


def _sweep_smooth_window(all_pos, axes, args):
    from .smooth import BRMSD_smooth
    _require(args, ['sigma'], 'smooth/window')
    values = _parse_sweep_values(args)
    sigma  = args.sigma
    rows   = []
    for W in values:
        logging.info(f"  W={W}  sigma={sigma}")
        _, weights, rmsds = BRMSD_smooth(
            all_pos, axes, sigma, args.maxiter, args.tol,
            half_width=W, window_type=args.window_type, n_jobs=args.n_jobs)
        mean_rmsd = float(rmsds.mean())
        eps       = 1e-300
        w_mean    = weights.mean(axis=0)
        entropy   = float(-np.sum(w_mean * np.log(w_mean + eps)))
        rows.append({
            'window_half_width': W,
            'mean_rmsd':         round(mean_rmsd, 4),
            'weight_entropy':    round(entropy, 6),
        })
    return rows


# ─── dispatch table ───────────────────────────────────────────────────────────

_DISPATCH = {
    ('align',   'sigma'):    _sweep_align_sigma,
    ('cluster', 'sigma'):    _sweep_cluster_sigma,
    ('cluster', 'tau'):      _sweep_cluster_tau,
    ('cluster', 'K'):        _sweep_cluster_K,
    ('domain',  'sigma'):    _sweep_domain_sigma,
    ('domain',  'K'):        _sweep_domain_K,
    ('domain',  'mask_thr'): _sweep_domain_mask_thr,
    ('smooth',  'sigma'):    _sweep_smooth_sigma,
    ('smooth',  'window'):   _sweep_smooth_window,
}


# ─── CLI entry point ──────────────────────────────────────────────────────────

def run(args):
    valid = _VALID_MODES.get(args.module, [])
    if args.mode not in valid:
        logging.error(
            f"Module '{args.module}' does not support mode '{args.mode}'. "
            f"Valid modes: {', '.join(valid)}")
        sys.exit(1)

    if args.out is None:
        args.out = f"{args.module}_{args.mode}_sweep.csv"

    u, sel, all_pos, axes = _load_common(args)

    key = (args.module, args.mode)

    if key == ('focus', 'mu'):
        _require(args, ['sigma', 'domain_sel'], 'focus/mu')
        dom     = u.select_atoms(args.domain_sel)
        dom_set = set(dom.indices)
        domain_mask = np.array([i in dom_set for i in sel.indices])
        rows = _sweep_focus_mu(all_pos, axes, domain_mask, args)
    else:
        rows = _DISPATCH[key](all_pos, axes, args)

    _write_csv(rows, args.out)

    if args.plot:
        plot_file = os.path.splitext(args.out)[0] + '.png'
        x_key = list(rows[0].keys())[0]
        _plot_sweep(rows, x_key, plot_file)

    logging.info("Done.")


# ─── argparse sub-parser ──────────────────────────────────────────────────────

def add_subparser(subparsers):
    from .core import CompactHelpFormatter
    p = subparsers.add_parser(
        'sweep',
        help='Parameter sweep for any BRMSD module',
        description=__doc__,
        formatter_class=CompactHelpFormatter)

    req = p.add_argument_group('Required')
    req.add_argument('-m',   '--module',
                     required=True, metavar=None,
                     choices=['align', 'cluster', 'focus', 'domain', 'smooth'],
                     help='BRMSD module to sweep: align | cluster | focus | domain | smooth')
    req.add_argument('--mode',
                     required=True, metavar=None,
                     choices=['sigma', 'tau', 'K', 'mu', 'mask_thr', 'window'],
                     help='Parameter to sweep (module-dependent; see DESCRIPTION above)')
    req.add_argument('-f',   '--intrajectory',
                     required=True, metavar=None,
                     help='Input trajectory (XTC/TRR/DCD)')
    req.add_argument('-s',   '--topology',
                     required=True, metavar=None,
                     help='Topology file (TPR/GRO/PDB/PSF)')

    io = p.add_argument_group('I/O')
    io.add_argument('-o',    '--out',
                    default=None, metavar=None,
                    help='Output CSV file [{module}_{mode}_sweep.csv]')
    io.add_argument('-g',    '--log',
                    default='BRMSD_sweep.log', metavar=None,
                    help='Log file [BRMSD_sweep.log]')

    sweep = p.add_argument_group('Sweep specification')
    sweep.add_argument('--range',
                       nargs=3, metavar=('MIN', 'MAX', 'N'),
                       help='Sweep N values from MIN to MAX (linear or log-spaced)')
    sweep.add_argument('--list',
                       nargs='+', metavar='VAL',
                       help='Explicit list of sweep values (alternative to --range)')
    sweep.add_argument('--log-space',
                       action='store_true', default=False,
                       dest='log_space',
                       help='Use log-spaced values with --range (recommended for σ sweeps)')

    sel = p.add_argument_group('Selection & axes')
    sel.add_argument('-sel', '--selection',
                     default='protein and not name H*', metavar=None,
                     help='MDAnalysis atom selection [protein and not name H*]')
    sel.add_argument('-dom', '--domain-sel',
                     default=None, metavar=None,
                     dest='domain_sel',
                     help='Focus domain selection (required for focus/mu)')
    sel.add_argument('-ax',  '--axes',
                     default='xyz', metavar=None,
                     help='Alignment axes: xyz, xy, z … [xyz]')

    traj = p.add_argument_group('Trajectory range')
    traj.add_argument('-b',  '--begin',
                      type=float, default=0.0, metavar=None,
                      help='Start time (ps) [0.0]')
    traj.add_argument('-e',  '--end',
                      type=float, default=None, metavar=None,
                      help='End time (ps) [last frame]')
    traj.add_argument('-dt', '--skiptime',
                      type=float, default=1.0, metavar=None,
                      help='Time stride (ps) [1.0]')

    fix = p.add_argument_group('Fixed parameters (required when not being swept)')
    fix.add_argument('--sigma',
                     type=float, default=None, metavar=None,
                     help='Fixed σ (Å); required when sweeping tau, K, mu, window, or mask_thr')
    fix.add_argument('--tau',
                     type=float, default=None, metavar=None,
                     help='Fixed τ; required for cluster/sigma and cluster/K')
    fix.add_argument('-K',   '--n-clusters',
                     type=int, default=None, metavar=None,
                     dest='n_clusters',
                     help='Fixed K (number of clusters); required for cluster/sigma and cluster/tau')
    fix.add_argument('-nd',  '--n-domains',
                     type=int, default=None, metavar=None,
                     dest='n_domains',
                     help='Fixed K (number of domains); required for domain/sigma and domain/mask_thr')
    fix.add_argument('-mu',  '--mu',
                     type=float, default=0.0, metavar=None,
                     help='Overlap penalty μ for domain simultaneous mode [0.0]')
    fix.add_argument('-mt',  '--mask-thr',
                     type=float, default=0.5, metavar=None,
                     dest='mask_thr',
                     help='Domain-assignment weight threshold for sigma/K sweeps [0.5]')
    fix.add_argument('-W',   '--half-width',
                     type=int, default=5, metavar=None,
                     dest='half_width',
                     help='Smooth window half-width W (frames); required for smooth/sigma [5]')
    fix.add_argument('-win', '--window-type',
                     default='triangular', metavar=None,
                     dest='window_type',
                     choices=['triangular', 'uniform'],
                     help='Smooth window shape: triangular | uniform [triangular]')
    fix.add_argument('-pm',  '--peeling-mode',
                     default='sequential', metavar=None,
                     dest='peeling_mode',
                     choices=['sequential', 'simultaneous'],
                     help='Domain sweep strategy: sequential | simultaneous [sequential]')

    alg = p.add_argument_group('Algorithm')
    alg.add_argument('-max', '--maxiter',
                     type=int, default=100, metavar=None,
                     help='Max iterations per sweep point [100]')
    alg.add_argument('-tol', '--tol',
                     type=float, default=1e-3, metavar=None,
                     help='Convergence tolerance [1e-3]')
    alg.add_argument('-nr',  '--n-restarts',
                     type=int, default=5, metavar=None,
                     dest='n_restarts',
                     help='Independent restarts per sweep point (cluster/domain) [5]')
    alg.add_argument('-j',   '--n-jobs',
                     type=int, default=-1, metavar=None,
                     dest='n_jobs',
                     help='Parallel workers (-1 = all cores) [-1]')
    alg.add_argument('--plot',
                     action='store_true', default=False,
                     help='Save a PNG diagnostic figure alongside the CSV [off]')
    alg.add_argument('-v',   '--verbose',
                     action='store_true', default=False,
                     help='Print per-iteration detail to screen [off]')

    p.set_defaults(func=run)
