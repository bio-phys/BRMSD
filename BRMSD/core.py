"""
BRMSD core.py
Shared utilities: trajectory I/O, weighted Kabsch rotation, logging, file backup.
"""

import os
import time
import logging
import argparse
import numpy as np


# Custom help formatter — suppresses metavar from option list

class CompactHelpFormatter(argparse.RawTextHelpFormatter):
    """RawTextHelpFormatter that hides metavar from the option list.

    Result: ``-th, --theta   description`` instead of
    ``-th THETA, --theta THETA   description``.
    The usage line still uses the default metavar so Python's internal
    assertion in _format_usage is satisfied on all Python versions.
    """
    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        return ', '.join(action.option_strings)


# Logging / file utilities

def setup_logging(logfile, verbose=False):
    """
    Configure root logger.

    The log file always receives DEBUG-level messages (full detail).
    The screen (stderr) receives:
      - INFO and above when verbose=False  (summaries, warnings, errors)
      - DEBUG and above when verbose=True  (per-iteration convergence, matrices)
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)


def backup_file(filepath):
    """Rename an existing file with a timestamp suffix before overwriting."""
    if not filepath or not os.path.isfile(filepath):
        return
    base, ext = os.path.splitext(filepath)
    backup_name = f"{base}_{time.strftime('%d-%m-%Y_%H-%M-%S')}{ext}"
    os.rename(filepath, backup_name)
    # logging may not be set up yet at call time — use print as fallback
    try:
        logging.info(f"Backed up {filepath} → {backup_name}")
    except Exception:
        print(f"Backed up {filepath} → {backup_name}")


def parse_axes(axes_str):
    """'xyz' → [0,1,2],  'xy' → [0,1],  'z' → [2], etc."""
    return [i for i, a in enumerate('xyz') if a in axes_str.lower()]


# Trajectory loading

def load_frames(u, sel_idx, frames):
    """
    Load positions of selected atoms for the requested frame indices.

    Parameters
    ----------
    u        : MDAnalysis Universe
    sel_idx  : array of atom indices (from u.select_atoms(...).indices)
    frames   : list of integer frame indices into u.trajectory

    Returns
    -------
    all_pos : ndarray, shape (n_frames, n_atoms, 3)
    times   : list of simulation times (ps) for each frame
    """
    all_pos = []
    times = []
    for f in frames:
        u.trajectory[f]
        all_pos.append(u.atoms.positions[sel_idx].copy())
        times.append(getattr(u.trajectory.ts, 'time', float(f)))
    return np.array(all_pos, dtype=float), times


# Weighted Kabsch rotation  (row-vector convention)

def compute_weighted_rotation(frame_pos, ref_pos, w, axes_idx):
    """
    Find the rigid-body rotation R, and centroids c1 (frame) and c2 (ref),
    that minimise  sum_alpha w_alpha || (x_alpha - c1) R + c2 - y_alpha ||^2

    Row-vector convention: aligned = (frame_pos[:,axes] - c1) @ R + c2

    Parameters
    ----------
    frame_pos : (N, 3) array  — mobile structure
    ref_pos   : (N, 3) array  — reference (average) structure
    w         : (N,)   array  — per-atom weights (need not sum to 1)
    axes_idx  : list of axis indices, e.g. [0,1,2] for xyz

    Returns
    -------
    R  : rotation matrix, shape (len(axes_idx), len(axes_idx))
    c1 : weighted centroid of frame_pos in selected axes
    c2 : weighted centroid of ref_pos in selected axes
    """
    X = frame_pos[:, axes_idx]   # (N, d)
    Y = ref_pos[:, axes_idx]     # (N, d)

    w_sum = np.sum(w)
    w_norm = w / w_sum if w_sum > 1e-300 else np.ones(len(w)) / len(w)

    c1 = w_norm @ X              # weighted centroid of mobile
    c2 = w_norm @ Y              # weighted centroid of reference

    Xc = X - c1
    Yc = Y - c2

    # Weighted cross-covariance H = Xc^T W Yc  (d x d)
    H = (Xc * w_norm[:, None]).T @ Yc

    U, _, Vht = np.linalg.svd(H)

    # Reflection correction: ensure det(R) = +1
    D = np.eye(U.shape[0])
    D[-1, -1] = np.linalg.det(U @ Vht)

    R = U @ D @ Vht
    return R, c1, c2


def align_frame(pos, ref_pos, w, axes_idx):
    """
    Apply weighted Kabsch alignment of `pos` onto `ref_pos`.

    Returns
    -------
    aln      : (N, len(axes_idx))  aligned coordinates (selected axes only)
    R, c1, c2 for downstream use
    """
    R, c1, c2 = compute_weighted_rotation(pos, ref_pos, w, axes_idx)
    aln = (pos[:, axes_idx] - c1) @ R + c2
    return aln, R, c1, c2


# Output writers

def write_average_structure(u, sel_idx, avg_coords, outfile, bfactors=None):
    """
    Write a single-frame PDB with selected atoms placed at avg_coords (N,3).
    Non-selected atoms keep their positions from the last loaded frame.

    Parameters
    ----------
    bfactors : (N,) array, optional
        Per-atom values written into the B-factor (tempfactor) column.
        If None the existing tempfactors are preserved unchanged.
        Useful for encoding per-atom weights or RMSF for visualisation:
        e.g. pass ``w * N`` so that uniform weight → 1.0, rigid > 1, flexible < 1.
    """
    import MDAnalysis as mda
    orig_pos = u.atoms.positions.copy()

    # tempfactors may not exist for TPR/XTC universes — initialise if absent
    try:
        orig_bfac = u.atoms.tempfactors.copy()
        has_bfac  = True
    except Exception:
        u.add_TopologyAttr('tempfactors', np.zeros(u.atoms.n_atoms))
        orig_bfac = None
        has_bfac  = False

    pos = orig_pos.copy()
    pos[sel_idx] = avg_coords
    u.atoms.positions = pos
    if bfactors is not None:
        # Indexed assignment on AtomGroup.tempfactors returns a copy — must
        # write directly into the topology array to actually persist the values.
        u._topology.tempfactors.values[:] = 0.0
        u._topology.tempfactors.values[sel_idx] = bfactors

    with mda.Writer(outfile, multiframe=False) as W:
        W.write(u.atoms)

    u.atoms.positions = orig_pos
    if has_bfac:
        u._topology.tempfactors.values[:] = orig_bfac
    logging.info(f"Wrote average structure → {outfile}")


def write_domain_rounds_pdb(u, sel_idx, results, outfile):
    """
    Write all identified domains as a single multi-model PDB.

    Each MODEL corresponds to one domain.  The B-factor column encodes
    ``w_alpha * N`` for that domain (uniform weight → 1.0, rigid > 1,
    flexible ≈ 0).  A final MODEL stores the max weight across all domains,
    labelled 'RIGIDITY MAP'.

    Load in PyMOL::

        load domain_rounds.pdb
        spectrum b, blue_white_red

    Load in VMD::

        mol new domain_rounds.pdb
        mol modcolor 0 0 Beta

    Parameters
    ----------
    u       : MDAnalysis Universe
    sel_idx : array of atom indices
    results : list of dicts from identify_domains (keys: round, avg, w, max_w)
    outfile : str
    """
    import MDAnalysis as mda

    N        = len(sel_idx)
    orig_pos = u.atoms.positions.copy()

    # Ensure tempfactors attribute exists
    try:
        orig_bfac = u.atoms.tempfactors.copy()
        has_bfac  = True
    except Exception:
        u.add_TopologyAttr('tempfactors', np.zeros(u.atoms.n_atoms))
        orig_bfac = None
        has_bfac  = False

    labels = []
    frames_data = []   # (avg_coords, bfactors, label) per MODEL

    for res in results:
        r    = res['round']
        bfac = res['w'] / res['w'].max() * 100
        n_rigid = int((res['w'] > res['w'].mean()).sum())
        frames_data.append((res['avg'], bfac,
                            f"Domain {r}: {n_rigid}/{N} atoms above mean weight  "
                            f"(w_max={res['w'].max():.4f})"))

    # final MODEL: cumulative max weight → overall rigidity map
    max_w = results[-1]['max_w']
    frames_data.append((results[-1]['avg'], max_w / max_w.max() * 100,
                        f"Rigidity map  "
                        f"(B-factor = max w normalised 0-100 over all {len(results)} domains)"))

    with mda.Writer(outfile, n_atoms=u.atoms.n_atoms, multiframe=True) as W:
        for avg_coords, bfac, lbl in frames_data:
            pos = orig_pos.copy()
            pos[sel_idx] = avg_coords
            u.atoms.positions = pos
            u.atoms.tempfactors          = 0.0
            u._topology.tempfactors.values[sel_idx] = bfac
            W.write(u.atoms)
            labels.append(lbl)

    # Prepend REMARK header (binary r/w to avoid codec issues)
    with open(outfile, 'rb') as fh:
        body = fh.read()
    header_lines = (
        f"REMARK BRMSD domain identification — "
        f"{len(results)} domains + rigidity map\n"
        f"REMARK B-factor = w_alpha / w_max * 100  "
        f"(rigid=100, flexible~0)\n"
    )
    for i, lbl in enumerate(labels):
        header_lines += f"REMARK MODEL {i+1}: {lbl}\n"
    with open(outfile, 'wb') as fh:
        fh.write(header_lines.encode('ascii', errors='replace'))
        fh.write(body)

    # restore Universe state
    u.atoms.positions = orig_pos
    if has_bfac:
        u.atoms.tempfactors = orig_bfac

    logging.info(f"Wrote domain rounds PDB ({len(frames_data)} models) → {outfile}")


def write_cluster_centers_pdb(u, sel_idx, centers, outfile, labels=None,
                              bfactors=None):
    """
    Write all cluster center structures as a single multi-model PDB.

    Each center becomes one MODEL/ENDMDL block, making the file directly
    loadable in PyMOL or VMD for side-by-side comparison.  Optional per-cluster
    metadata (membership fractions, dominant frame counts, etc.) is written as
    REMARK lines at the top of the file.

    Parameters
    ----------
    u         : MDAnalysis Universe
    sel_idx   : array of atom indices
    centers   : list of (N, 3) arrays, one per cluster
    outfile   : str  output PDB filename
    labels    : list of str, one per cluster (appended as REMARK lines)
    bfactors  : list of (N,) arrays, one per cluster, optional
        Per-atom values written into the B-factor column for each MODEL.
        Pass ``[w * N_atoms for w in w_ca]`` so that uniform weight → 1.0,
        rigid atoms > 1, flexible atoms < 1.
    """
    import MDAnalysis as mda

    orig_pos = u.atoms.positions.copy()
    try:
        orig_bfac = u.atoms.tempfactors.copy()
        has_bfac  = True
    except Exception:
        u.add_TopologyAttr('tempfactors', np.zeros(u.atoms.n_atoms))
        orig_bfac = None
        has_bfac  = False

    with mda.Writer(outfile, n_atoms=u.atoms.n_atoms, multiframe=True) as W:
        for k, coords in enumerate(centers):
            pos = orig_pos.copy()
            pos[sel_idx] = coords
            u.atoms.positions = pos
            if bfactors is not None:
                u._topology.tempfactors.values[:] = 0.0
                u._topology.tempfactors.values[sel_idx] = bfactors[k]
            W.write(u.atoms)

    u.atoms.positions = orig_pos
    if has_bfac:
        u._topology.tempfactors.values[:] = orig_bfac

    # Prepend REMARK header (binary r/w to avoid codec issues)
    with open(outfile, 'rb') as fh:
        body = fh.read()
    header_lines = f"REMARK BRMSD cluster centers — {len(centers)} cluster(s)\n"
    if labels:
        for lbl in labels:
            header_lines += f"REMARK   {lbl}\n"
    with open(outfile, 'wb') as fh:
        fh.write(header_lines.encode('ascii', errors='replace'))
        fh.write(body)

    logging.info(f"Wrote {len(centers)} cluster centers → {outfile}")


def write_aligned_trajectory(u, sel_idx, avg, w, axes_idx, outfile, frames):
    """
    Apply final rigid-body alignment (computed from sel_idx atoms) to ALL
    atoms and write the result to outfile.
    """
    import MDAnalysis as mda
    from tqdm import tqdm
    with mda.Writer(outfile, n_atoms=u.atoms.n_atoms) as W:
        for f in tqdm(frames, desc="Writing aligned trajectory"):
            u.trajectory[f]
            pos = u.atoms.positions.copy()
            sub = pos[sel_idx]
            R, c1, c2 = compute_weighted_rotation(sub, avg, w, axes_idx)
            pos[:, axes_idx] = (pos[:, axes_idx] - c1) @ R + c2
            u.atoms.positions = pos
            W.write(u.atoms)
    logging.info(f"Wrote aligned trajectory → {outfile}")


def write_smoothed_trajectory(u, smoothed_pos_full, outfile, frames):
    """
    Write a trajectory where selected atoms have been replaced by smoothed
    coordinates.  smoothed_pos_full: list/array of (n_all_atoms, 3) frames.
    """
    import MDAnalysis as mda
    from tqdm import tqdm
    with mda.Writer(outfile, n_atoms=u.atoms.n_atoms) as W:
        for k, f in enumerate(tqdm(frames, desc="Writing smoothed trajectory")):
            u.trajectory[f]
            u.atoms.positions = smoothed_pos_full[k]
            W.write(u.atoms)
    logging.info(f"Wrote smoothed trajectory → {outfile}")


def write_rmsd(rmsd_data, fname, times=None, header=""):
    """
    Write RMSD data to an xmgrace-compatible file.

    rmsd_data : list of (label, frame_idx, rmsd_value)

    Each label becomes one Grace dataset, preceded by a ``# label`` comment
    and terminated by ``&`` so the file can be read directly with::

        xmgrace rmsd.dat
    """
    from collections import defaultdict
    groups      = defaultdict(list)
    label_order = []
    for (lbl, k, val) in rmsd_data:
        if lbl not in groups:
            label_order.append(lbl)
        groups[lbl].append((k, val))

    with open(fname, 'w') as f:
        if header:
            f.write(f"# {header}\n")
        f.write("# time(ps)   RMSD(Angstrom)\n")
        for lbl in label_order:
            f.write(f"# {lbl}\n")
            for k, val in groups[lbl]:
                t = times[k] if times is not None else k
                f.write(f"{t:.6f}\t{val:.6f}\n")
            f.write("&\n")
    logging.info(f"Wrote RMSD → {fname}")


def write_rmsf(rmsf_data, sel_idx, fname, header=""):
    """
    Write RMSF data to an xmgrace-compatible file.

    rmsf_data : dict  label → (N,) array of per-atom RMSF values

    Each label is one Grace dataset terminated by ``&``::

        xmgrace rmsf.dat
    """
    with open(fname, 'w') as f:
        if header:
            f.write(f"# {header}\n")
        f.write("# atom_index   RMSF(Angstrom)\n")
        for lbl, vals in rmsf_data.items():
            f.write(f"# {lbl}\n")
            for j, atom in enumerate(sel_idx):
                f.write(f"{atom}\t{vals[j]:.6f}\n")
            f.write("&\n")
    logging.info(f"Wrote RMSF → {fname}")


def write_weights(weight_data, sel_idx, fname, header=""):
    """
    Write per-atom weights to an xmgrace-compatible file.

    weight_data : dict  label → (N,) weight array

    Each label is one Grace dataset terminated by ``&``::

        xmgrace weights.dat
    """
    with open(fname, 'w') as f:
        if header:
            f.write(f"# {header}\n")
        f.write("# atom_index   weight\n")
        for lbl, vals in weight_data.items():
            f.write(f"# {lbl}\n")
            for j, atom in enumerate(sel_idx):
                f.write(f"{atom}\t{vals[j]:.8e}\n")
            f.write("&\n")
    logging.info(f"Wrote weights → {fname}")


# Vectorised batch alignment  (no Python loop over frames)

def batch_align_frames(all_pos, ref_pos, w, axes_idx, return_R=False):
    """
    Align all M frames to ref_pos simultaneously using numpy's batched SVD.

    Eliminates the Python ``for`` loop over frames that dominates runtime in
    the single-frame Kabsch approach.  All M cross-covariance matrices are
    built and decomposed in a single LAPACK call.

    Parameters
    ----------
    all_pos  : (M, N, 3)   positions of selected atoms for every frame
    ref_pos  : (N, 3)      reference (average) structure
    w        : (N,)        per-atom weights (need not sum to 1)
    axes_idx : list        axis indices, e.g. [0, 1, 2] for xyz

    Returns
    -------
    aligned : (M, N, d)   rotated + translated frame coordinates
    dev_sum : (N,)        sum over frames of per-atom squared deviation
                          Σᵢ |ref_alpha − aligned_{i,alpha}|²
    """
    M, N, _ = all_pos.shape
    d        = len(axes_idx)

    X  = all_pos[:, :, axes_idx]          # (M, N, d)
    Y  = ref_pos[:, axes_idx]             # (N, d)

    w_sum = w.sum()
    wn    = w / w_sum if w_sum > 1e-300 else np.ones(N) / N

    # weighted centroids
    c1 = np.einsum('mnd,n->md', X, wn)   # (M, d)  per-frame weighted centroid
    c2 = wn @ Y                           # (d,)    reference weighted centroid

    # centred coordinates
    Xc = X - c1[:, np.newaxis, :]         # (M, N, d)
    Yc = Y - c2                           # (N, d)

    # batched weighted cross-covariance  H[m] = Xc[m]ᵀ W Yc
    H = np.einsum('mni,n,nj->mij', Xc, wn, Yc)   # (M, d, d)

    # batched SVD  (one LAPACK call for all M matrices)
    U, _s, Vht = np.linalg.svd(H)                 # (M,d,d), (M,d), (M,d,d)

    # reflection correction: ensure det(R) = +1 for every frame
    UVht = np.einsum('mij,mjk->mik', U, Vht)      # (M, d, d)
    det  = np.linalg.det(UVht)                     # (M,)
    D    = np.tile(np.eye(d), (M, 1, 1))           # (M, d, d)
    D[:, -1, -1] = det

    # rotation matrices R[m] = U[m] D[m] Vht[m]
    R = np.einsum('mij,mjk,mkl->mil', U, D, Vht)  # (M, d, d)

    # aligned coordinates: (Xc[m] @ R[m]) + c2
    aligned = np.einsum('mni,mij->mnj', Xc, R) + c2   # (M, N, d)

    # per-atom deviation sum  Σᵢ |Y - aligned_i|²
    diff    = Y[np.newaxis] - aligned                   # (M, N, d)
    dev_sum = np.einsum('mnd,mnd->n', diff, diff)       # (N,)

    if return_R:
        return aligned, dev_sum, R
    return aligned, dev_sum


def compute_rotations_per_frame(all_pos, avg, w, axes_idx):
    """
    Compute per-frame rotation matrices given fixed avg and w.

    Returns (M, 3, 3) array; axes-subset rotations are embedded into full 3×3.
    """
    M = all_pos.shape[0]
    _, _, R_sub = batch_align_frames(all_pos, avg, w, axes_idx, return_R=True)
    idx = np.array(axes_idx)
    R_full = np.tile(np.eye(3), (M, 1, 1))
    R_full[:, idx[:, None], idx[None, :]] = R_sub
    return R_full


def write_rotation_data(R_arr, times, outfile, fmt='rotvec', euler_seq='ZYX'):
    """
    Write per-frame rotation data in one of four representations.

    R_arr     : (M, 3, 3)  rotation matrices
    times     : (M,) array or None — time axis (ps); frame index used if None
    fmt       : 'matrix' | 'quaternion' | 'euler' | 'rotvec'
    euler_seq : Euler angle convention, e.g. 'ZYX' (ignored for other formats)
    """
    from scipy.spatial.transform import Rotation

    rots = Rotation.from_matrix(R_arr)

    if fmt == 'matrix':
        data = R_arr.reshape(-1, 9)
        col_header = 'R00 R01 R02 R10 R11 R12 R20 R21 R22'
        desc = 'rotation matrix (row-major)'
    elif fmt == 'quaternion':
        q = rots.as_quat()          # scipy: [x, y, z, w]
        data = q[:, [3, 0, 1, 2]]  # reorder to [w, x, y, z]
        col_header = 'qw qx qy qz'
        desc = 'quaternion (w x y z)'
    elif fmt == 'euler':
        data = rots.as_euler(euler_seq, degrees=False)
        col_header = f'alpha({euler_seq[0]}) beta({euler_seq[1]}) gamma({euler_seq[2]})'
        desc = f'Euler angles {euler_seq} (radians)'
    elif fmt == 'rotvec':
        data = rots.as_rotvec()
        col_header = 'omega_x omega_y omega_z'
        desc = 'rotation vector axis*phi (radians), |omega|=phi'
    else:
        raise ValueError(f"Unknown rotation format '{fmt}'. "
                         "Choose: matrix, quaternion, euler, rotvec")

    time_col = 'time(ps)' if times is not None else 'frame'
    with open(outfile, 'w') as f:
        f.write(f"# BRMSD rotations: {desc}\n")
        f.write(f"# {time_col}  {col_header}\n")
        for i, row in enumerate(data):
            t = float(times[i]) if times is not None else i
            vals = '  '.join(f'{v:.8f}' for v in row)
            f.write(f"{t:.4f}  {vals}\n")
    logging.info(f"Wrote rotation data ({fmt}) → {outfile}")
