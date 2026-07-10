"""
Shared pytest fixtures for the BRMSD test suite.

A small synthetic two-domain trajectory is generated in memory and written to a
temporary GRO + XTC once per session.  No binary data is committed to the repo;
the fixture is fully reproducible from the fixed random seed below.

System (30 CA atoms, one atom per residue, resid 1-30):
  * domain A (resid 1-15)  — the common rigid core, present in every frame
  * domain B (resid 16-30) — a rigid body that occupies two distinct
                             orientations relative to A (two conformational
                             states), giving the trajectory clustering and
                             domain-identification signal.
Every frame is additionally placed in a random global orientation, so alignment
has real work to do, and small Gaussian noise is added to every atom.
"""

import numpy as np
import pytest
import MDAnalysis as mda
from scipy.spatial.transform import Rotation


N_A = 15                       # domain A atom count (resid 1-15)
N_B = 15                       # domain B atom count (resid 16-30)
N = N_A + N_B
M = 50                         # frames
NOISE = 0.05                   # per-atom Gaussian noise (Angstrom)
SEED = 20240607


def _reference():
    rng = np.random.default_rng(SEED)
    a = rng.normal(0.0, 3.0, size=(N_A, 3))
    b = rng.normal(0.0, 3.0, size=(N_B, 3)) + np.array([15.0, 0.0, 0.0])
    return np.vstack([a, b])


def _build_coords():
    rng = np.random.default_rng(SEED + 1)
    ref = _reference()
    a_ref = ref[:N_A]
    b_ref = ref[N_A:]
    b_centroid = b_ref.mean(axis=0)

    # domain B second-state orientation: rigid rotation about its own centroid
    hinge = Rotation.from_euler('z', 40.0, degrees=True).as_matrix()
    b_state1 = (b_ref - b_centroid) @ hinge.T + b_centroid

    coords = np.zeros((M, N, 3))
    labels = np.zeros(M, dtype=int)
    for i in range(M):
        b = b_ref if i < M // 2 else b_state1
        labels[i] = 0 if i < M // 2 else 1
        frame = np.vstack([a_ref, b])
        # random global rigid placement
        Rg = Rotation.random(random_state=rng).as_matrix()
        t = rng.normal(0.0, 2.0, size=3)
        frame = frame @ Rg.T + t
        frame += rng.normal(0.0, NOISE, size=frame.shape)
        coords[i] = frame
    return ref, coords, labels


def _make_universe(ref, coords):
    resids = np.arange(1, N + 1)
    u = mda.Universe.empty(N, n_residues=N, atom_resindex=np.arange(N),
                           residue_segindex=np.zeros(N, dtype=int),
                           trajectory=True)
    u.add_TopologyAttr('name', ['CA'] * N)
    u.add_TopologyAttr('type', ['C'] * N)
    u.add_TopologyAttr('resname', ['ALA'] * N)
    u.add_TopologyAttr('resid', resids)
    u.add_TopologyAttr('segid', ['A'])
    u.load_new(coords, order='fac')
    u.dimensions = [200.0, 200.0, 200.0, 90.0, 90.0, 90.0]
    return u


@pytest.fixture(scope='session')
def synthetic_arrays():
    """Return (ref (N,3), coords (M,N,3), labels (M,)) — pure numpy, no files."""
    return _build_coords()


@pytest.fixture(scope='session')
def traj(tmp_path_factory, synthetic_arrays):
    """
    Write the synthetic system to GRO + XTC once and return a dict of paths and
    metadata for use by the CLI subprocess tests.
    """
    ref, coords, labels = synthetic_arrays
    d = tmp_path_factory.mktemp('brmsd_data')
    gro = d / 'system.gro'
    xtc = d / 'traj.xtc'

    u = _make_universe(ref, coords)
    u.trajectory[0]
    u.atoms.write(str(gro))
    with mda.Writer(str(xtc), n_atoms=N) as W:
        for _ in u.trajectory:
            W.write(u.atoms)

    return {
        'gro': str(gro),
        'xtc': str(xtc),
        'n_atoms': N,
        'n_frames': M,
        'domainA_sel': 'resid 1:15',
        'domainB_sel': 'resid 16:30',
    }
