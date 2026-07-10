"""Correctness tests for the alignment, focus, and smoothing free energies."""

import numpy as np

from BRMSD.align import BRMSD_iterate
from BRMSD.focus import BRMSD_focus_iterate
from BRMSD.smooth import BRMSD_smooth, triangular_window, uniform_window


def test_large_sigma_gives_uniform_weights(synthetic_arrays):
    _, coords, _ = synthetic_arrays
    N = coords.shape[1]
    _, w, _ = BRMSD_iterate(coords, [0, 1, 2], sigma=1e6, maxiter=50, tol=1e-4)
    # sigma -> infinity recovers standard RMSD (equal weights)
    assert np.allclose(w, 1.0 / N, atol=1e-3)


def test_theta_is_sample_size_independent(synthetic_arrays):
    """theta = M*sigma**2, so duplicating every frame must leave weights fixed."""
    _, coords, _ = synthetic_arrays
    _, w1, _ = BRMSD_iterate(coords, [0, 1, 2], sigma=0.3, maxiter=100, tol=1e-6)
    coords2 = np.concatenate([coords, coords], axis=0)     # 2*M frames
    _, w2, _ = BRMSD_iterate(coords2, [0, 1, 2], sigma=0.3, maxiter=100, tol=1e-6)
    assert np.allclose(w1, w2, atol=1e-6)


def test_rigid_core_gets_more_weight(synthetic_arrays):
    """Domain A (resid 1-15) is the common rigid core -> higher mean weight."""
    _, coords, _ = synthetic_arrays
    _, w, _ = BRMSD_iterate(coords, [0, 1, 2], sigma=0.3, maxiter=200, tol=1e-6)
    assert w[:15].mean() > w[15:].mean()


def test_focus_mu_ratio_zero_equals_align(synthetic_arrays):
    """mu_ratio = 0 must reproduce plain BRMSD alignment exactly."""
    _, coords, _ = synthetic_arrays
    N = coords.shape[1]
    mask = np.zeros(N, dtype=bool)
    mask[15:] = True
    _, w_align, _ = BRMSD_iterate(coords, [0, 1, 2], sigma=0.3, maxiter=100, tol=1e-7)
    _, w_focus, _ = BRMSD_focus_iterate(coords, [0, 1, 2], sigma=0.3, mu_ratio=0.0,
                                        domain_mask=mask, maxiter=100, tol=1e-7)
    assert np.allclose(w_align, w_focus, atol=1e-7)


def test_focus_raises_domain_weight(synthetic_arrays):
    """Positive mu_ratio concentrates weight inside the focus domain."""
    _, coords, _ = synthetic_arrays
    N = coords.shape[1]
    mask = np.zeros(N, dtype=bool)
    mask[15:] = True
    _, w0, _ = BRMSD_focus_iterate(coords, [0, 1, 2], 0.3, 0.0, mask, 100, 1e-6)
    _, w1, _ = BRMSD_focus_iterate(coords, [0, 1, 2], 0.3, 2.0, mask, 100, 1e-6)
    assert w1[mask].sum() > w0[mask].sum()


def test_window_functions_normalised():
    for fn in (triangular_window, uniform_window):
        p = fn(21, 10, 4)
        assert np.isclose(p.sum(), 1.0)
        assert (p >= 0).all()
    # triangular window peaks at the target frame
    p = triangular_window(21, 10, 4)
    assert p.argmax() == 10


def test_smooth_runs_and_reduces_fluctuation(synthetic_arrays):
    _, coords, _ = synthetic_arrays
    smoothed, weights, rmsds = BRMSD_smooth(
        coords, [0, 1, 2], sigma=0.5, maxiter=50, tol=1e-4,
        half_width=3, window_type='triangular', n_jobs=1)
    assert smoothed.shape == coords.shape
    assert weights.shape == (coords.shape[0], coords.shape[1])
    assert np.isfinite(rmsds).all()
