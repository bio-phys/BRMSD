"""Correctness tests for soft K-means clustering and domain identification."""

import numpy as np

from BRMSD.cluster import soft_kmeans, pairwise_center_rmsd, cluster_quality_metrics
from BRMSD.domain import identify_domains, identify_domains_sequential


def test_soft_kmeans_shapes_and_finite_G(synthetic_arrays):
    _, coords, _ = synthetic_arrays
    M, N, _ = coords.shape
    centers, w_ca, q_ai, hist, G, dev_ai = soft_kmeans(
        coords, 2, [0, 1, 2], sigma=0.3, tau=2.0, maxiter=100, tol=1e-4,
        n_restarts=3, n_jobs=1)
    assert centers.shape == (2, N, 3)
    assert w_ca.shape == (2, N)
    assert q_ai.shape == (2, M)
    assert np.isfinite(G)
    # responsibilities sum to 1 per frame
    assert np.allclose(q_ai.sum(axis=0), 1.0, atol=1e-6)
    # each cluster's weights sum to 1
    assert np.allclose(w_ca.sum(axis=1), 1.0, atol=1e-6)


def test_soft_kmeans_recovers_two_states(synthetic_arrays):
    """
    The trajectory has two conformational states -> both clusters populated.
    Clustering a transition needs a broad sigma: a small (concentrating) sigma
    down-weights the mobile domain that distinguishes the states and collapses
    the clusters, so a large sigma is used here.
    """
    _, coords, labels = synthetic_arrays
    _, _, q_ai, _, _, _ = soft_kmeans(
        coords, 2, [0, 1, 2], sigma=5.0, tau=1.0, maxiter=200, tol=1e-5,
        n_restarts=5, n_jobs=1)
    hard = np.argmax(q_ai, axis=0)
    # both clusters used
    assert len(np.unique(hard)) == 2
    # agreement with ground-truth labels up to cluster relabeling
    agree = max((hard == labels).mean(), (hard == (1 - labels)).mean())
    assert agree > 0.8


def test_cluster_quality_metrics_sane(synthetic_arrays):
    _, coords, _ = synthetic_arrays
    centers, w_ca, q_ai, _, _, dev_ai = soft_kmeans(
        coords, 2, [0, 1, 2], sigma=5.0, tau=1.0, maxiter=200, tol=1e-5,
        n_restarts=5, n_jobs=1)
    rmsd_ak = np.sqrt(np.maximum(dev_ai, 0))
    ctr = pairwise_center_rmsd(list(centers), [0, 1, 2])
    ch, db, s_a = cluster_quality_metrics(rmsd_ak, q_ai, ctr)
    assert np.isfinite(ch) and ch > 0
    assert np.isfinite(db) and db > 0
    assert ctr[0, 1] > 0    # the two states have distinct centers


def test_sequential_peeling_returns_requested_domains(synthetic_arrays):
    _, coords, _ = synthetic_arrays
    N = coords.shape[1]
    results = identify_domains_sequential(
        coords, [0, 1, 2], sigma=1.0, maxiter=100, tol=1e-4,
        n_domains=2, n_restarts=2, mask_thr=0.5, n_jobs=1)
    assert len(results) == 2
    for r in results:
        assert r['w'].shape == (N,)
        assert r['domain_mask'].sum() > 0


def test_sequential_peeling_finds_rigid_core_first(synthetic_arrays):
    """Domain A (rigid core, resid 1-15 -> idx 0-14) is peeled in round 1."""
    _, coords, _ = synthetic_arrays
    results = identify_domains_sequential(
        coords, [0, 1, 2], sigma=1.0, maxiter=100, tol=1e-4,
        n_domains=2, n_restarts=3, mask_thr=0.5, n_jobs=1)
    round1_mask = results[0]['domain_mask']
    frac_in_A = round1_mask[:15].sum() / max(round1_mask.sum(), 1)
    assert frac_in_A > 0.6


def test_simultaneous_domains_shapes(synthetic_arrays):
    _, coords, _ = synthetic_arrays
    N = coords.shape[1]
    results = identify_domains(
        coords, [0, 1, 2], sigma=1.0, mu=1.0, maxiter=100, tol=1e-4,
        n_domains=2, n_restarts=2, mask_thr=0.5, n_jobs=1)
    assert len(results) == 2
    for r in results:
        assert r['w'].shape == (N,)
        assert np.isfinite(r['G_final'])
