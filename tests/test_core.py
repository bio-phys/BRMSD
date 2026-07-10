"""Unit tests for core numerics: weighted Kabsch, batched alignment, helpers."""

import numpy as np
from scipy.spatial.transform import Rotation

from BRMSD.core import (compute_weighted_rotation, align_frame,
                        batch_align_frames, parse_axes)


def test_parse_axes():
    assert parse_axes('xyz') == [0, 1, 2]
    assert parse_axes('xy') == [0, 1]
    assert parse_axes('z') == [2]
    assert parse_axes('ZX') == [0, 2]


def test_weighted_kabsch_recovers_known_rotation():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(20, 3))
    R_true = Rotation.from_euler('xyz', [30, -15, 45], degrees=True).as_matrix()
    t = np.array([1.0, -2.0, 0.5])
    mobile = ref @ R_true.T + t                       # rotate + translate
    w = np.ones(20)

    aln, R, c1, c2 = align_frame(mobile, ref, w, [0, 1, 2])
    # aligned mobile should match the reference
    assert np.allclose(aln, ref, atol=1e-8)
    # recovered rotation maps mobile back onto ref: (mobile-c1) @ R = ref-c2
    assert np.allclose((mobile - c1) @ R + c2, ref, atol=1e-8)


def test_batch_matches_single_frame():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(12, 3))
    all_pos = np.array([ref @ Rotation.random(random_state=rng).as_matrix().T
                        + rng.normal(0, 0.1, 3) for _ in range(8)])
    w = rng.uniform(0.5, 1.5, size=12)

    aligned_b, dev_b = batch_align_frames(all_pos, ref, w, [0, 1, 2])
    for i in range(8):
        aln_s, _, _, _ = align_frame(all_pos[i], ref, w, [0, 1, 2])
        assert np.allclose(aligned_b[i], aln_s, atol=1e-9)
    # dev_sum is the per-atom sum over frames of squared deviation
    diff = ref[None] - aligned_b
    dev_ref = np.einsum('mnd,mnd->n', diff, diff)
    assert np.allclose(dev_b, dev_ref, atol=1e-9)


def test_rotation_is_proper():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(10, 3))
    mobile = rng.normal(size=(10, 3))
    _, R, _, _ = align_frame(mobile, ref, np.ones(10), [0, 1, 2])
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)   # no reflection
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)     # orthonormal
