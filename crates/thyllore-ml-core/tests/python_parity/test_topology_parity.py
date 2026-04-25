import numpy as np

import thyllore_ml_core as tml

from conftest import array_from_bits_2d


def _make_skeleton(skeleton_fixture):
    bone_names = skeleton_fixture["bone_names"]
    parent_indices = skeleton_fixture["parent_indices"]
    n = len(bone_names)
    matrices = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    return tml.PySkeleton.from_flat(bone_names, parent_indices, matrices)


def test_topology_matches_rust(skeleton_fixture):
    skel = _make_skeleton(skeleton_fixture)
    py_topology = tml.compute_topology(skel)

    rust_topology = array_from_bits_2d(skeleton_fixture["topology"])

    assert py_topology.shape == rust_topology.shape
    assert np.array_equal(
        py_topology.view(np.uint32),
        rust_topology.view(np.uint32),
    ), "Python and Rust topology outputs differ at bit level"


def test_tokenize_matches_rust(skeleton_fixture):
    skel = _make_skeleton(skeleton_fixture)
    py_tokens = tml.tokenize_bone_names(skel)

    rust_tokens = np.array(skeleton_fixture["tokens"], dtype=np.int64)

    assert py_tokens.shape == rust_tokens.shape
    assert np.array_equal(py_tokens, rust_tokens)


def test_bone_count(skeleton_fixture):
    skel = _make_skeleton(skeleton_fixture)
    assert skel.bone_count == len(skeleton_fixture["bone_names"])


def test_bone_name(skeleton_fixture):
    skel = _make_skeleton(skeleton_fixture)
    for i, expected in enumerate(skeleton_fixture["bone_names"]):
        assert skel.bone_name(i) == expected
