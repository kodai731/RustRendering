import numpy as np

import thyllore_ml_core as tml


def test_tokenize_bone_name_string_handles_basic_inputs():
    tokens = tml.tokenize_bone_name_string("LeftShoulder")
    assert tokens.shape == (32,)
    assert tokens.dtype == np.int64


def test_tokenize_bone_name_string_empty_returns_padding():
    tokens = tml.tokenize_bone_name_string("")
    assert tokens.shape == (32,)
    assert np.all(tokens == 0)
