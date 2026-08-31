from __future__ import annotations

import numpy as np

from industrial_tsfm.models.timesfm_peft import TimesFMTransformersModel


def test_timesfm_transformers_flatten_series_contract() -> None:
    context = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    target = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    flat_context, flat_target = TimesFMTransformersModel._flatten_series(context, target)
    assert flat_context.shape == (6, 4)
    assert flat_target.shape == (6, 2)
    np.testing.assert_array_equal(flat_context[0], context[0, :, 0])
    np.testing.assert_array_equal(flat_target[0], target[0, :, 0])
