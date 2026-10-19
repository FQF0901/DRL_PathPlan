import os
import numpy as np
import pytest

def test_expert_data_contract_if_present():
    path = "datasets/bc_200_v2/expert_bc.npz"
    if not os.path.exists(path):
        pytest.skip("dataset not present")
    z = np.load(path)
    for k in ("od", "od_mask", "ld", "ld_mask", "nav", "nav_mask", "ego", "ego_mask", "hist_valid", "action"):
        assert k in z.files, k
    assert z["od"].shape[1:] == (6, 16, 11)
    assert z["ld"].shape[1:] == (6, 16, 5, 9)
    assert z["action"].shape[1] == 4
    assert np.isfinite(z["action"]).all()
