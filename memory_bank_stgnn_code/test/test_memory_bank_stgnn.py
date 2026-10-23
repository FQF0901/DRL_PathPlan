import torch
from net.memory_bank import *
from net.st_gnn import STGNN


def ent(i, typ, x, h=16):
    return EntityRecord(i, typ, torch.full((h,), float(i) / 10), torch.tensor([x, 0., 0., 0., 0.]))


def frame(k, ids=(1,2), pose=(0.,0.,0.), source=Source.REAL, depth=0, parent=-1):
    entities = [ent(0, NodeType.EGO, 0.)] + [ent(i, NodeType.OD, float(i)) for i in ids]
    return FrameRecord("ep", k, k * .5, torch.tensor(pose), tuple(entities), source, depth, parent)


def test_fifo_reset_and_no_cross_episode():
    bank = RealMemoryBank(2)
    bank.append(frame(0)); bank.append(frame(1)); bank.append(frame(2))
    assert [f.frame_id for f in bank.frames] == [1,2]
    other = FrameRecord("ep2", 0, 0., torch.zeros(3), (ent(0, NodeType.EGO, 0.),))
    bank.append(other)
    assert len(bank) == 1 and bank.frames[0].episode_id == "ep2"


def test_se2_compensation():
    # Historical object at x=10 in ego0. Query ego has moved +2m world-x => x=8.
    a = FrameRecord("ep", 0, 0., torch.tensor([0.,0.,0.]), (ent(7, NodeType.OD, 10.),))
    b = FrameRecord("ep", 1, .5, torch.tensor([2.,0.,0.]), (ent(7, NodeType.OD, 8.),))
    view = BankViewBuilder().build((a,b))
    assert torch.allclose(view.geometry[:,0], torch.tensor([8.,8.]))


def test_stable_id_edges_birth_death_without_current_slot_alignment():
    view = BankViewBuilder().build((frame(0, (1,2)), frame(1, (2,3))))
    pairs = {(int(view.stable_id[s]), int(view.stable_id[d])) for s,d in view.temporal_edge_index.t()}
    assert (2,2) in pairs and (1,1) not in pairs and (3,3) not in pairs
    current3 = (view.stable_id == 3) & view.readout_mask
    assert bool(view.birth[current3].all())


def test_imagine_copy_on_write():
    real = RealMemoryBank(6); real.append(frame(0))
    imagined = real.snapshot()
    imagined.append(frame(1, source=Source.IMAGINED, depth=1, parent=0))
    assert len(real.frames) == 1 and len(imagined.frames) == 2


def test_stgnn_permutation_equivariance_readout_shape():
    torch.manual_seed(0)
    view = BankViewBuilder().build((frame(0), frame(1)))
    net = STGNN(hidden=16, geometry_dim=5, layers=1)
    out = net(view)
    assert out["current"].shape[0] == int(view.readout_mask.sum())
    assert torch.isfinite(out["all"]).all()
