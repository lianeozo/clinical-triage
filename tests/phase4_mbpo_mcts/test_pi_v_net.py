"""Policy / value network tests (Phase 4 spec §3b)."""
import torch
from triage_rl.config import PiVNetConfig
from phase4_mbpo_mcts.agents.pi_v_net import PiVNet

OBS_DIMS = [3, 3, 2, 5, 2, 2, 2, 4]
N_ACTIONS = 32


def test_outputs_shape_and_pi_normalized():
    cfg = PiVNetConfig(hidden_dim=32)
    net = PiVNet(obs_dims=OBS_DIMS, n_actions=N_ACTIONS, config=cfg, seed=0, device="cpu")
    obs_hist = torch.zeros(4, 5, len(OBS_DIMS), dtype=torch.long)
    act_hist = torch.zeros(4, 5, dtype=torch.long)
    logits, value = net.forward(obs_hist, act_hist)
    assert logits.shape == (4, N_ACTIONS)
    assert value.shape == (4,)
    pi = torch.softmax(logits, dim=-1)
    assert torch.allclose(pi.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_gradient_flow():
    cfg = PiVNetConfig(hidden_dim=32, lr=1e-2)
    net = PiVNet(obs_dims=OBS_DIMS, n_actions=N_ACTIONS, config=cfg, seed=0, device="cpu")
    obs_hist = torch.zeros(2, 3, len(OBS_DIMS), dtype=torch.long)
    act_hist = torch.zeros(2, 3, dtype=torch.long)
    logits, value = net.forward(obs_hist, act_hist)
    loss = logits.sum() + value.sum()
    loss.backward()
    for p in net.parameters():
        if p.requires_grad:
            assert p.grad is not None
