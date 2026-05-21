"""AlphaZero-style MCTS planner.

The planner is a pure-numpy/Python ``run_mcts`` function that interacts with
PyTorch components (the dynamics ensemble and the policy/value net) ONLY
through two callable closures supplied by the caller:

    model_step_fn(history, action, member_idx) -> (next_obs, reward, done)
        Uses the env's known reward and termination functions and samples
        the next observation from the chosen ensemble member.

    pi_v_fn(history) -> (pi_probs[n_actions], V_scalar)
        Evaluates the policy and value heads at a leaf history.

Keeping the planner closure-based means it is fully testable with simple
lambdas (see tests/phase4_mbpo_mcts/test_mcts.py) and carries no torch
dependency itself.

Spec: docs/superpowers/specs/2026-05-20-part4-mbpo-mcts-design.md §3c.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from triage_rl.config import MCTSConfig


# ----------------------------------------------------------------------------
# Public result type
# ----------------------------------------------------------------------------

@dataclass
class MCTSResult:
    """Aggregated outputs of one ``run_mcts`` call.

    Attributes:
        visit_counts: ``{action_idx: N(root, a)}``. Only actions that were
            visited at least once appear as keys.
        value: Mean backed-up return at the root,
            ``sum(W_a) / sum(N_a)``.
        root_q: ``{action_idx: Q(root, a)}`` for actions that were visited.
        mean_depth: Average number of edges traversed per simulation.
    """

    visit_counts: dict[int, int]
    value: float
    root_q: dict[int, float]
    mean_depth: float


# ----------------------------------------------------------------------------
# Internal node
# ----------------------------------------------------------------------------

class _Node:
    """A single search-tree node.

    Per-action statistics (N, W, Q, child pointers) are stored AT THE PARENT
    keyed by the action that leads to the child, in the standard AlphaZero /
    MuZero representation. The node also stores the per-edge step reward
    that came FROM the parent (used to seed backup of the discounted return)
    and a terminality flag.
    """

    __slots__ = (
        "parent",
        "parent_action",
        "obs",
        "history_so_far",
        "children",
        "N_a",
        "W_a",
        "Q_a",
        "prior",
        "is_expanded",
        "is_terminal",
        "step_reward_from_parent",
    )

    def __init__(
        self,
        parent: Optional["_Node"],
        parent_action: Optional[int],
        obs: np.ndarray,
        history_so_far: list,
        step_reward_from_parent: float = 0.0,
        is_terminal: bool = False,
    ) -> None:
        self.parent = parent
        self.parent_action = parent_action
        self.obs = obs
        self.history_so_far = history_so_far
        self.children: dict[int, "_Node"] = {}
        self.N_a: dict[int, int] = {}
        self.W_a: dict[int, float] = {}
        self.Q_a: dict[int, float] = {}
        self.prior: Optional[np.ndarray] = None
        self.is_expanded: bool = False
        self.is_terminal: bool = is_terminal
        self.step_reward_from_parent: float = float(step_reward_from_parent)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _puct_select(node: _Node, c_puct: float, n_actions: int) -> int:
    """Return the action that maximizes the PUCT score at ``node``.

    PUCT (AlphaZero):
        UCB(a) = Q(s,a) + c_puct * prior[a] * sqrt(sum_b N(s,b)) / (1 + N(s,a))

    Unvisited actions (N(s,a) == 0) get Q=0, so the prior * sqrt(N_total)
    term dominates and the planner explores in proportion to the prior.
    """
    prior = node.prior
    assert prior is not None, "PUCT called on an unexpanded node"
    total_N = sum(node.N_a.values())
    sqrt_total = math.sqrt(total_N) if total_N > 0 else 0.0

    best_a = -1
    best_score = -math.inf
    for a in range(n_actions):
        n_sa = node.N_a.get(a, 0)
        q_sa = node.Q_a.get(a, 0.0)
        # When total_N is 0 (first visit at this node), sqrt_total=0 and the
        # exploration term collapses; but the spec's convention is that the
        # PUCT prior term carries sqrt(N_total). To still discriminate on
        # the prior in that case, AlphaZero adds 1 inside the sqrt; we
        # follow MuZero's version with bare sqrt and accept ties broken by
        # action index, which is fine since priors will dominate the moment
        # any sibling is visited.
        u = c_puct * float(prior[a]) * sqrt_total / (1 + n_sa)
        score = q_sa + u
        if score > best_score:
            best_score = score
            best_a = a
    return best_a


def _expand(
    node: _Node,
    pi: np.ndarray,
    is_root: bool,
    rng: np.random.Generator,
    cfg: MCTSConfig,
    n_actions: int,
) -> None:
    """Mark ``node`` as expanded and set its prior.

    At the root, blend Dirichlet noise into ``pi`` per AlphaZero. For
    non-root nodes, the prior is ``pi`` directly.
    """
    if is_root and cfg.dirichlet_eps > 0.0:
        noise = rng.dirichlet([cfg.dirichlet_alpha] * n_actions)
        node.prior = (1.0 - cfg.dirichlet_eps) * pi + cfg.dirichlet_eps * noise
    else:
        node.prior = np.asarray(pi, dtype=np.float64).copy()
    node.is_expanded = True


# ----------------------------------------------------------------------------
# Public planner
# ----------------------------------------------------------------------------

def run_mcts(
    root_obs: np.ndarray,
    root_history: list,
    model_step_fn: Callable[[list, int, int], tuple[np.ndarray, float, bool]],
    pi_v_fn: Callable[[list], tuple[np.ndarray, float]],
    config: MCTSConfig,
    n_actions: int,
    n_ensemble_members: int,
    rng: np.random.Generator,
) -> MCTSResult:
    """Run AlphaZero-style MCTS from ``root_obs``.

    Args:
        root_obs: ``(V,)`` int observation at the root.
        root_history: Past ``(obs, action)`` pairs in this real-env episode.
        model_step_fn: ``(history, action, member_idx) -> (next_obs, r, done)``
            using the env's known reward and termination functions and the
            ensemble's sampled next-obs.
        pi_v_fn: ``history -> (pi_probs[n_actions], V_scalar)`` evaluating
            the policy/value net at a leaf history.
        config: ``MCTSConfig`` controlling n_simulations, c_puct, Dirichlet
            noise, discount, etc.
        n_actions: Total number of discrete actions.
        n_ensemble_members: K for Thompson-sampling the ensemble member
            per simulation.
        rng: ``np.random.Generator`` for noise + ensemble member sampling.

    Returns:
        ``MCTSResult`` with visit counts, root value, root Q-values, and
        mean tree depth across simulations.
    """
    # ----- Root expansion (must happen before the simulation loop) -----
    root = _Node(parent=None, parent_action=None, obs=root_obs, history_so_far=list(root_history))
    pi_root_raw, v_root = pi_v_fn(root.history_so_far)
    pi_root_raw = np.asarray(pi_root_raw, dtype=np.float64)
    assert pi_root_raw.shape == (n_actions,), (
        f"pi_v_fn must return a length-{n_actions} prior; got {pi_root_raw.shape}"
    )
    _expand(root, pi_root_raw, is_root=True, rng=rng, cfg=config, n_actions=n_actions)

    gamma = config.gamma
    depths: list[int] = []

    for _ in range(config.n_simulations):
        # Thompson-sampled ensemble member, fixed for this simulation.
        member_idx = int(rng.integers(0, n_ensemble_members))

        # ----- Selection: traverse from root until unexpanded or terminal -----
        node = root
        path_edges: list[tuple[_Node, int]] = []  # (parent_node, action)
        depth = 0
        while node.is_expanded and not node.is_terminal:
            a = _puct_select(node, config.c_puct, n_actions)
            path_edges.append((node, a))
            depth += 1

            if a in node.children:
                node = node.children[a]
            else:
                # Step the model; this creates a NEW child whose `step_reward_from_parent`
                # carries the per-edge reward used during backup.
                next_obs, r, done = model_step_fn(node.history_so_far, a, member_idx)
                child_history = node.history_so_far + [(node.obs, a)]
                child = _Node(
                    parent=node,
                    parent_action=a,
                    obs=np.asarray(next_obs),
                    history_so_far=child_history,
                    step_reward_from_parent=float(r),
                    is_terminal=bool(done),
                )
                node.children[a] = child
                node = child
                break  # newly created leaf — exit selection loop

        # ----- Leaf evaluation -----
        if node.is_terminal:
            # Terminal leaf: value is the env's terminal reward (the per-edge
            # reward stored when this node was created). Do NOT call V_psi.
            leaf_value = node.step_reward_from_parent
            # The edge reward will be re-added by the backup loop, so to avoid
            # double-counting we set G = 0 here and let backup do
            # G <- step_reward + gamma * G  on the final edge, yielding
            # G = node.step_reward_from_parent at the parent. Tracking this
            # cleanly: we feed leaf_value=0 if backup itself adds the edge
            # reward. We choose the convention where backup ADDS edge rewards
            # for every edge, so leaf_value seeds the bootstrap PAST the leaf.
            leaf_bootstrap = 0.0
        else:
            # Non-terminal unexpanded node: evaluate π/V here, expand the
            # node so future simulations can select through it.
            pi_leaf, v_leaf = pi_v_fn(node.history_so_far)
            pi_leaf = np.asarray(pi_leaf, dtype=np.float64)
            assert pi_leaf.shape == (n_actions,)
            _expand(node, pi_leaf, is_root=False, rng=rng, cfg=config, n_actions=n_actions)
            leaf_bootstrap = float(v_leaf)

        # ----- Backup: walk path_edges from leaf to root accumulating G -----
        # Convention: every edge contributes its `step_reward_from_parent`
        # (stored on the CHILD). leaf_bootstrap seeds G past the leaf node.
        G = leaf_bootstrap
        # Walk edges in reverse so we visit the deepest edge first.
        for parent_node, action_a in reversed(path_edges):
            child = parent_node.children[action_a]
            G = child.step_reward_from_parent + gamma * G
            n_new = parent_node.N_a.get(action_a, 0) + 1
            w_new = parent_node.W_a.get(action_a, 0.0) + G
            parent_node.N_a[action_a] = n_new
            parent_node.W_a[action_a] = w_new
            parent_node.Q_a[action_a] = w_new / n_new

        depths.append(depth)

    # ----- Assemble result -----
    visit_counts = {a: int(n) for a, n in root.N_a.items()}
    root_q = {a: float(q) for a, q in root.Q_a.items()}
    total_N = sum(visit_counts.values())
    total_W = sum(root.W_a.values())
    value = float(total_W / total_N) if total_N > 0 else float(v_root)
    mean_depth = float(np.mean(depths)) if depths else 0.0

    return MCTSResult(
        visit_counts=visit_counts,
        value=value,
        root_q=root_q,
        mean_depth=mean_depth,
    )
