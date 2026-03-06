"""Finite-Horizon SARSOP Solver.

Replaces the random/greedy belief collection of PBVI with a SARSOP-inspired
approach: an explicit belief tree is grown by sampling belief paths guided by
the gap between the sawtooth upper bound and the alpha-vector lower bound.

Each iteration performs:
  1. SAMPLE  — traverse one root-to-leaf path, choosing actions/observations
               by the gap signal.
  2. BACKUP  — bottom-up alpha-vector + sawtooth backups along the path.
  3. PRUNE   — (periodic) dominated alpha-vector removal.

The output .npz format is identical to finite_horizon_pbvi.py.
"""

import os
import time

import numpy as np

from backward_induction import (
    load_models,
    lp_prune_certified,
    lp_prune_sequential,
    pointwise_prune,
)
from finite_horizon_pbvi import (
    belief_update,
    collect_beliefs,
    compute_mdp_upper_bound,
    sawtooth_eval,
)


# ── Belief tree ────────────────────────────────────────────────────────────────

class BeliefNode:
    """One node in the SARSOP belief tree."""

    __slots__ = ("belief", "time_step", "parent", "action", "observation",
                 "children", "is_terminal")

    def __init__(self, belief, time_step, parent=None,
                 action=None, observation=None):
        self.belief      = belief
        self.time_step   = time_step
        self.parent      = parent
        self.action      = action        # action taken *from* parent
        self.observation = observation   # observation received
        self.children    = {}            # (action, obs) -> BeliefNode
        self.is_terminal = False


def count_tree_nodes(root):
    count = [0]
    def _visit(n):
        count[0] += 1
        for c in n.children.values():
            _visit(c)
    _visit(root)
    return count[0]


# ── Discounted backup helpers ──────────────────────────────────────────────────
# (Identical to finite_horizon_pbvi originals but with explicit gamma.)

def _pb_backup_single(b, prev_alpha, transition, observations, reward, gamma):
    """Point-based backup for a single belief with discount factor gamma."""
    A = transition.shape[0]
    Z = observations.shape[1]
    best_val = -np.inf
    best_alpha  = None
    best_action = 0

    for a in range(A):
        b_prop = transition[a].T @ b          # (S,)
        phi_a  = reward[a].copy()             # (S,)
        for z in range(Z):
            w        = observations[a, z] * b_prop
            scores   = prev_alpha @ w
            best_idx = int(np.argmax(scores))
            phi_a   += gamma * (
                transition[a] @ (observations[a, z] * prev_alpha[best_idx])
            )
        val = float(phi_a @ b)
        if val > best_val:
            best_val    = val
            best_alpha  = phi_a
            best_action = a

    return best_alpha, best_action


def _saw_backup_single(b, v_mdp_next, next_saw_b, next_saw_v,
                       transition, observations, reward, gamma):
    """Sawtooth backup for a single belief with discount factor gamma."""
    A = transition.shape[0]
    Z = observations.shape[1]
    best_u = -np.inf

    for a in range(A):
        b_prop = transition[a].T @ b
        u_a    = float(reward[a] @ b)
        for z in range(Z):
            w    = observations[a, z] * b_prop
            prob = w.sum()
            if prob < 1e-12:
                continue
            b_next = w / prob
            u_a   += gamma * prob * sawtooth_eval(
                b_next, v_mdp_next, next_saw_b, next_saw_v
            )
        best_u = max(best_u, u_a)

    return best_u


# ── Bound evaluators ───────────────────────────────────────────────────────────

def _v_lower(b, Gamma_t):
    """max alpha · b over alpha-vectors in Gamma_t. -inf if empty."""
    if len(Gamma_t) == 0:
        return -np.inf
    return float((Gamma_t @ b).max())


def _q_upper(b, a, t, v_mdp_ext, saw_b, saw_v,
             transition, observations, reward, gamma):
    """Q upper-bound for action a at belief b, time t (using sawtooth at t+1)."""
    Z      = observations.shape[1]
    b_prop = transition[a].T @ b
    q      = float(reward[a] @ b)
    for z in range(Z):
        w    = observations[a, z] * b_prop
        prob = w.sum()
        if prob < 1e-12:
            continue
        b_next = w / prob
        q += gamma * prob * sawtooth_eval(
            b_next, v_mdp_ext[t + 1], saw_b[t + 1], saw_v[t + 1]
        )
    return q


def _q_lower(b, a, t, Gamma, transition, observations, reward, gamma):
    """Q lower-bound for action a at belief b, time t (using alpha-vectors at t+1)."""
    Z      = observations.shape[1]
    b_prop = transition[a].T @ b
    q      = float(reward[a] @ b)
    for z in range(Z):
        w    = observations[a, z] * b_prop
        prob = w.sum()
        if prob < 1e-12:
            continue
        b_next = w / prob
        q += gamma * prob * _v_lower(b_next, Gamma[t + 1])
    return q


# ── Gap tolerance ──────────────────────────────────────────────────────────────

def _gap_tol(t, epsilon, H, gamma):
    """Acceptable gap at depth t.

    gamma < 1 : gamma^{-t} * epsilon  (looser at depth, tighter near leaves)
    gamma = 1 : max(epsilon, (H - t) / H * epsilon)
    """
    if gamma < 1.0:
        return (gamma ** (-t)) * epsilon
    return max(epsilon, (H - t) / H * epsilon)


# ── SAMPLE ─────────────────────────────────────────────────────────────────────

def _sample_path(node, L, U, epsilon, H, Gamma, saw_b, saw_v, v_mdp_ext,
                 transition, observations, reward, gamma, visited):
    """Recursively grow T_R along one path; collect visited (node, t) pairs."""
    t = node.time_step
    b = node.belief
    A = transition.shape[0]
    Z = observations.shape[1]

    # ── Termination ──────────────────────────────────────────────────────────
    if t == H - 1:
        visited.append((node, t))
        return

    V_upper = sawtooth_eval(b, v_mdp_ext[t], saw_b[t], saw_v[t])
    V_lower = _v_lower(b, Gamma[t])
    tol     = _gap_tol(t, epsilon, H, gamma)
    if V_upper <= max(U, V_lower + tol):
        return   # bounds already tight at this node

    # ── Action selection: argmax Q_upper ─────────────────────────────────────
    best_a   = 0
    best_qu  = -np.inf
    for a in range(A):
        qu = _q_upper(b, a, t, v_mdp_ext, saw_b, saw_v,
                      transition, observations, reward, gamma)
        if qu > best_qu:
            best_qu = qu
            best_a  = a
    a_star = best_a

    # ── Observation selection: argmax p(o|b,a*) * gap(o) ─────────────────────
    b_prop   = transition[a_star].T @ b
    best_o   = None
    best_wg  = -np.inf
    obs_info = {}   # z -> (b_next, prob)

    for z in range(Z):
        w    = observations[a_star, z] * b_prop
        prob = w.sum()
        if prob < 1e-12:
            continue
        b_next = w / prob
        v_up   = sawtooth_eval(b_next, v_mdp_ext[t + 1], saw_b[t + 1], saw_v[t + 1])
        v_lo   = _v_lower(b_next, Gamma[t + 1])
        wg     = prob * max(0.0, v_up - v_lo)
        obs_info[z] = (b_next, prob)
        if wg > best_wg:
            best_wg = wg
            best_o  = z

    if best_o is None:
        return   # no reachable observation

    z_star         = best_o
    b_child, p_star = obs_info[z_star]

    # ── Expand tree ───────────────────────────────────────────────────────────
    if (a_star, z_star) not in node.children:
        child             = BeliefNode(b_child, t + 1, node, a_star, z_star)
        child.is_terminal = (t + 1 == H - 1)
        node.children[(a_star, z_star)] = child
    else:
        child        = node.children[(a_star, z_star)]
        child.belief = b_child   # keep in sync with updated belief

    visited.append((node, t))

    # ── Target passing ────────────────────────────────────────────────────────
    # Best Q lower over all actions
    best_ql = max(_q_lower(b, a, t, Gamma, transition, observations, reward, gamma)
                  for a in range(A))

    L_prime = max(L, best_ql)
    U_prime = max(U, best_ql + tol)

    # Decompose L_prime into child target:
    #   L' = R(b,a*) + gamma * [p(o*) * L_child + sum_{o≠o*} p(o) * V_lo(b_o)]
    r_star   = float(reward[a_star] @ b)
    sib_lo   = sum(
        p * _v_lower(bn, Gamma[t + 1])
        for z, (bn, p) in obs_info.items() if z != z_star
    )
    if p_star > 1e-12 and gamma > 1e-12:
        L_child = (L_prime - r_star - gamma * sib_lo) / (gamma * p_star)
    else:
        L_child = L_prime

    # Decompose U_prime into child target:
    #   U' = R(b,a*) + gamma * [p(o*) * U_child + sum_{o≠o*} p(o) * V_up(b_o)]
    sib_up = sum(
        p * sawtooth_eval(bn, v_mdp_ext[t + 1], saw_b[t + 1], saw_v[t + 1])
        for z, (bn, p) in obs_info.items() if z != z_star
    )
    if p_star > 1e-12 and gamma > 1e-12:
        U_child = (U_prime - r_star - gamma * sib_up) / (gamma * p_star)
    else:
        U_child = U_prime

    # ── Recurse ───────────────────────────────────────────────────────────────
    _sample_path(child, L_child, U_child, epsilon, H,
                 Gamma, saw_b, saw_v, v_mdp_ext,
                 transition, observations, reward, gamma, visited)


def sample_once(root, Gamma, saw_b, saw_v, v_mdp_ext, epsilon, H,
                transition, observations, reward, gamma):
    """One SAMPLE pass. Returns list of (node, t) visited root-to-leaf."""
    b0 = root.belief
    L  = _v_lower(b0, Gamma[0])
    U  = L + epsilon
    visited = []
    _sample_path(root, L, U, epsilon, H, Gamma, saw_b, saw_v, v_mdp_ext,
                 transition, observations, reward, gamma, visited)
    return visited


# ── BACKUP ─────────────────────────────────────────────────────────────────────

def backup_along_path(visited, H, Gamma, gamma_actions,
                      saw_b, saw_v, v_mdp_ext,
                      transition, observations, reward, gamma):
    """Bottom-up alpha-vector and sawtooth backup for each node in visited."""
    for node, t in reversed(visited):
        b = node.belief

        # Alpha-vector backup using Gamma[t+1]
        alpha_vec, act_idx = _pb_backup_single(
            b, Gamma[t + 1], transition, observations, reward, gamma
        )
        Gamma[t]          = np.vstack([Gamma[t], alpha_vec.reshape(1, -1)])
        gamma_actions[t]  = np.append(gamma_actions[t], act_idx)

        # Sawtooth backup using saw anchors and MDP bound at t+1
        saw_val = _saw_backup_single(
            b, v_mdp_ext[t + 1],
            saw_b[t + 1], saw_v[t + 1],
            transition, observations, reward, gamma,
        )
        saw_b[t] = np.vstack([saw_b[t], b.reshape(1, -1)])
        saw_v[t] = np.append(saw_v[t], saw_val)


# ── PRUNE ──────────────────────────────────────────────────────────────────────

def prune_all(Gamma, gamma_actions, H, prune_fn):
    """Apply prune_fn to Gamma[t] for every timestep."""
    for t in range(H):
        if len(Gamma[t]) > 1:
            Gamma[t], gamma_actions[t] = prune_fn(Gamma[t], gamma_actions[t])


# ── Warm-start ─────────────────────────────────────────────────────────────────

def warm_start(initial_belief, n_traj, H, S,
               Gamma, gamma_actions, saw_b, saw_v, v_mdp_ext,
               transition, observations, reward, gamma, prune_fn, rng):
    """One backward PBVI pass over random trajectories to seed the bounds."""
    beliefs_per_t = collect_beliefs(
        initial_belief, n_traj, H, transition, observations, rng
    )

    next_saw_b = np.zeros((0, S))
    next_saw_v = np.zeros(0)

    for t in range(H - 1, -1, -1):
        bset = beliefs_per_t[t]   # (n_b, S)

        # Alpha-vector backup
        alphas  = []
        actions = []
        for b in bset:
            av, ai = _pb_backup_single(
                b, Gamma[t + 1], transition, observations, reward, gamma
            )
            alphas.append(av)
            actions.append(ai)
        raw_a = np.vstack(alphas)
        raw_i = np.array(actions, dtype=int)
        Gamma[t], gamma_actions[t] = prune_fn(raw_a, raw_i)

        # Sawtooth backup
        saw_vals = np.array([
            _saw_backup_single(
                b, v_mdp_ext[t + 1], next_saw_b, next_saw_v,
                transition, observations, reward, gamma,
            )
            for b in bset
        ])
        saw_b[t]   = bset
        saw_v[t]   = saw_vals
        next_saw_b = bset
        next_saw_v = saw_vals

    return beliefs_per_t


# ── Main solver ────────────────────────────────────────────────────────────────

def finite_horizon_sarsop(
    models_path,
    n_timesteps,
    initial_belief,
    output_path,
    epsilon=1e-3,
    max_iterations=10_000,
    max_time_s=300,
    prune_every=50,
    prune_fn=None,
    gamma=1.0,
    enable_tree_pruning=False,   # placeholder; tree pruning not yet implemented
    rng_seed=0,
    warmup_trajectories=200,
    status_every=100,
):
    """Run finite-horizon SARSOP and save results.

    Parameters
    ----------
    models_path          : str   path to .npz POMDP model
    n_timesteps          : int   H (horizon length)
    initial_belief       : (S,) starting belief
    output_path          : str   where to write the .npz output
    epsilon              : float target gap V̄(b₀) - V(b₀) < epsilon
    max_iterations       : int   SAMPLE/BACKUP iteration cap
    max_time_s           : float wall-clock budget in seconds
    prune_every          : int   prune alpha-vectors every K iterations
    prune_fn             : callable or None  (vectors, actions) -> (vectors, actions)
    gamma                : float discount factor (1.0 = undiscounted)
    enable_tree_pruning  : bool  (not yet implemented, reserved for future)
    rng_seed             : int
    warmup_trajectories  : int   random trajectories for initial bounds (0 = skip)
    status_every         : int   print a status line every N iterations

    Output .npz keys (same as finite_horizon_pbvi.py)
    ─────────────────────────────────────────────────
    alpha_vectors    : object array (T,) — alpha_vectors[t] is (n_t, S)
    actions          : object array (T,) — actions[t] is (n_t,) int
    action_names     : (A,) str array
    upper_bound      : (T, S) float array — MDP relaxation upper-bound vectors
    sawtooth_beliefs : object array (T,) — anchor beliefs for sawtooth
    sawtooth_values  : object array (T,) — anchor values  for sawtooth
    """
    if prune_fn is None:
        prune_fn = lambda v, a: (v, a)   # no-op

    rng = np.random.default_rng(rng_seed)
    H   = n_timesteps

    # ── 1. Load model ─────────────────────────────────────────────────────────
    transition, observations, reward, action_names = load_models(models_path)
    A, S, _ = transition.shape
    Z        = observations.shape[1]

    print(f"Model: A={A}, S={S}, Z={Z}, H={H}, gamma={gamma}")
    print(f"  epsilon={epsilon}, max_iter={max_iterations}, "
          f"max_time={max_time_s}s, prune_every={prune_every}")

    # ── 2. MDP upper bound ────────────────────────────────────────────────────
    v_mdp_arr = compute_mdp_upper_bound(transition, reward, H)   # (H, S)
    # Extend to H+1 with terminal zeros for uniform indexing as v_mdp_ext[t]
    v_mdp_ext = np.vstack([v_mdp_arr, np.zeros((1, S))])         # (H+1, S)

    # ── 3. Initialise bounds ──────────────────────────────────────────────────
    # Index range: t = 0 … H  (H is the terminal "step after the last action")
    Gamma         = [np.zeros((1, S)) for _ in range(H + 1)]
    gamma_actions = [np.zeros(1, dtype=int) for _ in range(H + 1)]
    saw_b         = [np.zeros((0, S)) for _ in range(H + 1)]
    saw_v         = [np.zeros(0)      for _ in range(H + 1)]
    # Gamma[H], saw_b[H], saw_v[H] stay as zeros/empty (terminal value = 0).

    # ── 4. Warm-start ─────────────────────────────────────────────────────────
    if warmup_trajectories > 0:
        print(f"\nWarm-start: {warmup_trajectories} random trajectories …")
        warm_start(
            initial_belief, warmup_trajectories, H, S,
            Gamma, gamma_actions, saw_b, saw_v, v_mdp_ext,
            transition, observations, reward, gamma, prune_fn, rng,
        )
        b0       = initial_belief
        lo_ws    = _v_lower(b0, Gamma[0])
        up_ws    = sawtooth_eval(b0, v_mdp_ext[0], saw_b[0], saw_v[0])
        print(f"  lower(b0)={lo_ws:.4f}  upper(b0)={up_ws:.4f}  "
              f"gap={up_ws - lo_ws:.4f}")

    # ── 5. Belief tree root ───────────────────────────────────────────────────
    root             = BeliefNode(initial_belief, 0)
    root.is_terminal = (H == 1)

    # ── 6. Main SARSOP loop ───────────────────────────────────────────────────
    hdr = (f"\n{'Iter':>6}  {'Gap(b0)':>10}  {'Lower(b0)':>10}  "
           f"{'Upper(b0)':>10}  {'#alpha':>8}  {'#nodes':>7}  {'Time(s)':>8}")
    print(hdr)
    print("-" * 70)

    t_start = time.time()
    final_iter = 0

    for it in range(max_iterations):
        elapsed = time.time() - t_start

        # ── Current gap at b₀ ─────────────────────────────────────────────
        b0      = root.belief
        lo      = _v_lower(b0, Gamma[0])
        up      = sawtooth_eval(b0, v_mdp_ext[0], saw_b[0], saw_v[0])
        gap     = up - lo

        if it % status_every == 0:
            n_alpha = sum(len(Gamma[t]) for t in range(H))
            n_nodes = count_tree_nodes(root)
            print(f"{it:>6}  {gap:>10.4f}  {lo:>10.4f}  {up:>10.4f}  "
                  f"{n_alpha:>8}  {n_nodes:>7}  {elapsed:>8.1f}")

        # ── Termination ───────────────────────────────────────────────────
        if gap <= epsilon:
            print(f"\nConverged at iteration {it}: gap={gap:.6f} ≤ {epsilon}")
            final_iter = it
            break
        if elapsed >= max_time_s:
            print(f"\nTime budget reached at iteration {it}: gap={gap:.6f}")
            final_iter = it
            break

        # ── SAMPLE ───────────────────────────────────────────────────────
        visited = sample_once(
            root, Gamma, saw_b, saw_v, v_mdp_ext, epsilon, H,
            transition, observations, reward, gamma,
        )

        if not visited:
            # Sampler found nothing to improve; this can happen when the
            # tree is already exhausted at the current epsilon — continue
            # (a future iteration may open new paths after pruning)
            continue

        # ── BACKUP ───────────────────────────────────────────────────────
        backup_along_path(
            visited, H, Gamma, gamma_actions,
            saw_b, saw_v, v_mdp_ext,
            transition, observations, reward, gamma,
        )

        # ── PRUNE ────────────────────────────────────────────────────────
        if (it + 1) % prune_every == 0:
            prune_all(Gamma, gamma_actions, H, prune_fn)

    else:
        final_iter = max_iterations - 1

    # ── 7. Final status ───────────────────────────────────────────────────────
    b0      = root.belief
    lo      = _v_lower(b0, Gamma[0])
    up      = sawtooth_eval(b0, v_mdp_ext[0], saw_b[0], saw_v[0])
    gap     = up - lo
    elapsed = time.time() - t_start
    n_alpha = sum(len(Gamma[t]) for t in range(H))
    n_nodes = count_tree_nodes(root)

    # Final prune
    prune_all(Gamma, gamma_actions, H, prune_fn)
    n_alpha_pruned = sum(len(Gamma[t]) for t in range(H))

    print(f"\n{'─' * 70}")
    print(f"Iterations : {final_iter + 1}")
    print(f"Gap(b0)    : {gap:.6f}  (target: {epsilon})")
    print(f"Lower(b0)  : {lo:.6f}")
    print(f"Upper(b0)  : {up:.6f}")
    print(f"Alpha-vecs : {n_alpha} → {n_alpha_pruned} (after final prune)")
    print(f"Tree nodes : {n_nodes}")
    print(f"Wall time  : {elapsed:.1f}s")

    # ── 8. Save ───────────────────────────────────────────────────────────────
    alpha_arr = np.empty(H, dtype=object)
    acts_arr  = np.empty(H, dtype=object)
    saw_b_arr = np.empty(H, dtype=object)
    saw_v_arr = np.empty(H, dtype=object)
    for t in range(H):
        alpha_arr[t] = Gamma[t]
        acts_arr[t]  = gamma_actions[t]
        saw_b_arr[t] = saw_b[t]
        saw_v_arr[t] = saw_v[t]

    np.savez(
        output_path,
        alpha_vectors    = alpha_arr,
        actions          = acts_arr,
        action_names     = action_names,
        upper_bound      = v_mdp_arr,
        sawtooth_beliefs = saw_b_arr,
        sawtooth_values  = saw_v_arr,
    )
    print(f"Saved {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(script_dir, "model_fatigue.npz")
    output_path = os.path.join(
        script_dir, "alpha_vector_sarsop_fatigue.npz"
    )

    _data          = np.load(models_path, allow_pickle=True)
    initial_belief = _data["initial_belief"]

    finite_horizon_sarsop(
        models_path,
        n_timesteps=30,
        initial_belief=initial_belief,
        output_path=output_path,
        epsilon=1e-3,
        max_iterations=10_000,
        max_time_s=300,
        prune_every=50,
        prune_fn=lp_prune_certified,
        gamma=0.95,
        enable_tree_pruning=False,
        rng_seed=0,
        warmup_trajectories=200,
        status_every=100,
    )
