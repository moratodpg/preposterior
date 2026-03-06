"""Finite-Horizon Point-Based Value Iteration (PBVI).

Approximates the finite-horizon POMDP value function over a sampled set of
reachable belief points.  Alpha-vector count is bounded by the belief set size
(n_trajectories) rather than growing exponentially as in exact backward
induction.

Output format is identical to backward_induction.py.
"""

import os

import numpy as np

from backward_induction import (
    load_models,
    lp_prune_certified,
    lp_prune_sequential,
    pointwise_prune,
)


# ── Belief update ──────────────────────────────────────────────────────────────

def belief_update(b, a, z, transition, observations, eps=1e-12):
    """Bayesian belief update after action *a* and observation *z*.

    Parameters
    ----------
    b            : (S,) current belief
    a            : int  action index
    z            : int  observation index
    transition   : (A, S, S)
    observations : (A, Z, S)  — O[a, z, s'] = P(z | a, s')

    Returns
    -------
    b_new : (S,) updated belief (equals *b* if observation is impossible)
    prob  : float  P(z | b, a)
    """
    b_propagated = transition[a].T @ b          # (S,)  P(s' | b, a)
    b_unnorm     = observations[a, z] * b_propagated
    prob         = b_unnorm.sum()
    if prob < eps:
        return b, 0.0
    return b_unnorm / prob, float(prob)


# ── Belief collection (forward pass) ──────────────────────────────────────────

def collect_beliefs(b0, n_trajectories, n_timesteps, transition, observations, rng):
    """Simulate forward trajectories to collect reachable belief points.

    Parameters
    ----------
    b0             : (S,) initial belief
    n_trajectories : int
    n_timesteps    : int  T (total horizon length)
    transition     : (A, S, S)
    observations   : (A, Z, S)
    rng            : np.random.Generator

    Returns
    -------
    beliefs_per_t : list of length T, each element is an ndarray (n_t, S)
                    with the deduplicated belief points reached at that step,
                    including t = T-1 (the beliefs at the last decision point).
    """
    A = transition.shape[0]
    Z = observations.shape[1]

    # Each trajectory carries one current belief vector
    trajs = np.tile(b0, (n_trajectories, 1))   # (n_trajectories, S)

    beliefs_per_t = [None] * n_timesteps

    for t in range(n_timesteps - 1):
        # Store current beliefs (deduplicated)
        beliefs_per_t[t] = np.unique(trajs, axis=0)

        # Advance each trajectory
        actions = rng.integers(0, A, size=n_trajectories)
        new_trajs = np.empty_like(trajs)

        for i in range(n_trajectories):
            a = int(actions[i])
            b = trajs[i]

            # Sample observation: P(z | b, a)
            b_prop  = transition[a].T @ b              # (S,)
            obs_probs = (observations[a] * b_prop).sum(axis=1)   # (Z,)
            obs_sum   = obs_probs.sum()
            if obs_sum < 1e-12:
                obs_probs = np.ones(Z) / Z
            else:
                obs_probs /= obs_sum

            z = int(rng.choice(Z, p=obs_probs))
            b_new, prob = belief_update(b, a, z, transition, observations)
            new_trajs[i] = b_new

        trajs = new_trajs

    # Store the actual beliefs reached at t = T-1.
    # These are still needed as backup targets; the zero terminal *value* is
    # handled separately by initialising gamma = zeros in the solver.
    beliefs_per_t[n_timesteps - 1] = np.unique(trajs, axis=0)

    return beliefs_per_t


def collect_beliefs_greedy(b0, n_trajectories, n_timesteps, transition,
                           observations, alpha_vectors, actions_per_t, rng):
    """Collect reachable beliefs by following the current greedy policy.

    At each timestep t the action is chosen as

        a = actions_per_t[t][argmax(alpha_vectors[t] @ b)]

    i.e. the action attached to the alpha-vector that dominates at belief b.
    Observations are still sampled stochastically from the model, so the
    resulting trajectories cover the subtree of beliefs reachable under the
    current policy.

    Parameters
    ----------
    alpha_vectors  : list of length T, each (n_t, S)
    actions_per_t  : list of length T, each (n_t,) int

    Returns
    -------
    beliefs_per_t  : list of length T, each (n_t, S)  (deduplicated)
    """
    Z = observations.shape[1]
    trajs = np.tile(b0, (n_trajectories, 1))
    beliefs_per_t = [None] * n_timesteps

    for t in range(n_timesteps - 1):
        beliefs_per_t[t] = np.unique(trajs, axis=0)
        new_trajs = np.empty_like(trajs)

        alpha_t   = alpha_vectors[t]    # (n_t, S)
        actions_t = actions_per_t[t]    # (n_t,)

        for i in range(n_trajectories):
            b = trajs[i]
            # Greedy action: pick the alpha-vector that maximises value at b,
            # then read off the action stored with it.
            best_idx = int(np.argmax(alpha_t @ b))
            a        = int(actions_t[best_idx])

            b_prop    = transition[a].T @ b
            obs_probs = (observations[a] * b_prop).sum(axis=1)
            obs_sum   = obs_probs.sum()
            if obs_sum < 1e-12:
                obs_probs = np.ones(Z) / Z
            else:
                obs_probs /= obs_sum

            z            = int(rng.choice(Z, p=obs_probs))
            b_new, _     = belief_update(b, a, z, transition, observations)
            new_trajs[i] = b_new

        trajs = new_trajs

    beliefs_per_t[n_timesteps - 1] = np.unique(trajs, axis=0)
    return beliefs_per_t


# ── MDP upper bound ───────────────────────────────────────────────────────────

def compute_mdp_upper_bound(transition, reward, n_timesteps):
    """MDP relaxation upper bound on V*(b, t).

    Solves the fully-observable MDP (state known perfectly).  Since knowing
    the state can only help, V_MDP(s,t) ≥ V*(b,t) component-wise, so

        U(b, t) = upper_bound[t] @ b  ≥  V*(b, t)  for all beliefs b.

    Parameters
    ----------
    transition  : (A, S, S)
    reward      : (A, S)
    n_timesteps : int

    Returns
    -------
    upper_bound : (T, S)  one value vector per timestep;
                  upper_bound[t] @ b is the upper bound at time t for belief b.
    """
    A, S, _ = transition.shape
    upper_bound = np.zeros((n_timesteps, S))
    v_next      = np.zeros(S)                       # terminal value = 0

    for t in range(n_timesteps - 1, -1, -1):
        # Q(a, s) = R(a, s) + sum_{s'} T(a, s, s') * v_next(s')
        q              = reward + transition @ v_next   # (A, S, S) @ (S,) → (A, S)
        upper_bound[t] = q.max(axis=0)                 # (S,)
        v_next         = upper_bound[t]

    return upper_bound


# ── Sawtooth upper bound ──────────────────────────────────────────────────────

def sawtooth_eval(b, v_mdp_t, anchor_beliefs, anchor_values):
    """Evaluate the sawtooth upper bound at a single belief b.

    For each anchor (bᵢ, uᵢ) we form the piecewise-linear function

        φᵢ(b) = U_MDP(b) + λᵢ(b) · (uᵢ − U_MDP(bᵢ))

    where  λᵢ(b) = min_{s : bᵢ[s]>0} ( b[s] / bᵢ[s] )  is the largest
    scalar such that b − λ·bᵢ ≥ 0 component-wise.  The combined bound is

        U_saw(b) = min( U_MDP(b) , min_i φᵢ(b) )

    Derivation: any belief b can be written as λ·bᵢ + (1−λ)·bₑ for a
    boundary belief bₑ.  Because V* is convex and U_MDP ≥ V* everywhere,
        V*(b) ≤ λ·V*(bᵢ) + (1−λ)·V*(bₑ)
              ≤ λ·uᵢ + (1−λ)·U_MDP(bₑ)
               = U_MDP(b) + λ·(uᵢ − U_MDP(bᵢ))   (linearity of U_MDP)

    Parameters
    ----------
    b              : (S,)
    v_mdp_t        : (S,)  MDP value vector for the current timestep
    anchor_beliefs : (n, S)
    anchor_values  : (n,)  upper-bound values at the anchor beliefs; must
                           satisfy anchor_values ≤ v_mdp_t @ anchor_beliefs

    Returns
    -------
    float : upper bound on V*(b, t)
    """
    u_mdp = float(v_mdp_t @ b)
    if len(anchor_beliefs) == 0:
        return u_mdp

    # corrections[i] = uᵢ − U_MDP(bᵢ)  ≤ 0  (sawtooth never exceeds MDP)
    corrections = anchor_values - (anchor_beliefs @ v_mdp_t)  # (n,)

    # λᵢ(b) = min_{s: bᵢ[s]>0} b[s]/bᵢ[s]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(anchor_beliefs > 0,
                          b[np.newaxis, :] / anchor_beliefs, np.inf)  # (n, S)
    min_ratios = ratios.min(axis=1)   # (n,)

    phi = u_mdp + corrections * min_ratios   # (n,)  ≤ u_mdp
    return float(min(u_mdp, phi.min()))


def sawtooth_backup(beliefs, v_mdp_next, next_anchor_beliefs, next_anchor_values,
                    transition, observations, reward):
    """Compute sawtooth upper-bound values for a set of belief points.

    For each belief b, evaluates

        u(b, t) = max_a [ R(b, a) + Σ_z P(z|b,a) · U_saw(b'(b,a,z), t+1) ]

    where U_saw(·, t+1) is the sawtooth bound at the *next* timestep.
    Because U_saw(·, t+1) ≥ V*(·, t+1), the result is ≥ V*(b, t) (upper bound).

    Parameters
    ----------
    beliefs             : (n_b, S)
    v_mdp_next          : (S,)  MDP value vector for t+1  (zeros at terminal)
    next_anchor_beliefs : (n, S)  anchor beliefs for t+1  (empty at terminal)
    next_anchor_values  : (n,)  anchor values for t+1     (empty at terminal)
    transition          : (A, S, S)
    observations        : (A, Z, S)
    reward              : (A, S)

    Returns
    -------
    values : (n_b,)  upper-bound values, one per belief
    """
    A = transition.shape[0]
    Z = observations.shape[1]
    values = np.empty(len(beliefs))

    for i, b in enumerate(beliefs):
        best_u = -np.inf
        for a in range(A):
            b_prop = transition[a].T @ b       # (S,)
            u_a    = float(reward[a] @ b)
            for z in range(Z):
                w    = observations[a, z] * b_prop   # (S,)
                prob = w.sum()
                if prob < 1e-12:
                    continue
                b_next = w / prob
                u_a += prob * sawtooth_eval(
                    b_next, v_mdp_next,
                    next_anchor_beliefs, next_anchor_values,
                )
            best_u = max(best_u, u_a)
        values[i] = best_u

    return values


# ── Point-based backup ────────────────────────────────────────────────────────

def point_based_backup_single(b, prev_alpha, transition, observations, reward):
    """PBVI backup for a single belief point.

    Parameters
    ----------
    b          : (S,) belief
    prev_alpha : (N, S) alpha-vectors from the next timestep
    transition : (A, S, S)
    observations: (A, Z, S)
    reward     : (A, S)

    Returns
    -------
    alpha_vec  : (S,) best alpha-vector for this belief
    action_idx : int  corresponding action index
    """
    A = transition.shape[0]
    Z = observations.shape[1]

    best_val   = -np.inf
    best_alpha = None
    best_action = 0

    for a in range(A):
        b_propagated = transition[a].T @ b      # (S,)
        phi_a = reward[a].copy()                # (S,)

        for z in range(Z):
            w        = observations[a, z] * b_propagated   # (S,)
            # which alpha-vector maximises the expected value for this (a, z)?
            scores   = prev_alpha @ w                       # (N,)
            best_idx = int(np.argmax(scores))
            # contribution to phi_a (over all states s)
            phi_a += transition[a] @ (observations[a, z] * prev_alpha[best_idx])

        val = float(phi_a @ b)
        if val > best_val:
            best_val    = val
            best_alpha  = phi_a
            best_action = a

    return best_alpha, best_action


def point_based_backup(beliefs, prev_alpha, prev_alpha_actions,
                       transition, observations, reward):
    """Vectorised wrapper: run point_based_backup_single for each belief.

    Parameters
    ----------
    beliefs           : (n_b, S)
    prev_alpha        : (N, S) alpha-vectors from t+1
    prev_alpha_actions: (N,)  action indices (not used in backup, kept for
                              API symmetry)

    Returns
    -------
    alpha_vectors : (n_b, S)
    action_indices: (n_b,) int
    """
    alphas  = []
    actions = []
    for b in beliefs:
        alpha_vec, action_idx = point_based_backup_single(
            b, prev_alpha, transition, observations, reward
        )
        alphas.append(alpha_vec)
        actions.append(action_idx)

    return np.vstack(alphas), np.array(actions, dtype=int)


# ── Main solver ───────────────────────────────────────────────────────────────

def finite_horizon_pbvi(
    models_path,
    n_timesteps,
    initial_belief,
    output_path,
    n_trajectories=500,
    n_iterations=1,
    prune_fn=None,
    rng_seed=0,
):
    """Run finite-horizon PBVI and save results.

    Parameters
    ----------
    models_path    : str   path to .npz POMDP model
    n_timesteps    : int   T (horizon length)
    initial_belief : (S,) starting belief for trajectory collection
    output_path    : str   where to write the .npz output
    n_trajectories : int   number of forward trajectories per iteration
    n_iterations   : int   number of PBVI iterations (default 1).
                     Iteration 1 uses random trajectories.  Each subsequent
                     iteration follows the greedy policy from the previous
                     backward pass and merges the new beliefs into the existing
                     sets — so belief sets grow monotonically and the lower
                     bound can only improve.
    prune_fn       : callable or None
                     Signature: (vectors, actions) -> (vectors, actions).
                     Use lp_prune_certified, lp_prune_sequential, or
                     pointwise_prune.  Pass None to skip pruning.
    rng_seed       : int

    Output .npz keys
    ----------------
    alpha_vectors     : object array (T,) — alpha_vectors[t] is (n_t, S)
    actions           : object array (T,) — actions[t] is (n_t,) int indices
    action_names      : (A,) str array
    upper_bound       : (T, S) float array — MDP relaxation upper bound vectors
    sawtooth_beliefs  : object array (T,) — anchor beliefs for sawtooth; [t] is (n_t, S)
    sawtooth_values   : object array (T,) — anchor values  for sawtooth; [t] is (n_t,)

    Loading example
    ---------------
    data         = np.load('output.npz', allow_pickle=True)
    alpha        = data['alpha_vectors']      # (T,), each (n_t, S)
    actions      = data['actions']            # (T,), each (n_t,)
    action_names = data['action_names']
    ub           = data['upper_bound']        # (T, S)  — MDP bound vectors
    saw_b        = data['sawtooth_beliefs']   # (T,), each (n_t, S)
    saw_v        = data['sawtooth_values']    # (T,), each (n_t,)

    # Evaluate bounds at timestep t for belief b:
    from finite_horizon_pbvi import sawtooth_eval
    lower     = (alpha[t] @ b).max()                      # PBVI lower bound
    upper_mdp = ub[t] @ b                                 # MDP upper bound
    upper_saw = sawtooth_eval(b, ub[t], saw_b[t], saw_v[t])  # sawtooth (tighter)
    gap       = upper_saw - lower
    action    = action_names[actions[t][np.argmax(alpha[t] @ b)]]
    """
    if prune_fn is None:
        prune_fn = lambda v, a: (v, a)   # identity — no pruning

    rng = np.random.default_rng(rng_seed)

    # 1. Load model
    transition, observations, reward, action_names = load_models(models_path)
    S = transition.shape[1]

    # 2. MDP relaxation upper bound (fully-observable DP — very fast)
    upper_bound = compute_mdp_upper_bound(transition, reward, n_timesteps)

    # 3. Initial belief collection — random trajectories
    print(f"[iter 1/{n_iterations}] Collecting beliefs: "
          f"{n_trajectories} random trajectories × {n_timesteps} steps …")
    beliefs_per_t = collect_beliefs(
        initial_belief, n_trajectories, n_timesteps, transition, observations, rng
    )

    alpha_vectors_list = [None] * n_timesteps
    actions_list       = [None] * n_timesteps
    saw_beliefs_list   = [None] * n_timesteps
    saw_values_list    = [None] * n_timesteps

    cw  = 11   # column width for value columns
    hdr = (f"{'t':>4}  {'#alpha':>7}  {'#beliefs':>9}  "
           f"{'lower(b0)':>{cw}}  {'saw_ub(b0)':>{cw}}  "
           f"{'mdp_ub(b0)':>{cw}}  {'gap_saw':>9}")

    # 4. Iterative backward passes with belief expansion
    for iteration in range(n_iterations):
        if iteration > 0:
            # Expand belief sets by following the current greedy policy
            print(f"\n[iter {iteration + 1}/{n_iterations}] "
                  f"Expanding beliefs: {n_trajectories} greedy trajectories …")
            new_beliefs = collect_beliefs_greedy(
                initial_belief, n_trajectories, n_timesteps,
                transition, observations,
                alpha_vectors_list, actions_list, rng,
            )
            for t in range(n_timesteps):
                beliefs_per_t[t] = np.unique(
                    np.vstack([beliefs_per_t[t], new_beliefs[t]]), axis=0
                )

        # Backward pass — reset both bounds from the terminal each iteration
        gamma            = np.zeros((1, S))
        gamma_actions    = np.zeros(1, dtype=int)
        next_saw_beliefs = np.zeros((0, S))   # no anchors at terminal
        next_saw_values  = np.zeros(0)
        next_v_mdp       = np.zeros(S)        # terminal value = 0

        print(hdr)
        for t in range(n_timesteps - 1, -1, -1):
            # ── Lower bound (alpha-vectors) ──────────────────────────────────
            raw_alpha, raw_actions = point_based_backup(
                beliefs_per_t[t], gamma, gamma_actions,
                transition, observations, reward,
            )
            gamma, gamma_actions  = prune_fn(raw_alpha, raw_actions)
            alpha_vectors_list[t] = gamma
            actions_list[t]       = gamma_actions

            # ── Upper bound (sawtooth) ───────────────────────────────────────
            saw_vals = sawtooth_backup(
                beliefs_per_t[t], next_v_mdp,
                next_saw_beliefs, next_saw_values,
                transition, observations, reward,
            )
            saw_beliefs_list[t] = beliefs_per_t[t]
            saw_values_list[t]  = saw_vals

            # Advance sawtooth state for t-1
            next_saw_beliefs = beliefs_per_t[t]
            next_saw_values  = saw_vals
            next_v_mdp       = upper_bound[t]

            # ── Report ───────────────────────────────────────────────────────
            lower     = float((gamma @ initial_belief).max())
            upper_mdp = float(upper_bound[t] @ initial_belief)
            upper_saw = sawtooth_eval(initial_belief, upper_bound[t],
                                      saw_beliefs_list[t], saw_values_list[t])
            print(f"{t:>4}  {len(gamma):>7}  {len(beliefs_per_t[t]):>9}  "
                  f"{lower:>{cw}.4f}  {upper_saw:>{cw}.4f}  "
                  f"{upper_mdp:>{cw}.4f}  {upper_saw - lower:>9.4f}")

    # 5. Pack into object arrays and save
    alpha_arr = np.empty(n_timesteps, dtype=object)
    acts_arr  = np.empty(n_timesteps, dtype=object)
    saw_b_arr = np.empty(n_timesteps, dtype=object)
    saw_v_arr = np.empty(n_timesteps, dtype=object)
    for t in range(n_timesteps):
        alpha_arr[t] = alpha_vectors_list[t]
        acts_arr[t]  = actions_list[t]
        saw_b_arr[t] = saw_beliefs_list[t]
        saw_v_arr[t] = saw_values_list[t]

    np.savez(output_path,
             alpha_vectors=alpha_arr, actions=acts_arr,
             action_names=action_names, upper_bound=upper_bound,
             sawtooth_beliefs=saw_b_arr, sawtooth_values=saw_v_arr)
    print(f"Saved {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(script_dir, "model_simplified_fatigue.npz")
    output_path = os.path.join(script_dir, "alpha_vector_pbvi_simplified_fatigue.npz")

    _data          = np.load(models_path, allow_pickle=True)
    initial_belief = _data["initial_belief"]

    n_trajectories = 10_000

    # Pruning options:
    #   lp_prune_certified  — exact, fastest for large sets (recommended)
    #   lp_prune_sequential — exact, simpler
    #   pointwise_prune     — fast but incomplete (removes obvious duplicates)
    #   None                — no pruning (keeps all n_trajectories vectors)
    prune_fn = lp_prune_certified

    finite_horizon_pbvi(
        models_path,
        n_timesteps=10,
        initial_belief=initial_belief,
        output_path=output_path,
        n_trajectories=n_trajectories,
        n_iterations=5,
        prune_fn=prune_fn,
        rng_seed=0,
    )
