import os

import numpy as np
from scipy.optimize import linprog


def load_models(path):
    """Load POMDP arrays from a .npz file.

    Returns
    -------
    transition  : (A, S, S)
    observations: (A, Z, S)
    reward      : (A, S)
    action_names: (A,) str array
    """
    data = np.load(path, allow_pickle=True)
    return (
        data["transition"],
        data["observations"],
        data["reward"],
        data["action_names"],
    )


def pointwise_prune(vectors, actions):
    """Remove vectors pointwise dominated by any other in the set.

    v_i is removed if there exists v_j such that v_j[s] >= v_i[s] for all s
    and v_j[s] > v_i[s] for at least one s.  O(M^2 * S) but fully vectorised
    — much cheaper than an LP solve.
    """
    if len(vectors) <= 1:
        return vectors, actions

    keep = np.ones(len(vectors), dtype=bool)
    for i in range(len(vectors)):
        if not keep[i]:
            continue
        diff = vectors - vectors[i]                          # (M, S)
        dominated = np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
        dominated[i] = False
        if dominated.any():
            keep[i] = False

    return vectors[keep], actions[keep]


def lp_prune(vectors, actions, tol=1e-8):
    """LP-based upper-convex-hull pruning.

    Applies pointwise_prune as a fast pre-filter, then for each remaining
    candidate v_i solves:
        maximize  ε
        s.t.  b · (v_i − v_j) ≥ ε   ∀ j ≠ i
              Σ b_k = 1,  b_k ≥ 0

    Keep v_i only if the optimal ε > tol.  One sequential pass suffices.
    """
    vectors, actions = pointwise_prune(vectors, actions)

    S = vectors.shape[1]
    survivors = list(range(len(vectors)))
    i = 0

    while i < len(survivors):
        idx_i = survivors[i]
        vi = vectors[idx_i]

        others_idx = [survivors[j] for j in range(len(survivors)) if j != i]

        if len(others_idx) == 0:
            i += 1
            continue

        n_others = len(others_idx)

        # Variables: x = [b_0, ..., b_{S-1}, ε]
        # Minimize -ε  →  c = [0, ..., 0, -1]
        c = np.zeros(S + 1)
        c[-1] = -1.0

        # (v_j - v_i) · b + ε ≤ 0  for each j ≠ i
        A_ub = np.zeros((n_others, S + 1))
        for k, idx_j in enumerate(others_idx):
            A_ub[k, :S] = vectors[idx_j] - vi
            A_ub[k, S] = 1.0
        b_ub = np.zeros(n_others)

        # sum(b) = 1
        A_eq = np.zeros((1, S + 1))
        A_eq[0, :S] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, None)] * S + [(None, None)]

        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
        )

        # Keep if strictly optimal somewhere (ε = −result.fun > tol)
        if result.status == 0 and -result.fun > tol:
            i += 1
        else:
            survivors.pop(i)

    return vectors[survivors], actions[survivors]


def backup(prev_vectors, transition, observations, reward):
    """One backward-induction step with incremental pruning.

    Instead of enumerating all A * N^Z combinations up front, processes one
    observation at a time and prunes after each cross-product step:

        for each action a:
            set ← {R[a] + T[a] @ (O[a,0] ⊙ γ)  :  γ ∈ Γ}   # N vectors
            for z = 1 … Z-1:
                set ← lp_prune({v + T[a] @ (O[a,z] ⊙ γ)  :  v ∈ set, γ ∈ Γ})

    The pruned per-action sets are unioned and returned for a final cross-action
    lp_prune in backward_induction.

    Parameters
    ----------
    prev_vectors : (N, S)
    transition   : (A, S, S)
    observations : (A, Z, S)
    reward       : (A, S)

    Returns
    -------
    vectors : (M, S)   partially pruned candidates (one set per action, unioned)
    actions : (M,)     integer action indices
    """
    A, S, _ = transition.shape
    Z = observations.shape[1]

    all_vectors = []
    all_actions = []

    for a in range(A):
        # contrib[z] shape (N, S): T[a] @ (O[a,z] ⊙ γ_n) for each n
        # = (O[a,z] * prev_vectors) @ T[a].T  (vectorised over N)
        contrib = [
            (observations[a, z] * prev_vectors) @ transition[a].T
            for z in range(Z)
        ]

        # Initialise with z=0
        N = len(prev_vectors)
        intermediate = reward[a] + contrib[0]           # (N, S)
        int_actions  = np.full(N, a, dtype=int)
        intermediate, int_actions = lp_prune(intermediate, int_actions)

        # Cross-product + prune for z = 1 … Z-1
        for z in range(1, Z):
            M = len(intermediate)
            # All (M * N) pairwise sums, shape (M*N, S)
            candidates = (
                intermediate[:, np.newaxis, :] + contrib[z][np.newaxis, :, :]
            ).reshape(M * len(prev_vectors), S)
            cand_actions = np.full(len(candidates), a, dtype=int)
            intermediate, int_actions = lp_prune(candidates, cand_actions)

        all_vectors.append(intermediate)
        all_actions.append(int_actions)

    return np.vstack(all_vectors), np.concatenate(all_actions)


def backward_induction(models_path, n_timesteps, output_path):
    """Run full backward induction and save results.

    Initialises with a single zero terminal vector, then iterates from
    t = T−1 down to t = 0, calling backup + lp_prune at each step.

    Output .npz keys
    ----------------
    alpha_vectors : object array (T,) — alpha_vectors[t] is (n_t, S)
    actions       : object array (T,) — actions[t] is (n_t,) int indices
    action_names  : (A,) str array

    Loading example
    ---------------
    data         = np.load('alpha_vectors.npz', allow_pickle=True)
    alpha        = data['alpha_vectors']   # shape (T,), each element (n_t, S)
    actions      = data['actions']         # shape (T,), each element (n_t,)
    action_names = data['action_names']

    # Evaluate policy at timestep t for belief b:
    values = alpha[t] @ b
    action = action_names[actions[t][np.argmax(values)]]
    """
    transition, observations, reward, action_names = load_models(models_path)
    S = transition.shape[1]

    gamma = np.zeros((1, S))  # terminal: single zero vector

    alpha_vectors_list = [None] * n_timesteps
    actions_list       = [None] * n_timesteps

    for t in range(n_timesteps - 1, -1, -1):
        raw_vectors, raw_actions = backup(gamma, transition, observations, reward)
        gamma, gamma_actions = lp_prune(raw_vectors, raw_actions)
        alpha_vectors_list[t] = gamma
        actions_list[t]       = gamma_actions
        print(f"t={t}: {len(gamma)} alpha-vectors after LP pruning")

    alpha_arr = np.empty(n_timesteps, dtype=object)
    acts_arr  = np.empty(n_timesteps, dtype=object)
    for t in range(n_timesteps):
        alpha_arr[t] = alpha_vectors_list[t]
        acts_arr[t]  = actions_list[t]

    np.savez(output_path, alpha_vectors=alpha_arr, actions=acts_arr,
             action_names=action_names)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(script_dir, "model_simplified_fatigue.npz")
    output_path = os.path.join(script_dir, "alpha_vector_simplified_fatigue.npz")
    backward_induction(models_path, n_timesteps=20, output_path=output_path)
