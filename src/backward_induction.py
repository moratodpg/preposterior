import itertools
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


def lp_prune(vectors, actions, tol=1e-8):
    """LP-based upper-convex-hull pruning.

    For each candidate vector v_i, solve:
        maximize  ε
        s.t.  b · (v_i − v_j) ≥ ε   ∀ j ≠ i
              Σ b_k = 1,  b_k ≥ 0

    Keep v_i only if the optimal ε > tol.
    One sequential pass suffices.
    """
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
    """One backward-induction step.

    For every action a and every combination c ∈ {0,…,|Γ|−1}^Z:
        α[s] = R[a, s] + Σ_z  (T[a] @ (O[a, z] ⊙ Γ[c_z]))[s]

    Parameters
    ----------
    prev_vectors : (N, S)  alpha-vectors from the next timestep
    transition   : (A, S, S)
    observations : (A, Z, S)
    reward       : (A, S)

    Returns
    -------
    raw_vectors : (A * N^Z, S)
    raw_actions : (A * N^Z,)  integer action indices
    """
    A = transition.shape[0]
    Z = observations.shape[1]
    N = prev_vectors.shape[0]

    raw_vectors = []
    raw_actions = []

    for a in range(A):
        for c in itertools.product(range(N), repeat=Z):
            alpha = reward[a].astype(float).copy()
            for z in range(Z):
                alpha += transition[a] @ (observations[a, z] * prev_vectors[c[z]])
            raw_vectors.append(alpha)
            raw_actions.append(a)

    return np.array(raw_vectors), np.array(raw_actions)


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
    models_path = os.path.join(script_dir, "models.npz")
    output_path = os.path.join(script_dir, "alpha_vectors.npz")
    backward_induction(models_path, n_timesteps=2, output_path=output_path)
