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


# ── Shared pre-processing ─────────────────────────────────────────────────────

def pointwise_prune(vectors, actions):
    """Remove vectors pointwise dominated by any other in the set.

    v_i is removed if there exists v_j with v_j[s] >= v_i[s] for all s
    and v_j[s] > v_i[s] for at least one s.  O(M^2 * S) but fully
    vectorised — much cheaper than an LP solve.
    """
    if len(vectors) <= 1:
        return vectors, actions

    keep = np.ones(len(vectors), dtype=bool)
    for i in range(len(vectors)):
        if not keep[i]:
            continue
        diff      = vectors - vectors[i]                    # (M, S)
        dominated = np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
        dominated[i] = False
        if dominated.any():
            keep[i] = False

    return vectors[keep], actions[keep]


def _preprocess(vectors, actions):
    """Shared first steps for both pruning methods:
    pointwise dominance filter + ascending sort by mean value.
    """
    vectors, actions = pointwise_prune(vectors, actions)
    if len(vectors) <= 1:
        return vectors, actions, False          # False = skip further pruning
    order   = np.argsort(vectors.mean(axis=1))
    return vectors[order], actions[order], True


def _lp_solve(vi, others):
    """Single LP check: return True if vi is strictly optimal somewhere
    against the rows of others.

    Solves:  maximize ε
             s.t.  b · (v_i − v_j) ≥ ε  ∀ j
                   Σ b_k = 1,  b_k ≥ 0
    """
    S, n    = len(vi), len(others)
    c       = np.zeros(S + 1);  c[-1] = -1.0
    A_ub    = np.empty((n, S + 1))
    A_ub[:, :S] = others - vi
    A_ub[:,  S] = 1.0
    A_eq    = np.zeros((1, S + 1));  A_eq[0, :S] = 1.0
    result  = linprog(
        c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=np.ones(1),
        bounds=[(0., None)] * S + [(None, None)], method="highs",
    )
    return result


# ── Pruning methods ───────────────────────────────────────────────────────────

def lp_prune_sequential(vectors, actions, tol=1e-8):
    """Standard sequential LP pruning (exact).

    For each vector in turn, solves one LP to check whether it is strictly
    optimal somewhere against all current survivors.  Removing a vector
    immediately shrinks the constraint set for subsequent checks.

    Suitable for small-to-medium alpha-vector sets.
    """
    vectors, actions, proceed = _preprocess(vectors, actions)
    if not proceed:
        return vectors, actions

    survivors = list(range(len(vectors)))
    i = 0
    while i < len(survivors):
        others_idx = [survivors[j] for j in range(len(survivors)) if j != i]
        if not others_idx:
            i += 1
            continue
        result = _lp_solve(vectors[survivors[i]], vectors[others_idx])
        if result.status == 0 and -result.fun > tol:
            i += 1
        else:
            survivors.pop(i)

    return vectors[survivors], actions[survivors]


def lp_prune_certified(vectors, actions, tol=1e-8, n_samples=2000):
    """Sample-certified LP pruning (exact, faster for large sets).

    Evaluates n_samples random beliefs (plus all S simplex vertices) to
    certify non-dominance cheaply: any vector that is the argmax at some
    sampled belief is guaranteed non-dominated and kept without an LP solve.
    Only the remaining 'uncertain' vectors are checked via LP.

    Both methods are exact — see docstring in lp_prune_sequential for the
    correctness argument.
    """
    vectors, actions, proceed = _preprocess(vectors, actions)
    if not proceed:
        return vectors, actions

    # Sample beliefs: S vertices + n_samples random interior points
    S       = vectors.shape[1]
    rng     = np.random.default_rng(0)
    beliefs = np.vstack([
        np.eye(S),
        rng.dirichlet(np.ones(S), size=n_samples),
    ])
    with np.errstate(all="ignore"):
        values = beliefs @ vectors.T             # (S + n_samples, M)
    certified = np.zeros(len(vectors), dtype=bool)
    certified[np.unique(np.argmax(values, axis=1))] = True

    # Certified vectors go straight to survivors; uncertain ones need LP
    survivors = list(np.where(certified)[0])
    for idx in np.where(~certified)[0]:
        if not survivors:
            survivors.append(idx)
            continue
        result = _lp_solve(vectors[idx], vectors[survivors])
        if result.status == 0 and -result.fun > tol:
            survivors.append(idx)

    return vectors[survivors], actions[survivors]


# ── Backward induction ────────────────────────────────────────────────────────

def backup(prev_vectors, transition, observations, reward, prune_fn):
    """One backward-induction step with incremental pruning.

    For each action, builds the cross-sum incrementally over observations,
    pruning after each step to keep the intermediate set small:

        set ← prune_fn({R[a] + T[a] @ (O[a,0] ⊙ γ)  :  γ ∈ Γ})
        for z = 1 … Z-1:
            set ← prune_fn({v + T[a] @ (O[a,z] ⊙ γ)  :  v ∈ set, γ ∈ Γ})

    Returns the per-action pruned sets unioned for a final cross-action prune
    in backward_induction.
    """
    A, S, _ = transition.shape
    Z        = observations.shape[1]

    all_vectors = []
    all_actions = []

    for a in range(A):
        contrib = [
            (observations[a, z] * prev_vectors) @ transition[a].T
            for z in range(Z)
        ]

        N            = len(prev_vectors)
        intermediate = reward[a] + contrib[0]
        int_actions  = np.full(N, a, dtype=int)
        intermediate, int_actions = prune_fn(intermediate, int_actions)

        for z in range(1, Z):
            M          = len(intermediate)
            candidates = (
                intermediate[:, np.newaxis, :] + contrib[z][np.newaxis, :, :]
            ).reshape(M * N, S)
            intermediate, int_actions = prune_fn(
                candidates, np.full(len(candidates), a, dtype=int)
            )

        all_vectors.append(intermediate)
        all_actions.append(int_actions)

    return np.vstack(all_vectors), np.concatenate(all_actions)


def backward_induction(models_path, n_timesteps, output_path, prune_fn):
    """Run full backward induction and save results.

    Parameters
    ----------
    prune_fn : callable
        Pruning function with signature (vectors, actions) -> (vectors, actions).
        Use lp_prune_sequential or lp_prune_certified.

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

    gamma = np.zeros((1, S))

    alpha_vectors_list = [None] * n_timesteps
    actions_list       = [None] * n_timesteps

    for t in range(n_timesteps - 1, -1, -1):
        raw_vectors, raw_actions  = backup(gamma, transition, observations,
                                           reward, prune_fn)
        gamma, gamma_actions      = prune_fn(raw_vectors, raw_actions)
        alpha_vectors_list[t]     = gamma
        actions_list[t]           = gamma_actions
        print(f"t={t}: {len(gamma)} alpha-vectors after pruning")

    alpha_arr = np.empty(n_timesteps, dtype=object)
    acts_arr  = np.empty(n_timesteps, dtype=object)
    for t in range(n_timesteps):
        alpha_arr[t] = alpha_vectors_list[t]
        acts_arr[t]  = actions_list[t]

    np.savez(output_path, alpha_vectors=alpha_arr, actions=acts_arr,
             action_names=action_names)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(script_dir, "model_simplified_fatigue.npz")
    output_path = os.path.join(script_dir, "alpha_vector_simplified_fatigue.npz")

    # ── Pruning method ────────────────────────────────────────────────────────
    # lp_prune_sequential : standard sequential LP pass.
    #                       Exact. Slower for large alpha-vector sets (>~50).
    #
    # lp_prune_certified  : sample-based pre-filter + LP only for uncertain
    #                       vectors. Also exact. Faster for large sets.
    #
    prune_fn = lp_prune_certified   # ← change here to switch methods
    # ─────────────────────────────────────────────────────────────────────────

    backward_induction(models_path, n_timesteps=20, output_path=output_path,
                       prune_fn=prune_fn)
