import os
import numpy as np

# Transition model (identical for all actions in the pile-selection problem)
# Shape: (S, S) — rows = from-state, cols = to-state
transition_deterioration = np.array([
    [0.95, 0.05, 0.00],
    [0.00, 0.95, 0.05],
    [0.00, 0.00, 1.00],
])

transition_repair = np.array([
    [0.95, 0.05, 0.0],
    [0.95, 0.05, 0.0],
    [0.95, 0.05, 0.0],
])

# Observation models — shape: (Z, S)  rows = observations, cols = states
# Z=3: (40ft, 45ft, 50ft)   S=4: (short_t0, long_t0, short_t1, long_t1)
observation_no_info = np.array([
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
])

observation_inspection = np.array([
    [0.8, 0.15, 0.05],
    [0.2, 0.85, 0.95],
])

# A=6 actions; each action gets its own observation model
action_names = np.array([
    "do_nothing",
    "inspect",
    "repair",
])

obs_per_action = [
    observation_no_info,  # action 0
    observation_inspection,    # action 1
    observation_no_info,  # action 2
]
observations = np.stack(obs_per_action, axis=0)  # (A, Z, S)

act_per_action = [
    transition_deterioration,  # action 0
    transition_deterioration,    # action 1
    transition_repair,  # action 2
]
transition = np.stack(act_per_action, axis=0)  # (A, S, S)

# -------- #
# Inspection: 1
# Repair: 20
# Failure: 1000
# -------- #

# Reward model — shape: (A, S)
reward = np.array([
    [  0.0, -50.0,    0.0],
    [ -1.0, -51.0,   -1.0],
    [-20.0, -20.0,  -20.0],
])

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "model_simplified_fatigue.npz")
np.savez(output_path, transition=transition, observations=observations,
         reward=reward, action_names=action_names)

print(f"Saved {output_path}")
print(f"  transition:   {transition.shape}")
print(f"  observations: {observations.shape}")
print(f"  reward:       {reward.shape}")
print(f"  action_names: {action_names}")
