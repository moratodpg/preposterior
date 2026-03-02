import os
import numpy as np

# Transition model (identical for all actions in the pile-selection problem)
# Shape: (S, S) — rows = from-state, cols = to-state
transition_base = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# Observation models — shape: (Z, S)  rows = observations, cols = states
# Z=3: (40ft, 45ft, 50ft)   S=4: (short_t0, long_t0, short_t1, long_t1)
observation_no_info = np.array([
    [0.5, 0.5, 0.5, 0.5],
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5, 0.5],
])

observation_drill = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
])

observation_sonic = np.array([
    [0.6, 0.1, 0.6, 0.1],
    [0.3, 0.2, 0.3, 0.2],
    [0.1, 0.7, 0.1, 0.7],
])

# A=6 actions; each action gets its own observation model
action_names = np.array([
    "order_short_no_info",
    "order_short_drill",
    "order_short_sonic",
    "order_long_no_info",
    "order_long_drill",
    "order_long_sonic",
])

obs_per_action = [
    observation_no_info,  # action 0
    observation_drill,    # action 1
    observation_sonic,    # action 2
    observation_no_info,  # action 3
    observation_drill,    # action 4
    observation_sonic,    # action 5
]
observations = np.stack(obs_per_action, axis=0)  # (A, Z, S)

A = len(action_names)
transition = np.tile(transition_base[np.newaxis], (A, 1, 1))  # (A, S, S)

# Reward model — shape: (A, S)
reward = np.array([
    [  0.0,   0.0,    0.0, -400.0],
    [-50.0, -50.0,  -50.0, -450.0],
    [-20.0, -20.0,  -20.0, -420.0],
    [  0.0,   0.0, -100.0,    0.0],
    [-50.0, -50.0, -150.0,  -50.0],
    [-20.0, -20.0, -120.0,  -20.0],
])

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "models.npz")
np.savez(output_path, transition=transition, observations=observations,
         reward=reward, action_names=action_names)

print(f"Saved {output_path}")
print(f"  transition:   {transition.shape}")
print(f"  observations: {observations.shape}")
print(f"  reward:       {reward.shape}")
print(f"  action_names: {action_names}")
