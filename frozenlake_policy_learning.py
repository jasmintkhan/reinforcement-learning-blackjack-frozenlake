"""
Frozen Lake RL Agent (Model-Based)
-----------------------------------
Implements data collection, value iteration, policy extraction, and simulation
for solving FrozenLake-v1 using Gymnasium.
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict
import pickle

# Initialize the environment
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode=None)

# =================================================
# Stage 1: Collecting Training Data with Random Policy
# =================================================

# Parameters for data collection
num_episodes = 1000
state_count = env.observation_space.n  # Total number of states
action_count = env.action_space.n  # Total number of actions

# Data collection structures
transition_counts = defaultdict(lambda: np.zeros(state_count))  # Count transitions to each state
reward_sums = defaultdict(lambda: np.zeros(state_count))  # Sum of rewards for each state transition
transition_counts_total = defaultdict(lambda: np.zeros(action_count))  # Total transitions for each (state, action)

print("Stage 1: Starting random policy data collection... Will print status every 100 episodes...")
# Run episodes to collect data
for episode in range(num_episodes):
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes}")
    state, _ = env.reset()
    done = False

    while not done:
        # Choose a random action
        action = env.action_space.sample()
        
        # Step in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Collect data for the transition
        transition_counts[(state, action)][next_state] += 1
        reward_sums[(state, action)][next_state] += reward
        transition_counts_total[(state, action)][action] += 1

        # Move to the next state
        state = next_state

env.close()

# Convert defaultdicts to regular dictionaries for pickling
transition_counts = dict(transition_counts)
reward_sums = dict(reward_sums)
transition_counts_total = dict(transition_counts_total)

# Estimate the transition function T(s'|s,a)
T = {}
for (state, action), counts in transition_counts.items():
    total_transitions = counts.sum()
    if total_transitions > 0:
        T[(state, action)] = counts / total_transitions  # Normalize to get probabilities

# Estimate the reward function R(s,a,s')
R = {}
for (state, action), rewards in reward_sums.items():
    total_transitions = transition_counts[(state, action)].sum()
    if total_transitions > 0:
        R[(state, action)] = rewards / total_transitions  # Average reward

# Save the collected data using pickle
with open('T.pkl', 'wb') as f:
    pickle.dump(T, f)
with open('R.pkl', 'wb') as f:
    pickle.dump(R, f)

print("Training data saved.")

# Print sample outputs
print("\nSample transition probabilities T(s'|s,a):")
for key, value in list(T.items())[:5]:  # Display a few 
    print(f"State {key[0]}, Action {key[1]}: {value}")

print("\nSample reward function R(s,a,s'):")
for key, value in list(R.items())[:5]:  # Display a few 
    print(f"State {key[0]}, Action {key[1]}: {value}")

# ==================================
# Stage 2: Implementing Value Iteration
# ==================================

# Value Iteration parameters
gamma = 0.9  # Discount factor
theta = 1e-6  # Convergence threshold

# Initialize value function
V = np.zeros(state_count)

print("\nStage 2: Starting value iteration...")
while True:
    delta = 0
    for state in range(state_count):
        v = V[state]
        # Bellman update
        max_value = float('-inf')
        for action in range(action_count):
            if (state, action) in T:
                value = sum(T[(state, action)][next_state] * (R[(state, action)][next_state] + gamma * V[next_state])
                            for next_state in range(state_count))
                max_value = max(max_value, value)
                #print(f"State {state}, Action {action}, Calculated Value: {value}")
        
        V[state] = max_value if max_value != float('-inf') else 0  # Update value function
        delta = max(delta, abs(v - V[state]))
        #print(f"    Updated V[{state}] from {v:.6f} to {V[state]:.6f}")
    
    # Check for convergence
    if delta < theta:
        break

print("Value iteration completed.")
print("\nOptimal value function:")
print(V)

# ====================================
# Stage 3: Extract Optimal Policy from Value Function
# ====================================

# Initialize policy
policy = np.zeros(state_count, dtype=int)

print("\nStage 3: Starting policy extraction...")
for state in range(state_count):
    best_action = None
    best_value = float('-inf')
    for action in range(action_count):
        if (state, action) in T:
            value = sum(T[(state, action)][next_state] * (R[(state, action)][next_state] + gamma * V[next_state])
                        for next_state in range(state_count))
            if value > best_value:
                best_value = value
                best_action = action
            #print(f"State {state}, Action {action}, Expected Value: {value}")
    policy[state] = best_action if best_action is not None else 0  # Default to action 0 if no valid action found
    #print(f"    State {state}, Action {best_action}, Highest Value: {best_value}")

print("Policy extraction completed.")
print("\nOptimal policy:")
print(policy)

# Save the policy
with open('optimal_policy.pkl', 'wb') as f:
    pickle.dump(policy, f)

print("Optimal policy saved.")

# ===========================================================
# Stage 4: Acting Inside the Environment with the Optimal Policy
# ===========================================================

# Re-initialize the environment to avoid using the closed environment from Q2.2
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="human")  # Change render_mode to None for faster testing

# Load the optimal policy
with open('optimal_policy.pkl', 'rb') as f:
    policy = pickle.load(f)

# Parameters for running the simulation
num_simulation_episodes = 50  # Number of episodes to simulate
successes = 0  # Variable to track the number of successful episodes

print("\nStage 4: Running simulation with the extracted optimal policy...")

# Run the simulation using the optimal policy
for episode in range(num_simulation_episodes):
    print(f"\nEpisode {episode + 1}/{num_simulation_episodes}")
    state, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0  # Total reward for the current episode

    while not done:
        # Choose action based on the optimal policy
        action = policy[state]

        # Step in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Print the current step's details
        step_count += 1
        total_reward += reward
        #print(f"  Step {step_count}: State {state} -> Action {action} -> Next State {next_state} -> Reward {reward}")

        # Move to the next state
        state = next_state

    # Check if the episode was successful
    if total_reward == 1.0:
        successes += 1

    # Print episode summary
    print(f"Episode {episode + 1} completed. Total Reward: {total_reward}")

# Print the number of successful episodes
print(f"\nNumber of successful episodes: {successes}/{num_simulation_episodes} ({(successes / num_simulation_episodes) * 100:.2f}%)")

# Close the environment safely
env.close()

print("\nOptimal policy simulation completed.")
