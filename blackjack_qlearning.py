"""
Blackjack Q-Learning Agent
--------------------------
Trains a reinforcement learning agent to play Blackjack using the Gymnasium library.
The agent learns an optimal policy using the Q-learning algorithm and saves the Q-table for future use.
"""

import numpy as np
import gymnasium as gym

# Initialize the environment
env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")

# Initialize Q-table
state_space_size = (32, 11, 2)
action_space_size = env.action_space.n
Q = np.zeros(state_space_size + (action_space_size,))

# Parameters for Q-learning
alpha = 0.2    # Learning rate
gamma = 0.9    # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995
episodes = 1000  # Total training episodes

# Track rewards for visualization
reward_log = []

# Track wins and losses
wins = 0
losses = 0

print(f"\nQ-learning training started with {episodes} Episodes...")
print("     - Will print W/L status every episode")
print("     - Will print average reward every 100 episodes")
print("     - Total win rate will be calculated at the end\n")

# Q-learning algorithm
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0  # Sum of rewards for the current episode

    state = (state[0], state[1], int(state[2]))  # Convert state to Q-table format

    while not done:
        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # Step in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward  # Accumulate reward

        next_state = (next_state[0], next_state[1], int(next_state[2]))

        # Q-value update
        if not done:
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        else:
            Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])

        state = next_state

    # Track win/loss
    if total_reward > 0:
        wins += 1
        print(f"Episode {episode + 1}: Win")
    else:
        losses += 1
        print(f"Episode {episode + 1}: Loss")

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Log reward
    reward_log.append(total_reward)

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(reward_log[-100:])  # Calculate average reward over the last 100 episodes
        print(f"- Average Reward (Last 100): {avg_reward:.2f} \n- Epsilon: {epsilon:.3f}")

# Calculate and print total win rate
win_rate = wins / episodes * 100
print(f"\n***Total Win Rate: {win_rate:.2f}% ({wins} wins out of {episodes} episodes)***\n")

# Save the Q-table
np.save('Q_table_blackjack.npy', Q)

env.close()
print("Training completed. Q-table saved as 'Q_table_blackjack.npy'.")
