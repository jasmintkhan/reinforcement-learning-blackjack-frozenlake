# reinforcement-learning-blackjack-frozenlake

This repo contains two reinforcement learning projects using **Gymnasium environments**:

1. **Blackjack with Q-Learning**
2. **Frozen Lake with Value Iteration & Policy Extraction**

Both agents are written in Python and designed to be run directly from the command line.

---

## `blackjack_qlearning.py`

A model-free Q-learning agent that learns how to play Blackjack by interacting with the environment and updating its Q-table over time.

<details>
<summary>Watch the training preview</summary>

<video src="https://github.com/user-attachments/assets/935c5c68-911c-415a-9179-bc426f3d4e4b" controls autoplay muted loop style="max-width: 100%; height: auto;">
Your browser does not support the video tag.
</video>

</details>




### What it does:
- Initializes the Blackjack environment from Gymnasium.
- Uses an **epsilon-greedy policy** to balance exploration and exploitation.
- Tracks rewards, win/loss status, and gradually decays exploration over episodes.
- Saves the final Q-table as `Q_table_blackjack.npy`.

### To run:
```bash
python blackjack_qlearning.py
```

You'll see win/loss logs per episode and average rewards every 100 episodes.

---

## `frozenlake_policy_learning.py`

A model-based reinforcement learning agent that solves the FrozenLake-v1 environment using data-driven **value iteration**.

<details>
<summary>Watch the training preview</summary>

<video src="https://github.com/user-attachments/assets/dc292874-2588-49a4-ae57-3445d54c429c" controls autoplay muted loop style="max-width: 100%; height: auto;">
Your browser does not support the video tag.
</video>

</details>

### What it does:
#### Stage 1: Random Data Collection
- Collects transition probabilities and reward estimates using a random policy.

#### Stage 2: Value Iteration
- Computes the optimal value function using the Bellman update.

#### Stage 3: Policy Extraction
- Extracts the best action to take in each state from the computed values.

#### Stage 4: Simulation
- Runs the agent in the environment using the learned policy to evaluate success rate.

### To run:
```bash
python frozenlake_policy_learning.py
```

You'll see simulation progress printed to the console along with the number of successful episodes.

---

## Requirements

- Python 3.8+
- `gymnasium`
- `numpy`
- `pickle`

Install with:
```bash
pip install gymnasium numpy
```

---

## Why this project?

These small but complete examples show the difference between:
- **Model-free RL** (Q-learning)
- **Model-based RL** (transition + reward modeling + value iteration)

Great for learning core RL concepts without needing complex setups.

---

## License

MIT — feel free to use or adapt for your own learning or projects.
