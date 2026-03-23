# Week 5: Advantage Actor-Critic (A2C) for Continuous Control

## Team Members

| Name |
|------|
| Varun Kasa |
| Srinjana Nag |
| Gaurav Dalal |

---

## Assignment Overview

This assignment extends the Advantage Actor-Critic (A2C) implementation from
class — originally designed for discrete action spaces (CartPole) — to handle
continuous action spaces in MuJoCo robotics environments. The implementation
compares three advantage estimation methods across three environments and
performs a hyperparameter grid search to find the optimal configuration for
each environment.

---

## Problem Statement

The base implementation provided by the professor uses a `CategoricalPolicy`
that samples from a discrete softmax distribution over a finite set of actions.
Robotics environments in MuJoCo require continuous actions — for example, joint
torques — which cannot be represented by a categorical distribution.

The assignment requires:

1. Extending A2C to handle continuous action spaces
2. Testing on three MuJoCo robotics environments
3. Comparing three advantage estimation methods per environment
4. Plotting learning curves for all methods on each environment
5. Performing a hyperparameter grid search over five hyperparameters

---

## Environments

| Environment | Description | Obs Dim | Act Dim | Action Bounds |
|-------------|-------------|---------|---------|---------------|
| HalfCheetah-v4 | 2D cheetah learns to run forward | 17 | 6 | [-1.0, 1.0] |
| Hopper-v4 | One-legged robot learns to hop forward | 11 | 3 | [-1.0, 1.0] |
| Walker2d-v4 | Bipedal robot learns to walk forward | 17 | 6 | [-1.0, 1.0] |

---

## Key Extension: Discrete to Continuous Actions

The professor's `CategoricalPolicy` was replaced with a `GaussianPolicy` that:

- Outputs a mean action vector via a two-layer MLP with tanh activations
- Maintains a learned log standard deviation parameter per action dimension
- Squashes actions through tanh to enforce the [-1.0, 1.0] action bounds
- Applies the tanh change-of-variables correction to the log probability

The policy gradient update uses the corrected log probability:

```
log pi(a|s) = log N(atanh(a) | mu(s), sigma) - sum(log(1 - a^2 + eps))
```

The critic was also deepened from one hidden layer to two hidden layers with
tanh activations to handle the higher dimensional MuJoCo state spaces.

---

## Advantage Estimation Methods

All three methods are implemented via a single lambda parameter in the GAE formula:

```
A_t = delta_t + gamma * lambda * (1 - d_t) * A_{t+1}
```

| Method | Lambda | Bias | Variance |
|--------|--------|------|----------|
| 1-step TD | 0.0 | High | Low |
| Monte Carlo | 1.0 | Low | High |
| GAE | 0.95 | Medium | Medium |

---

## Results Summary

### Algorithm Comparison (150 epochs)

| Environment | 1-step TD | Monte Carlo | GAE |
|-------------|-----------|-------------|-----|
| HalfCheetah-v4 | -130.01 | -174.28 | -142.13 |
| Hopper-v4 | 334.16 | 231.60 | 364.75 |
| Walker2d-v4 | 268.30 | 280.77 | 304.64 |

GAE was the best or near-best method in two out of three environments and
never the worst in any environment, confirming its theoretical advantage as
the optimal bias-variance tradeoff for advantage estimation.

### Optimal Hyperparameters (Grid Search over 144 runs)

| Environment | num_envs | policy_lr | value_lr | gamma | lambda | Best Return |
|-------------|----------|-----------|----------|-------|--------|-------------|
| HalfCheetah-v4 | 16 | 1e-3 | 3e-4 | 0.95 | 0.95 | -63.50 |
| Hopper-v4 | 8 | 1e-3 | 1e-3 | 0.99 | 0.95 | 194.38 |
| Walker2d-v4 | 16 | 1e-3 | 1e-3 | 0.95 | 0.95 | 268.15 |

Universal findings: `policy_lr=1e-3` and `lambda=0.95` were optimal across
every single environment without exception.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── a2c.py                      # Professor's original discrete A2C implementation
├── a2c_continuous.ipynb        # Main assignment notebook (run in Google Colab)
├── learning_curves.png         # Learning curve comparison plots (Cell 4 output)
└── grid_search_results.png     # Grid search bar charts (Cell 6 output)
```

---

## Notebook Structure

| Cell | Content |
|------|---------|
| Cell 1 | Install dependencies and verify MuJoCo environments |
| Cell 2 | GaussianPolicy, ValueEstimator, VectorizedEnvWrapper |
| Cell 3 | A2C training loop — all three environments and methods |
| Cell 4 | Learning curve plots |
| Cell 5 | Hyperparameter grid search (144 runs) |
| Cell 6 | Grid search results plot and optimal config summary |

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Set runtime to GPU (T4 or A100 recommended)
3. Run cells in order from Cell 1 to Cell 6
4. Cell 3 takes approximately 20-25 minutes on T4
5. Cell 5 takes approximately 25-30 minutes on T4

### Option 2: Local

```bash
pip install -r requirements.txt
jupyter notebook a2c_continuous.ipynb
```

Note: MuJoCo requires a working system installation of the MuJoCo physics
engine. On Linux this is handled automatically by `gymnasium[mujoco]`. On
macOS and Windows additional steps may be required.

---

## Training Configuration

```python
CONFIG = dict(
    num_envs         = 8,
    policy_lr        = 3e-4,
    value_lr         = 1e-3,
    gamma            = 0.99,
    epochs           = 150,
    train_v_iters    = 40,
    rollout_traj_len = 1024,
)
```

---

## Key Findings

**GAE is the most consistent advantage estimation method.** It achieved the
best or near-best final performance in two out of three environments and showed
the most stable learning curves overall. This aligns with the theoretical
motivation — by blending bias and variance through the lambda parameter, GAE
avoids the failure modes of both extremes.

**Monte Carlo is environment-dependent.** It performed competitively on
Walker2d but struggled significantly on HalfCheetah. High variance gradients
are particularly damaging when the reward signal is dense and continuous over
long time horizons.

**1-step TD is a reliable baseline.** It never failed catastrophically and
produced consistent learning in all three environments. Its low variance makes
it particularly stable on difficult environments like HalfCheetah.

**policy_lr=1e-3 and lambda=0.95 are universal defaults.** These two
hyperparameters were optimal across every environment in the grid search and
should be the starting point for any new MuJoCo continuous control task.

**gamma should be tuned per environment.** Hopper preferred 0.99 while
HalfCheetah and Walker2d preferred 0.95, reflecting differences in episode
length and reward structure.

---

## References

1. Sutton and Barto, Reinforcement Learning: An Introduction
   - Section 6.1: TD Prediction
   - Section 6.2: Advantages of TD
   - Section 7.1: n-step TD prediction
   - Section 12.1: Lambda-return
   - Section 12.2: TD(lambda) return
   - Section 13.5: Actor-Critic methods

2. Mnih et al., Asynchronous Methods for Deep Reinforcement Learning (A3C paper)

3. Schulman et al., High-Dimensional Continuous Control Using Generalized
   Advantage Estimation (GAE paper)

4. Gymnasium MuJoCo Documentation: https://gymnasium.farama.org/environments/mujoco/
