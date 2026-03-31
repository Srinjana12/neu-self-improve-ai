# Week 06: Poly-PPO on MiniGrid FourRooms

 Week06 implements and compares four training stages on `MiniGrid-FourRooms-v0`:

- Behavior Cloning (pretraining from expert demonstrations)
- REINFORCE fine-tuning
- PPO fine-tuning
- Poly-PPO fine-tuning (diversity-aware branch rollouts)

Main notebook:

- `Week06_Implemement_Poly_PPO.ipynb`

## Project Goal

Train an agent in FourRooms using a pretrained policy and compare how different policy-gradient variants improve:

- Average reward
- Success rate (%)

The notebook also compares final metrics against paper target numbers.

## What Is Implemented

- Full-observation wrapper for FourRooms, flattened and normalized input.
- Shared `ActorCritic` network with categorical policy head and value head.
- Expert planner using shortest-path search in grid state space.
- Behavior cloning pretraining from generated expert trajectories.
- REINFORCE baseline.
- PPO baseline with clipping and GAE.
- Poly-PPO variant:
  - Saves environment snapshots during PPO collection.
  - Rolls out multiple branch trajectories (vines) from snapshots.
  - Scores groups using normalized return and diversity of visited rooms.
  - Replaces local advantages with a Poly signal over a short window.

## Dependencies

The notebook installs dependencies with:

```bash
pip install gymnasium minigrid torch numpy matplotlib pandas tqdm
```

Recommended:

- Python 3.9+
- A virtual environment
- Jupyter Notebook or VS Code Notebook support

## How To Run

1. Open `Week06_Implemement_Poly_PPO.ipynb`.
2. Run cells top-to-bottom.
3. Wait for the training cell (REINFORCE, PPO, Poly-PPO) to finish.
4. View the final comparison table and success-rate plot.

## Notebook Flow

- Install/import and set random seeds.
- Build environment wrappers and helper utilities.
- Define model and RL utility functions.
- Generate expert dataset and run behavior cloning pretraining.
- Train REINFORCE, PPO, and Poly-PPO from pretrained weights.
- Evaluate each method over 100 episodes.
- Compare against paper targets and plot success-rate trends.

## Outputs

Running the notebook produces:

- Pretraining history (`bc_hist`)
- Fine-tuning histories (`reinforce_hist`, `ppo_hist`, `polyppo_hist`)
- Final comparison DataFrame (`comparison_df`)
- Success-rate training plot

## Notes

- The notebook filename contains a typo (`Implemement`).
- Computation can take time because multiple algorithms are trained in one run.
- Poly-PPO adds extra rollout cost due to snapshot branching (`N`, `n`, `M`, `W`).
