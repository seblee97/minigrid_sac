# MiniGrid RL + MDP Reductions

A small, self-contained RL codebase for training agents on **MiniGrid** environments via the **Gymnasium** API, with an SB3-inspired structure:

- Algorithms: **PPO**, **A2C**, **DQN**, **QRDQN**
- Policies: **MlpPolicy** and **CnnPolicy**
- MiniGrid helpers: wrappers + vector env utilities
- MDP reduction utilities:
  - **Exact partition refinement** (bisimulation-style) for small/known MDPs
  - **Empirical MDP extraction** from an environment via sampling
  - **Approximate bisimulation metrics** (Sinkhorn-regularized OT as a scalable proxy)
  - **State aggregation / homomorphism heuristics** based on embeddings + stability checks

This aims to be readable research code rather than a fully battle-tested RL library.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart

Train PPO on MiniGrid:

```bash
python -m rlgrid.scripts.train   --algo ppo --env MiniGrid-Empty-8x8-v0   --total-steps 200000 --n-envs 8 --obs image
```

Train DQN:

```bash
python -m rlgrid.scripts.train   --algo dqn --env MiniGrid-Empty-8x8-v0   --total-steps 200000 --obs image
```

Compute an empirical reduced MDP via state aggregation:

```bash
python -m rlgrid.scripts.reduce_mdp   --env MiniGrid-Empty-8x8-v0 --obs image   --steps 100000 --method sinkhorn_bisim
```

## Notes / Design

- MiniGrid is Gymnasium-native (Farama). The docs show you can create envs via `gym.make("MiniGrid-...")`.  
- The MiniGrid docs also show training with SB3 is straightforward; this repo provides SB3-like ergonomics without depending on SB3.  
- QRDQN follows the SB3-Contrib idea/API.

References:
- MiniGrid docs & training tutorial: https://minigrid.farama.org/  
- Stable Baselines3 architecture notes: https://stable-baselines3.readthedocs.io/  
- SB3-Contrib QRDQN: https://sb3-contrib.readthedocs.io/  

## File layout

- `rlgrid/algos/`: PPO/A2C/DQN/QRDQN
- `rlgrid/policies/`: MLP/CNN policies + distributions
- `rlgrid/envs/`: MiniGrid wrappers + vector-env helpers
- `rlgrid/mdp/`: empirical MDP extraction + reductions
- `rlgrid/scripts/`: CLI entrypoints (train, evaluate, reduce_mdp)

