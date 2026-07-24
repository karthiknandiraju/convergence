# Code framework and execution environments

## Shared reinforcement-learning framework

- Deep Q-Network with replay buffer and target network
- Adam optimizer
- Nine discrete actions
- 500 training episodes and 300 frozen-policy test episodes per seed
- Collision RMST evaluated through a fixed 500-step censoring horizon
- Epsilon-Greedy DQN, NoisyNet DQN, and DQN + RND baselines
- SafetyPool memory and action filtering used during training only
- Frozen greedy DQN evaluation without a test-time safety mask

## HighwayEnv execution record

- Python 3.12.3
- PyTorch 2.11.0+cu128
- HighwayEnv 1.12.0
- Gymnasium
- CUDA execution
- Adapter: `framework/highway/highway_env_adapter.py`
- SafetyPool implementation:
  `policy/SafetyPool_Highway_Karthikeya27adv8956.py`
- Baseline trainer: `framework/highway/train_canonical_baselines.py`

The Highway framework requirements are pinned in
`framework/highway/requirements.txt`. Seed-specific runtime manifests embedded
in the source archive also record HighwayEnv 1.12.0 and CUDA 12.8.

## MetaDrive execution record

- Ubuntu 26.04 LTS
- Python 3.11.15
- PyTorch 2.7.1+cu128
- NumPy 1.26.4
- Gymnasium 1.3.0
- MetaDrive Simulator 0.4.3
- CUDA 12.8
- NVIDIA GeForce RTX 5050 Laptop GPU

The exact package list, Conda environment, protocol, and system record are in
`framework/metadrive/configuration/`.
