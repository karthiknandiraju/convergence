set3_v2 outputs

Experiments in order:
1. Noisy + Count
2. RND + Count
3. Ensemble Own + Noisy/RND expert candidates

Neural setup: Set 3 agents from src.dqn_agent with count-based intrinsic reward and RND where applicable.
Testing is frozen: no optimizer, replay, RND, or target updates.
Reward graphs use all test episodes and environment reward only.
Convergence graphs use training data only.

Main CSVs:
- all_experiments_train_episode_rewards.csv
- all_experiments_test_episode_rewards.csv
- all_experiments_episode_block_logs.csv
- all_experiments_runtime_logs.csv
- all_experiments_learning_rate_summary.csv

Figures:
- set3_v2_dashboard.png / .pdf
- figures_ieee/*.png and *.pdf

Configuration:
train_episodes=500
test_episodes=300
max_episode_steps=500
epsilon=0.2
learning_rate=5e-05
rnd_beta=0.01
count_beta=0.05
convergence_threshold_fraction=0.95
convergence_window=10
