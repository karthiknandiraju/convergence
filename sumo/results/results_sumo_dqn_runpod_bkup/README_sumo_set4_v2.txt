SUMO set4_v2 DQN outputs

Experiments in order:
1. Epsilon Greedy
2. Median 50

Neural setup:
DQN + RND + Count-Based intrinsic reward
+ Target Network + Replay Buffer + Adam.
No NoisyNet.

Testing is frozen:
- no optimizer updates
- no replay updates
- no RND updates
- no count updates
- no target-network updates
- environment reward only

Main CSVs:
- all_experiments_train_episode_rewards.csv
- all_experiments_test_episode_rewards.csv
- all_experiments_episode_block_logs.csv
- all_experiments_runtime_logs.csv
- all_experiments_learning_rate_summary.csv

Figures:
- figures_ieee/*.png

Configuration:
train_episodes=500
test_episodes=300
max_episode_steps=500
epsilon=0.2
learning_rate=5e-05
gamma=0.99
rnd_beta=0.01
count_beta=0.05
