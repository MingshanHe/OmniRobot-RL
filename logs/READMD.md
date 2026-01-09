0. 2026-01-09_11-37-40_ppo_torch
原本的方向控制和forward速度控制


1. 2026-01-09_16-28-28_ppo_torch
total_reward = forward_reward + alignment_reward - distance_reward*0.1

2. 2026-01-09_16-42-46_ppo_torch
total_reward = alignment_reward - distance_reward*0.1
但是这部分有一些小问题：1）机器人z轴没有和cmd的z值对其，可能会导致有稳态误差 2）0.1 discount 可能会有问题

3. 2026-01-09_16-59-14_ppo_torch
total_reward = alignment_reward - distance_reward
修改了z轴问题，以及0.1discount问题，但是仍然存在一些问题，打算忽略alightment reward 只用distance reward确保目标位置训练的问题能够解决

4. 2026-01-09_17-14-40_ppo_torch
用了更新的distance reward
total_reward = alignment_reward - 2.0 distance_reward
想只用distance_reward试一下

5. 2026-01-09_17-37-08_ppo_torch
total_reward = 5*distance_reward + success_bonus

6. total_reward = 5*distance_reward
改到100个env

7. 2026-01-09_17-51-28_ppo_torch
total_reward = 5*distance_reward

8. 2026-01-09_18-09-41_ppo_torch
将5->1000，数值变大

9. 2026-01-09_18-36-44_ppo_torch
修改bug后训练貌似没有什么大问题
