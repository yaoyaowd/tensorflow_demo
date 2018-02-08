import gym
import numpy as np

env = gym.make('FrozenLake-v0')
print 'observation space:', env.observation_space.n
print 'action space:', env.action_space.n
Q = np.zeros([env.observation_space.n, env.action_space.n])

NUM_EPOCHS = 20000
NUM_STEPS = 99
ALPHA = 0.8
DISCOUNT = 0.95

rewards = []
for i in range(NUM_EPOCHS):
    s = env.reset()
    reward = 0
    died = False
    for j in range(NUM_STEPS):
        a = np.argmax(Q[s,:] +
                      np.random.randn(1, env.action_space.n) * (1./(i + 1.0)))
        next_s, r, died, _ = env.step(a)
        Q[s, a] = (1 - ALPHA) * Q[s, a] + ALPHA * (r + DISCOUNT * np.max(Q[next_s, :]))
        reward += r
        s = next_s
        if died:
            break
    rewards.append(reward)

average_rewards = []
total_reward = 0
for i in range(len(rewards)):
    total_reward += rewards[i]
    if i % 1000 == 0:
        average_rewards.append(total_reward / 1000)
        total_reward = 0
print "Score over time:", average_rewards
print "Final Q-Table:", Q

