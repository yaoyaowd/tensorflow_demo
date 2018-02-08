import gym
import numpy as np
import tensorflow as tf

env = gym.make('FrozenLake-v0')
tf.reset_default_graph()

input = tf.placeholder(shape=[1, env.observation_space.n], dtype=tf.float32)
w = tf.Variable(tf.random_uniform([env.observation_space.n, env.action_space.n], 0, 0.01))
output = tf.matmul(input, w)
predict = tf.argmax(output, 1)

nextQ = tf.placeholder(shape=[1, env.action_space.n], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - output))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

alpha = 0.99
err = 0.1
rewards = []

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(1,20001):
        s = env.reset()
        reward = 0
        died = False
        for j in range(99):
            a, allQ = sess.run([predict, output],
                               feed_dict={input: np.identity(env.observation_space.n)[s:s+1]})
            if np.random.rand() < err:
                a[0] = env.action_space.sample()
            s2, r, d, _  = env.step(a[0])

            Q2 = sess.run(output, feed_dict={input: np.identity(env.observation_space.n)[s2:s2+1]})
            maxQ = np.max(Q2)
            targetQ = allQ
            targetQ[0,a[0]] = r + alpha * maxQ
            _ = sess.run(optimizer, feed_dict={input: np.identity(env.observation_space.n)[s:s+1],
                                               nextQ: targetQ})
            reward += r
            s = s2

            if d:
                err = 1. / ((i / 50) + 10)
                break

        rewards.append(reward)
        if i % 1000 == 0:
            print sum(rewards)
            rewards = []

print rewards
