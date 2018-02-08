# https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149
# https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html

import numpy as np
import tensorflow as tf

bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)

def pull_arm(k):
    result = np.random.randn(1)
    if result > bandits[k]:
        return 1
    else:
        return -1

weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights, 0)
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
loss = -tf.log(responsible_weight)*reward_holder
update = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

total_rewards = np.zeros(num_bandits)
e = 0.1
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(1001):
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)
        reward = pull_arm(action)

        _, resp, w = sess.run([update, responsible_weight, weights],
                              feed_dict={reward_holder: [reward],
                                         action_holder: [action]})
        total_rewards[action] += reward
        if i % 50 == 0:
            print "Total rewards: ", total_rewards
            print "Weights: ", w
