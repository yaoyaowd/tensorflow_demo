import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

ENV_NAME = 'CartPole-v0'
EPISODE = 2000
STEP = 300

class DeepQNetwork():
    def __init__(self,
                 env,
                 hidden_size=8,
                 learning_rate=0.01,
                 gamma=0.99):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.history = []

        # Define network
        self.state_in = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, hidden_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, self.action_dim,
                                           biases_initializer=None,
                                           activation_fn=tf.nn.softmax)
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actual_action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * self.action_dim \
                       + self.actual_action
        self.actual_output = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        self.loss = -tf.reduce_mean(tf.log(self.actual_output)*self.reward)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)

        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def action(self, state):
        output_value = self.session.run(self.output,
                                        feed_dict={self.state_in: [state]})[0]
        action = np.random.choice(output_value, p=output_value)
        action = np.argmax(output_value == action)
        return action

    def update(self, state, action , reward):
        self.history.append([state, action, reward])

    def discount_reward(self, r):
        discounted_reward = np.zeros_like(r)
        running_reward = 0
        for t in reversed(xrange(0, r.size)):
            running_reward = running_reward * self.gamma + r[t]
            discounted_reward[t] = running_reward
        return discounted_reward

    def train(self):
        h = np.array(self.history)
        h[:,2] = self.discount_reward(h[:,2])
        self.history = []

        feed_dict = {self.reward: h[:,2],
                     self.actual_action: h[:,1],
                     self.state_in: np.vstack(h[:,0])}
        self.session.run([self.train_op], feed_dict=feed_dict)

def main():
    env = gym.make(ENV_NAME)
    agent = DeepQNetwork(env, 8)
    print 'action space:', env.action_space
    print 'observation space:', env.observation_space

    rewards = []
    for e in xrange(EPISODE):
        state = env.reset()
        running_reward = 0
        for i in xrange(STEP):
            action = agent.action(state)
            next_state, reward, died, _ = env.step(action)
            running_reward += reward
            if e % 100 == 0:
                env.render()
            agent.update(state, action, reward)
            state = next_state
            if died:
                agent.train()
                break
        rewards.append(running_reward)
        if len(rewards) == 100:
            print sum(rewards) / len(rewards)
            rewards = []

if __name__ == '__main__':
    main()
