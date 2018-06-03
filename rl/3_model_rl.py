import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import gym


env = gym.make('CartPole-v0')
hidden = 8
dimension = 4
learning_rate = 1e-2
gamma = 0.99
decay_rate = 0.99
model_batch_size = 3
real_batch_size = 3

class PolicyNetwork(object):
    def __init__(self):
        # policy network
        self.observations = tf.placeholder(tf.float32, [None, 4], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')
        self.reward = tf.placeholder(tf.float32, name='reward_signal')
        l1 = slim.fully_connected(self.observations,
                                  hidden,
                                  biases_initializer=None,
                                  activation_fn=tf.nn.relu)
        self.score = slim.fully_connected(l1,
                                          1,
                                          biases_initializer=None)
        self.probability = tf.nn.sigmoid(self.score)
        loglike = tf.log(self.input_y * (self.input_y - self.probability)
                         + (1 - self.input_y) * (self.input_y + self.probability))
        loss = -tf.reduce_mean(loglike * self.reward)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.w1grad = tf.placeholder(tf.float32, name='batch_grad1')
        self.w2grad = tf.placeholder(tf.float32, name='batch_grad2')
        batch_grad = [self.w1grad, self.w2grad]

        self.tvars = tf.trainable_variables()
        self.newgrads = tf.gradients(loss, self.tvars)
        self.update = self.optimizer.apply_gradients(zip(batch_grad, self.tvars))

class ModelNetwork(object):
    def __init__(self):
        # model network
        self.input_data = tf.placeholder(tf.float32, [None, 5])
        self.previous_state = tf.placeholder(tf.float32, [None, 5], name='previous_state')
        self.true_observation = tf.placeholder(tf.float32, [None, 4], name="true_observation")
        self.true_reward = tf.placeholder(tf.float32, [None, 1], name="true_reward")
        self.true_done = tf.placeholder(tf.float32, [None, 1], name="true_done")

        w1m = tf.get_variable(name='w1m', shape=[5, 256], initializer=tf.contrib.layers.xavier_initializer())
        b1m = tf.Variable(tf.zeros([256]), name='b1m')
        layer1m = tf.nn.relu(tf.matmul(self.previous_state, w1m) + b1m)

        w2m = tf.get_variable(name='w2m', shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
        b2m = tf.Variable(tf.zeros([256]), name='b2m')
        layer2m = tf.nn.relu(tf.matmul(layer1m, w2m) + b2m)

        wo = tf.get_variable("wo", shape=[256, 4], initializer=tf.contrib.layers.xavier_initializer())
        wr = tf.get_variable("wr", shape=[256, 1], initializer=tf.contrib.layers.xavier_initializer())
        wd = tf.get_variable("wd", shape=[256, 1], initializer=tf.contrib.layers.xavier_initializer())
        bo = tf.Variable(tf.zeros([4]), name="bo")
        br = tf.Variable(tf.zeros([1]), name="br")
        bd = tf.Variable(tf.zeros([1]), name="bd")

        self.predicted_observation = tf.matmul(layer2m, wo, name="predicted_observation") + bo
        self.predicted_reward = tf.matmul(layer2m, wr, name="predicted_reward") + br
        self.predicted_done = tf.sigmoid(tf.matmul(layer2m, wd, name="predicted_done") + bd)

        self.predicted_state = tf.concat(values=[
            self.predicted_observation, self.predicted_reward, self.predicted_done], axis=1)
        self.observation_loss = tf.square(self.true_observation - self.predicted_observation)
        self.reward_loss = tf.square(self.true_reward - self.predicted_reward)
        self.done_loss = -tf.log(tf.multiply(self.predicted_done, self.true_done) +
                                 tf.multiply(1-self.predicted_done, 1-self.true_done))
        self.model_loss = tf.reduce_mean(self.observation_loss + self.done_loss + self.reward_loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update = self.optimizer.minimize(self.model_loss)

def reset_grad_buffer(grad_buffer):
    for ix, grad in enumerate(grad_buffer):
        grad_buffer[ix] = grad * 0
    return grad_buffer

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def step_model(sess, model, xs, action):
    to_feed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1, 5])
    my_predict= sess.run([model.predicted_state], feed_dict={model.previous_state: to_feed})
    reward = my_predict[0][:, 4]
    observation = my_predict[0][:, 0:4]
    observation[:,0] = np.clip(observation[:, 0], -2.4, 2.4)
    observation[:,2] = np.clip(observation[:, 2], -0.4, 0.4)
    done_p = np.clip(my_predict[0][:, 5], 0, 1)
    if done_p > 0.1 or len(xs) >= 300:
        done = True
    else:
        done = False
    return observation, reward, done


xs, ys, ds, drs = [], [], [], [] # input x, decision, done, rewards
running_reward = None
reward_sum = 0
episode_number = 1;
real_episodes = 1
batch_size = real_batch_size

draw_from_model = False
train_the_model = True
train_the_policy = False
switch_point = 1

with tf.Session() as sess:
    policy = PolicyNetwork()
    model = ModelNetwork()
    sess.run(tf.initialize_all_variables())
    grad_buffer = sess.run(policy.tvars)
    grad_buffer = reset_grad_buffer(grad_buffer)

    observation = env.reset()
    x = observation

    while episode_number <= 5000:
        x = np.reshape(observation, [1, 4])
        tf_prob = sess.run(policy.probability, feed_dict={policy.observations: x})
        action = 1 if np.random.uniform() < tf_prob else 0

        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        if draw_from_model == False:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done = step_model(sess, model, xs, action)

        reward_sum += reward
        ds.append(done * 1)
        drs.append(reward)

        if done:
            if draw_from_model == False:
                real_episodes += 1
            episode_number += 1

            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)
            xs, ys, ds, drs = [], [], [], []

            if train_the_model:
                actions = np.array([np.abs(y-1) for y in epy][:-1])
                state_prevs = epx[:-1, :]
                state_prevs = np.hstack([state_prevs, actions])
                state_nexts = epx[1:, :]
                rewards = np.array(epr[1:, :])
                dones = np.array(epd[1:, :])
                state_nextsAll = np.hstack([state_nexts, rewards, dones])

                feed_dict = {model.previous_state: state_prevs,
                             model.true_observation: state_nexts,
                             model.true_done: dones,
                             model.true_reward: rewards}
                loss, pState, _ = sess.run([model.model_loss, model.predicted_state, model.update], feed_dict)

            if train_the_policy:
                discounted_epr = discount_rewards(epr).astype('float32')
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                tGrad = sess.run(policy.newgrads,
                                    feed_dict={policy.observations: epx,
                                               policy.input_y: epy,
                                               policy.reward: discounted_epr})

                if np.sum(tGrad[0] == tGrad[0]) == 0:
                    break
                for ix, grad in enumerate(tGrad):
                    grad_buffer[ix] += grad

            if switch_point + batch_size == episode_number:
                switch_point = episode_number
                if train_the_policy:
                    sess.run(policy.update, feed_dict={policy.w1grad: grad_buffer[0], policy.w2grad: grad_buffer[1]})
                    grad_buffer = reset_grad_buffer(grad_buffer)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                if draw_from_model == False:
                    print 'World Perf: Episode %d. Reward %f. action: %f. mean reward %f.' % (
                        real_episodes, reward_sum / real_batch_size, action, running_reward / real_batch_size)
                    if reward_sum / batch_size > 200:
                        break
                reward_sum = 0

                if episode_number > 100:
                    draw_from_model = not draw_from_model
                    train_the_model = not train_the_model
                    train_the_policy = not train_the_policy

            if draw_from_model:
                observation = np.random.uniform(-0.1, 0.1, [4])
                batch_size = model_batch_size
            else:
                observation = env.reset()
                batch_size = real_batch_size















