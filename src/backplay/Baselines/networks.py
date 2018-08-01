from keras import backend as K
import tensorflow as tf
from keras.layers import Dense
import numpy as np


class MLP_Network:
    """
    MLP network taking an environment's observation as input and outputs a policy and value
    """

    def __init__(self, sess, obs_space, act_space):
        self.sess = sess

        self.obs_data = tf.placeholder(dtype=tf.float32, shape=obs_space.shape)
        obs_data = tf.expand_dims(self.obs_data, axis=0)

        with tf.variable_scope('model'):

            dense1 = Dense(64, activation='relu')(obs_data)
            dense2 = Dense(64, activation='relu')(dense1)

            self.policy_output = Dense(act_space.n, activation='softmax')(dense2)
            self.value_output = Dense(1)(dense2)

    def act(self, obs):
        """ Get Action, Value from Observation """
        policy_dist, value = self.sess.run([self.policy_output, self.value_output], feed_dict={self.obs_data: obs})
        action = np.random.choice(len(policy_dist[0]), 1, p=policy_dist[0])[0]
        value = value[0][0]
        # print(policy_dist[0])
        return action, value
