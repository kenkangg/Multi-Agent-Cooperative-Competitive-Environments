import tensorflow as tf
from networks import *

class PPOAgent:
    """
    Proximal Policy Optimization

    """

    def __init__(self, env, network='mlp'):
        sess = tf.get_default_session()

        self.network = self.build_model(sess=sess, network=network, obs_space=env.observation_space,
                                        act_space=env.action_space)

        A = tf.placeholder(tf.float32, [None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        tf.global_variables_initializer().run(session=sess)




    def build_model(self, sess, network, obs_space, act_space):
        if network == 'mlp':
            return MLP_Network(sess, obs_space=obs_space, act_space=act_space)

class Runner:
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = agent.network.act(observation)[0]

            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
