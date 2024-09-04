import gymnasium as gym
import numpy as np


class FeatureInverter(gym.Wrapper):
    """
    This wrapper is used to change the order of features in the state representation of an environment
    It has no effect on the dynamics of the environment
    It is mainly used for visualisation: if an environment has more than 2 state features,
    it makes it possible to choose which of the features are the first two, because only the
    first two will be visualized with a portrait visualization.
    A concrete example is CartPole: we would like to visualize position and pole angle, and pole angle
    is only the third feature.
    We specify the rank of two features to be inverted and their place is exchanged in the observation
    vector output at each step, including reset.
    """

    def __init__(self, env, f1, f2):
        """
        :param env: the environment to be wrapped
        :param f1: the rank of the first feature to be inverted
        :param f2:  the rank of the second feature to be inverted
        """
        super(FeatureInverter, self).__init__(env)
        self.f1 = f1
        self.f2 = f2

        low_space = env.observation_space.low
        high_space = env.observation_space.high
        tmp = low_space[self.f1]
        low_space[self.f1] = low_space[self.f2]
        low_space[self.f2] = tmp
        tmp = high_space[self.f1]
        high_space[self.f1] = high_space[self.f2]
        high_space[self.f2] = tmp
        self.observation_space.low = low_space
        self.observation_space.high = high_space

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        tmp = observation[self.f1]
        observation[self.f1] = observation[self.f2]
        observation[self.f2] = tmp
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        tmp = observation[self.f1]
        observation[self.f1] = observation[self.f2]
        observation[self.f2] = tmp
        return observation, info


class FeatureFilterWrapper(gym.Wrapper):
    """
    This wrapper is used to remove a feature in the state representation of an environment
    It has no effect on the dynamics of the environment
    It is mainly used to make an environment non-Markovian
    it makes it possible to choose which feature to remove.
    A concrete example is CartPole: we would like to get observations with only the position and pole angle,
    removing velocity and pole angular velocity.
    """

    def __init__(self, env, rank):
        """
        :param env: the environment to be wrapped
        :param rank: the rank of the feature we want to remove
        """
        super(FeatureFilterWrapper, self).__init__(env)
        self.rank = rank

        low_space = env.observation_space.low
        high_space = env.observation_space.high
        # print(env.observation_space.shape)
        low_space = np.delete(low_space, rank)
        high_space = np.delete(high_space, rank)
        self.observation_space.low = low_space
        self.observation_space.high = high_space
        self.observation_space._shape = low_space.shape
        # print("space : ", self.observation_space.low, self.observation_space.high)
        # print("shape : ", self.observation_space.shape)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = np.delete(observation, self.rank)
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = np.delete(observation, self.rank)
        return observation, info


class TimeExtensionWrapper(gym.Wrapper):
    """
    This wrapper is used to temporally extend an environment
    so that it takes N actions as input and only plays the first
    And it returns two successive observations, by memorizing the previous one from the inner environment
    It makes it possible to build temporally extended policies and critics in a transparent way.
    """

    def __init__(self, env):
        """
        :param env: the environment to be wrapped
        :param rank: the rank of the feature we want to remove
        """
        super(TimeExtensionWrapper, self).__init__(env)

        low_space = env.observation_space.low
        high_space = env.observation_space.high
        new_low = np.concatenate((low_space, low_space))
        new_high = np.concatenate((high_space, high_space))
        self.observation_space.low = new_low
        self.observation_space.high = new_high
        self.observation_space._shape = new_low.shape
        # print("obs space : ", self.observation_space.low, self.observation_space.high)
        # print("shape : ", self.observation_space.shape)
        self.state_memory = np.zeros(
            self.observation_space.shape[0]
        )  # TODO: étendre à un nombre d'états quelconques

        # action_space = gym.spaces.MultiDiscrete(np.array(env.action_space.n, env.action_space.n))
        action_space = gym.spaces.MultiDiscrete(
            [env.action_space.n, env.action_space.n]
        )  # TODO: étendre à un nombre d'actions quelconques
        self.action_space = action_space
        # print("action space : ", self.action_space)

    def step(self, action):
        list_actions = np.split(action, 2)
        local_action = list_actions[0]
        # print("action locale :", local_action)
        observation, reward, terminated, truncated, info = self.env.step(
            local_action[0]
        )  # TODO: [0] pas generique
        # observation = np.delete(observation, self.rank)
        full_obs = np.concatenate((observation, self.state_memory))
        self.state_memory = observation
        return full_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.state_memory = np.zeros(self.observation_space.shape[0])
        full_obs = np.concatenate((observation, self.state_memory))
        # print(full_obs)
        return full_obs, info


def evaluate(mdp):
    x, _ = mdp.reset()
    done = False
    reward = 0

    while not done:
        # Perform a step of the MDP
        u = env.action_space.sample()
        print("action", u)
        obs, r, done, *_ = mdp.step(u)
        print(obs, u, r)
        reward += r
    return reward


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # env = FeatureInverter(env, 0, 1)
    obs, _ = env.reset()
    # print("inv", obs)
    # print("##############")
    env = FeatureFilterWrapper(FeatureFilterWrapper(env, 3), 1)
    obs, _ = env.reset()
    print(obs)
    env = TimeExtensionWrapper(env)
    obs, _ = env.reset()
    print(obs)
    evaluate(env)
