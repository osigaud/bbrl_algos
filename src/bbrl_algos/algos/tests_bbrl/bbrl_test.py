import copy
import torch
import numpy as np

from functools import partial

from bbrl.utils.chrono import Chrono

from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent, PrintAgent

from bbrl_algos.models.loggers import Logger
from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl_algos.models.stochastic_actors import RandomContinuousOneDActor
from bbrl_algos.models.actors import ContinuousDeterministicActor
from bbrl.agents.gymnasium import make_env, ParallelGymAgent
from gymnasium.wrappers import AutoResetWrapper

import gymnasium as gym


class BBRLTest:
    def __init__(self, n_envs):
        self.env_agent = ParallelGymAgent(
            partial(make_env, "Pendulum-v1"),
            n_envs,
            include_last_state=True,
            seed=1,
        )
        obs_size, act_size = self.env_agent.get_obs_and_actions_sizes()
        # self.actor = ContinuousDeterministicActor(obs_size, [10, 10], act_size, seed=1)
        self.actor = RandomContinuousOneDActor()
        self.agent = TemporalAgent(Agents(self.env_agent, self.actor, PrintAgent()))

    def test(self):
        wp = Workspace()
        taille = 1000
        self.agent(wp, n_steps=taille)
        assert wp.time_size() == taille
        tr_wp = wp.get_transitions()
        assert tr_wp.time_size() == taille, f"real size = {wp.time_size()}"


if __name__ == "__main__":
    testeur = BBRLTest(1)
    testeur.test()
