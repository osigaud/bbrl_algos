#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import numpy as np
from typing import List

import hydra
import optuna
from omegaconf import DictConfig

# %%
import torch
import torch.nn as nn

# %%
import gymnasium as gym

from bbrl.visu.plot_critics import plot_discrete_q, plot_critic

from bbrl.utils.functional import gae
from bbrl.utils.chrono import Chrono

# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt

from tabularmazemdp.toolbox import sample_categorical
from tabularmazemdp.mdp import Mdp
from bbrl_gymnasium.envs.maze_mdp import MazeMDPEnv
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from functools import partial
from bbrl_algos.models.hyper_params import launch_optuna


matplotlib.use("TkAgg")


def actor_critic_averaging(cfg, logger=None, trial=None) -> float:
    v = np.zeros(cfg.nb_repeats)
    for i in range(cfg.nb_repeats):
        v[i] = actor_critic_v(cfg)
    return v.mean()


def actor_critic_v(cfg) -> float:
    mdp = gym.make("MazeMDP-v0", kwargs={"width": 5, "height": 5, "ratio": 0.2})
    nb_episodes = cfg.nb_episodes
    alpha_critic = cfg.alpha_critic
    alpha_actor = cfg.alpha_actor
    render = cfg.render
    v = np.zeros(mdp.nb_states)  # initial action values are set to 0
    policy = (
        np.ones((mdp.nb_states, mdp.action_space.n)) / mdp.action_space.n
    )  # action probabilities are uniform

    mdp.timeout = cfg.timeout  # max episode length

    if render:
        video_recorder = VideoRecorder(mdp, "videos/Actor-Critic.mp4", enabled=render)
        mdp.init_draw("SaActor-Critic", recorder=video_recorder)

    for _ in range(nb_episodes):
        # Draw the first state of episode i using a uniform distribution over all states
        x, _ = mdp.reset(uniform=True)
        done = False

        while not done:
            if render:
                mdp.draw_v(v, recorder=video_recorder)

            # Perform a step of the MDP
            u = sample_categorical(policy[x])
            y, r, done, *_ = mdp.step(u)

            # Update the state-action value function with bootstrap
            delta = r + mdp.gamma * v[y] * (1 - done) - v[x]
            v[x] = v[x] + alpha_critic * delta
            # Update the policy
            policy[x, u] = max(10e-8, policy[x, u] + alpha_actor * delta)

            total = sum(policy[x])
            for u1 in range(mdp.action_space.n):
                policy[x, u1] = policy[x, u1] / total

            # Update the agent position
            x = y

    if render:
        mdp.draw_v(v, recorder=video_recorder)
        video_recorder.close()

    return np.linalg.norm(v)


def eval_gridsearch(cfg):
    size = 20  # number of values to try for alpha actor and alpha critic
    ech = np.around(np.linspace(0.1, 1, size), 2)
    vals = np.zeros((size, size))
    i = 0
    for alpha_critic in ech:
        j = 0
        # OmegaConf does not support float64
        cfg.alpha_critic = np.float64(alpha_critic).item()
        for alpha_actor in ech:
            print(alpha_critic, alpha_actor)
            # OmegaConf does not support float64
            cfg.alpha_actor = np.float64(alpha_actor).item()
            vals[i, j] = actor_critic_averaging(cfg)
            j = j + 1
        i = i + 1
    plt.figure()
    plt.title("Landscape alpha_actor, alpha_critic")
    plt.imshow(np.where(vals > 0, vals, -1), cmap="RdYlGn_r")
    plt.xlabel("alpha_critic")
    plt.ylabel("alpha_actor")
    plt.xticks(range(size), labels=ech, rotation=90)
    plt.yticks(range(size), labels=ech)
    plt.colorbar()
    indx1, indx2 = np.unravel_index(vals.argmax(), vals.shape)
    alpha_critic = ech[indx1]
    alpha_actor = ech[indx2]
    max_val = vals[indx1, indx2]
    print()
    print(
        f"alpha critic: {alpha_critic}, alpha actor: {alpha_actor}, best_val: {max_val}"
    )
    # plt.savefig(f'alphas.png')
    plt.show()


@hydra.main(
    config_path="./configs/",
    # config_name="maze.yaml",
    config_name="maze_optuna.yaml",
)
def main(cfg: DictConfig):
    if "optuna" in cfg:
        launch_optuna(cfg, actor_critic_averaging)
    else:
        print(cfg)
        eval_gridsearch(cfg)


if __name__ == "__main__":
    main()
