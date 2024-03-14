from functools import partial

from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent, PrintAgent

from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl_algos.models.stochastic_actors import RandomContinuousOneDActor
from bbrl.agents.gymnasium import make_env, ParallelGymAgent


class BBRLTest:
    def __init__(self, n_envs):
        self.env_agent = ParallelGymAgent(
            partial(make_env, "Pendulum-v1", autoreset=True),
            n_envs,
            include_last_state=True,
            seed=1,
        )
        self.actor = RandomContinuousOneDActor()
        self.agent = TemporalAgent(Agents(self.env_agent, self.actor, PrintAgent()))

    def test(self):
        wp = Workspace()
        taille = 520
        self.agent(wp, n_steps=taille)
        assert wp.time_size() == taille
        tr_wp = wp.get_transitions()

        print(tr_wp.variables["env/timestep"].size)
        nb_trans = tr_wp.variables["env/timestep"].size[1]
        # assert tr_wp.time_size() == taille, f"real size = {tr_wp.time_size()}"
        if taille > 200:
            factor = int(taille / 200)
            assert nb_trans == taille - 1 - factor, f"real size = {nb_trans} / {taille}"
        else:
            assert nb_trans == taille - 1, f"real size = {nb_trans} / {taille}"
        rb = ReplayBuffer(300)
        rb.put(tr_wp)
        print(rb.size(), rb.is_full)


if __name__ == "__main__":
    testeur = BBRLTest(1)
    testeur.test()
