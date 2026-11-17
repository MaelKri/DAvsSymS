"""Expert scripté de l'équipe 4 (osigaud/pystk_project), adapté à supertuxkart/full-v0."""

from .team4.agent4 import Agent4


class Team4Expert:
    def __init__(self):
        self.agent = Agent4(env=None)

    def reset(self):
        a = self.agent
        a.isEnd = False
        for e in (a.expert_rescue, a.expert_banana_dodge, a.expert_esquive_adv, a.expert_pilot):
            e.reset()

    def __call__(self, obs):
        return self.agent.choose_action(obs)
