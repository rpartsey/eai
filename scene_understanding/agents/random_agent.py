from gym.spaces import Discrete
import habitat


class RandomAgent(habitat.Agent):
    def __init__(self, config):
        self._action_spaces = Discrete(len(config.TASK.POSSIBLE_ACTIONS))

    def reset(self):
        pass

    def act(self, observations):
        return {"action": self._action_spaces.sample()}
