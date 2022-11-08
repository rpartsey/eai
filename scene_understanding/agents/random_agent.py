from gym.spaces import Discrete
import habitat


class RandomAgent(habitat.Agent):
    def __init__(self, config):
        self._action_spaces = Discrete(len(config.TASK.POSSIBLE_ACTIONS))

    def reset(self):
        pass

    def act(self, observations):
        return {"action": self._action_spaces.sample()}


class RandomObjectDetectionAgent(RandomAgent):
    def __init__(self, config):
        super().__init__(config)
        # init object detection attributes here

    def reset(self):
        # reset object detection attributes here
        # recall, that reset is called at the episode reset (i.e. before new episode starts)
        pass

    def do_object_detection(self, observations):
        # define object detection here
        raise NotImplementedError

    def act(self, observations):
        # do object detection here:
        self.do_object_detection(observations)
        return super().act(observations)
