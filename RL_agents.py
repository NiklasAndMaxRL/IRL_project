from typing import List
import numpy as np


class ValueIteration:

    def __init__(self, states: List[str], terminal_states: List[str], gamma: float = 1, threshold: float = 1e-2) -> None:

        # initialise value function randomly
        self.value_function = {state: np.random.uniform() for state in states}
        # set value of terminal states to 0
        for terminal in terminal_states:
            self.value_function[terminal] = 0

        # save hyperparameters
        self.gamma = gamma
        self.threshold = threshold


class QLearningAgent:

    def __init__(self) -> None:
        pass
