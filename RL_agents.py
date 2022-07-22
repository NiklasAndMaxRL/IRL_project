from typing import Dict, List, Tuple, Any
import numpy as np


class ValueIteration:

    def __init__(self,
                 states: List[Any],
                 terminal_states: List[Any],
                 actions: List[Any],
                 gamma: float = 1,
                 threshold: float = 1e-2) -> None:

        # initialise value function randomly
        self.value_function = {state: 0 for state in states}
        # set value of terminal states to 0
        for terminal in terminal_states:
            self.value_function[terminal] = 0

        self.actions = actions

        # save hyperparameters
        self.gamma = gamma
        self.threshold = threshold

        self.converged = False

    def get_optimal_action(self, action_state_pairs: Dict[Any, Any]):
        """Return the optimal action from a Dict with actions-state pairs"""
        # Check https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/ to understand the one-liner
        state_values = [self.value_function[state] for state in action_state_pairs.values()]
        return max(zip(state_values, action_state_pairs.keys()))[1]

    def get_state_value(self, state):
        return self.value_function[state]

    def set_state_value(self, state, new_value):
        self.value_function[state] = new_value




class QLearningAgent:

    def __init__(self) -> None:
        pass
