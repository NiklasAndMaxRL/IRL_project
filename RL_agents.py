from typing import Callable, Dict, List, Any, Tuple
import numpy as np


class PolicyExecutingAgent:

    def __init__(self) -> None:
        pass

    def set_policy(self, policy: Dict[Any, Any]):
        self.policy = policy

    def get_policy(self):
        return self.policy

    def select_action(self, state):
        return self.policy[state]


class ValueIterationAgent(PolicyExecutingAgent):

    def __init__(self,
                 states: List[Any],
                 terminal_states: List[Any],
                 reward_function: Dict[Any, float],
                 actions: List[Any],
                 gamma: float = 1,
                 threshold: float = 1e-4) -> None:

        # initialise value function to zero (consider random initialization?)
        self._value_function = {state: .0 for state in states}
        # set value of terminal states to 0
        self._terminal_states = terminal_states
        for terminal in self._terminal_states:
            self._value_function[terminal] = .0

        self._reward_func = reward_function

        self.actions = actions

        # save hyperparameters
        self.gamma = gamma
        self.threshold = threshold
        self.value_unchanged_counter = 0
        self.value_converged = False  # this flag indicates convergence by value

    def reset_agent(self):
        self._value_function = {state: .0 for state in self._value_function}
        self.value_unchanged_counter = 0
        self.value_converged = False

    def get_value_function(self) -> Dict[Any, float]:
        return self._value_function

    def set_value_function(self, new_value_function: Dict[Any, float]):
        if list(self._value_function.keys()) == list(new_value_function.keys()):
            self._value_function = new_value_function
        else:
            raise ValueError("Invalid value function. The states differ but they should be the same on both value functions.")

    def get_optimal_action(self, action_state_pairs: Dict[Any, Any]) -> Any:
        """Return the optimal action from a Dict with actions-state pairs"""
        state_values = [self._reward_func[state] + self.gamma * self._value_function[state] for state in action_state_pairs.values()]
        # Check https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/ to understand the one-liner
        return max(zip(state_values, action_state_pairs.keys()))[1]

    def get_state_value(self, state) -> float:
        return self._value_function[state]

    def set_state_value(self, state, new_value):
        """Set the value of state to new_value. Also check for convergence.

        If the updated values are close to the old ones (by .threshold) for one full round, set the .converged flag to True"""
        if np.isclose(self._value_function[state], new_value, atol=self.threshold):
            self.value_unchanged_counter += 1
            if self.value_unchanged_counter > len(self._value_function):
                self.value_converged = True
                self.value_unchanged_counter = 0
        else:
            self.value_unchanged_counter = 0

        self._value_function[state] = new_value

    def construct_greedy_policy(self, get_action_state_pairs: Callable):
        """Construct a policy out of the current value function. To do so, a function that takes a state and returns a Dict
        with the action-state pairs is needed (provided by the environment."""
        policy = {}
        for state in self._value_function:
            if state in self._terminal_states:
                policy[state] = self.actions[0]  # for consistency, on termial states we take the first action in the list. This is arbitrary but necessary
            else:
                policy[state] = self.get_optimal_action(get_action_state_pairs(state))
        self.policy = policy
        return self.policy


class QLearningAgent(PolicyExecutingAgent):

    def __init__(self,
                 states: List[Any],
                 terminal_states: List[Any],
                 reward_function: Dict[Any, float],
                 actions: List[Any],
                 gamma: float = 1,
                 lr: float = 1.0,
                 epsilon: float = 0.1,
                 threshold: float = 1e-2) -> None:

        # initialise value function randomly
        self._Q_function = {state: {action: np.random.rand() for action in actions} for state in states}
        # set value of terminal states to 0
        self._terminal_states = terminal_states
        for terminal in self._terminal_states:
            for action in actions:
                self._Q_function[terminal][action] = .0

        #print(self._Q_function[(0,0)][(1,0)])

        self._reward_func = reward_function

        self.actions = actions

        # save hyperparameters
        self.gamma = gamma
        self.learning_rate = lr
        self.threshold = threshold
        self.epsilon = epsilon #exploration

        self.converged = False

    def get_Q_function(self) -> Dict[Any, float]:
        return self._Q_function

    def update_Q_value(self, old_state:Tuple[int], new_state:Tuple[int], action:Tuple[int]) -> int:
        q_value = self.get_state_action_value(state=old_state, action=action)
        q_value = q_value + self.learning_rate * (self._reward_func[new_state] + self.gamma * np.max(list(self.get_Q_function()[new_state].values())) - q_value )
        self.set_state_action_value(old_state, action, q_value)
        return q_value

    def get_greedy_action(self, state) -> Tuple[int]:
        """ Returns the tuple of the action with the highest Q-Value from the given state """
        return list(self.get_Q_function()[state])[np.argmax(list(self.get_Q_function()[state].values()))]

    def get_state_action_value(self, state, action) -> float:
        return self._Q_function[state][action]

    def set_state_action_value(self, state, action, new_value):
        self._Q_function[state][action] = new_value

    def construct_greedy_policy(self, get_action_state_pairs: Callable):
        """Construct a policy out of the current value function. To do so, a function that takes a state and returns a Dict
        with the action-state pairs is needed (provided by the environment."""
        policy = {}
        for state in self._Q_function:
            if state in self._terminal_states:
                policy[state] = self.actions[0]  # for consistency, on terminal states we take the first action in the list. This is arbitrary but necessary
            else:
                policy[state] = self.get_greedy_action(state)
        self.policy = policy
        return self.policy
