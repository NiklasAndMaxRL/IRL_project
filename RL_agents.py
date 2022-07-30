from typing import Callable, Dict, List, Any


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
                 threshold: float = 1e-2) -> None:

        # initialise value function randomly
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

        self.converged = False

    def get_value_function(self) -> Dict[Any, float]:
        return self._value_function

    def get_optimal_action(self, action_state_pairs: Dict[Any, Any]) -> Any:
        """Return the optimal action from a Dict with actions-state pairs"""
        state_values = [self._reward_func[state] + self.gamma * self._value_function[state] for state in action_state_pairs.values()]
        # Check https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/ to understand the one-liner
        return max(zip(state_values, action_state_pairs.keys()))[1]

    def get_state_value(self, state) -> float:
        return self._value_function[state]

    def set_state_value(self, state, new_value):
        self._value_function[state] = new_value

    def construct_policy(self, get_action_state_pairs: Callable):
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


class QLearningAgent:

    def __init__(self,
                 states: List[Any],
                 terminal_states: List[Any],
                 reward_function: Dict[Any, float],
                 actions: List[Any],
                 gamma: float = 1,
                 lr: float = 1.0,
                 threshold: float = 1e-2) -> None:

        # initialise value function randomly
        self._Q_function = {state: {action: 0. for action in actions} for state in states}
        # set value of terminal states to 0
        self._terminal_states = terminal_states
        for terminal in self._terminal_states:
            for action in actions:
                self._Q_function[terminal] = .0

        self._reward_func = reward_function

        self.actions = actions

        # save hyperparameters
        self.gamma = gamma
        self.learning_rate = lr
        self.threshold = threshold

        self.converged = False

    def get_Q_function(self) -> Dict[Any, float]:
        return self._Q_function

    def get_optimal_action(self, action_state_pairs: Dict[Any, Any]) -> Any:
        """Return the optimal action from a Dict with actions-state pairs"""
        state_values = [self._reward_func[state] + self.gamma * self._value_function[state] for state in action_state_pairs.values()]
        
        for state in action_state_pairs:
            self.get_Q_function()[state]

        self.Q[state_idx, action_idx] + self.learning_rate * (self.R['win'] + self.discount * np.max(self.Q[next_state_idx, :]) - self.Q[state_idx, action_idx] )

        # Check https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/ to understand the one-liner
        return max(zip(state_values, action_state_pairs.keys()))[1]

    def get_state_action_value(self, state) -> float:
        return self._Q_function[state]

    def set_state_action_value(self, state, new_value):
        self._Q_function[state] = new_value

    def construct_policy(self, get_action_state_pairs: Callable):
        """Construct a policy out of the current value function. To do so, a function that takes a state and returns a Dict
        with the action-state pairs is needed (provided by the environment."""
        policy = {}
        for state in self._Q_function:
            if state in self._terminal_states:
                policy[state] = self.actions[0]  # for consistency, on terminal states we take the first action in the list. This is arbitrary but necessary
            else:
                policy[state] = self.get_optimal_action(get_action_state_pairs(state))
        self.policy = policy
        return self.policy
