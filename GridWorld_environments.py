from typing import Any, List, Dict, Tuple
import numpy as np


class Grid_World:
    """Grid World in 2D with deterministic actions.
    """

    GOAL_REWARD = 1
    DEADLY_TRAP_REWARD = -1
    INT_GOAL_REWARD = 0.2
    STEP_REWARD = -0.04
    WALL = -1

    def __init__(self,
                 size: Tuple[int],
                 start: Tuple[int] = (0, 0),
                 walls: List[Tuple[int]] = [],
                 traps: List[Tuple[int]] = [],
                 intermediate_goals: List[Tuple[int]] = [],
                 goals: List[Tuple[int]] = [],
                 randomize_board: bool = False):

        self.possible_actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # South, East, North, West
        self.actions_to_str_map = {(1, 0): "v", (0, 1): ">", (-1, 0): "^", (0, -1): "<", None: "-"}
        self.gameover, self.win, self.lose = False, False, False

        self.n_rows, self.n_cols = size
        self.player_pos = start

        # generate random objects or store passed lists of objects
        if randomize_board:
            N_total = self.n_rows * self.n_cols

            # get random walls - 5% of total fields will be walls
            self.walls = []
            for _ in range(int(N_total * 0.05)):
                candidate = (np.random.choice(self.n_rows), np.random.choice(self.n_cols))
                while candidate in self.walls:
                    candidate = (np.random.choice(self.n_rows), np.random.choice(self.n_cols))
                self.walls.append(candidate)

            # get random traps - 5% of total fields will be traps
            self.traps = []
            for _ in range(int(N_total * 0.05)):
                candidate = (np.random.choice(self.n_rows), np.random.choice(self.n_cols))
                while candidate in (self.walls + self.traps):
                    candidate = (np.random.choice(self.n_rows), np.random.choice(self.n_cols))
                self.traps.append(candidate)

            # get random intermediate goals - 2.5% of total fields will be traps
            self.int_goals = []
            for _ in range(int(N_total * 0.025)):
                candidate = (np.random.choice(self.n_rows), np.random.choice(self.n_cols))
                while candidate in (self.walls + self.traps + self.int_goals):
                    candidate = (np.random.choice(self.n_rows), np.random.choice(self.n_cols))
                self.int_goals.append(candidate)

            # get random goals - for now just one goal
            self.goals = []
            for _ in [1]:
                candidate = (np.random.choice(self.n_rows), np.random.choice(self.n_cols))
                while candidate in (self.walls + self.traps + self.int_goals + self.goals):
                    candidate = (np.random.choice(self.n_rows), np.random.choice(self.n_cols))
                self.goals.append(candidate)
        else:
            self.walls = walls
            self.traps = traps
            self.int_goals = intermediate_goals
            self.goals = goals

        # construct board
        self.board = np.zeros((self.n_rows, self.n_cols))
        # populate board
        for trap in self.traps:
            self.board[trap] = self.DEADLY_TRAP_REWARD
        for int_goal in self.int_goals:
            self.board[int_goal] = self.INT_GOAL_REWARD
        for goal in self.goals:
            self.board[goal] = self.GOAL_REWARD

        # state space
        self._state_space = [(i, j) for i in range(self.n_rows) for j in range(self.n_cols) if (i, j) not in self.walls]
        # terminal states
        self._terminal_states = self.traps + self.goals

    def get_board(self):
        return self.board

    def set_board(self, new_board):
        self.board = new_board

    def get_state_reward(self, state: Tuple[int]):
        return self.board[state]

    def set_reward_func(self, new_reward_func):
        for state in self._state_space:
            self.board[state] = new_reward_func[state]

    def get_reward_func(self):
        return {state: self.board[state] for state in self._state_space}

    def get_action_space(self):
        return self.possible_actions

    def get_state_space(self):
        return self._state_space

    def get_terminal_states(self):
        return self._terminal_states

    def get_action_state_pairs(self, state: Tuple[int]) -> Dict[Tuple[int], Tuple[int]]:
        """Return a Dict with the possible actions and the results states at a given state"""
        return {action: self.get_new_state_on_action(old_state=state, action=action) for action in self.possible_actions}

    def _get_random_action(self):
        return self.possible_actions[np.random.choice([x for x, _ in enumerate(self.possible_actions)])]  # optimize!!!

    def construct_random_policy(self):
        policy = {}
        for state in self._state_space:
            if state in self._terminal_states:
                policy[state] = self.possible_actions[0]  # for consistency, on termial states we take the first action in the list. This is arbitrary but necessary
            else:
                policy[state] = self._get_random_action()
        return policy

    def reset_env(self, state):
        if state in self._state_space:
            self.gameover, self.win, self.lose = False, False, False
            self.player_pos = state
            self.check_gameover()
        else:
            raise ValueError(f"Invalid state '{state}' passed to reset_env()")

    def check_gameover(self):
        if self.player_pos in self.traps:
            self.gameover, self.win, self.lose = True, False, True
        if self.player_pos in self.goals:
            self.gameover, self.win, self.lose = True, True, False
        return self.gameover, self.win, self.lose

    def get_new_state_on_action(self, old_state: Tuple[int], action: Tuple[int]):
        """Return the new state after taking action when in old_state"""
        # calculate the new state
        new_state = (old_state[0] + action[0], old_state[1] + action[1])

        # check if the new position is out of bounds
        if (new_state[0] < 0 or new_state[0] > self.n_rows - 1) or (new_state[1] < 0 or new_state[1] > self.n_cols - 1):
            new_state = old_state
        # check if there is a wall in the new positon
        if new_state in self.walls:
            new_state = old_state

        return new_state

    def take_action(self, action, verbose=False):
        if action not in self.possible_actions:
            print(f"Invalid action '{action}'. Please choose one from '{self.possible_actions}'")
            return self.player_pos, 0, self.gameover, self.win, self.lose  # player_pos, reward, gameover, win, lose

        # calculate the new positon
        self.player_pos = self.get_new_state_on_action(old_state=self.player_pos, action=action)
        # collect reward in new position
        reward = self.board[self.player_pos]
        # check gameover
        self.check_gameover()

        if verbose:
            print(f"Taken action '{action}'. New state is '{self.player_pos}' and received a reward...")
        return self.player_pos, reward, self.gameover, self.win, self.lose  # player_pos, reward, gameover, win, lose

    def generate_trajectories(self, policy: Dict[Any, Any], n_traj: int, max_traj_length: int):
        trajs = []
        for _ in range(n_traj):
            initial_state = self._state_space[np.random.choice(range(len(self._state_space)))]
            self.reset_env(state=initial_state)
            traj = [initial_state]

            for _ in range(max_traj_length):
                if self.gameover:  # check for gameover first, in case the initial state is already terminal
                    break
                action = policy[self.player_pos]
                self.take_action(action=action)
                traj.append(self.player_pos)
            trajs.append(traj)

        self.reset_env(state=self._state_space[0])
        return trajs

    def display_q_function(self, q_func: Dict[Tuple[int], float]):
        q_func_arr = np.zeros((self.n_rows, self.n_cols, len(self.possible_actions)))

        for state in q_func:
            for act_idx, action in enumerate(self.possible_actions):
                q_func_arr[state[0], state[1], act_idx] = q_func[state][action]

        print("Q function:")
        print(q_func_arr)

    def display_value_function(self, value_func: Dict[Tuple[int], float]):
        value_func_arr = np.zeros((self.n_rows, self.n_cols))

        for state in value_func:
            value_func_arr[state] = value_func[state]

        print("Value function:")
        print(value_func_arr)

    def display_policy(self, policy: Dict[Tuple[int], Tuple[int]]):

        policy_str = {state: self.actions_to_str_map[action] for state, action in policy.items()}

        policy_arr = np.zeros((self.n_rows, self.n_cols), str)

        for state in policy_str:
            policy_arr[state] = policy_str[state]

        for state in self._terminal_states:
            policy_arr[state] = "x"

        print("Policy:")
        print(policy_arr)
