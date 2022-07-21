from typing import List, Tuple
import numpy as np


class Grid_World():
    """Grid World in 2D with deterministic actions.

        # board:
        # 0 is empty
        # -1 is wall
        # -10 is trap
        # +1 is intermediate_goal
        # +10 is goal
        # #2 is player
    """

    GOAL_REWARD = 10
    DEADLY_TRAP_REWARD = -10
    INTERMEDIATE_GOAL_REWARD = 1
    WALL = -1

    def __init__(self,
                 size: Tuple[int],
                 start: Tuple[int] = (0, 0),
                 walls: List[Tuple[int]] = [],
                 traps: List[Tuple[int]] = [],
                 intermediate_goals: List[Tuple[int]] = [],
                 goals: List[Tuple[int]] = []):

        self.possible_actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.gameover, self.win, self.lose = False, False, False

        self.n_rows, self.n_cols = size
        self.player_pos = start

        # store lists of objects
        self.walls = walls
        self.traps = traps
        self.int_goals = intermediate_goals
        self.goals = goals

        # construct board
        self.board = np.zeros((self.n_rows, self.n_cols))
        # populate board
        for wall in walls:
            self.board[wall] = self.WALL
        for trap in traps:
            self.board[trap] = self.DEADLY_TRAP_REWARD
        for int_goal in intermediate_goals:
            self.board[int_goal] = self.INTERMEDIATE_GOAL_REWARD
        for goal in goals:
            self.board[goal] = self.GOAL_REWARD

    def _get_board(self):
        return self.board

    def _get_random_action(self):
        return self.possible_actions[np.random.choice([x for x, _ in enumerate(self.possible_actions)])]  # optimize!!!

    def check_gameover(self):
        if self.player_pos in self.traps:
            self.gameover = True
            self.win = False
        if self.player_pos in self.goals:
            self.gameover = True
            self.lose = True
        return self.gameover, self.win, self.lose

    def take_action(self, action):
        if action not in self.possible_actions:
            print(f"Invalid action '{action}'. Please choose one from '{self.possible_actions}'")
            return self.player_pos, 0, self.gameover, self.win, self.lose  # player_pos, reward, gameover, win, lose

        old_pos = self.player_pos

        # calculate the new positon
        new_pos = (self.player_pos[0] + action[0], self.player_pos[1] + action[1])

        # check if the new position is out of bounds
        if (new_pos[0] < 0 or new_pos[0] > self.n_rows - 1) or (new_pos[1] < 0 or new_pos[1] > self.n_cols - 1):
            new_pos = old_pos
        # check if there is a wall in the new positon
        if new_pos in self.walls:
            new_pos = old_pos
        # new position is valid, therefore store it
        self.player_pos = new_pos

        # collect reward in new position
        reward = self.board[self.player_pos]

        # check gameover
        self.check_gameover()

        print(f"Action '{action}' taken at position {old_pos}. New position is '{self.player_pos}' and received a reward...")
        return self.player_pos, reward, self.gameover, self.win, self.lose  # player_pos, reward, gameover, win, lose
