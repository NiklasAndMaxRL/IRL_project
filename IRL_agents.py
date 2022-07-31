from typing import List, Any, Tuple, Union
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import linprog


class IRL_from_sampled_trajectories:

    def __init__(self,
                 d: Tuple[int],
                 env_ranges: Tuple[Tuple[float]],
                 env_discrete_size: Tuple[int],
                 penalty_factor: float,
                 gamma: float,
                 cov: Union[float, np.ndarray] = 5) -> None:
        """_summary_

        Args:
            d (Tuple[int]): number of approximating functions per dimension (rows, cols)
            env_ranges (Tuple[Tuple[float]]): min and max ranges of the environment. Used for proper scaling of distances.
            env_discrete_size (Tuple[int]): size of the space distretization grid (rows, cols)
            gamma (float): _description_
            cov (Union[float, np.ndarray], optional): covariance of the approximating function (default is Gaussian). Defaults to 5.
        """
        # unpack input tuples
        self.d_rows, self.d_cols = d
        (self.env_rows_min, self.env_rows_max), (self.env_cols_min, self.env_cols_max) = env_ranges
        self.env_n_rows, self.env_n_cols = env_discrete_size

        if self.d_rows < self.env_n_rows or self.d_cols < self.env_n_cols:
            print(f"It is recommended that the number of approximating functions '{d}' be larger than" +
                  f"the discrete space size '{env_discrete_size}' over all dimensions.")

        # determine the approximating funcs centers
        d_row_centers, step = np.linspace(self.env_rows_min, self.env_rows_max, self.d_rows, endpoint=False, retstep=True)
        # d_row_centers += step / 2
        d_col_centers, step = np.linspace(self.env_cols_min, self.env_cols_max, self.d_cols, endpoint=False, retstep=True)
        # d_col_centers += step / 2
        # aggregate to one list
        self.d_centers = [(i, j) for i in d_row_centers.round(4) for j in d_col_centers.round(4)]

        # # determine the discrete state space
        # env_row_centers, step = np.linspace(self.env_rows_min, self.env_rows_max, self.env_n_rows, endpoint=False, retstep=True)
        # # env_row_centers += step / 2
        # env_col_centers, step = np.linspace(self.env_cols_min, self.env_cols_max, self.env_n_cols, endpoint=False, retstep=True)
        # # env_col_centers += step / 2
        # # aggregate to one list
        # self.env_mesh = [(i, j) for i in env_row_centers.round(4) for j in env_col_centers.round(4)]

        self.penalty_factor = penalty_factor
        self.gamma = gamma
        self.approx_func_cov = cov

        self.alphas = np.random.uniform(-1, 1, size=len(self.d_centers))

    def _approx_func(self, x, center):
        return multivariate_normal.pdf(x=x, mean=center, cov=self.approx_func_cov)

    def get_alphas(self):
        return self.alphas.tolist()

    def compute_value_estimate(self, trajs: List[List[Any]]) -> List[float]:
        """Given a List of trajectories, return a List with the value estimates for each approximating function."""
        value_estimates = []
        for approx_center in self.d_centers:
            temp_value = 0

            for traj in trajs:
                for i, state in enumerate(traj):
                    temp_value += self.gamma**i * self._approx_func(x=state, center=approx_center)

            temp_value /= len(trajs)
            value_estimates.append(temp_value)

        return value_estimates

    def compute_action_value_estimate(self, trajs: List[List[Any]]) -> List[float]:
        """Given a List of trajectories, return a List with the value estimates for each approximating function."""
        action_value_estimates = []
        for approx_center in self.d_centers:
            temp_value = 0

            for traj in trajs:
                for i, state in enumerate(traj):
                    temp_value += self.gamma**i * self._approx_func(x=state, center=approx_center)

            temp_value /= len(trajs)
            action_value_estimates.append(temp_value)

        return action_value_estimates

    def solve_lp(self, target_estimate, candidate_estimates):
        """Solve the Linear programing task at hand:
        Maximize: sum over i of p(V_target - V_candidate_i) s.t. |alpha_i| <= 1
        Where V_target and V_candiate_i are the product of the estimates times the alphas
        And where p is x if x >= 0 and 2x if x < 0
        """

        target_value = np.dot(np.array(target_estimate), self.alphas)  # this returns a scalar
        candidate_values = np.dot(np.array(candidate_estimates), self.alphas)  # this returns an array of len(candidate_estimates)

        lp_input = np.zeros(len(self.d_centers))

        for i in range(len(candidate_estimates)):
            if target_value - candidate_values[i] >= 0:
                lp_input += np.array(candidate_estimates[i]) - np.array(target_estimate)
            else:
                lp_input += self.penalty_factor * (np.array(candidate_estimates[i]) - np.array(target_estimate))

        res = linprog(lp_input, bounds=(-1, 1), method="simplex")

        self.alphas = np.array(res.x)
        return self.alphas

    def construct_reward_function(self, alphas: List[float]):
        """Given a list of alphas (of lenght equal to the number of approximating functions), construct a reward function
        on the discrete environment mesh."""
        # new_reward = np.zeros((self.env_n_rows, self.env_n_cols))
        new_reward = {}

        for i in range(self.env_n_rows):
            for j in range(self.env_n_cols):
                temp_value = 0

                for k, approx_center in enumerate(self.d_centers):
                    temp_value += alphas[k] * self._approx_func(x=(i, j), center=approx_center)

                new_reward[(i, j)] = temp_value

        return new_reward

# aa = IRL_from_sampled_trajectories(d=(2, 34), env_ranges=((0, 5), (0, 5)), env_discrete_size=(6, 6))
