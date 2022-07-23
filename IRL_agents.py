from typing import List, Any, Tuple, Union
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import linprog


class IRL_from_sampled_trajectories:

    def __init__(self,
                 d: Tuple[int],
                 env_ranges: Tuple[Tuple[float]],
                 env_discrete_size: Tuple[int],
                 gamma: float,
                 cov: Union[float, np.ndarray] = 2) -> None:
        # unpack input tuples
        self.d_rows, self.d_cols = d
        (self.env_rows_min, self.env_rows_max), (self.env_cols_min, self.env_cols_max) = env_ranges
        self.env_n_rows, self.env_n_cols = env_discrete_size

        if self.d_rows < self.env_n_rows or self.d_cols < self.env_n_cols:
            print(f"It is recommended that the number of approximating functions '{d}' be larger than" +
                  f"the discrete space size '{env_discrete_size}'")

        # determine the approximating funcs centers
        d_row_centers, step = np.linspace(self.env_rows_min, self.env_rows_max, self.d_rows, endpoint=False, retstep=True)
        # d_row_centers += step / 2
        d_col_centers, step = np.linspace(self.env_cols_min, self.env_cols_max, self.d_cols, endpoint=False, retstep=True)
        # d_col_centers += step / 2
        # aggregate to one list
        self.d_centers = [(i, j) for i in d_row_centers for j in d_col_centers]

        self.gamma = gamma
        self.approx_func_cov = cov

    def _approx_func(self, x, center):
        return multivariate_normal.pdf(x=x, mean=center, cov=self.approx_func_cov)

    def compute_value_estimate(self, trajs: List[List[Any]]):
        value_estimates = []
        for approx_center in self.d_centers:
            temp_value = 0

            for traj in trajs:
                for i, step in enumerate(traj):
                    temp_value += self.gamma**i * self._approx_func(x=step, center=approx_center)

            temp_value /= len(trajs)
            value_estimates.append(temp_value)

        return value_estimates

    def solve_lp(self, target_estimate, candidate_estimates):
        c = [0] * len(self.d_centers) + [-1] * (len(candidate_estimates))
        A = [[0 for _ in range(len(c))] for _ in range(2 * len(candidate_estimates))]
        b = [0] * (2 * len(candidate_estimates))
        bound = [(-1, 1) for _ in range(len(self.d_centers))] + [(None, None) for _ in range(len(candidate_estimates))]

        for i in range(len(candidate_estimates)):
            A[2 * i][len(candidate_estimates) + i] = 1
            A[2 * i + 1][len(candidate_estimates) + i] = 1

            for j in range(len(candidate_estimates)):
                A[2 * i][j]     = -target_estimate[j] + candidate_estimates[i][j]
                A[2 * i + 1][j] = 4 * (-target_estimate[j] + candidate_estimates[i][j])

        res = linprog(c, A_ub=A, b_ub=b, bounds=bound)
        return res['x']


# aa = IRL_from_sampled_trajectories(d=(2, 34), env_ranges=((0, 5), (0, 5)), env_discrete_size=(6, 6))
