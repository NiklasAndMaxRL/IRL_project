import argparse
from typing import Dict, Any, List
import numpy as np

from GridWorld_environments import Grid_World
from RL_agents import ValueIterationAgent
from IRL_agents import IRL_from_sampled_trajectories

GAMMA = 0.95
VALUE_ITERATION_TRAINING_N = 50

NUMBER_OF_TRAJECTORIES = 100
MAXIMUM_TRAJECTORY_LENGTH = 30

GW_SIZE = (4, 4)
GW_TRAPS = [(2, 3), (1, 2)]
GW_GOALS = [(3, 3)]


def train_value_iteration(gw_env: Grid_World):
    vi_agent = ValueIterationAgent(states=gw_env.get_state_space(),
                                   terminal_states=gw_env.get_terminal_states(),
                                   actions=gw_env.get_action_space())

    for trap in gw_env.traps:
        vi_agent.set_state_value(state=trap, new_value=gw_env.DEADLY_TRAP_REWARD)
    for goal in gw_env.goals:
        vi_agent.set_state_value(state=goal, new_value=gw_env.GOAL_REWARD)

    iters = 0
    while iters < VALUE_ITERATION_TRAINING_N and not vi_agent.converged:

        for state in gw_env.get_state_space():

            if state in gw_env.get_terminal_states():
                continue

            opt_act = vi_agent.get_optimal_action(action_state_pairs=gw_env.get_action_state_pairs(state=state))
            next_state = gw_env.get_new_state_on_action(old_state=state, action=opt_act)
            next_state_value = vi_agent.get_state_value(state=next_state)

            vi_agent.set_state_value(state=state, new_value=(GAMMA * next_state_value))

        iters += 1
        # print(f"Iteration {iters}")
        # print(vi_agent.get_value_function())

    print("Board:")
    print(gw_env.get_board())

    gw_env.display_value_function(value_func=vi_agent.get_value_function())

    vi_agent.construct_policy(gw_env.get_action_state_pairs)

    gw_env.display_policy(policy=vi_agent.get_policy())

    return vi_agent.get_policy()


def irl_reward_estimation(env: Grid_World, optimal_trajectories: List[List[Any]]):

    irl_agent = IRL_from_sampled_trajectories(d=GW_SIZE,
                                              env_ranges=((0, GW_SIZE[0]), (0, GW_SIZE[1])),
                                              env_discrete_size=GW_SIZE,
                                              gamma=GAMMA)

    # step 2: given optimal trajectories, compute the value estimate
    optimal_value_estimate = irl_agent.compute_value_estimate(trajs=optimal_trajectories)

    # step 3: generate trajectories and compute the value estimate for a random policy
    candidate_policies = [env.construct_random_policy()]
    random_trajectories = env.generate_trajectories(policy=candidate_policies[0],
                                                    n_traj=NUMBER_OF_TRAJECTORIES,
                                                    max_traj_length=MAXIMUM_TRAJECTORY_LENGTH)
    random_value_estimate = irl_agent.compute_value_estimate(trajs=random_trajectories)

    # step 4:
    alphas = irl_agent.solve_lp(optimal_value_estimate, [random_value_estimate])

    print(alphas)

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-vi", "--value-iteration", required=False, default=False, action="store_true")
    parser.add_argument("-gt", "--generate-trajectories", type=int, required=False, default=0)
    parser.add_argument("-irl", "--inverse-rl", required=False, default=False, action="store_true")

    args = parser.parse_args()
    print(f"Passed args: {args}")

    environment = Grid_World(size=GW_SIZE, traps=GW_TRAPS, goals=GW_GOALS)

    if args.value_iteration:
        print("Training via value iteration...")
        greepy_policy = train_value_iteration(gw_env=environment)
    else:
        # load from file (?)
        greepy_policy = {}

    if args.generate_trajectories:
        print(f"Generating {NUMBER_OF_TRAJECTORIES} trajectories...")
        trajectories = environment.generate_trajectories(policy=greepy_policy,
                                                         n_traj=NUMBER_OF_TRAJECTORIES,
                                                         max_traj_length=MAXIMUM_TRAJECTORY_LENGTH)

    if args.inverse_rl:
        print("IRL from samples...")
        estimated_reward = irl_reward_estimation(env=environment, optimal_trajectories=trajectories)

    print("Closing up the arena...")
