import argparse
from typing import Dict, Any
import numpy as np

from GridWorld_environments import Grid_World
from RL_agents import ValueIterationAgent

GAMMA = 0.95

VALUE_ITERATION_TRAINING_N = 50

GW_SIZE = (4, 4)
GW_TRAPS = [(2, 3), (1, 2)]
GW_GOALS = [(3, 3)]


def train_value_iteration(gw_env: Grid_World):
    print("Training RL with Value Iteration...")

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


def generate_trajectories(env: Grid_World, policy: Dict[Any, Any], n_traj: int, max_traj_length: int = 30):

    trajs = []
    state_space = env.get_state_space()

    for i in range(n_traj):

        initial_state = state_space[np.random.choice(range(len(state_space)))]
        traj = [initial_state]
        env.reset_env(state=initial_state)

        for _ in range(max_traj_length):

            action = policy[env.player_pos]
            env.take_action(action=action)
            traj.append(env.player_pos)
            if env.gameover:
                break

        print(traj)
        trajs.append(traj)

    return trajs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-vi", "--value-iteration", required=False, default=False, action="store_true")
    parser.add_argument("-gt", "--generate-trajectories", type=int, required=False, default=0)

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
        print("Generating trajectories...")
        trajectories = generate_trajectories(env=environment, policy=greepy_policy, n_traj=args.generate_trajectories)

    print("Closing the RL_arena...")
