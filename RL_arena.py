import argparse
from typing import Any, List

from GridWorld_environments import Grid_World
from RL_agents import ValueIterationAgent, QLearningAgent
from IRL_agents import IRL_from_sampled_trajectories

GAMMA = 0.95
VALUE_ITERATION_TRAINING_N = 500

NUMBER_OF_TRAJECTORIES = 50
MAXIMUM_TRAJECTORY_LENGTH = 50

GW_SIZE = (5, 5)
GW_TRAPS = []
GW_GOALS = [(0, 4)]


def train_value_iteration(gw_env: Grid_World):
    vi_agent = ValueIterationAgent(states=gw_env.get_state_space(),
                                   terminal_states=gw_env.get_terminal_states(),
                                   reward_function=gw_env.get_reward_func(),
                                   actions=gw_env.get_action_space(),
                                   gamma=GAMMA)

    # for trap in gw_env.traps:
    #     vi_agent.set_state_value(state=trap, new_value=gw_env.DEADLY_TRAP_REWARD)
    # for goal in gw_env.goals:
    #     vi_agent.set_state_value(state=goal, new_value=gw_env.GOAL_REWARD)

    iters = 0
    while iters < VALUE_ITERATION_TRAINING_N and not vi_agent.converged:

        for state in gw_env.get_state_space():

            if state in gw_env.get_terminal_states():
                continue

            opt_act = vi_agent.get_optimal_action(action_state_pairs=gw_env.get_action_state_pairs(state=state))
            next_state = gw_env.get_new_state_on_action(old_state=state, action=opt_act)
            next_state_value = vi_agent.get_state_value(state=next_state)

            vi_agent.set_state_value(state=state, new_value=(gw_env.get_state_reward(state=next_state) + GAMMA * next_state_value))

        iters += 1
        # print(f"Iteration {iters}")
        # print(vi_agent.get_value_function())

    print("Board:")
    print(gw_env.get_board())

    gw_env.display_value_function(value_func=vi_agent.get_value_function())

    vi_agent.construct_policy(gw_env.get_action_state_pairs)

    gw_env.display_policy(policy=vi_agent.get_policy())

    return vi_agent.get_policy()


def train_q_learning(gw_env: Grid_World):
    ql_agent = QLearningAgent(states=gw_env.get_state_space(),
                                   terminal_states=gw_env.get_terminal_states(),
                                   reward_function=gw_env.get_reward_func(),
                                   actions=gw_env.get_action_space(),
                                   gamma=GAMMA)

    # for trap in gw_env.traps:
    #     vi_agent.set_state_value(state=trap, new_value=gw_env.DEADLY_TRAP_REWARD)
    # for goal in gw_env.goals:
    #     vi_agent.set_state_value(state=goal, new_value=gw_env.GOAL_REWARD)

    iters = 0
    while iters < VALUE_ITERATION_TRAINING_N and not ql_agent.converged:

        for state in gw_env.get_state_space():

            if state in gw_env.get_terminal_states():
                continue

            opt_act = ql_agent.get_optimal_action(state, action_state_pairs=gw_env.get_action_state_pairs(state=state))
            next_state = gw_env.get_new_state_on_action(old_state=state, action=opt_act)
            next_q_value = ql_agent.get_state_action_value(state=next_state, action=opt_act)

            ql_agent.set_state_action_value(state=state, action=opt_act, new_value=(gw_env.get_state_reward(state=next_state) + GAMMA * next_q_value))

        iters += 1
        # print(f"Iteration {iters}")
        # print(vi_agent.get_value_function())

    print("Board:")
    print(gw_env.get_board())

    gw_env.display_q_function(q_func=ql_agent.get_Q_function())

    ql_agent.construct_policy(gw_env.get_action_state_pairs)

    gw_env.display_policy(policy=ql_agent.get_policy())

    return ql_agent.get_policy()



def irl_reward_estimation(env: Grid_World, optimal_trajectories: List[List[Any]]):

    irl_agent = IRL_from_sampled_trajectories(d=(20, 20),
                                              env_ranges=((0, GW_SIZE[0]), (0, GW_SIZE[1])),
                                              env_discrete_size=GW_SIZE,
                                              penalty_factor=2,
                                              gamma=GAMMA)

    # step 2: given optimal trajectories, compute the value estimate
    optimal_value_estimate = irl_agent.compute_value_estimate(trajs=optimal_trajectories)
    print("Optimal value estimates from optimal trajectory:\n", optimal_value_estimate)

    # step 3: generate trajectories and compute the value estimate for a random policy
    candidate_policies = [env.construct_random_policy()]
    candidate_value_estimates = []

    # while True:
    for i in range(100):
        candidate_trajectories = env.generate_trajectories(policy=candidate_policies[-1],
                                                           n_traj=NUMBER_OF_TRAJECTORIES,
                                                           max_traj_length=MAXIMUM_TRAJECTORY_LENGTH)
        candidate_value_estimates.append(irl_agent.compute_value_estimate(trajs=candidate_trajectories))

        # step 4: obtain new alphas
        irl_agent.solve_lp(optimal_value_estimate, candidate_value_estimates)

        # step 5: construct new reward function from the alphas
        reward_func = irl_agent.construct_reward_function(alphas=irl_agent.get_alphas())

        # step 6: find optimal policy under new reward function and add to 'candidate_policies' list
        env.set_reward_func(reward_func)
        candidate_policies.append(train_value_iteration(gw_env=env))

        print(f"Iteration {i}...")
        # print(f"Alphas ({len(irl_agent.get_alphas())}):\n", np.array(irl_agent.get_alphas()).reshape(GW_SIZE))
        print("Latest rewardfunc:\n", env.get_board())
        env.display_policy(policy=candidate_policies[-1])
        print("============================================================\n" * 2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-vi", "--value-iteration", required=False, default=False, action="store_true")
    parser.add_argument("-ql", "--q-learning", required=False, default=False, action="store_true")
    parser.add_argument("-gt", "--generate-trajectories", required=False, default=False, action="store_true")
    parser.add_argument("-irl", "--inverse-rl", required=False, default=False, action="store_true")

    args = parser.parse_args()
    print(f"Passed args: {args}")

    environment = Grid_World(size=GW_SIZE, traps=GW_TRAPS, goals=GW_GOALS)

    if args.value_iteration:
        print("Training via value iteration...")
        greedy_policy = train_value_iteration(gw_env=environment)
    elif args.q_learning:
        print("Training via q-learning...")
        greedy_policy = train_q_learning(gw_env=environment)        
    else:
        # load from file (?)
        greedy_policy = {}

    if args.generate_trajectories:
        print(f"Generating {NUMBER_OF_TRAJECTORIES} trajectories...")
        trajectories = environment.generate_trajectories(policy=greedy_policy,
                                                         n_traj=NUMBER_OF_TRAJECTORIES,
                                                         max_traj_length=MAXIMUM_TRAJECTORY_LENGTH)

    if args.inverse_rl:
        print("IRL from samples...")
        estimated_reward = irl_reward_estimation(env=environment, optimal_trajectories=trajectories)

    print("Closing up the arena...")
