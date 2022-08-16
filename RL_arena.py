import argparse
from typing import Any, List, Callable

from GridWorld_environments import Grid_World
from RL_agents import ValueIterationAgent, QLearningAgent
from IRL_agents import IRL_from_sampled_trajectories

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

GAMMA = 0.95
VALUE_ITERATION_TRAINING_N = 50
IRL_TRAINING_N = 5

NUMBER_OF_TRAJECTORIES = 40
MAXIMUM_TRAJECTORY_LENGTH = 50

# GW_SIZE = (5, 5)
GW_SIZES = [(5, 5)]  # [(x, x) for x in np.arange(5,11, 5)]
GW_TRAPS = []
GW_GOALS = [(4, 4)]


def train_value_iteration(gw_env: Grid_World, verbose=False):
    vi_agent = ValueIterationAgent(states=gw_env.get_state_space(),
                                   terminal_states=gw_env.get_terminal_states(),
                                   reward_function=gw_env.get_reward_func(),
                                   actions=gw_env.get_action_space(),
                                   gamma=GAMMA)

    iters = 0
    while iters < VALUE_ITERATION_TRAINING_N and not vi_agent.value_converged:

        for state in gw_env.get_state_space():

            if state in gw_env.get_terminal_states():
                continue

            opt_act = vi_agent.get_optimal_action(action_state_pairs=gw_env.get_action_state_pairs(state=state))
            next_state = gw_env.get_new_state_on_action(old_state=state, action=opt_act)
            next_state_value = vi_agent.get_state_value(state=next_state)

            vi_agent.set_state_value(state=state, new_value=(gw_env.get_state_reward(state=next_state) + GAMMA * next_state_value))

        iters += 1

    if verbose:
        gw_env.display_value_function(value_func=vi_agent.get_value_function())

    vi_agent.construct_greedy_policy(gw_env.get_action_state_pairs)

    if verbose:
        gw_env.display_policy(policy=vi_agent.get_policy())

    return vi_agent.get_policy()


def train_q_learning(gw_env: Grid_World, n_episodes=1000, verbose=False, policy="eps_greedy", eps=0.2, max_episode_len=100, gamma=0.95):
    ql_agent = QLearningAgent(states=gw_env.get_state_space(),
                              terminal_states=gw_env.get_terminal_states(),
                              reward_function=gw_env.get_reward_func(),
                              actions=gw_env.get_action_space(),
                              gamma=gamma)
    
    # init episodes
    episodes = []
    
    # Define state_space without terminal states for getting starting position
    state_space = deepcopy(gw_env.get_state_space()) # all states
    terminal_states = gw_env.get_terminal_states()
    for terminal_state in terminal_states:
        state_space.remove(terminal_state) # not non absorbing state_space
    
    for n in range(n_episodes):
        
        episode = []
        
        # random starting position
        start_idx = (np.random.choice(len(state_space)))
        start = state_space[start_idx]
        
        episode.append(start)
        
        i = 0
        terminal = False
        
        while ( ( i < max_episode_len ) and ( not terminal ) ):
            i += 1
            
            # Choose Action from S derived by given policy
            if policy == "eps_greedy":
                if np.random.uniform() < (1-eps):
                    # Choose greedy action -> highest Q-Value
                    chosen_action = ql_agent.get_greedy_action(episode[-1])
                else:
                    # Choose random action form action space
                    action_space = gw_env.get_action_space()
                    chosen_action = action_space[np.random.choice(len(action_space))]
            
            new_state = gw_env.get_new_state_on_action(episode[-1], chosen_action)
            
            # Reward is taken from Q_learning agent -> it knows the reward function from the environment
            ql_agent.update_Q_value(episode[-1], new_state, chosen_action)
            
            episode.append(new_state)
            
            if new_state in terminal_states:
                terminal = True
        
        episodes.append(episode)
    
    if verbose:
        gw_env.display_q_function(q_func=ql_agent.get_Q_function())

    ql_agent.construct_greedy_policy(gw_env.get_action_state_pairs)

    if verbose:
        gw_env.display_policy(policy=ql_agent.get_policy())

    return ql_agent.get_policy()         


def irl_reward_estimation(env: Grid_World, optimal_trajectories: List[List[Any]], train_func: Callable):

    # prepare reference reward function
    reward_func_ref = deepcopy(env.get_board())
    reward_func_preds = []
    print('reward_func_ref\n', reward_func_ref)

    minmax_scaler = MinMaxScaler(feature_range=(-1, 1))

    reward_func_ref_shape = reward_func_ref.shape
    reward_func_ref = minmax_scaler.fit_transform(reward_func_ref.reshape(-1, 1)).reshape(reward_func_ref_shape)
    print('MinMax scaled reward_func_ref:\n', reward_func_ref)

    irl_agent = IRL_from_sampled_trajectories(d=(GW_SIZE[0] * 4, GW_SIZE[1] * 4),
                                              env_ranges=((0, GW_SIZE[0]), (0, GW_SIZE[1])),
                                              env_discrete_size=GW_SIZE,
                                              penalty_factor=2,
                                              gamma=GAMMA)

    # step 2: given optimal trajectories, compute the value estimate
    print("Computing value estimates for optimal trajectories...")
    optimal_value_estimate = irl_agent.compute_value_estimate(trajs=optimal_trajectories)

    candidate_policies = [env.construct_random_policy()]
    candidate_value_estimates = []

    # while True:
    for i in range(IRL_TRAINING_N):
        print(f"Iteration {i}...")

        # step 3: generate trajectories and compute the value estimate for a random policy
        print("Generating trajectories for the candidate policy...")
        candidate_trajectories = env.generate_trajectories(policy=candidate_policies[-1],
                                                           n_traj=NUMBER_OF_TRAJECTORIES,
                                                           max_traj_length=MAXIMUM_TRAJECTORY_LENGTH)
        print("Computing value estimates for condidate trajectories...")
        candidate_value_estimates.append(irl_agent.compute_value_estimate(trajs=candidate_trajectories))

        # step 4: obtain new alphas
        print("Solving linear programming...")
        irl_agent.solve_lp(optimal_value_estimate, candidate_value_estimates)

        # step 5: construct new reward function from the alphas
        reward_func = irl_agent.construct_reward_function(alphas=irl_agent.get_alphas())

        # step 6: find optimal policy under new reward function and add to 'candidate_policies' list
        env.set_reward_func(reward_func)
        print("Latest non-Scaled reward func:\n", env.get_board())
        scaled_board = minmax_scaler.fit_transform(env.get_board().reshape(-1, 1)).reshape(env.get_board().shape)
        env.set_board(new_board=scaled_board)

        reward_func_preds.append(env.get_board())

        candidate_policies.append(train_func(gw_env=env, verbose=True))  # train_value_iteration(gw_env=env))

        print("Latest Scaled reward func:\n", reward_func_preds[-1])
        env.display_policy(policy=candidate_policies[-1])
        print("============================================================\n" * 2)

    return {'reference_reward_func': reward_func_ref, 'policy_pred': np.mean(np.array([list(pol.values()) for pol in candidate_policies ]), axis=0), 'avg_predicted_reward_func': np.mean(np.array(reward_func_preds), axis=0)}


def calc_value_distance(value_estimates_ref, value_estimates_pred):
    return np.linalg.norm(np.array(value_estimates_ref) - np.array(value_estimates_pred))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-vi", "--value-iteration", required=False, default=False, action="store_true")
    parser.add_argument("-ql", "--q-learning", required=False, default=False, action="store_true")
    parser.add_argument("-gt", "--generate-trajectories", required=False, default=False, action="store_true")
    parser.add_argument("-irl", "--inverse-rl", required=False, default=False, action="store_true")
    parser.add_argument("-plt", "--plots", required=False, default=False, action="store_true")

    args = parser.parse_args()
    print(f"Passed args: {args}")

    ref_reward_funcs = []
    avg_pred_reward_funcs = []
    reward_loss = []
    policy_loss = []

    for GW_SIZE in GW_SIZES:
        environment = Grid_World(size=GW_SIZE, traps=GW_TRAPS, goals=GW_GOALS, randomize_board=False)

        train_func = train_value_iteration

        if args.value_iteration:
            print("Training via value iteration...")
            greedy_policy = train_value_iteration(gw_env=environment, verbose=True)
        elif args.q_learning:
            print("Training via q-learning...")
            greedy_policy = train_q_learning(gw_env=environment, verbose=True)
            train_func = train_q_learning
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
            estimated_rewards = irl_reward_estimation(env=environment, optimal_trajectories=trajectories, train_func=train_func)
            ref_reward_funcs.append(estimated_rewards['reference_reward_func'])
            avg_pred_reward_funcs.append(estimated_rewards['avg_predicted_reward_func'])
            # Using default value for reward loss -> Frobenius for matrices and L2-loss for vectors
            reward_loss.append(np.linalg.norm(estimated_rewards['reference_reward_func'] - estimated_rewards['avg_predicted_reward_func']))
            # Using L1-Loss for policy loss as described by Ng and Russel in 2000
            policy_loss.append(np.linalg.norm(estimated_rewards['policy_pred'] - np.array(list(greedy_policy.values())), ord=1 ))
            
            print('**********************************************')
            print('*****************REWARD LOSS******************')
            print(reward_loss)
            print('**********************************************')
            print('*****************POLICY LOSS*************************')
            print(policy_loss)
            print('**********************************************')

    print('reward_loss \n', reward_loss)
    plt.plot(reward_loss)
    plt.savefig('reward_loss.png')

    print("Closing up the arena...")
