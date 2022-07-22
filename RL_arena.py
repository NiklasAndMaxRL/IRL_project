import argparse

from GridWorld_environments import Grid_World
from RL_agents import ValueIteration


def train_value_iteration():
    print("Training RL with Value Iteration...")

    gw_env = Grid_World(size=(4, 4), traps=[(2, 3), (1, 2)], goals=[(3, 3)])

    vi_agent = ValueIteration(states=gw_env.get_state_space(),
                              terminal_states=gw_env.get_terminal_states(),
                              actions=gw_env.get_action_space())

    for trap in gw_env.traps:
        vi_agent.set_state_value(state=trap, new_value=gw_env.DEADLY_TRAP_REWARD)
    for goal in gw_env.goals:
        vi_agent.set_state_value(state=goal, new_value=gw_env.GOAL_REWARD)

    iters = 0
    while iters < 50 and not vi_agent.converged:

        for state in gw_env.get_state_space():

            if state in gw_env.get_terminal_states():
                continue

            opt_act = vi_agent.get_optimal_action(action_state_pairs=gw_env.get_action_state_pairs(state=state))
            next_state = gw_env.get_new_state_on_action(old_state=state, action=opt_act)
            next_state_value = vi_agent.get_state_value(state=next_state)

            vi_agent.set_state_value(state=state, new_value=(next_state_value + gw_env.STEP_REWARD))

        iters += 1
        # print(f"Iteration {iters}")
        # print(vi_agent.get_value_function())

    print("Board:")
    print(gw_env.get_board())

    gw_env.display_value_function(value_func=vi_agent.get_value_function())
    gw_env.display_policy(value_func=vi_agent.get_value_function(),
                          get_action_func=vi_agent.get_optimal_action)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-vi", "--value-iteration", action="store_true", default=False)

    args = parser.parse_args()
    print(f"Passed args: {args}")

    if args.value_iteration:
        train_value_iteration()

    print("Closing the RL_arena...")
