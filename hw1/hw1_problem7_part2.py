import numpy as np

GRID_SIZE = 4
GOAL_REWARD = 1
PENALTY_REWARD = -1
DISCOUNT_FACTOR = 0.95


def reward(state):
    if state == (1, 2):
        return GOAL_REWARD
    elif state == (2, 2):
        return PENALTY_REWARD
    else:
        return 0


def transition(state, action):
    i, j = state
    new_i, new_j = i + action[0], j + action[1]

    if (i, j) in [(1, 1), (3, 2), (3, 3), (1, 2)]:  # Mountain and lightning cells
        return state

    if 0 <= new_i < GRID_SIZE and 0 <= new_j < GRID_SIZE:
        if (new_i, new_j) == (0, 0):  # Starting position
            return state
        elif (new_i, new_j) == (1, 0):
            return state
        elif (new_i, new_j) == (2, 0):
            return state
        elif (new_i, new_j) == (0, 1):
            return state
        elif (new_i, new_j) == (0, 2):
            return state
        elif (new_i, new_j) == (0, 3):
            return state
        elif (new_i, new_j) == (1, 3):
            return state
        elif (new_i, new_j) == (2, 1):
            return state
        elif (new_i, new_j) == (2, 4):
            return state
        elif (new_i, new_j) == (4, 1):
            return state
        elif (new_i, new_j) == (4, 2):
            return state
        elif (new_i, new_j) == (4, 3):
            return state
        elif (new_i, new_j) == (0, 4):
            return state
        elif (new_i, new_j) == (4, 0):
            return state
        elif (new_i, new_j) == (4, 4):
            return state
        else:
            return new_i, new_j
    else:
        return state


def policy(state):
    if state in [(1, 1), (3, 2), (3, 3)]:  # Mountain cells
        return (0, 0)  
    elif state == (1, 2):
        return (0, 0)
    elif state == (0, 0):
        return (0, 1)
    elif state == (1, 0):
        return (0, 1)
    elif state == (2, 0):
        return (1, 0)
    elif state == (0, 1):
        return (1, 0)
    elif state == (0, 2):
        return (1, 0)
    elif state == (0, 3):
        return (0, 0)
    elif state == (1, 3):
        return (0, -1)
    elif state == (2, 1):
        return (0, -1)
    elif state == (2, 4):
        return (-1, 0)
    elif state == (4, 1):
        return (-1, 0)
    elif state == (4, 2):
        return (-1, 0)
    elif state == (4, 3):
        return (0, -1)
    elif state == (0, 4):
        return (0, 0)
    elif state == (4, 0):
        return (0, 0)
    elif state == (4, 4):
        return (0, 0)
    else:
        return (0, 0)


def value_iteration(reward, transition, policy, discount_factor, epsilon=0.01):
    # Initialize value function with zeros
    V = np.zeros((GRID_SIZE, GRID_SIZE))

    while True:
        delta = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                state = (i, j)
                action = policy(state)
                next_state = transition(state, action)
                v_old = V[i, j]
                V[i, j] = (
                    reward(state) + discount_factor * V[next_state[0], next_state[1]]
                )
                delta = max(delta, abs(v_old - V[i, j]))

        if delta < epsilon:
            break

    return V


def extract_policy(V):
    policy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=object)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state = (i, j)
            best_action = None
            best_value = float("-inf")

            # Iterate over possible actions and select the one with the highest value
            for action in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_state = transition(state, action)
                value = (
                    reward(state) + DISCOUNT_FACTOR * V[next_state[0], next_state[1]]
                )
                if value > best_value:
                    best_value = value
                    best_action = action

            policy[i, j] = best_action

    return policy


def main():
    V = value_iteration(reward, transition, policy, DISCOUNT_FACTOR)

    optimal_policy = extract_policy(V)

    print("Optimal Policy:")
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            print(optimal_policy[i, j], end="\t")
        print()


if __name__ == "__main__":
    main()
