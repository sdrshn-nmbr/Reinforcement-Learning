import numpy as np
import matplotlib.pyplot as plt

STATES_COUNT = 25
GRID_LENGTH = 5
STATES = np.arange(STATES_COUNT).reshape((GRID_LENGTH, GRID_LENGTH))
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTIONS_COUNT = len(ACTIONS)
ACTIONS_MAP = {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN", LEFT: "LEFT"}
GOAL_STATE = 4
LIGHTNING_STATE = 7
MOUNTAIN_STATE = [6, 17, 18]
GAMMA = 0.95
REWARDS = np.zeros(STATES_COUNT)
REWARDS[GOAL_STATE] = 1
REWARDS[LIGHTNING_STATE] = -1
POLICY = [
    RIGHT,
    RIGHT,
    RIGHT,
    RIGHT,
    UP,
    LEFT,
    UP,
    LEFT,
    LEFT,
    UP,
    UP,
    UP,
    RIGHT,
    RIGHT,
    RIGHT,
    UP,
    DOWN,
    DOWN,
    DOWN,
    UP,
    UP,
    RIGHT,
    RIGHT,
    UP,
    UP,
]


def is_accessible(i, j):
    if i < 0 or i >= GRID_LENGTH:
        return False
    if j < 0 or j >= GRID_LENGTH:
        return False
    if STATES[i, j] in MOUNTAIN_STATE:
        return False
    return True


def create_P_pi_matrix(policy):
    P = np.zeros((STATES_COUNT, STATES_COUNT))
    for i in range(GRID_LENGTH):
        for j in range(GRID_LENGTH):
            state_index = STATES[i, j]
            policy_action = policy[state_index]
            adjacent_states = {}
            adjacent_states[UP] = (
                STATES[i - 1, j] if is_accessible(i - 1, j) else state_index
            )
            adjacent_states[RIGHT] = (
                STATES[i, j + 1] if is_accessible(i, j + 1) else state_index
            )
            adjacent_states[DOWN] = (
                STATES[i + 1, j] if is_accessible(i + 1, j) else state_index
            )
            adjacent_states[LEFT] = (
                STATES[i, j - 1] if is_accessible(i, j - 1) else state_index
            )
            if state_index == GOAL_STATE or state_index == LIGHTNING_STATE:
                P[state_index, state_index] = 1
            else:
                for act in ACTIONS:
                    if act == policy_action:
                        P[state_index, adjacent_states[act]] += 0.85
                    else:
                        P[state_index, adjacent_states[act]] += 0.05
    return P


def evaluate_policy(policy):
    P_pi = create_P_pi_matrix(policy)
    return P_pi, np.linalg.inv(np.identity(STATES_COUNT) - GAMMA * P_pi) @ REWARDS


def create_T_matrix():
    T = np.zeros((ACTIONS_COUNT, STATES_COUNT, STATES_COUNT))
    T[UP, :, :] = create_P_pi_matrix([UP for i in range(STATES_COUNT)])
    T[RIGHT, :, :] = create_P_pi_matrix([RIGHT for i in range(STATES_COUNT)])
    T[DOWN, :, :] = create_P_pi_matrix([DOWN for i in range(STATES_COUNT)])
    T[LEFT, :, :] = create_P_pi_matrix([LEFT for i in range(STATES_COUNT)])
    return T


def problem_7():
    P_pi, V_analytical = evaluate_policy(POLICY)
    print("Analytical Policy Evaluation")
    print(np.round(V_analytical.reshape(GRID_LENGTH, GRID_LENGTH), 3))
    steps = 149
    errors = np.zeros(steps + 1)
    V = np.zeros(STATES_COUNT)
    for k in range(steps):
        errors[k] = np.max(np.abs(V - V_analytical))
        V = REWARDS + GAMMA * P_pi @ V
    errors[steps] = np.max(np.abs(V - V_analytical))

    print("Iterative Policy Evaluation")
    print(np.round(V.reshape(GRID_LENGTH, GRID_LENGTH), 3))

    plt.plot(errors)
    plt.title("Error in the value function from policy evaluation")
    plt.xlabel("Iteration")
    plt.ylabel(r"$||V_t - V^\pi|||_\infty$")
    plt.show()


def problem_4():
    T_matrix = np.array([[0.6, 0.2, 0.2], [0.1, 0, 0.9], [0.3, 0, 0.7]])

    # Computing the eigne vector corresponding to eigen value of 1
    evals, evecs = np.linalg.eig(T_matrix.T)
    evec1 = evecs[:, np.isclose(evals, 1)]
    evec1 = evec1[:, 0]
    stationary = evec1 / evec1.sum()
    stationary = stationary.real
    stationary = np.array(stationary)
    print("Stationary distribution of the Markov chain: \n", stationary)
    mu0 = np.array([1, 0, 0])
    error_list = []
    mut = mu0
    print("Beleif at t = 2: \n", (mut @ T_matrix) @ T_matrix)
    for t in range(1, 100):
        error_list.append(np.linalg.norm(mut - stationary, 1))
        mut = mut @ T_matrix

    plt.plot(error_list)
    plt.title("L1 norm difference of beliefs w.r.t stationary distribution")
    plt.xlabel("Time")
    plt.ylabel(r"$||\mu_t - \hat{\mu}||_1$")
    plt.show()


if __name__ == "__main__":
    problem_4()
    problem_7()
