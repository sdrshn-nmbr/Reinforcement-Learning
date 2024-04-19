import numpy as np
from hw2_problem4_part2 import compute_Heff

# Transition probabilities -> P[s, a, s']
P = np.array(
    [
        [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]],
        [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]],
        [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
    ]
)

# Reward function -> R[s, a]
# a = g -> 0, a = h -> 1
R = np.array([[0, 0], [0, 0], [0, 1]])

GAMMA = 0.95

pi_t = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
pi_b = np.array([[0.85, 0.15], [0.88, 0.12], [0.1, 0.9]])

# Compute the effective horizon
h_eff = compute_Heff(P, R, pi_t, pi_b)

def off_policy_mc_eval(P, R, pi_t, pi_b, h_eff, num_trajectories=50):
    num_states, num_actions, _ = P.shape
    V_hat_pi_t = {}

    for state in range(num_states):
        V_hat_pi_t[state] = 0

    for _ in range(num_trajectories):
        episode = []
        state = 0  # Starting state
        rho = 1.0  # ISR
        G = 0

        # Generate trajectory of length equal to effective horizon
        for t in range(h_eff):
            action = np.random.choice(np.arange(num_actions), p=pi_b[state])
            next_state = np.random.choice(np.arange(num_states), p=P[state][action])
            reward = R[next_state][action]
            episode.append((state, action, reward))
            state = next_state
            if state == 2:  # If terminal state reached
                break

        # Calculate returns using importance sampling
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = GAMMA * G + reward
            rho *= pi_t[state][action] / pi_b[state][action]
            if t < h_eff:
                V_hat_pi_t[state] += (rho * G) / num_trajectories

    return V_hat_pi_t

# Perform off-policy Monte Carlo evaluation for the target policy
v_hat_pi_t = off_policy_mc_eval(P, R, pi_t, pi_b, h_eff)

# Get the estimated value function at state s=1
v_hat_pi_t_1 = v_hat_pi_t[1]

# Perform off-policy Monte Carlo evaluation for the target policy in Part 1 to get the true value function
value_function_target_true = off_policy_mc_eval(P, R, pi_t, pi_b, h_eff, num_trajectories=10_000)

# Get the true infinite-horizon value function at state s=1
v_pi_t_1_true = value_function_target_true[1]

# Compute the error
error = np.abs(v_hat_pi_t_1 - v_pi_t_1_true)
print("Error between the estimated value function and the true value function at state s=1:", error)
