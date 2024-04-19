import numpy as np

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


def compute_Heff(P, R, pi_t, pi_b, threshold=0.1, max_steps=1000):
    num_states, num_actions, _ = P.shape
    T = 1
    while True:
        V_pi_t = np.zeros(num_states)
        V_pi_b = np.zeros(num_states)

        for _ in range(10000):  # Run a fixed number of episodes for estimation
            episode = []
            state = 0  # Start from initial state
            for t in range(T):  # Limit the episode to T time steps
                action = np.random.choice(np.arange(num_actions), p=pi_b[state]) # np.arrange returns an array of evenly spaced values
                next_state = np.random.choice(np.arange(num_states), p=P[state][action])
                reward = R[next_state][action]
                episode.append((state, action, reward))
                state = next_state
                if state == 2:  # If terminal state reached
                    break

            rho = 1.0  # Importance Sampling Ratio (ISR)
            G = 0
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = GAMMA * G + reward
                rho *= pi_t[state][action] / pi_b[state][action]
                if t < T:
                    V_pi_t[state] += rho * (G - V_pi_t[state]) / (t + 1)
                    V_pi_b[state] += (1 / 10000) * (G - V_pi_b[state])
        max_error = np.max(np.abs(V_pi_t - V_pi_b))
        if max_error < threshold or T >= max_steps:
            break
        T += 1
    return T

# Compute the effective horizon
effective_horizon = compute_Heff(P, R, pi_t, pi_b)
print("Effective Horizon:", effective_horizon)