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


def off_policy_mc_eval(P, R, pi_t, pi_b, num_episodes=10000):
    num_states, num_actions, _ = P.shape
    V_pi_t = np.zeros(num_states)
    V_pi_b = np.zeros(num_states)

    C = np.zeros(num_states)

    for _ in range(num_episodes):
        episode = []
        state = 0  # Start from initial state
        while True:
            action = np.random.choice(np.arange(num_actions), p=pi_b[state])
            next_state = np.random.choice(np.arange(num_states), p=P[state][action])
            reward = R[next_state][action]
            episode.append((state, action, reward))
            state = next_state
            if state == 2:  # If terminal state reached
                break

        rho = 1.0  # ISR
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = GAMMA * G + reward
            rho *= pi_t[state][action] / pi_b[state][action]
            C[state] += rho
            V_pi_t[state] += (rho / C[state]) * (G - V_pi_t[state])
            if action != np.argmax(pi_t[state]):
                break

        # Calculate V_pi_b for comparison
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = GAMMA * G + reward
            V_pi_b[state] += (1 / num_episodes) * (G - V_pi_b[state])

    return V_pi_t, V_pi_b


# Perform off-policy Monte Carlo evaluation
V_pi_t, V_pi_b = off_policy_mc_eval(P, R, pi_t, pi_b)
print("Value function for target policy V_pi_t:", V_pi_t)
print("Value function for behavior policy V_pi_b:", V_pi_b)