import numpy as np

P_true = np.array(
    [
        [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]],
        [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]],
        [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
    ]
)

num_samples = 100
states = [1, 2, 3]
actions = ["g", "h"]

# initialize the estimated transition function
P_hat = np.zeros((3, 2, 3))

# sample each s, a pair and estimate the transition probabilities
for s in states:
    for a_idx, a in enumerate(actions):
        # generate samples using the true transition probabilities
        samples = np.random.choice([1, 2, 3], size=num_samples, p=P_true[s - 1, a_idx])

        # count the occurrences of each s' in the samples
        for s_prime in states:
            count = np.sum(samples == s_prime)
            P_hat[s - 1, a_idx, s_prime - 1] = count / num_samples

print("Estimated transition function Pˆ:")
for s in states:
    for a_idx, a in enumerate(actions):
        print(f"P^(.|{s}, {a}) = {P_hat[s - 1, a_idx]}")

pi = ["g", "g", "h"]

# compute the L1-difference for each state under policy pi
l1_diff = np.zeros(3)
for s in range(3):
    a_idx = 0 if pi[s] == "g" else 1
    l1_diff[s] = np.sum(np.abs(P_hat[s, a_idx] - P_true[s, a_idx]))

print("\nL1-difference under policy pi:")
for s in range(3):
    print(f"|P^(.|{s+1}, {pi[s]}) - P(.|{s+1}, {pi[s]})|_1 = {l1_diff[s]:.4f}")

R = np.array([[0, 0], [0, 0], [0, 1]])

gamma = 0.95

# set up the system of linear equations
A = np.eye(3)
b = np.zeros(3)

for s in range(3):
    a_idx = 0 if pi[s] == "g" else 1
    A[s] -= gamma * P_hat[s, a_idx]
    b[s] = R[s, a_idx]

# solve system of linear equations
V_hat_pi = np.linalg.solve(A, b)

print("\nEstimated value function V^ pi:")
for s in range(3):
    print(f"V^ π({s+1}) = {V_hat_pi[s]:.4f}")


diff = np.abs(V_hat_pi[0] - 0.8752)

# Print the exact difference
print(f"\nExact difference between the value functions in the initial state s = 1:")
print(f"|V^ pi(1) - V pi(1)| = {diff:.4f}")

rho_bar_pi_mu_0 = np.array([2.9, 3.8285, 13.2715])

# compute the upper bound using the simulation lemma
upper_bound = (1 / (1 - gamma)) * np.sum(rho_bar_pi_mu_0 * l1_diff)

print(f"\nUpper bound on |V^ pi(1) - V pi(1)| using the simulation lemma:")
print(f"|V^ pi(1) - V pi(1)| <= {upper_bound:.4f}")
