import numpy as np

# original transition probabilities under policy pi
P_pi = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]])

# new transition probabilities under policy pi
P_prime_pi = np.array([[0.15, 0.7, 0.15], [0.05, 0.05, 0.9], [0.15, 0.8, 0.05]])

# compute the L1-difference for each state
l1_diff = np.sum(np.abs(P_prime_pi - P_pi), axis=1)

# print the L1-difference for each state
for i, diff in enumerate(l1_diff, 1):
    print(f"s = {i}: |P′(.|{i}, pi({i})) - P(.|{i}, pi({i}))|₁ = {diff:.1f}")
