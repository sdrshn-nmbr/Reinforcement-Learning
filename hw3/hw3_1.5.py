import numpy as np

# Normalized state occupancy measure
rho_bar_pi_mu_0 = np.array([0.1450, 0.1914, 0.6636])

# L1-difference between transition probabilities
l1_diff = np.array([0.2, 0.2, 1.5])

gamma = 0.95

# compute the upper bound using the simulation lemma
upper_bound = (1 / (1 - gamma)) * np.sum(rho_bar_pi_mu_0 * l1_diff)

print(f"|V′π(1) - Vπ(1)| = {upper_bound:.4f}")