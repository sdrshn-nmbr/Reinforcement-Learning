import numpy as np

# Define the discount factor, initial state distribution, and probability transition matrix
gamma = 0.95
mu_0 = np.array([[1], [0], [0]])
P_pi = np.array([[0.1, 0.8, 0.1],
                 [0.1, 0.1, 0.8],
                 [0.1, 0.1, 0.8]])

# Compute the transpose of P_pi
P_pi_t = P_pi.T

# Compute (I - gamma * P_pi^T)
I = np.eye(3)
I_minus_gamma_P_pi_t = I - gamma * P_pi_t

# Compute (I - gamma * P_pi^T)^-1
I_minus_gamma_P_pi_t_inv = np.linalg.inv(I_minus_gamma_P_pi_t)

# Compute rho_pi_mu_0 = (I - gamma * P_pi^T)^-1 * mu_0
rho_pi_mu_0 = np.dot(I_minus_gamma_P_pi_t_inv, mu_0)

print("State occupancy measure:")
for i, rho in enumerate(rho_pi_mu_0, 1):
    print(f"rho pi mu_0 ({i}) ≈ {rho[0]:.4f}")
    
Normalization = np.sum(rho_pi_mu_0)

norm_rho_pi_mu_0 = rho_pi_mu_0 / Normalization

print("\nNormalized state occupancy measure:")
for i, rho in enumerate(norm_rho_pi_mu_0, 1):
    print(f"norm_rho_pi_mu_0 ({i}) ≈ {rho[0]:.4f}")