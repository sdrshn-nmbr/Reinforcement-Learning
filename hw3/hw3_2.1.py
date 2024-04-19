import numpy as np

# Transition probabilities and rewards
P = np.array(
    [
        [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]],
        [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]],
        [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
    ]
)
R = np.array([[0, 0], [0, 0], [0, 1]])

# Policy
pi = np.array([0, 0, 1])

gamma = 0.95

# set up system of linear equations
A = np.eye(3) - gamma * P[np.arange(3), pi]
b = R[np.arange(3), pi]

# solve system of linear equations
V_pi = np.linalg.solve(A, b)

for i, v in enumerate(V_pi, 1):
    print(f"V π({i}) ≈ {v:.4f}")
