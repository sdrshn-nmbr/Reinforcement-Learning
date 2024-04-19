import numpy as np

transition_matrix = np.array([
    [0.6, 0.2, 0.2],
    [0.1, 0, 0.9],
    [0.3, 0, 0.7]
])

n = transition_matrix.shape[0]

def compute_stationary_distribution(transition_matrix):
    # Transpose the matrix and subtract the identity matrix
    Q = (np.transpose(transition_matrix) - np.identity(n))

    # Create the augmented matrix for normalization
    augmented_matrix = np.vstack((Q, np.ones(n)))

    # Solve system of linear equations
    stationary_distribution = np.linalg.lstsq(augmented_matrix, [0]*n + [1], rcond=None)[0]

    return stationary_distribution

print("Stationary distribution:", compute_stationary_distribution(transition_matrix))