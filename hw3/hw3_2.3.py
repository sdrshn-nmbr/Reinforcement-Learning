import numpy as np

P_true = np.array([
    [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]],
    [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]],
    [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]
])

num_samples = 100
states = [1, 2, 3]
actions = ['g', 'h']

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

pi = ['g', 'g', 'h']

# compute the L1-difference for each state under policy pi
l1_diff = np.zeros(3)
for s in range(3):
    a_idx = 0 if pi[s] == 'g' else 1
    l1_diff[s] = np.sum(np.abs(P_hat[s, a_idx] - P_true[s, a_idx]))

print("\nL1-difference under policy pi:")
for s in range(3):
    print(f"|P^(.|{s+1}, {pi[s]}) - P(.|{s+1}, {pi[s]})|_1 = {l1_diff[s]:.4f}")