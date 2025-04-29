'''

1. [50 points] Let us consider the mini-gridworld example.
Gridworld Setup:
• Three states S: A, B, C
• Utility of each state: U(A) = +3, U(B) = −2, U(C) = +1
• Actions A: L (Left) and R (Right) for moving left and right, respectively
• The agent moves:
– In the chosen direction with probability 0.8
– In the opposite direction with probability 0.2
• Whenever the agent hit a wall in this maze, it return to its original location,
and get the reward from this location. E.g. if the agent move left at location
A, it will stay at A, and receive reward 3.
(a) [10 points] Programming for Markov Decision Process Implement this MDP
in python, construct a function called “step” which responds action choices
at different states with successor states and reward achieved. Print out the
outputs of the step function for all action-state combinations.
(b) [10 points] Value Iteration Using value iteration to find V2 (given discount
factor γ = 0.5) Hint: Refer to slide 18 of Lecture 4.
(c) [10 points] Programming for Value Iteration Implement value iteration in
python to solve this MDP (given discount factor γ = 0.5). Print out the final
policy Print out the final policy (i.e. π (A), π (B), π (C)).
(d) [10 points] Policy Iteration Using value iteration find π2 (given initial policy
π1 = (R, R, R) discount factor γ = 0.5) Hint: Refer to slide 38 of Lecture 4.
(e) [10 points] Programming for Policy Iteration Implement policy iteration in
python to solve this MDP (given initial policy π1 = (R, R, R), and discount
factor γ = 0.5). Print out the final policy (i.e. π (A), π (B), π (C)).

'''

# MDP Formulation
import numpy as np


S = ['A', 'B', 'C']

A = ['L', 'R']

R = {'A': 3, 'B': -2, 'C': 1}

T = {
    'A': {'L': {'A': 0.8, 'B': 0.2, 'C': 0.0}, 'R': {'A': 0.2, 'B': 0.8, 'C': 0.0}},
    'B': {'L': {'A': 0.8, 'B': 0.0, 'C': 0.2}, 'R': {'A': 0.2, 'B': 0.0, 'C': 0.8}},
    'C': {'L': {'A': 0.0, 'B': 0.8, 'C': 0.2}, 'R': {'A': 0.0, 'B': 0.2, 'C': 0.8}},
}

# Step function
def step(state, action):
    next_state = np.random.choice(S, p=[T[state][action][s_] for s_ in S])
    reward = R[next_state]
    return next_state, reward

# Value Iteration
def value_iteration(discount_factor=0.5, max_iterations=1000):
    V = {s: 0 for s in S}
    for _ in range(max_iterations):
        delta = 0
        V_new = V.copy()
        for s in S:
            v = V[s]
            V_new[s] = max(sum(T[s][a][s_] * (R[s_] + discount_factor * V[s_]) for s_ in S) for a in A)
            delta = max(delta, abs(v - V_new[s]))
        V = V_new
        if delta < 1e-8:
            break
    policy = {s: max(A, key=lambda a: sum(T[s][a][s_] * (R[s_] + discount_factor * V[s_]) for s_ in S)) for s in S}
    return policy, V


# Policy Iteration
def policy_iteration(discount_factor=0.5, max_iterations=1000):
    policy = {s: 'R' for s in S}  # Initial policy π0 = (R, R, R)
    V = {s: 0 for s in S}

    for _ in range(max_iterations):
        policy, V = value_iteration(discount_factor)
        stable = True
        new_policy = {}
        for s in S:
            old_action = policy[s]
            new_policy[s] = max(A, key=lambda a: sum(T[s][a][s_] * (R[s_] + discount_factor * V[s_]) for s_ in S))
            if old_action != new_policy[s]:
                stable = False
        policy = new_policy
        if stable:
            break

    return policy, V


print("(a) Step function output - MDP Implementation:")
for s in S:
    for a in A:
        print(f"State {s}, Action {a} -> {step(s, a)}")

print("(b) Value Iteration - Computing V2:")
policy_vi, V_vi = value_iteration(max_iterations=2)
print("State Values:", V_vi)

print("(c) Value Iteration - Final Policy:")
policy_vi, V_vi = value_iteration()
print("Optimal Policy:", policy_vi)

print("(d) Policy Iteration - Computing π2:")
print("Solved by hand")

print("(e) Policy Iteration - Final Policy:")
policy_pi, V_pi = policy_iteration()
print("Optimal Policy:", policy_pi)
print("State Values:", V_pi)
