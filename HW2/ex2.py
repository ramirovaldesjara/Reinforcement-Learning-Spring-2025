'''

2. [50 points] Still consider the mini-gridworld example, but consider the fact a robot
moving this gridword has no access to its actual location, but only observation
of a wall (1) or no wall (0). Your initial belief state is b0 (S) = (1/4, 1/4, 1/2)
(a) [10 points] Compute the updated belief state after observing a wall and taking
action R.
(b) [10 points] Programming for Belief Update Implement the belief update
(slide 11 of lecture 5) in python. Print out the result of updated belief
state after observing a wall and taking action R. Compare with the result in
(a).
(c) [10 points] Compute the probability of observing a wall in the new belief
state.
(d) [10 points] Compute the probability of not observing a wall in the new belief
state.
(e) [10 points] Programming for Belief Transition Function Implement belief
transition model (slide 23 of lecture 5) in python. Print out transition of
belief state after taking action R with b0. Compare your results with the
ones in (c) and (d).


'''



S = ['A', 'B', 'C']
A = ['L', 'R']

R = {'A': 3, 'B': -2, 'C': 1}

T = {
 'A': {'L': {'A': 0.8, 'B': 0.2, 'C': 0.0}, 'R': {'A': 0.2, 'B': 0.8, 'C': 0.0}},
 'B': {'L': {'A': 0.8, 'B': 0.0, 'C': 0.2}, 'R': {'A': 0.2, 'B': 0.0, 'C': 0.8}},
 'C': {'L': {'A': 0.0, 'B': 0.8, 'C': 0.2}, 'R': {'A': 0.0, 'B': 0.2, 'C': 0.8}},
}

# Observation probabilities (Sensor Model)
O = {
 'A': {1: 0.5, 0: 0.5},
 'B': {1: 0.0, 0: 1.0},
 'C': {1: 0.5, 0: 0.5},
}

# Initial belief state
belief = {'A': 1 / 4, 'B': 1 / 4, 'C': 1 / 2}


def update_belief(belief, action, observation):
 updated_belief = {}
 total = 0

 for s_prime in S:
  weighted_probabilities = sum(T[s][action][s_prime] * belief[s] for s in S)
  updated_belief[s_prime] = O[s_prime][observation] * weighted_probabilities
  total += updated_belief[s_prime]

 # Normalize
 for s in S:
  updated_belief[s] /= total if total > 0 else 1

 return updated_belief


def probability_of_observation(belief, action, observation):
 return sum(O[s_prime][observation] * sum(T[s][action][s_prime] * belief[s] for s in S) for s_prime in S)


def belief_transition(belief, action):
 new_belief = {s_prime: sum(T[s][action][s_prime] * belief[s] for s in S) for s_prime in S}
 return new_belief


# Compute updated belief after observing a wall and taking action R
updated_belief = update_belief(belief, 'R', 1)
print("(b) Updated Belief State after observing a wall and taking action R:", updated_belief)

# Compute probability of observing a wall in the new belief state
prob_wall = probability_of_observation(updated_belief, 'R', 1)
print("(c) Probability of observing a wall:", prob_wall)

# Compute probability of not observing a wall in the new belief state
prob_no_wall = probability_of_observation(updated_belief, 'R', 0)
print("(d) Probability of not observing a wall:", prob_no_wall)

# Compute belief transition after taking action R
belief_after_transition = belief_transition(belief, 'R')
print("(e) Belief Transition after taking action R:", belief_after_transition)
