import numpy as np
import matplotlib.pyplot as plt

'''
Let us consider the mini-gridworld example.
Gridworld Setup:
• Three states S: A, B, C
• Utility of each state: U(A) = +3, U(B) = −2, U(C) = +1
• Actions A: L (Left) and R (Right) for moving left and right, respectively
• The agent starts at state B
• The agent moves:
– In the chosen direction with probability 0.7
– In the opposite direction with probability 0.2
– Does not move at all with probability 0.1
• Whenever the agent hit a wall in this maze, it return to its original location, and get
the reward from this location. E.g. if the agent move left at location A, it will stay at
A, and receive reward 3.
(a) [15 points] Implement a simple method to get EU(action, state) for an arbitrary set
of utility values using python. Submit your code, and printout EU for all action-state
combinations.
(b) [15 points] Equipped with the method in step (a), find the action that returns the
maximum expected utility. Submit your code, and printout optimal actions for all
states.
(c) [15 points] Suppose you are operating a robot in this maze. It starts at state B, and
is trying to maximize its overall utility achieved with two step movements (e.g. if it
moves from B to C and stay at C, its overall utility is 1 + 1 = 2). Implement an ϵgreedy action selection method to solve the gridworld problem using python. Try it
with ϵ = 0.1, 0.5, and0.9. Submit your code, and draw a curve with overall reward
achieved in action selection loop being y axis, and action selection loops beig the x axis
(i.e. the convergence curves).

'''

utilities = {
    'A': 3,
    'B': -2,
    'C': 1
}

prob_forward = 0.7
prob_backward = 0.2
prob_static = 0.1

actions = ['L', 'R']
states = ['A', 'B', 'C']

transitions = {
    'A': {'L': ('A', 'A', 'B'), 'R': ('B', 'A', 'A')},
    'B': {'L': ('A', 'B', 'C'), 'R': ('C', 'B', 'A')},
    'C': {'L': ('B', 'C', 'C'), 'R': ('C', 'C', 'B')}
}

def calculate_expected_utility(action, state):
    if state not in transitions or action not in transitions[state]:
        return 0

    chosen_move, static_move, opposite_move = transitions[state][action]

    expected_value = (
            prob_forward * utilities.get(chosen_move, 0) +
            prob_static * utilities.get(static_move, 0) +
            prob_backward * utilities.get(opposite_move, 0)
    )

    return expected_value

for state in states:
    for action in actions:
        eu_value = calculate_expected_utility(action, state)
        print(f"EU({action}, {state}) = {eu_value:.2f}")

def find_optimal_actions():
    optimal_actions = {}

    for state in states:
        best_action = None
        best_utility = float('-inf')

        for action in actions:
            eu_value = calculate_expected_utility(action, state)
            if eu_value > best_utility:
                best_utility = eu_value
                best_action = action

        optimal_actions[state] = (best_action, best_utility)

    return optimal_actions

optimal_actions = find_optimal_actions()

for state, (action, value) in optimal_actions.items():
    print(f"Optimal Action for {state}: {action} (EU = {value:.2f})")


def choose_action_epsilon(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        return max(Q[state], key=Q[state].get)


def transition(state, action):
    next_state = np.random.choice(
        transitions[state][action],
        p=[prob_forward, prob_static, prob_backward]
    )
    reward = utilities.get(next_state, 0)
    return next_state, reward


def run_episode(Q, epsilon, alpha=0.2):
    total_reward = 0
    s = 'B'
    for _ in range(2):
        a = choose_action_epsilon(s, Q, epsilon)
        s_next, r = transition(s, a)
        total_reward += r

        # Update Q(s, a)
        Q[s][a] += alpha * (r - Q[s][a])

        s = s_next

    return total_reward


def run_simulation(epsilon, action_selection_loops):
    total_rewards = []
    Q_values = {state: {action: 0 for action in actions} for state in states}

    for _ in range(action_selection_loops):
        total_rewards.append(run_episode(Q_values, epsilon))

    return total_rewards


epsilons = [0.1, 0.5, 0.9]
action_selection_loops = 1000
results = {epsilon: run_simulation(epsilon, action_selection_loops) for epsilon in epsilons}

plt.figure(figsize=(10, 6))
for epsilon, rewards in results.items():
    x_values = np.arange(1, action_selection_loops + 1)
    y_values = np.cumsum(rewards) / x_values
    plt.plot(x_values, y_values, label=f'ϵ={epsilon}')

plt.xlabel('Action selection loops')
plt.ylabel('Average Cumulative Reward')
plt.title('Convergence of ϵ-Greedy Action Selection')
plt.legend()
plt.grid(True)
plt.show()



