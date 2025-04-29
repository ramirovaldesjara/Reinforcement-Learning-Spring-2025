
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Updated SmartGrid Environment
class GridWorld:
    def __init__(self, grid_size=4, goal_state=11, obstacles={7, 13}, gamma=0.9):
        self.grid_size = grid_size
        self.state_space = grid_size * grid_size
        self.action_space = 4  # Up, Down, Left, Right
        self.goal_state = goal_state
        self.obstacles = obstacles
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.state = 0  # Start at top-left corner
        return self.state

    def step(self, action):
        old_state = self.state
        row, col = divmod(self.state, self.grid_size)
        if action == 0:  # Up
            row = max(row - 1, 0)
        elif action == 1:  # Down
            row = min(row + 1, self.grid_size - 1)
        elif action == 2:  # Left
            col = max(col - 1, 0)
        elif action == 3:  # Right
            col = min(col + 1, self.grid_size - 1)

        new_state = row * self.grid_size + col

        if new_state in self.obstacles:
            reward = -1
            new_state = old_state  # bounce back
            done = False
        elif new_state == self.goal_state:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.state = new_state
        return new_state, reward, done

# One-hot encoding
def one_hot(state, size):
    x = np.zeros(size, dtype=np.float32)
    x[state] = 1.0
    return torch.tensor(x)

# Policy network
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# REINFORCE algorithm
def reinforce(env, policy, episodes=500, gamma=0.9, lr=1e-2, use_baseline=False, batch_size=1):
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    reward_history = []
    log_probs_batch = []
    returns_batch = []

    for episode in range(episodes):
        log_probs = []
        rewards = []
        state = env.reset()
        done = False

        while not done:
            state_tensor = one_hot(state, env.state_space)
            probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Compute returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        log_probs_batch.append(torch.stack(log_probs))
        returns_batch.append(returns)

        # Only update when enough episodes are collected
        if (episode + 1) % batch_size == 0:
            log_probs_batch = torch.cat(log_probs_batch)
            returns_batch = torch.cat(returns_batch)

            if use_baseline:
                baseline = returns_batch.mean()
                loss = -(log_probs_batch * (returns_batch - baseline)).sum()
            else:
                loss = -(log_probs_batch * returns_batch).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_probs_batch = []
            returns_batch = []

        reward_history.append(sum(rewards))

    return reward_history



def main():
    env = GridWorld()

    # (a) REINFORCE without baseline
    policy1 = PolicyNet(env.state_space, env.action_space)
    rewards_no_baseline = reinforce(env, policy1, use_baseline=False, batch_size=1)

    plt.figure()
    plt.plot(rewards_no_baseline)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE without Baseline")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reinforce_no_baseline.png")
    plt.show()

    # (b) REINFORCE with baseline, batch=1
    env = GridWorld()
    policy2 = PolicyNet(env.state_space, env.action_space)
    rewards_baseline = reinforce(env, policy2, use_baseline=True, batch_size=1)

    plt.figure()
    plt.plot(rewards_baseline)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE with Baseline (Batch Size = 1)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reinforce_baseline_batch1.png")
    plt.show()

    # (c.1) REINFORCE, batch=100
    env = GridWorld()
    policy3 = PolicyNet(env.state_space, env.action_space)
    rewards_batch100 = reinforce(env, policy3, use_baseline=False, batch_size=100)

    plt.figure()
    plt.plot(rewards_batch100)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE without Baseline (Batch Size = 100)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reinforce__wo_baseline_batch100.png")
    plt.show()

    # (c.2) REINFORCE, batch=400
    env = GridWorld()
    policy4 = PolicyNet(env.state_space, env.action_space)
    rewards_batch400 = reinforce(env, policy4, use_baseline=False, batch_size=400)

    plt.figure()
    plt.plot(rewards_batch400)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE without Baseline (Batch Size = 400)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reinforce_wo_baseline_batch400.png")
    plt.show()


if __name__ == "__main__":
    main()


