
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Environment setup for SmartGrid with discounted return
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
        self.state = 0
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
            new_state = old_state
            done = False
        elif new_state == self.goal_state:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.state = new_state
        return new_state, reward, done

# One-hot encoding of state
def one_hot(state, size):
    x = np.zeros(size, dtype=np.float32)
    x[state] = 1.0
    return torch.tensor(x)

# Actor Network Definition
class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.actor = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.actor(x), dim=-1)
        return policy

# Critic Network Definition
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.critic(x)
        return value

# A2C Algorithm with Instant Reward + Discounted Future Value (TD(0))
def a2c_td0(env, actor, critic, episodes=500, gamma=0.9, lr=1e-3, batch_size=1):
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    reward_history = []

    for episode in range(episodes):
        log_probs = []
        rewards = []
        values = []
        states = []
        state = env.reset()
        done = False

        # Collect data for this episode
        while not done:
            state_tensor = one_hot(state, env.state_space)
            policy_probs = actor(state_tensor)
            value = critic(state_tensor)
            dist = torch.distributions.Categorical(policy_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            states.append(state)
            state = next_state

        # Calculate TD(0) targets: G_t = r_t + gamma * V(s_{t+1})
        # Q2 use Mont Carlo returns to calculate advantage (Hint: refer to Lecture 10, slide 34)
        actor_loss = 0
        critic_loss = 0
        for t in range(len(rewards)):
            if t == len(rewards) - 1:  # Last step in the episode
                td_target = rewards[t]
            else:
                td_target = rewards[t] + gamma * values[t + 1]

            # Compute the advantage as the difference between target and value
            advantage = td_target - values[t]

            # Actor and Critic Losses
            actor_loss -= log_probs[t] * advantage
            critic_loss += advantage**2

        # Perform optimization for actor and critic separately
        actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)  # Retain graph for critic loss
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        reward_history.append(sum(rewards))


        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{episodes}, Total Reward: {reward_history[-1]}")

    return reward_history

def a2c_mc(env, actor, critic, episodes=500, gamma=0.9, lr=1e-3, batch_size=1):
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    reward_history = []

    for episode in range(episodes):
        log_probs = []
        rewards = []
        values = []
        states = []
        state = env.reset()
        done = False

        # Collect data for this episode
        while not done:
            state_tensor = one_hot(state, env.state_space)
            policy_probs = actor(state_tensor)
            value = critic(state_tensor)
            dist = torch.distributions.Categorical(policy_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            states.append(state)
            state = next_state


        # Monte Carlo returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        actor_loss = 0
        critic_loss = 0
        for t in range(len(rewards)):
            advantage = returns[t] - values[t]
            actor_loss -= log_probs[t] * advantage
            critic_loss += advantage**2


        actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        reward_history.append(sum(rewards))

        if (episode + 1) % 50 == 0:
            print(f"[MC] Episode {episode+1}/{episodes}, Total Reward: {reward_history[-1]}")

    return reward_history


def main():
    # Initialize the environment and separate networks
    env = GridWorld()
    actor = Actor(env.state_space, env.action_space)
    critic = Critic(env.state_space)
    # Run the A2C algorithm using TD(0) return (Instant Reward + Discounted Future Value)
    reward_curve_td0 = a2c_td0(env, actor, critic, episodes=500, gamma=0.9, lr=1e-3)
    # Plotting the convergence curve
    plt.figure(figsize=(10, 6))
    plt.plot(reward_curve_td0, label='A2C with TD(0) Returns')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("A2C with TD(0) Returns and Separate Actor and Critic Networks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("a2c_td0.png")
    plt.show()



    env = GridWorld()
    actor_mc = Actor(env.state_space, env.action_space)
    critic_mc = Critic(env.state_space)
    # Run the A2C algorithm using Monte Carlo returns
    reward_curve_mc = a2c_mc(env, actor_mc, critic_mc, episodes=500, gamma=0.9, lr=1e-3)
    # Plotting the convergence curve
    plt.figure(figsize=(10, 6))
    plt.plot(reward_curve_td0, label='A2C with TD(0) Returns')
    plt.plot(reward_curve_mc, label='A2C with Monte Carlo Returns')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("A2C Comparison: TD(0) vs Monte Carlo Returns")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("a2c_mc.png")
    plt.show()



if __name__ == "__main__":
    main()




