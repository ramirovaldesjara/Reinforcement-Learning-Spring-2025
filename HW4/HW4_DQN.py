import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Environment
class GridWorld:
    def __init__(self, grid_size=4, goal_state=11, obstacles={7, 13}, gamma=0.9):
        self.grid_size = grid_size
        self.state_space = grid_size * grid_size
        self.action_space = 4
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
        if action == 0:
            row = max(row - 1, 0)
        elif action == 1:
            row = min(row + 1, self.grid_size - 1)
        elif action == 2:
            col = max(col - 1, 0)
        elif action == 3:
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

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)

# Fitted Q-Learning (no replay, no target network)
def fitted_q_learning(env, q_net, episodes=500, gamma=0.9, lr=1e-3, batch_size=64):
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    reward_curve = []
    # buffer = ReplayBuffer()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = []

        while not done:
            state_tensor = torch.eye(env.state_space)[state].unsqueeze(0)
            q_values = q_net(state_tensor)
            action = torch.argmax(q_values).item() if np.random.rand() > 0.1 else np.random.choice(env.action_space)

            next_state, reward, done = env.step(action)
            total_reward += reward
            steps.append((state, action, reward, next_state, done))
            state = next_state
        # Q3.2 Add replay buffer to the code (hint: buffer.push(state, action, reward, next_state, done))

        # Train at end of episode
        for state, action, reward, next_state, done in steps:
            state_tensor = torch.eye(env.state_space)[state].unsqueeze(0)
            next_state_tensor = torch.eye(env.state_space)[next_state].unsqueeze(0)

            q_target = q_net(state_tensor).detach()
            q_next = q_net(next_state_tensor).detach()
            target_value = reward + (0 if done else gamma * torch.max(q_next).item())

            q_target[0, action] = target_value
            q_pred = q_net(state_tensor)

            loss = criterion(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        reward_curve.append(total_reward)
        # Q3.3 Add target network to the code

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, Total Reward: {total_reward}")

    return reward_curve


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


def fitted_q_learning_replay(env, q_net, target_net=None, episodes=500, gamma=0.9, lr=1e-3, batch_size=64):
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    reward_curve = []
    buffer = ReplayBuffer()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.eye(env.state_space)[state].unsqueeze(0)
            q_values = q_net(state_tensor)
            action = torch.argmax(q_values).item() if np.random.rand() > 0.1 else np.random.choice(env.action_space)

            next_state, reward, done = env.step(action)
            total_reward += reward
            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states_tensor = torch.eye(env.state_space)[states]
                next_states_tensor = torch.eye(env.state_space)[next_states]

                q_values = q_net(states_tensor)

                if target_net is None:
                    q_next = q_net(next_states_tensor).detach()
                else:
                    q_next = target_net(next_states_tensor).detach()

                q_targets = q_values.clone().detach()

                for idx in range(batch_size):
                    target_value = rewards[idx] + (0 if dones[idx] else gamma * torch.max(q_next[idx]))
                    q_targets[idx, actions[idx]] = target_value

                loss = criterion(q_values, q_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        reward_curve.append(total_reward)

        # Update target network if provided
        if target_net is not None and (ep + 1) % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (ep + 1) % 50 == 0:
            print(f"[Replay] Episode {ep+1}, Total Reward: {total_reward}")

    return reward_curve



def main():
    # 3.a Original (no replay, no target)
    env = GridWorld()
    q_net = QNetwork(state_size=env.state_space, action_size=env.action_space)
    rewards = fitted_q_learning(env, q_net, episodes=500)
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward per Episode")
    plt.title("Fitted Q-Learning (No Replay, No Target Net)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fitted_q_no_replay_no_target.png")
    plt.close()

    # 3.b Replay buffer
    env = GridWorld()
    q_net_replay = QNetwork(state_size=env.state_space, action_size=env.action_space)
    rewards_replay = fitted_q_learning_replay(env, q_net_replay, episodes=500)
    plt.plot(rewards_replay)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward per Episode")
    plt.title("Fitted Q-Learning with Replay Buffer")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fitted_q_with_replay.png")
    plt.close()

    # 3.c Replay buffer + target network a
    env = GridWorld()
    q_net_target = QNetwork(state_size=env.state_space, action_size=env.action_space)
    target_net_target = QNetwork(state_size=env.state_space, action_size=env.action_space)
    target_net_target.load_state_dict(q_net_target.state_dict())
    rewards_target = fitted_q_learning_replay(env, q_net_target, target_net=target_net_target, episodes=500)
    plt.plot(rewards_target)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward per Episode")
    plt.title("Fitted Q-Learning with Replay Buffer and Target Network")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fitted_q_with_replay_target.png")
    plt.close()



if __name__ == "__main__":
    main()


