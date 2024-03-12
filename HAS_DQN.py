import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 设置是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN网络作为Q-网络
class QNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetwork, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv_net(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv_net(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# 经验回放
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, input_shape, n_actions, replay_buffer):
        self.q_network = QNetwork(input_shape, n_actions).to(device)
        self.target_q_network = QNetwork(input_shape, n_actions).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters())
        self.replay_buffer = replay_buffer
        self.n_actions = n_actions
    
    def update_policy(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.tensor(states, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.uint8).to(device)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_q_network(next_states).max(1)[0]
        expected_q_values = rewards + 0.99 * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32).to(device)
            q_values = self.q_network(state)
            action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)
        return action
    
    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

# 主循环
def train_dqn(env, agent, n_episodes, batch_size):
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_episode = lambda episode: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * episode / epsilon_decay)
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            epsilon = epsilon_by_episode(episode)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.update_policy(batch_size)
        
        if episode % 10 == 0:
            agent.update_target_network()
            print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {epsilon}")
