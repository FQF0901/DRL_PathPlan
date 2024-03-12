## ================================== DQN ====================================
import random
import gym
# import gymnasium as gym # select gym or gymnasium
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

# ---------------------------- ReplayBuffer ---------------------------
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 双端队列,先进先出，类似list但更小更快

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))   # ()是创建元组tuple

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions) # 相当于分列打包
        return np.array(state), action, reward, np.array(next_state), done  # 经测试有没有np.array强制转换似乎不影响

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    
# ----------------------------- Q net ---------------------------------
class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):  # state维度，128个全连接隐藏层，action维度
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)   # 设置全连接层，参数为全连接层的输入/出神经元个数
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 1个隐藏层，相当于有2个全连接

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数（该网络只有一层隐藏层，因此激活层位于隐藏层和输出层之间）
        return self.fc2(x)
    
# ----------------------------- ConvolutionalQnet ---------------------------------
class ConvolutionalQnet(torch.nn.Module):
    ''' 加入卷积层的Q网络 '''
    def __init__(self, action_dim, in_channels=4):  # 对于灰度图像，“in_channels”将为 1，对于 RGB 图像，“in_channels”将为 3
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)  # in_channels个输入，“32”输出通道数，“kernel_size=8”卷积核大小，“stride=4”核移动步长
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)   # “32”输入通道、“64”输出通道、内核大小“4”和步幅“2”
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)   # “64”输入通道、“64”输出通道、内核大小“3”和步幅“1”。
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512) # 全连接层，它以“7 * 7 * 64”作为输入大小（根据前面卷积层的输出形状计算），以“512”作为输出大小
        self.head = torch.nn.Linear(512, action_dim)    # “self.head”表示另一个具有“512”输入神经元和“action_dim”输出神经元的全连接层，负责为每个动作生成Q值

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x))
        return self.head(x)

# ------------------------------ DQN ----------------------------------
class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        # ---------------- Q net ----------------
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # ---------------- Convolutional Qnet ----------------
        # self.q_net = ConvolutionalQnet(state_dim, hidden_dim,
        #                   self.action_dim).to(device)  # Q网络
        # # 目标网络
        # self.target_q_net = ConvolutionalQnet(state_dim, hidden_dim,
        #                          self.action_dim).to(device)
        # ---------------- Adma optimizer ----------------
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)    # parameters代表一个模型的可学习参数
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            # torch.tensor创建张量，是可以存储和操作数值数据的多维数组，张量和普通数组的区别如下：
            # 1. 同质数据：张量存储同质数据，这使得 GPU 计算更加高效，因为硬件可以并行处理多个数据点
            # 2. 内存分配：PyTorch 中的张量被分配在连续的内存块中，使它们更容易存储在 GPU 内存中并允许更快的计算
            # 3. 广播：PyTorch 中的张量支持广播，即使张量的形状不相同，也允许张量之间进行元素级运算
            # 4. 自动微分：PyTorch中的张量支持自动微分，这是训练深度学习模型的关键组成部分
            # 5. GPU加速：当在 GPU 上执行张量运算时，数据从 CPU 传输到 GPU 内存，在那里可以由数千个线程并行处理
            state = torch.tensor([state], dtype=torch.float).to(self.device)    # allows to specify the data type and the device (CPU or GPU) on which the tensor should reside
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # view()用于重塑张量而不更改其基础数据(连续内存，避免数据重复并提升效率)
        # View(-1, 1)表示把张量resharp为n行1列的形式
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # gathers elements from self.q_net(states) tensor along dimension 1 (columns) based on the actions tensor
        q_values = self.q_net(states).gather(1, actions)  # Q值：得到该state下可行actions的所有Q value
        # 下个状态的最大Q值, 每行self.target_q_net(next_states)会有2个动作的Q value，即batch_size × 2
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)    # view(-1, 1):把tensor从左到右从上到下排成1列
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标，batch_size × 1
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数，1 × 1

        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 根据 均方误差损失函数 进行 反向传播 更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

# ----------------------------------- train DQN ----------------------------------
lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name)    # , render_mode="human"
random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            # state, _= env.reset()      # handcode fqf
            state, _ = env.reset(seed=0)
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _, _ = env.step(action)   # changed by fqf
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过500后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
# ------------------------ DQN visualization ------------------------
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)   # Lib in 'rl_utils'
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

## ===================================== Post-processing =====================================
# MODEL_PATH = 'model.pth'  # 后缀名为 .pth
# torch.save(net, MODEL_PATH) # 直接使用torch.save()函数即可

# net = torch.load(MODEL_PATH)