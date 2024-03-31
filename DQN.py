## ================================== DQN ====================================
import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import Env
import utils
import ParaCfg
import time
import logging
from torch.optim.lr_scheduler import StepLR

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
# class Qnet(torch.nn.Module):
#     ''' 只有一层隐藏层的Q网络 '''
#     def __init__(self, state_dim, hidden_dim, action_dim):  # state维度，128个全连接隐藏层，action维度
#         super(Qnet, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, hidden_dim)   # 设置全连接层，参数为全连接层的输入/出神经元个数
#         self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 1个隐藏层，相当于有2个全连接

#     def forward(self, x):
#         x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数（该网络只有一层隐藏层，因此激活层位于隐藏层和输出层之间）
#         return self.fc2(x)
class Qnet(torch.nn.Module):
    ''' 带有残差连接的多层隐藏层Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim, num_layers):  
        super(Qnet, self).__init__()
        
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 输入层到第一个隐藏层的全连接层
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])  # 多层隐藏层
        
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 最后一个隐藏层到输出层的全连接层
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一个隐藏层
        
        # 多层隐藏层
        for layer in self.hidden_layers:
            residual = x
            x = F.relu(layer(x) + residual)  # 残差连接结合ReLU激活函数

        return self.fc2(x)  # 输出层
# ------------------------------ DQN ----------------------------------
class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon_max, target_update, num_layers, device):
        self.action_dim = action_dim
        # ---------------- Q net ----------------
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim, num_layers).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim, num_layers).to(device)
        # ---------------- Adma optimizer ----------------
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=5000, gamma=0.8)  # 定义学习率调度器
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon_max  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state, EnvState, trainproc, CutActSpc):  # 可变的epsilon-贪婪策略采取动作
        # Collision value
        expdNode_list = []
        current_node = ParaCfg.Node(EnvState.StartPntStep[0], \
                                    EnvState.StartPntStep[1], \
                                    EnvState.StartPntStep[2], 0, 0, None)
        expdNode_list = utils.expandNode(current_node)  # expand child nodes from curnt node
        Cc_values_array = ([])
        obstacles = EnvState.ObjRect + EnvState.OthVehRect
        for nodes in expdNode_list:
            if (not utils.is_overlap_node(nodes, obstacles, 0.0, 0.0)):
                Cc_values_array = np.append(Cc_values_array, 0)
            else:
                Cc_values_array = np.append(Cc_values_array, -1)  # 碰撞惩罚，用于变相裁剪，-1表示不可选

        # action selection
        if np.random.random() < (self.epsilon * (1 - trainproc)):   # np.random.random() < (self.epsilon * (1 - trainproc))
            NotOvlpidx_array = []
            NotOvlpidx_array = np.where(Cc_values_array != -1)[0]   # 找到所有非 -1 的元素的索引

            if NotOvlpidx_array.size > 0:  
                action = np.random.choice(NotOvlpidx_array)    # 从非 -1 的元素idx中随机选择一个idx
            else:   # 如果所有action都碰撞
                action = np.random.randint(self.action_dim)

        else:
            # torch.tensor创建张量，是可以存储和操作数值数据的多维数组
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            
            if CutActSpc:
                # DQN net value
                q_values = self.q_net(state)  # 获取单个状态的所有动作的 Q 值
                q_values_array = q_values.detach().cpu().numpy()
                q_values_array = q_values_array[0,:]

                action = utils.combineDqnMcts(q_values_array, Cc_values_array)

            else:
                action = self.q_net(state).argmax().item()    # select optimal action by dqn

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
# ---------------------- #
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# ---------------------- #
lr = 0.01   # 0.005
num_episodes = 10000
hidden_dim = 128
num_layers = 3
gamma = 0.98
epsilon_max = 0.1
target_update = min(100, num_episodes / 100)
buffer_size = 10000
minimal_size = 500
batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = Env.Env()
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = 128
action_dim = 6
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon_max, 
            target_update, num_layers, device)

return_list = []
DQN_DoneCause = ParaCfg.DQNPostProc()

for i in range(10):
    # ---------------------- #
    logging.debug(" ***** 第 %s个for loop ***** ", i)
    # ---------------------- #
    with tqdm(total=int(num_episodes / 10), desc='Itr %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state, EnvState = env.reset()
            done = False
            # ---------------------- #
            logging.debug(" *** 第 %s个for loop里, 第%s个epsd *** ", i, i_episode)
            # ---------------------- #
            while not done:
                agent.scheduler.step()    # 动态调整学习率
                action = agent.take_action(state, EnvState, trainproc=(i_episode + i * 10) / num_episodes, CutActSpc = 1)
                next_state, reward, done, info, EnvState = env.step(action)   # changed by fqf
                # plt.close('all')  # 用于每一步的观测
                # env.show()
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # ---------------------- #
                logging.debug(" --- action: %s, reward: %s, done: %s, info: %s, episode_return: %s", \
                              action, reward, done, info, episode_return)
                # ---------------------- #
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
            DQN_DoneCause = utils.doneCausePropt(done, DQN_DoneCause, num_episodes / 100)

            if not (done >> 2) & 1: # PathNotFnd but done, need log and debug
                env.show('%s _ %s' % (i, i_episode))
            plt.close('all')

            if (i_episode + 1) % 30 == 0:
                pbar.set_postfix({
                    'epsd':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[int(- num_episodes / 100):]),
                    'StepCnt':
                    '%.3f' % (DQN_DoneCause.donePct_StepCnt_list[-1]),
                    'ActOvlp':
                    '%.3f' % (DQN_DoneCause.donePct_ActVehOvlp_list[-1]),
                    'PathFnd':
                    '%.3f' % (DQN_DoneCause.donePct_PathFnd_list[-1]),
                    'VehOutMap':
                    '%.3f' % (DQN_DoneCause.donePct_VehOutMap_list[-1])
                })
            pbar.update(1)

    # ---------------------- #
    print("StepCnt: %.3f, ActOvlp: %.3f, PathFnd: %.3f, VehOutMap: %.3f" % (
        DQN_DoneCause.donePct_StepCnt_list[-1], 
        DQN_DoneCause.donePct_ActVehOvlp_list[-1], 
        DQN_DoneCause.donePct_PathFnd_list[-1], 
        DQN_DoneCause.donePct_VehOutMap_list[-1])
        )
    # ---------------------- #
# ------------------------ DQN visualization ------------------------
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on HAS')
plt.savefig('return_plot.svg', format='svg')  # 保存图像为 PNG 格式
plt.show()

# mv_return = rl_utils.moving_average(return_list, 9)   # Lib in 'rl_utils'
# plt.plot(episodes_list, mv_return)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('DQN on HAS')
# plt.show()

episodes_list = list(range(len(DQN_DoneCause.donePct_StepCnt_list)))
plt.plot(episodes_list, DQN_DoneCause.donePct_StepCnt_list)
plt.xlabel('Episodes')
plt.ylabel('donePct_StepCnt')
plt.title('StepCnt')
plt.savefig('StepCnt_plot.svg', format='svg')  # 保存图像为 PNG 格式
plt.show()

episodes_list = list(range(len(DQN_DoneCause.donePct_ActVehOvlp_list)))
plt.plot(episodes_list, DQN_DoneCause.donePct_ActVehOvlp_list)
plt.xlabel('Episodes')
plt.ylabel('donePct_ActVehOvlp')
plt.title('ActVehOvlp')
plt.savefig('ActVehOvlp_plot.svg', format='svg')  # 保存图像为 PNG 格式
plt.show()

episodes_list = list(range(len(DQN_DoneCause.donePct_PathFnd_list)))
plt.plot(episodes_list, DQN_DoneCause.donePct_PathFnd_list)
plt.xlabel('Episodes')
plt.ylabel('donePct_PathFnd')
plt.title('PathFnd')
plt.savefig('PathFnd_plot.svg', format='svg')  # 保存图像为 PNG 格式
plt.show()

episodes_list = list(range(len(DQN_DoneCause.donePct_VehOutMap_list)))
plt.plot(episodes_list, DQN_DoneCause.donePct_VehOutMap_list)
plt.xlabel('Episodes')
plt.ylabel('donePct_VehOutMap')
plt.title('VehOutMap')
plt.savefig('VehOutMap_plot.svg', format='svg')  # 保存图像为 PNG 格式
plt.show()

## ===================================== Post-processing =====================================
# MODEL_PATH = 'D:\01_Work\DRL_PathPlan\model.pth'  # 后缀名为 .pth
# torch.save(DQN, MODEL_PATH) # 直接使用torch.save()函数即可

# net = torch.load(MODEL_PATH)