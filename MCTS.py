# 导入所需的库
import numpy as np
import random
import torch
import utils
import ParaCfg
import Env
import logging
import tqdm
import collections

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

# ---------------------------- MCTS Tree ---------------------------
class MCTS:
    def __init__(self):
        self.c_puct = 5
        self.root_node = ParaCfg.MctsNode()
        self.grid_cells = [[] for _ in range(ParaCfg.HASParam.grid_num ** 2)]   # used for check repeat state

    def select_node(self, curt_node):
        # 通过PUCT公式选择子节点中最有价值的节点
        best_value = float("-inf")
        selected_node = None
        for child_node in curt_node.children:
            exploitation_term = child_node.Q
            exploration_term = np.sqrt(child_node.HasNode.parent.visit_count) / (1 + child_node.visit_count)
            puct_value = exploitation_term + self.c_puct * child_node.P * exploration_term  # PUCT公式

            if puct_value > best_value and child_node.P != 0:
                best_value = puct_value
                selected_node = child_node
        
        if selected_node == None:  # 如何所有child都不vaild，则该node不应该选择
            curt_node.P = 0
            # curt_node.Q = 0 # 不能给float("-inf")，太小在回溯时会过于影响父节点。干脆不给人工值
        else:
            curt_idx = utils.get_grid_index(curt_node.x, curt_node.y)   # used for check repeat state
            self.grid_cells[curt_idx].append(curt_node)

        return selected_node

    def expand_node(self, curt_node, EnvInfo):   # 这里要把不合法的动作概率全部设置为0，并补充P和Q
        new_node = ParaCfg.MctsNode()
        new_HasNode_list = utils.expandNode(curt_node)
        cnt = 0

        for HasNode in new_HasNode_list:
            new_node.HasNode = HasNode
            DnnQ, DnnP = utils.DNN(new_node.HasNode)    # 需要补充DNN
            new_node.Q = DnnQ
            new_node.P = DnnP

            EnvAction = utils.EnvDRL_ActionMapping(cnt)
            new_node.HasNode.g_cost = new_node.HasNode.g_cost + utils.EnvReward(EnvAction, EnvInfo)  # 用于最后的真值Q（基于Path的评估）
            cnt = cnt + 1

            EnvInfoState = EnvInfo.state
            if utils.ChildNotVaild(new_node, EnvInfoState, self.grid_cells):   # ovlp和RepeatMove，P为0
                new_node.P = 0  
                # new_node.Q = 0 # 不能给float("-inf")，太小在回溯时会过于影响父节点。干脆不给人工值

            curt_node.children.append(new_node) # 都填进去，只是不vaild的不选择

    def backpropagate(self, node, DnnQ):
        # 反向传播，更新节点的信息（访问次数、累计奖励等）
        while node is not None:
            node.Q = (node.Q * node.visit_count + DnnQ) / (node.visit_count + 1) # node.Q = total_reward / visit_count or DnnQ
            node.visit_count += 1
            node = node.parent

    def is_leaf(self, node):
        """检查是否是叶节点，即没有被扩展的节点"""
        NoChildFlag = node.children == None # 没子节点
         
        if not NoChildFlag: # 有子节点但均不vaild
            ChildNotVaildFlag = sum(child.P for child in node.children) == 0

        return NoChildFlag or ChildNotVaildFlag    # 这里还要增加判断children的P是否不为0
    
    def simulate(self, done, EnvInfo):  # 这是一个完整的plan流程
        Has_node = ParaCfg.HasNode(EnvInfo.SlotPntInit[0], \
                                        EnvInfo.SlotPntInit[1], \
                                        EnvInfo.SlotPntInit[2], 0, 0, None)
        node = ParaCfg.MctsNode()
        node.HasNode = Has_node

        while not done: # DQN不需要考虑openlist=[]，但MCTS和HAS需要考虑
            while True: # 探索选择，直到找到叶节点
                if self.is_leaf(node):
                    LeafNode = node
                    break
                node = self.select_node(node)
                
            self.expand_node(LeafNode, EnvInfo)   # 这里要判断是否pathfound和openlist
            self.backpropagate(LeafNode)

        node.Q = node.HasNode.g_cost  # 如果终止了，就应该给出真值用于更新DNN的Q。需要细致的评判轨迹的优劣
        self.backpropagate(node)

        # 这里应该增加replay_buffer的存储：state, action, reward, next_state, done。其中reward应该就是node.Q？
    

# ----------------------------------- Training Process ----------------------------------
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

                MCTS.simulate(done, EnvInfo)    # 需要确认这里EnvInfo好还是EnvState好。另这是个off policy的方案
                replay_buffer.add(state, action, reward, next_state, done)
                episode_return += reward
                # ---------------------- #
                logging.debug(" --- action: %s, reward: %s, done: %s, info: %s, episode_return: %s", \
                              action, reward, done, info, episode_return)
                # ---------------------- #
                # 当buffer数据的数量超过500后,才进行网络训练
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