# Notebook

原本想要解决的问题：是DRL对hybrid A star的搜索进行加速，即DRL直接根据当前state给出optimal action（而非人工定义的启发函数），得到next state，然后再给optimal action，。。。直到得到轨迹

=====================================

### 方案1：DQN
1. 输入为128维（15个障碍物 * 4个角点 *2个坐标x/y）+ 起点坐标（x, y, yaw） + 终点坐标（x, y, yaw）+ 补了两个0凑成128
2. 输出为6个动作（右前，中前，左前，右后，中后，左后）
3. 网络为5层Resnet
4. 给了碰撞、episode耗时、原地打转的惩罚，和发现路径的奖励等

**遇到的问题：DRL倾向于直接撞障碍物或者原地打转**

=====================================

### 方案2：DQN + cut action space

方案1中人为设计的奖惩，难以在撞障碍物和原地打转之间平衡，引入如下方案2：
1. 撞障碍物：变相的裁剪到action space：在给出optimal action之前要先做overlap校验，有碰撞的动作不可选。该方案可以避免DRL的老是向障碍物上撞
2. 原地打转：借用hybrid A star上grid的概念，将已经探索到的grid（这个grid map是3维的：x, y, yaw）设置为不可选（有点像围棋上已经下子的点不能被再下子，实际上也是对action space的裁剪）

上述方案里的第1点已经实施，**让DRL尽量跑满episode充分探索。但发现训练10000次发现该方案基本没有任何提高和降低**。因此提出方案3：

=====================================

### 方案3：MCTS + DNN

1. 针对方案2撞障碍物，修改action space后不奏效的问题，应该是DRL没有学到足够有用的东西：**当前的方案有点稀疏奖励，只在轨迹生成的时刻进行奖励**，即方案设计上只对最后的select action进行奖励，而在path found之前的action/state给的reward都是负值（因为没找到轨迹且有episode耗时等惩罚）。
个人认为上述方案是有问题的，**会让DRL只知道最后一步的action和state是好的，但不知道如何到达最后一个state。应该借用alphago的方案，当path found后回溯，该episode下所有action和state都应奖励**。

2. 针对方案2原地打转要引入hybrid A star的grid的问题，虽然也是是对action space的裁剪，但这个裁剪要求DRL知道之前state是什么（即本次episode是否探索过该位置），这个MDP本质相悖。因此在想是否要引入MCTS以simulation的方式更好的剔除重复动作，给出action的价值

3. AlphaZero不是任何典型的DRL方法，其重点在MCTS，DNN仅用于2处：一是对MCTS进行宽度和深度上的裁剪，二是用于逼近和存储MCTS信息。
   
4. AlphaZero为何同时拥有policy net和value net？实际可以用value net做policy net的活儿（即给出先验概率进行MCTS的宽度裁剪），但一是在巨大action space的情况下效率低下；二是他俩本质是在干两个不同的事情，用不同的网络头会更适合，并在一起实践效果不好。详见AlphaZero作者本人的解释：https://www.reddit.com/r/reinforcementlearning/comments/1b1te73/help_me_understand_why_use_a_policy_net_instead/

5. 从DRL的角度思考AlphaZero，MCTS是解决了DRL中最难解决的reward问题，即稀疏/延时奖励下如何准确及时的给出reward。除了MCTS也可以使用IM解决reward的问题

6. DNN是对MCTS的逼近和存储，那么DNN对启发函数的优化上限是MCTS找到轨迹的性能（如果MCTS在某些case下找不到轨迹，那DNN就没有该case下可逼近的有价值的Q）。**那么为何某些场景下人可以找到泊车轨迹而MCTS/HAS找不到**？给出方案5
   
=====================================

### 方案4：
 
某些case下人可以找到轨迹，但MCTS/HAS找不到轨迹的根本原因有两个：
- cycle counter > max_search_cnt
- openlist = []
  
其他如spacelimit等全是人为附加的场景，和规划本身无关

1. cycle counter > max_search_cnt问题本质有两个：
   1. 一是max_search_cnt不能放太大因为会有计算力和实时性问题，【增加硬件算力，改善启发函数】
   2. 二是启发函数不够好，没法在很短的时间内给出有用的node，而只能暴力搜索【改善启发函数】
   
2. openlist = []分两种：
   1. 一是确实空间逼仄，动作又是离散的，无法充分利用空间导致给不出轨迹。【动力由离散 -> 连续】
   2. 二是有些case默认算法下规划会open list =[]，但修改heuristic func后却能出轨迹。原因是grid尺寸大，且启发函数不好，导致non-optimal node先占领了某些grid，使得后续拓展出来的optimal node无法占领该grid，而RS又是解析解导致给不出无碰撞轨迹。【改善启发函数，缩小grid尺寸(其极限是动作连续)】

可以看出上述两个问题有3个方案：
1. 增加硬件算力：难以实现
2. 改善启发函数：MCTS + DNN已经解决
3. **改为连续动作空间：方案4需要解决的问题**

因此引入DDPG
