"""
@author: Fqf
@time: 20240618
@file: Train.py
@description: Training DNN
"""

import random
import collections
import numpy as np
from Dnn import PolicyValueNet
import pickle
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils

# ==========================================================
# ========================= Train ==========================
# ==========================================================

class TrainPipeline:
    
    def __init__(self, init_model=None) -> None:
        # 1. init paras
        self.batch_size = 1000
        self.data_buffer = collections.deque(maxlen = self.batch_size)
        self.epochs = 5
        self.lr = 1e-3
        self.lr_multiplier = 1  # Adjusting learning rate based on KL adaptation
        self.kl_targ = 0.02
        self.game_batch_num = 30
        self.savenet_freq = min(5, self.game_batch_num / 2)

        # 2. Load model
        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print('已加载上次最终模型')
            except:
                print('模型路径不存在，从零开始训练')
                self.policy_value_net = PolicyValueNet()
        else:
            print('从零开始训练')
            self.policy_value_net = PolicyValueNet()

# --------------------- Policy Evaluate --------------------
    """Evaluate the capabilities of the policy value network"""
    def net_evaluate(self):
        pass

# ---------------------- Policy Update ---------------------
    """Train the network and update the parameters"""
    def net_update(self):
        # 1. Preparing training data
        mini_batch = random.sample(self.data_buffer, self.batch_size)

        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype('float32')

        act_probs_batch = [data[1] for data in mini_batch]
        act_probs_batch = np.array(act_probs_batch).astype('float32')

        value_batch = [data[2] for data in mini_batch]
        value_batch = np.array(value_batch).astype('float32')

        # 2. Performance under the initial network(for comparison of training progress)
        old_act_probs_batch, old_value_batch = self.policy_value_net.policy_value_batch(state_batch)

        # 3. Training network
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, act_probs_batch, value_batch, self.lr * self.lr_multiplier)

            new_act_probs_batch, new_value_batch = self.policy_value_net.policy_value_batch(state_batch)

            kl = np.mean(np.sum(old_act_probs_batch * (np.log(old_act_probs_batch + 1e-10) - np.log(new_act_probs_batch + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # If the KL divergence is bad, terminate the for loop
                print(utils.HighLightRedMsg('KL divergence is too bad, For loop is terminated !'))
                break

        # 4. Adaptive learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # 5. Print parameters to monitor training progress
        print(("kl:{:.5f}," "lr_multiplier:{:.3f}," "loss:{}," "entropy:{},").format(kl, self.lr_multiplier, loss, entropy))

        return loss, entropy

# --------------------- Train Pipeline ---------------------
    """A complete training process"""
    def run(self):
        try:
            for i in range(self.game_batch_num):
                # 1. Loading data
                try:
                    with open('Mcts_Train_Data_buffer.pkl', 'rb') as data_dict:
                        data_file = pickle.load(data_dict)
                        self.data_buffer = data_file['data_buffer']
                        # self.iters = data_file['iters']   # [important]
                        del data_file
                    print('Import data from buffer_pkl success !')
                    break
                except:
                    time.sleep(30)  # To avoid resource competition or external dependency
                    print(utils.HighLightRedMsg('Failed to load data_buffer by pickle, waiting 30s! game_batch_num {} '.format(i)))

                # 2. Training net
                # print('step i {}: '.format(self.iters))   # [important]
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.net_update()

                    self.policy_value_net.save_model('crnt_policy_value_net.pkl')

                #  3. Save net
                if (i + 1) % self.savenet_freq == 0:
                    print("Save Net, : game_batch_num {}".format(i))
                    self.policy_value_net.save_model('TmpNet/crnt_policy_value_net_{}.pkl'.format(i))

        except KeyboardInterrupt:
            print(utils.HighLightRedMsg('\n\rQuit'))


# -------------------------- Test --------------------------
training_pipeline = TrainPipeline(init_model='crnt_policy_value_net.pkl')
training_pipeline.run()