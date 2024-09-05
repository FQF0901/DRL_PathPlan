"""
@author: Fqf
@time: 20240618
@file: Train.py
@description: Training DNN
"""

import collections
import numpy as np
import pandas as pd
from PIL import Image
import ast
from Dnn import PolicyValueNet
import DrlUtil
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils
from Util import Config

# ==========================================================
# ======================== DataSet =========================
# ==========================================================

class CustomDataset(Dataset):   # 它继承自torch.utils.data.Dataset，并实现其中的两个方法：__len__和__getitem__
    def __init__(self, csv_file, img_folder, transform=None):
        self.labels_df = pd.read_csv(csv_file)  # self.labels_df是一个DataFrame对象，其中包含图片名和对应的目标标签
        self.img_folder = img_folder
        self.transform = transform  # 这是一个可选参数，用于在数据加载时对图像进行预处理（如缩放、裁剪、标准化等）

    def __len__(self):  # 返回数据集中样本的数量
        return len(self.labels_df)

    def __getitem__(self, idx): # 它接受一个索引idx，并返回对应的图像和标签
        img_name = os.path.join(self.img_folder, self.labels_df.iloc[idx, 0])   # 获取DataFrame中第idx行、第0列的图片名
        image = Image.open(img_name).convert('RGB') # 使用Pillow库的Image.open打开图片，并将其转换为RGB模式
        
        label_str = self.labels_df.iloc[idx, 1]
        label_list = ast.literal_eval(label_str)
        label = np.array(label_list, dtype='float')

        if self.transform:  # self.transform通常是一个torchvision.transforms.Compose对象，包含多个图像预处理步骤
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float)


# ==========================================================
# ========================= Train ==========================
# ==========================================================

class TrainPipeline:
    
    def __init__(self, init_model=None, batch_size=32, epoch_num=10) -> None:
        # 1. init paras
        self.batch_size = batch_size
        self.data_buffer = collections.deque(maxlen = 10000)
        self.epoch_num = epoch_num
        self.mse_targ = 10
        self.savenet_freq = 50

        self.policy_value_net = PolicyValueNet(model_file=init_model)

        # # 2. Load model
        # if init_model:
        #     try:
        #         self.policy_value_net = PolicyValueNet(model_file=init_model)
        #         print(utils.HighLightGreenMsg('已加载上次最终模型'))
        #     except:
        #         print(utils.HighLightRedMsg('模型路径不存在，从零开始训练'))
        #         self.policy_value_net = PolicyValueNet()
        # else:
        #     print(utils.HighLightRedMsg('从零开始训练'))
        #     self.policy_value_net = PolicyValueNet()

# --------------------- Policy Evaluate --------------------
    """Evaluate the capabilities of the policy value network"""
    def net_evaluate(self):
        pass

# ---------------------- Policy Update ---------------------
    """Train the network and update the parameters"""
    def net_update(self, epoch, writer):
        # 1. Preparing training data
        with tqdm(total=len(self.data_buffer), dynamic_ncols=True, desc='Train Progress Bar') as pbar:

            running_loss = 0.0

            for batch_idx, (state_batch, value_batch) in enumerate(self.data_buffer):

                value_batch = np.array(value_batch).astype('float32')

                # 2. Performance under the initial network(for comparison of training progress)
                old_value_batch = self.policy_value_net.policy_value_eval_batch(state_batch)

                # 3. Training network
                try:
                    loss = self.policy_value_net.train_step(state_batch, value_batch)
                except Exception as e:
                    print(utils.HighLightRedMsg(f"Error during training step: {e}"))
                    continue

                new_value_batch = self.policy_value_net.policy_value_eval_batch(state_batch)

                par_update_loss = F.mse_loss(input=new_value_batch, target=old_value_batch)
                if par_update_loss > self.mse_targ * 4:
                    print(utils.HighLightRedMsg('KL divergence is too bad, For loop is terminated !'))
                    break
                
                running_loss += loss.item()

                # 4. Save net
                if (epoch * len(self.data_buffer) + batch_idx + 1) % self.savenet_freq == 0:
                    # print("Save Net, : epoch_num {}".format(epoch))
                    mdl_name = os.path.join(Config.StorePath.train_dataset_path, 'policy_value_net_{}.pkl'.format(epoch * len(self.data_buffer) + batch_idx))
                    self.policy_value_net.save_model(mdl_name)
                
                # 5. Tensorboard
                writer.add_scalar('Loss/train', loss.item(), epoch * len(self.data_buffer) + batch_idx)
                current_lr = self.policy_value_net.optimizer.param_groups[0]['lr']  # 获取当前学习率
                writer.add_scalar('Learning Rate', current_lr, epoch * len(self.data_buffer) + batch_idx)

                # 6. Progress Bar
                cycle_interval = 10
                if batch_idx % cycle_interval == 0:
                        pbar.set_postfix({
                            'episode': '%d' % (epoch)
                            })
                        
                        pbar.update(cycle_interval)

            # 7. Print parameters to monitor training progress
            # print(("par_update_loss:{:.3f}," "current_lr:{:.3f}," "loss:{}").format(par_update_loss, current_lr, loss))

        return running_loss

# --------------------- Train Pipeline ---------------------
    """A complete training process"""
    def run(self, csv_file, img_folder):
        print(utils.HighLightGreenMsg('运行 train.run()'))
        try:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.policy_value_net.optimizer, gamma=0.999)
            writer = SummaryWriter(log_dir=DrlUtil.generate_new_train_dir(os.path.join(os.getcwd(), 'logs'), 
                                                                          DrlUtil.find_existing_train_dirs(os.path.join(os.getcwd(), 'logs'))))

            # 1. Create a dataset
            transform = transforms.Compose([transforms.Resize((224, 384)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            dataset = CustomDataset(csv_file=csv_file, img_folder=img_folder, transform=transform)

            for epoch in range(self.epoch_num):
                # 2. Loading data
                self.data_buffer = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4)

                # 3. Training net
                loss_sum = self.net_update(epoch, writer)
                scheduler.step()

                # 4. Post process
                writer.add_scalar('Loss_sum/train/average', loss_sum / len(self.data_buffer), epoch)
                # print(f'Epoch {epoch+1}/{self.epoch_num}, Loss: {loss:.4f}, lr: {self.policy_value_net.optimizer.param_groups[0]['lr']}')

            writer.close()
            print('===== Train done ! =====')

        except KeyboardInterrupt:
            print(utils.HighLightRedMsg('\n\rQuit'))


# -------------------------- Test --------------------------
if __name__ == '__main__':

    net_model = os.path.join(Config.StorePath.train_dataset_path, 'policy_value_net.pkl')
    csv_path = os.path.join(Config.StorePath.train_dataset_path, 'label.csv')
    img_path = os.path.join(Config.StorePath.train_dataset_path, 'images')

    training_pipeline = TrainPipeline(init_model=net_model, 
                                          batch_size=32,
                                          epoch_num=10)       
    training_pipeline.run(csv_file=csv_path, 
                          img_folder=img_path)

    # tensorboard --logdir=logs/train