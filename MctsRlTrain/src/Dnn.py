"""
@author: Fqf
@time: 20240618
@file: Dnn.py
@description: Policy & Value Deep Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import DrlCfg
from torch.cuda.amp import autocast

# ==========================================================
# =============== BackboneNet and Multi-head ===============
# ==========================================================
class Net(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 1. Initial reading of BEV feature,
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.conv_bn = nn.BatchNorm2d(num_features=32)

        # 2. BackboneNet: ResNet extraction features
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4_bn = nn.BatchNorm2d(num_features=64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv5_bn = nn.BatchNorm2d(num_features=64)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv6_bn = nn.BatchNorm2d(num_features=64)

        # 3. Value head
        self.value_conv = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 12 * 7, DrlCfg.TreePara.StrActDim * DrlCfg.TreePara.GearActDim * DrlCfg.TreePara.DistActDim)

        # 4. adjust_channels
        self.adj_chl1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):

        # 1. Initial reading of BEV feature
        y = F.relu(self.conv_bn(self.conv(x)))

        # 2. BackboneNet: ResNet extraction features
        x = y
        y = F.relu(self.conv1_bn(self.conv1(y)))
        y = F.relu(x + self.conv2_bn(self.conv2(y)))
        y = F.max_pool2d(y, kernel_size=(2, 2), stride=(2, 2))

        x = y
        y = F.relu(self.conv3_bn(self.conv3(y)))
        y = F.relu(self.adj_chl1(x) + self.conv4_bn(self.conv4(y)))
        y = F.max_pool2d(y, kernel_size=(2, 2), stride=(2, 2))

        x = y
        y = F.relu(self.conv5_bn(self.conv5(y)))
        y = F.relu(x + self.conv6_bn(self.conv6(y)))
        y = F.max_pool2d(y, kernel_size=(2, 2), stride=(2, 2))

        # 3. Value head
        x = y

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = F.max_pool2d(value, kernel_size=(2, 2), stride=(2, 2))
        value = torch.reshape(value, [-1, 16 * 12 * 7])
        value = self.value_fc1(value)
        value = F.relu(value)
        value = F.dropout(value, 0.3)
        value = F.tanh(value)
        
        return value

# ==========================================================
# =================== Policy & Value Net ===================
# ==========================================================
class PolicyValueNet:
    def __init__(self, model_file=None):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.policy_value_net = Net().to(self.device)
        self.l2_const = 2e-3    # L2 Regularization
        self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.l2_const)

        if model_file:
            state_dict = torch.load(model_file, map_location=torch.device(self.device))
            self.policy_value_net.load_state_dict(state_dict)  # Load model parameters

# ------------------- Policy & Value net -------------------
    """ Using neural networks to evaluate policy and value in batch states """
    def policy_value_eval_batch(self, state_batch):
        self.policy_value_net.eval()    # Set the neural network to evaluation mode, which usually turns off specific layers used in training (such as Dropout or BatchNorm)
        # state_batch = torch.tensor(state_batch).float().to(self.device)
        state_batch = state_batch.clone().detach().float().to(self.device)

        # .detach(): Generate a new tensor that shares data storage with the original tensor 
        # but no longer has the history of gradient calculations (i.e. it becomes a leaf tensor). 
        # This is usually done to prevent the propagation of gradients.
        if self.device == 'cuda':   # When there is no GPU, a pop-up window will be displayed. This is used to clear the pop-up window.
            with autocast():    # Use mixed precision (FP16) calculations to reduce computational cost and memory usage.
                value_batch = self.policy_value_net(state_batch)
        else:
            value_batch = self.policy_value_net(state_batch)
        
        # value_batch = value_batch.cpu().detach().numpy()

        return value_batch

# ------------------------ Save net ------------------------
    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)

# ------------------------- Train --------------------------
    """Excute one-step Training"""
    def train_step(self, state_batch, value_batch):

        # 1. Data preparation
        self.policy_value_net.train()   # Set the neural network model to training mode(such as Dropout or BatchNorm)

        # state_batch = torch.tensor(state_batch).float().to(self.device)
        state_batch = state_batch.clone().detach().float().to(self.device)
        value_batch = torch.tensor(value_batch).float().to(self.device)

        # 2. Training parameter settings
        self.optimizer.zero_grad()  # Clear the gradient cache in the optimizer

        # 3. Forward propagation
        value = self.policy_value_net(state_batch)

        # 4. Calculation of loss
        value_loss = F.mse_loss(input=value, target=value_batch)   # The loss is calculated using the mean squared error (MSE) loss function
        loss = value_loss # The optimizer already includes a penalty term for L2 regularization (weight decay)

        # 5. Back Propagation
        loss.backward()
        self.optimizer.step()
        
        return loss.detach().cpu().numpy()

# ==========================================================
# ===================== Visualization ======================
# ==========================================================

if __name__ == "__main__":
    model = Net()
    summary(model, (3, 224, 384))  # The input dimensions are (batch_size, channels, height, width). Can add historical expansion node through batch_size