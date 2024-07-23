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
import Config
from torch.cuda.amp import autocast

# ==========================================================
# =============== BackboneNet and Multi-head ===============
# ==========================================================
'''
# ---------------------- ResNet Block ----------------------

class ResNetBlock(nn.Module):
    
    # This convolutional layer can extract num_filters features on the feature map
    def __init__(self, input_channels=128, num_channels=256, use_1x1conv=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=num_channels)
        self.conv1_act = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=num_channels)
        self.conv2_act = nn.ReLU()

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=(1, 1), stride=(1, 1))
        else:
            self.conv3 = None

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # Adding Max Pooling layer

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)

        y = self.conv2(y)
        y = self.conv2_bn(y)
        if self.conv3:
            x = self.conv3(x)
        y = x + y
        y = self.conv2_act(y)

        y = self.pool(y)  # Applying pooling after the residual connection

        return y

# --------------- BackboneNet and Multi-head ---------------

class Net(nn.Module):

    def __init__(self, num_channels=128, num_res_blocks=7, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 1. Initial reading of BEV feature
        self.conv = nn.Conv2d(in_channels=3, out_channels=num_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_bn = nn.BatchNorm2d(num_features=num_channels)
        self.conv_act = nn.ReLU()

        # 2. BackboneNet: ResNet extraction features
        # self.res_blocks = nn.ModuleList([ResNetBlock(input_channels=256, num_channels=256, use_1x1conv=False) for _ in range(num_res_blocks)])
        self.res_blocks = nn.ModuleList([
            ResNetBlock(num_channels, 128, False),
            ResNetBlock(128, 128, False),
            ResNetBlock(128, 256, True),
            ResNetBlock(256, 256, False),
            ResNetBlock(256, 256, False),
            # ResNetBlock(256, 512, True),
        ])

        # 3. Policy & Value head
        self.policy_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.policy_bn = nn.BatchNorm2d(128)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(128 * 4 * 4, Config.TreePara.GearActDim*Config.TreePara.StrActDim*Config.TreePara.DistActDim)

        self.value_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(128)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(128 * 4 * 4, 128)
        self.value_act2 = nn.ReLU()
        self.value_dropout = nn.Dropout(p=0.5)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # 1. Initial reading of BEV feature
        x = self.conv(x)
        x = self.conv_bn(x)
        x = self.conv_act(x)

        # 2. BackboneNet: ResNet extraction features
        for block in self.res_blocks:
            x = block(x)

        # 3. Policy & Value head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = torch.reshape(policy, [-1, 128 * 4 * 4])
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy)

        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = torch.reshape(value, [-1, 128 * 4 * 4])
        value = self.value_fc1(value)
        value = self.value_act1(value)
        value = self.value_dropout(value)
        value = self.value_fc2(value)
        value = F.tanh(value)

        return x
'''

class Net(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 1. Initial reading of BEV feature
        self.conv = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_bn = nn.BatchNorm2d(num_features=128)

        # 2. BackboneNet: ResNet extraction features
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_features=128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4_bn = nn.BatchNorm2d(num_features=128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv5_bn = nn.BatchNorm2d(num_features=256)
        self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv6_bn = nn.BatchNorm2d(num_features=256)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv7_bn = nn.BatchNorm2d(num_features=256)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv8_bn = nn.BatchNorm2d(num_features=256)

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv9_bn = nn.BatchNorm2d(num_features=256)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv10_bn = nn.BatchNorm2d(num_features=256)

        # 3. Policy & Value head
        self.policy_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.policy_bn = nn.BatchNorm2d(128)
        self.policy_fc = nn.Linear(128 * 4 * 4, Config.TreePara.GearActDim*Config.TreePara.StrActDim*Config.TreePara.DistActDim)

        self.value_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(128)
        self.value_fc1 = nn.Linear(128 * 4 * 4, 128)
        self.value_fc2 = nn.Linear(128, 1)

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
        y = F.relu(x + self.conv4_bn(self.conv4(y)))
        y = F.max_pool2d(y, kernel_size=(2, 2), stride=(2, 2))

        x = y
        y = F.relu(self.conv5_bn(self.conv5(y)))
        y = F.relu(self.conv5_1(x) + self.conv6_bn(self.conv6(y)))
        y = F.max_pool2d(y, kernel_size=(2, 2), stride=(2, 2))

        x = y
        y = F.relu(self.conv7_bn(self.conv7(y)))
        y = F.relu(x + self.conv8_bn(self.conv8(y)))
        y = F.max_pool2d(y, kernel_size=(2, 2), stride=(2, 2))

        x = y
        y = F.relu(self.conv9_bn(self.conv9(y)))
        y = F.relu(x + self.conv10_bn(self.conv10(y)))
        y = F.max_pool2d(y, kernel_size=(2, 2), stride=(2, 2))

        # 3. Policy & Value head
        x = y

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = torch.reshape(policy, [-1, 128 * 4 * 4])
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = torch.reshape(value, [-1, 128 * 4 * 4])
        value = self.value_fc1(value)
        value = F.relu(value)
        value = F.dropout(value, 0.5)
        value = self.value_fc2(value)
        value = F.tanh(value)
        
        return policy, value
    

# ==========================================================
# =================== Policy & Value Net ===================
# ==========================================================
class PolicyValueNet:

    def __init__(self, model_file=None, use_gpu=True, device = 'cuda'):
        self.use_gpu = use_gpu
        self.device = device
        self.policy_value_net = Net().to(self.device)
        self.l2_const = 2e-3    # L2 Regularization
        self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.l2_const)

        if model_file: 
            self.policy_value_net.load_state_dict(torch.load(model_file))   # Load model parameters

# ------------------- Policy & Value net -------------------
    """ Using neural networks to evaluate policy and value in batch states """
    def policy_value_batch(self, state_batch):
        self.policy_value_net.eval()    # Set the neural network to evaluation mode, which usually turns off specific layers used in training (such as Dropout or BatchNorm)
        state_batch = torch.tensor(state_batch).to(self.device)

        log_act_probs_batch, value_batch = self.policy_value_net(state_batch)
        log_act_probs_batch, value_batch = log_act_probs_batch.cpu(), value_batch.cpu()

        # .detach(): Generate a new tensor that shares data storage with the original tensor 
        # but no longer has the history of gradient calculations (i.e. it becomes a leaf tensor). 
        # This is usually done to prevent the propagation of gradients.
        act_probs_batch = np.exp(log_act_probs_batch.detach().numpy())  # Convert logarithmic probability to original probability
        value_batch = value_batch.detach().numpy()

        return act_probs_batch, value_batch
    
    """ Using neural networks to evaluate policy and value in a single state """
    def policy_value_single(self, state):
        self.policy_value_net.eval()

        # .ascontiguousarray(): ensures that the array is stored in memory in a continuous manner, which can improve the efficiency of data processing.
        current_state = np.ascontiguousarray(state.reshape(-1, 9, 10, 9)).astype('float16') # Need to adapt [important]
        current_state = torch.as_tensor(current_state).to(self.device)

        with autocast():    # Use mixed precision (FP16) calculations to reduce computational cost and memory usage.
            log_act_probs, value = self.policy_value_net(current_state)
        log_act_probs, value = log_act_probs.cpu() , value.cpu()

        # astype('float16'): Low-precision floating-point format, usually used to reduce storage space 
        # and speed up calculations when memory and computing resources are limited
        # .flatten(): Used to reduce a multidimensional array to a one-dimensional array, 
        # ensuring that the probability distribution log_act_probs is converted to a one-dimensional array act_probs
        act_probs = np.exp(log_act_probs.detach().numpy().astype('float16').flatten())
        legal_positions = state.availables  # Get a list of legal actions, Need to adapt [important]
        act_probs = zip(legal_positions, act_probs[legal_positions])    # Need to adapt [important]
        value = value.detach().numpy()

        return act_probs, value

# ------------------------ Save net ------------------------
    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)

# ------------------------- Train --------------------------
    """Excute one-step Training"""
    def train_step(self, state_batch, act_probs, value_batch, lr=0.002):

        # 1. Data preparation
        self.policy_value_net.train()   # Set the neural network model to training mode(such as Dropout or BatchNorm)

        state_batch = torch.tensor(state_batch).to(self.device)
        act_probs = torch.tensor(act_probs).to(self.device)
        value_batch = torch.tensor(value_batch).to(self.device)

        # 2. Training parameter settings
        self.optimizer.zero_grad()  # Clear the gradient cache in the optimizer
        for params in self.optimizer.param_groups:
            params['lr'] = lr

        # 3. Forward propagation
        log_act_probs, value = self.policy_value_net(state_batch)

        # 4. Calculation of loss
        value = torch.reshape(value, shape=[-1])
        value_loss = F.mse_loss(input=value, target=value_batch)   # The loss is calculated using the mean squared error (MSE) loss function
        policy_loss = -torch.mean(torch.sum(act_probs * log_act_probs, dim=1))
        loss = value_loss + policy_loss # The optimizer already includes a penalty term for L2 regularization (weight decay)

        # 5. Back Propagation
        loss.backward()
        self.optimizer.step()
        
        # 6. Evaluate the Net
        with torch.no_grad():
            # Cross entropy is used to measure the difference between two probability distributions.
            entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)) 

        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()

# ==========================================================
# ===================== Visualization ======================
# ==========================================================

model = Net()
summary(model, (3, 512, 384))  # The input dimensions are (batch_size, channels, height, width)