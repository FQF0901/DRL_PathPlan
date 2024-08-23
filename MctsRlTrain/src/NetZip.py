"""
@author: Fqf
@time: 20240711
@file: NetZip.py
@description: Model compression and deployment
"""

import torch
import torch.nn.utils.prune as prune
import torchvision.models as models

# ==========================================================
# ======================= Net Pruning ======================
# ==========================================================

# ----------------- Magnitude-based Pruning ----------------

# 加载ResNet模型
model = models.resnet18(pretrained=True)

# 定义要进行剪枝的卷积层
module = model.conv1

# 在conv1层上进行重要性剪枝，裁剪掉20%的权重
prune.l1_unstructured(module, name="weight", amount=0.2)

# 剪枝后需要移除剪枝参数
prune.remove(module, "weight")

# -------------------- Channel Pruning ---------------------

# 加载ResNet模型
model = models.resnet18(pretrained=True)

# 定义要进行通道剪枝的卷积层（示例中以resnet18的第一个卷积层为例）
module = model.layer1[0].conv1

# 获取要剪枝的维度（第1维是输入通道数）
num_channels = module.weight.shape[1]

# 定义要剪掉的通道数目（裁剪掉一半的通道）
num_channels_to_prune = num_channels // 2

# 使用通道剪枝工具剪枝
prune.l1_unstructured(module, name="weight", amount=num_channels_to_prune)

# 剪枝后需要移除剪枝参数
prune.remove(module, "weight")

# ------------------- Structured Pruning -------------------

# 加载ResNet模型
model = models.resnet18(pretrained=True)

# 定义要进行结构化剪枝的卷积层（示例中以resnet18的第一个卷积层为例）
module = model.layer1[0].conv1

# 定义要剪枝的维度（例如第0维是卷积核的数量）
dim = 0

# 使用结构化剪枝剪枝，此处示例为裁剪掉20%的卷积核
prune.ln_structured(module, name="weight", amount=0.2, n=2, dim=dim)

# 剪枝后需要移除剪枝参数
prune.remove(module, "weight")


# ==========================================================
# ================= Knowledge Distillation =================
# ==========================================================

