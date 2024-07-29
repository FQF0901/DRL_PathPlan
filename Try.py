import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 创建一个 500x500 的白色背景 RGB 图像: 一个像素代表2cm，覆盖范围为10*10m
image = np.ones((500, 500, 3), dtype=np.uint8) * 255

# 绘制三个空心矩形框，并调整每个矩形的 RGB 颜色强度
rect1 = patches.Rectangle((100, 100), 100, 100, linewidth=2, edgecolor=(1, 0, 0), facecolor='none')  # 红色
rect2 = patches.Rectangle((200, 200), 100, 100, linewidth=2, edgecolor=(0, 1, 0), facecolor='none')  # 绿色
rect3 = patches.Rectangle((300, 300), 100, 100, linewidth=2, edgecolor=(0, 0, 1), facecolor='none')  # 蓝色

# 将矩形框添加到图像中
fig, ax = plt.subplots()
ax.imshow(image)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)

# 添加网格
plt.grid(True)

# 关闭坐标轴
ax.axis('on')

# 保存图像为 CNN 可识别的格式（例如，PNG）
plt.savefig('CnnInput.png')

# 显示图像
plt.show()
