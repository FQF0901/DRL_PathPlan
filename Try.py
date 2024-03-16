import numpy as np

# 生成示例坐标数据
x1 = np.random.uniform(4, 6)
y1 = np.random.uniform(-0.8, -1.2)

x2 = np.random.uniform(-7, -6)
y2 = np.random.uniform(-0.8, -1.2)

x3 = np.random.uniform(-4, 0)
y3 = np.random.uniform(4, 6)

x4 = np.random.uniform(-1, 4)
y4 = np.random.uniform(4, 6)

x5 = np.random.uniform(3, 6)
y5 = np.random.uniform(4, 6)

# 创建示例数组
arr = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]])

# 指定要抽取的组数
n = 3

# 从数组中进行 n 组随机抽样
sampled_indices = np.random.choice(arr.shape[0], size=n, replace=False)
sampled_coordinates = arr[sampled_indices]

for coord in sampled_coordinates:
    print(coord[0])
