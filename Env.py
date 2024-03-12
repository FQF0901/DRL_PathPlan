import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import get_rectangle_corners, check_overlap

class Env:
    def __init__(self):
        self.rectangles = []  # 存储随机数量矩形的角点
        self.slot = None  # 存储特定矩形（slot）的角点
    
    def reset(self):
        self.rectangles.clear()  # 清空之前的矩形数据
        n_rectangles = np.random.randint(0, 20)  # 随机确定矩形的数量（1~64）
        
        for _ in range(n_rectangles):
            x = np.random.uniform(-8, 8)  # 中心点坐标在-8到8之间
            y = np.random.uniform(-4, 8)  # 中心点坐标在-8到8之间
            yaw = np.random.uniform(0, 360)  # 旋转角度在0到360度之间
            length, width = np.random.uniform(0.1, 0.5, 2)  # 长度和宽度在0.1到0.5之间
            
            corners = get_rectangle_corners(x, y, yaw, length, width)
            self.rectangles.append(corners)
        
        # 生成特定范围内的矩形（slot）
        x = np.random.uniform(-4, 4)  # 中心点坐标在-8到8之间
        y = np.random.uniform(-1, 1)  # 中心点坐标在-8到8之间
        yaw = np.random.uniform(-15, 15)  # 旋转角度在0到360度之间
        length = np.random.uniform(4.8, 5.6)  # 长度和宽度在0.1到0.5之间
        width = np.random.uniform(2, 2.6)  # 长度和宽度在0.1到0.5之间
        self.slot = get_rectangle_corners(x, y, yaw, length, width)
        
        # 检查并移除与slot存在重叠的矩形
        self.rectangles = [rect for rect in self.rectangles if not check_overlap(rect, self.slot)]
        
        return self.rectangles, self.slot
    
    def step(self, action):
        new_state = None
        reward = None
        done = False
        info = {}
        return new_state, reward, done, info
    
    def show(self):
        """绘制所有矩形和slot。"""
        if not self.rectangles and self.slot is None:
            print("No rectangles to show. Please call reset() first.")
            return
        
        fig, ax = plt.subplots()
        for corners in self.rectangles:
            polygon = patches.Polygon(corners, closed=True, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)
        
        if self.slot is not None:
            slot_polygon = patches.Polygon(self.slot, closed=True, edgecolor='g', facecolor='none')
            ax.add_patch(slot_polygon)
        
        buffer = 1
        all_corners = np.vstack(self.rectangles + [self.slot])
        xlim = (min(all_corners[:, 0]) - buffer, max(all_corners[:, 0]) + buffer)
        ylim = (min(all_corners[:, 1]) - buffer, max(all_corners[:, 1]) + buffer)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.grid(True)
        plt.axis("equal")
        plt.show()

# 示例使用
env = Env()
env.reset()
env.show()
