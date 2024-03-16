import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import ParaCfg

class Env:
    def __init__(self):
        self.veh_rear_x = 0
        self.veh_rear_y = 0

        self.rectangles = []  # 存储随机数量矩形的角点
        self.veh = None
        self.slot = None  # 存储特定矩形（slot）的角点
    
    def reset(self):
        self.rectangles.clear()  # 清空之前的矩形数据
        n_rectangles = np.random.randint(0, 20)  # 随机确定矩形的数量（1~64）
        
        # 生成特定范围内的obj矩形
        for _ in range(n_rectangles):
            x = np.random.uniform(-8, 8)  # 中心点坐标在-8到8之间
            y = np.random.uniform(-4, 8)  # 中心点坐标在-8到8之间
            yaw = np.random.uniform(0, 360)  # 旋转角度在0到360度之间
            length, width = np.random.uniform(0.1, 0.5, 2)  # 长度和宽度在0.1到0.5之间
            
            corners = utils.get_rectangle_corners(x, y, yaw, length, width)
            self.rectangles.append(corners)

        # 生成特定范围内的host veh
        self.veh_rear_x = np.random.uniform(-5, 5)  # 后轴坐标
        self.veh_rear_y = np.random.uniform(-1, 3)  # 后轴坐标
        self.veh_rear_yaw = np.random.uniform(-45, 45)  # 后轴坐标
        self.veh = utils.get_Veh_corners(self.veh_rear_x, self.veh_rear_y, self.veh_rear_yaw)
        
        # 生成特定范围内的矩形（slot）
        yaw = np.random.uniform(-0, 0)  # 旋转角度在0到360度之间
        length = np.random.uniform(4.8, 5.6)  # 长度和宽度在0.1到0.5之间
        width = np.random.uniform(2, 2.6)  # 长度和宽度在0.1到0.5之间
        x = 0
        y = -width / 2
        yaw = np.random.uniform(-0, 0)  # 旋转角度在0到360度之间
        self.slot = utils.get_rectangle_corners(x, y, yaw, length, width)
        
        # 检查并移除与 host veh 和 slot 存在重叠的obj矩形
        self.rectangles = [rect for rect in self.rectangles if not utils.check_overlap(rect, self.veh)]
        self.rectangles = [rect for rect in self.rectangles if not utils.check_overlap(rect, self.slot)]
        
        return self.rectangles, self.slot, self.veh
    
    def step(self, action):
        new_state = None
        reward = None
        done = False
        info = {}
        return new_state, reward, done, info
    
    def show(self):
        """绘制所有obj,slot和host veh"""
        if not self.rectangles and self.slot is None:
            print("No rectangles to show. Please call reset() first.")
            return

        fig, ax = plt.subplots()
        for corners in self.rectangles:
            polygon = patches.Polygon(corners, closed=True, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)

        if self.slot is not None:
            slot_polygon = patches.Polygon(self.slot, closed=True, edgecolor='b', facecolor='none')
            ax.add_patch(slot_polygon)
        
        veh_polygon = patches.Polygon(self.veh, closed=True, edgecolor='g', facecolor='none')
        ax.add_patch(veh_polygon)
        
        # 获取车辆后轴中心坐标
        veh_center = np.array([self.veh_rear_x, self.veh_rear_y])
        veh_center_circle = plt.Circle(veh_center, 0.05, color='g')  # 以绿色表示车辆后轴中心
        ax.add_artist(veh_center_circle)

        buffer = 1
        all_corners = np.vstack(self.rectangles + [self.slot])
        all_corners = np.vstack([all_corners, self.veh])
        
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