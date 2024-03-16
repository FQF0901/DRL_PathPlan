import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import ParaCfg
import random

class Env:
    def __init__(self):
        self.obj = []  # 存储随机数量obj的角点
        self.other_veh = []  # 存储随机数量other_veh的角点
        self.host_veh = None
        self.slot = None  # 存储特定矩形（slot）的角点

        self.SP = []
        self.TP = []
    
    def reset(self):
        self.obj.clear()  # 清空之前的矩形数据
        n_obj = np.random.randint(0, 10)  # 随机确定矩形的数量（1~64）
        
        # 生成特定范围内的obj矩形
        for _ in range(n_obj):
            x = np.random.uniform(-8, 8)
            y = np.random.uniform(-2, 6)
            yaw = np.random.uniform(0, 360)
            length, width = np.random.uniform(0.1, 0.5, 2)
            
            corners = utils.get_rectangle_corners(x, y, yaw, length, width)
            self.obj.append(corners)

        # 生成特定范围内的host veh(基于后轴)
        host_veh_rear_x = np.random.uniform(-5, 5)
        host_veh_rear_y = np.random.uniform(-1, 3)
        host_veh_rear_yaw = np.random.uniform(-45, 45)
        self.host_veh = utils.get_Veh_corners(host_veh_rear_x, host_veh_rear_y, host_veh_rear_yaw)
        
        # 生成特定范围内的矩形（slot）
        length = np.random.uniform(4.8, 5.6)
        width = np.random.uniform(2, 2.6)
        slot_x = 0
        slot_y = -width / 2
        slot_yaw = np.random.uniform(-10, 10)
        self.slot = utils.get_rectangle_corners(slot_x, slot_y, slot_yaw, length, width)

        # 生成slot周围的other veh
        length = np.random.uniform(4.4, 5.0)
        width = np.random.uniform(1.8, 2.1)
                       
        x1 = np.random.uniform(5, 6)
        y1 = np.random.uniform(-0.8, -1.2)
        yaw1 = np.random.uniform(-10, 10)

        x2 = np.random.uniform(-7, -4)
        y2 = np.random.uniform(-0.8, -1.2)
        yaw2 = np.random.uniform(-10, 10)

        x3 = np.random.uniform(-6, -4)
        y3 = np.random.uniform(4, 6)
        yaw3 = np.random.uniform(-10, 10)

        x4 = np.random.uniform(-1, 1)
        y4 = np.random.uniform(4, 6)
        yaw4 = np.random.uniform(-10, 10)

        x5 = np.random.uniform(4, 6)
        y5 = np.random.uniform(4, 6)
        yaw5 = np.random.uniform(-10, 10)

        n = random.randint(0, 5)
        arr = np.array([[x1, y1, yaw1], [x2, y2, yaw2], [x3, y3, yaw3], [x4, y4, yaw4], [x5, y5, yaw5]])

        # 从数组中进行 n 组随机抽样
        sampled_indices = np.random.choice(arr.shape[0], size=n, replace=False)
        sampled_coordinates = arr[sampled_indices]

        for coord in sampled_coordinates:
            corners = utils.get_rectangle_corners(coord[0], coord[1], coord[2], length, width)
            self.other_veh.append(corners)
        
        # 检查并移除与 host veh 和 slot 存在重叠的obj矩形
        self.obj = [rect for rect in self.obj if not utils.check_overlap(rect, self.host_veh)]
        self.obj = [rect for rect in self.obj if not utils.check_overlap(rect, self.slot)]
        self.other_veh = [rect for rect in self.other_veh if not utils.check_overlap(rect, self.host_veh)]

        if self.other_veh is not None:
            new_obj = [rect for rect in self.obj if not any(utils.check_overlap(rect, veh) for veh in self.other_veh)]
            self.obj = new_obj

        # 生成SP和TP
        self.SP = np.array([host_veh_rear_x, host_veh_rear_y, host_veh_rear_yaw])
        self.TP = utils.get_TP(self.slot, slot_yaw)
        
        return self.obj, self.slot, self.host_veh, self.other_veh
    
    def step(self, action):
        new_state = None
        reward = None
        done = False
        info = {}
        return new_state, reward, done, info
    
    def show(self):
        """绘制所有obj,slot,host veh和other veh"""
        if not self.obj and self.slot is None:
            print("No obj to show. Please call reset() first.")
            return

        fig, ax = plt.subplots()
        for corners in self.obj:
            polygon = patches.Polygon(corners, closed=True, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)

        for corners in self.other_veh:
            polygon = patches.Polygon(corners, closed=True, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)

        if self.slot is not None:
            slot_polygon = patches.Polygon(self.slot, closed=True, edgecolor='b', facecolor='none')
            ax.add_patch(slot_polygon)
        
        host_veh_polygon = patches.Polygon(self.host_veh, closed=True, edgecolor='g', facecolor='none')
        ax.add_patch(host_veh_polygon)
        
        # 绘制SP和TP
        SP_circle = plt.Circle(self.SP, 0.05, color='g')  # 以绿色表示SP
        ax.add_artist(SP_circle)
        TP_circle = plt.Circle(self.TP, 0.05, color='b')  # 以蓝色表示TP
        ax.add_artist(TP_circle)

        buffer = 1
        all_corners = np.vstack(self.obj + [self.slot])
        all_corners = np.vstack([all_corners, self.host_veh])
        
        # xlim = (min(all_corners[:, 0]) - buffer, max(all_corners[:, 0]) + buffer)
        # ylim = (min(all_corners[:, 1]) - buffer, max(all_corners[:, 1]) + buffer)
        xlim = (ParaCfg.MapParam.xmin, ParaCfg.MapParam.xmax)
        ylim = (ParaCfg.MapParam.ymin, ParaCfg.MapParam.ymax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        plt.show()

        # 保存图片到指定路径
        save_path = "image.jpg"
        plt.savefig(save_path)

# 示例使用
# env = Env()
# env.reset()
# env.show()