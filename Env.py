import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import ParaCfg
import random
import math

class EnvInfo:
    def __init__(self):
        self.ObjRect = []  # 存储随机数量obj的角点
        self.OthVehRect = []  # 存储随机数量other_veh的角点

        self.VehPntInit = []
        self.VehRectInit = None
        self.SlotPntInit = []
        self.SlotRectInit = None

        self.StartPntStep = []
        self.StartRectStep = None
        self.TgtPntStep = []
        self.TgtRectStep = None

class Env:
    def __init__(self):
        self.EnvInfo = EnvInfo()
    
    def reset(self):
        self.EnvInfo.ObjRect.clear()  # 清空之前的矩形数据
        n_ObjRect = np.random.randint(0, 10)  # 随机确定矩形的数量（1~64）
        
        # 生成特定范围内的obj矩形
        for _ in range(n_ObjRect):
            x = np.random.uniform(-8, 8)
            y = np.random.uniform(-2, 6)
            yaw = np.random.uniform(0, math.pi)
            length, width = np.random.uniform(0.1, 0.5, 2)
            
            corners = utils.get_rectangle_corners(x, y, yaw, length, width)
            self.EnvInfo.ObjRect.append(corners)

        # 生成特定范围内的host veh(基于后轴)
        host_veh_rear_x = np.random.uniform(-5, 5)
        host_veh_rear_y = np.random.uniform(-1, 3)
        host_veh_rear_yaw = np.random.uniform(math.radians(-45), math.radians(45))
        self.EnvInfo.VehRectInit = utils.get_Veh_corners(host_veh_rear_x, host_veh_rear_y, host_veh_rear_yaw)
        
        # 生成特定范围内的矩形（slot）
        length = np.random.uniform(4.8, 5.6)
        width = np.random.uniform(2, 2.6)
        slot_x = 0
        slot_y = -width / 2
        slot_yaw = np.random.uniform(math.radians(-10), math.radians(10))
        self.EnvInfo.SlotRectInit = utils.get_rectangle_corners(slot_x, slot_y, slot_yaw, length, width)

        # 生成slot周围的other veh
        length = np.random.uniform(4.4, 5.0)
        width = np.random.uniform(1.8, 2.1)
                       
        x1 = np.random.uniform(5, 6)
        y1 = np.random.uniform(-0.8, -1.2)
        yaw1 = np.random.uniform(math.radians(-10), math.radians(10))

        x2 = np.random.uniform(-7, -4)
        y2 = np.random.uniform(-0.8, -1.2)
        yaw2 = np.random.uniform(math.radians(-10), math.radians(10))

        x3 = np.random.uniform(-6, -4)
        y3 = np.random.uniform(4, 6)
        yaw3 = np.random.uniform(math.radians(-10), math.radians(10))

        x4 = np.random.uniform(-1, 1)
        y4 = np.random.uniform(4, 6)
        yaw4 = np.random.uniform(math.radians(-10), math.radians(10))

        x5 = np.random.uniform(4, 6)
        y5 = np.random.uniform(4, 6)
        yaw5 = np.random.uniform(math.radians(-10), math.radians(10))

        n = random.randint(0, 5)
        arr = np.array([[x1, y1, yaw1], [x2, y2, yaw2], [x3, y3, yaw3], [x4, y4, yaw4], [x5, y5, yaw5]])

        # 从数组中进行 n 组随机抽样
        sampled_indices = np.random.choice(arr.shape[0], size=n, replace=False)
        sampled_coordinates = arr[sampled_indices]

        for coord in sampled_coordinates:
            corners = utils.get_rectangle_corners(coord[0], coord[1], coord[2], length, width)
            self.EnvInfo.OthVehRect.append(corners)
        
        # 检查并移除与 host veh 和 slot 存在重叠的obj矩形
        self.EnvInfo.ObjRect = [rect for rect in self.EnvInfo.ObjRect if not utils.check_overlap(rect, self.EnvInfo.VehRectInit)]
        self.EnvInfo.ObjRect = [rect for rect in self.EnvInfo.ObjRect if not utils.check_overlap(rect, self.EnvInfo.SlotRectInit)]
        self.EnvInfo.OthVehRect = [rect for rect in self.EnvInfo.OthVehRect if not utils.check_overlap(rect, self.EnvInfo.VehRectInit)]

        if self.EnvInfo.OthVehRect is not None:
            new_ObjRect = [rect for rect in self.EnvInfo.ObjRect if not any(utils.check_overlap(rect, veh) for veh in self.EnvInfo.OthVehRect)]
            self.EnvInfo.ObjRect = new_ObjRect

        # 生成VehPoint和SlotPoint
        self.EnvInfo.VehPntInit = np.array([host_veh_rear_x, host_veh_rear_y, host_veh_rear_yaw])
        self.EnvInfo.SlotPntInit = utils.get_TP(self.EnvInfo.SlotRectInit, slot_yaw)

        # 根据park mode给出拓展起点和终点，下为prkin
        self.EnvInfo.TgtPntStep = self.EnvInfo.VehPntInit
        self.EnvInfo.StartPntStep = self.EnvInfo.SlotPntInit
        
        return self.EnvInfo
    
    def step(self, action):
        new_state = utils.EnvNextState(action, self.EnvInfo)    # return next EnvInfo
        reward = utils.EnvReward(action, self.EnvInfo)
        done = False
        info = {}
        return new_state, reward, done, info
    
    def show(self):
        """绘制所有obj,slot,host veh和other veh"""
        if not self.EnvInfo.ObjRect and self.EnvInfo.SlotRect is None:
            print("No obj to show. Please call reset() first.")
            return

        fig, ax = plt.subplots()
        for corners in self.EnvInfo.ObjRect:
            polygon = patches.Polygon(corners, closed=True, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)

        for corners in self.EnvInfo.OthVehRect:
            polygon = patches.Polygon(corners, closed=True, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)

        if self.EnvInfo.SlotRect is not None:
            slot_polygon = patches.Polygon(self.EnvInfo.SlotRect, closed=True, edgecolor='b', facecolor='none')
            ax.add_patch(slot_polygon)
        
        host_veh_polygon = patches.Polygon(self.EnvInfo.VehRectInit, closed=True, edgecolor='g', facecolor='none')
        ax.add_patch(host_veh_polygon)
        
        # 绘制SP和TP
        SP_circle = plt.Circle(self.EnvInfo.VehPntInit, 0.05, color='g')  # 以绿色表示SP
        ax.add_artist(SP_circle)
        TP_circle = plt.Circle(self.EnvInfo.SlotPntInit, 0.05, color='b')  # 以蓝色表示TP
        ax.add_artist(TP_circle)

        buffer = 1
        all_corners = np.vstack(self.EnvInfo.ObjRect + [self.EnvInfo.SlotRect])
        all_corners = np.vstack([all_corners, self.EnvInfo.VehRectInit])
        
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