import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import ParaCfg
import random
import math

class Env:
    def __init__(self):
        self.EnvInfo = ParaCfg.EnvInfo()
    
    def reset(self):
        self.EnvInfo.State.ObjRect.clear()  # 清空之前的矩形数据
        self.EnvInfo.State.OthVehRect.clear()  # 清空之前的矩形数据
        self.EnvInfo.State.StartPntStep = []
        self.EnvInfo.State.StartRectStep = None
        self.EnvInfo.State.TgtPntStep = []
        self.EnvInfo.State.TgtRectStep = None
        self.EnvInfo.VehPntInit = []
        self.EnvInfo.VehRectInit = None
        self.EnvInfo.SlotPntInit = []
        self.EnvInfo.SlotRectInit = None
        self.EnvInfo.action_z = []
        self.EnvInfo.Reward_z = []
        self.EnvInfo.ActionVehOvlp = False
        self.EnvInfo.PathFnd = False
        self.EnvInfo.StepCnt = 0

        n_ObjRect = np.random.randint(0, 3)  # 随机确定矩形的数量（0~10）
        
        # 生成特定范围内的obj矩形
        for _ in range(n_ObjRect):
            x = np.random.uniform(-8, 8)
            y = np.random.uniform(-2, 6)
            yaw = np.random.uniform(0, math.pi)
            length, width = np.random.uniform(0.1, 0.5, 2)
            
            corners = utils.get_rectangle_corners(x, y, yaw, length, width)
            self.EnvInfo.State.ObjRect.append(corners)

        # 生成特定范围内的host veh(基于后轴)
        veh_rear_x = np.random.uniform(-5, 5)
        veh_rear_y = np.random.uniform(-1, 3)
        veh_rear_yaw = np.random.uniform(math.radians(-45), math.radians(45))
        self.EnvInfo.VehRectInit = utils.get_Veh_corners(veh_rear_x, veh_rear_y, veh_rear_yaw ,0.1 ,0.1)
        
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

        x2 = np.random.uniform(-7, -5)
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

        n = random.randint(5, 5)    # 随机抽样0 ~ 5个
        arr = np.array([[x1, y1, yaw1], [x2, y2, yaw2], [x3, y3, yaw3], [x4, y4, yaw4], [x5, y5, yaw5]])

        # 从数组中进行 n 组随机抽样
        sampled_indices = np.random.choice(arr.shape[0], size=n, replace=False)
        sampled_coordinates = arr[sampled_indices]

        for coord in sampled_coordinates:
            corners = utils.get_rectangle_corners(coord[0], coord[1], coord[2], length, width)
            self.EnvInfo.State.OthVehRect.append(corners)
        
        # 检查并移除与 host veh 和 slot 存在重叠的obj矩形
        self.EnvInfo.State.ObjRect = [rect for rect in self.EnvInfo.State.ObjRect if not utils.is_overlap_Rect(rect, self.EnvInfo.VehRectInit)]
        self.EnvInfo.State.ObjRect = [rect for rect in self.EnvInfo.State.ObjRect if not utils.is_overlap_Rect(rect, self.EnvInfo.SlotRectInit)]
        self.EnvInfo.State.OthVehRect = [rect for rect in self.EnvInfo.State.OthVehRect if not utils.is_overlap_Rect(rect, self.EnvInfo.VehRectInit)]

        if self.EnvInfo.State.OthVehRect is not None:
            new_ObjRect = [rect for rect in self.EnvInfo.State.ObjRect if not any(utils.is_overlap_Rect(rect, veh) for veh in self.EnvInfo.State.OthVehRect)]
            self.EnvInfo.State.ObjRect = new_ObjRect

        # 生成VehPoint和SlotPoint
        self.EnvInfo.VehPntInit = np.array([veh_rear_x, veh_rear_y, veh_rear_yaw])
        self.EnvInfo.SlotPntInit = utils.get_TP(self.EnvInfo.SlotRectInit, slot_yaw)

        # 根据park mode给出拓展起点和终点，下为prkin
        self.EnvInfo.State.TgtPntStep = self.EnvInfo.VehPntInit
        self.EnvInfo.State.StartPntStep = self.EnvInfo.SlotPntInit

        # 初始化action和reward
        self.EnvInfo.action_z = np.array([0, 0])    # 左正右负，前正后负
        self.EnvInfo.Reward_z = 0
        self.EnvInfo.VehOvlp = False
        self.EnvInfo.PathFnd = False
        self.EnvInfo.StepCnt = 0
        
        state = utils.EnvDRL_StateMapping(self.EnvInfo.State)

        return state
    
    def step(self, action):
        action = utils.EnvDRL_ActionMapping(action)

        self.EnvInfo.StepCnt = self.EnvInfo.StepCnt + 1

        self.EnvInfo = utils.EnvNextState(action, self.EnvInfo)    # return next ParaCfg.EnvInfo
        self.EnvInfo = utils.EnvReward(action, self.EnvInfo)
        done = self.EnvInfo.StepCnt >= ParaCfg.HASParam.maxEpsd \
                or self.EnvInfo.ActionVehOvlp \
                or self.EnvInfo.PathFnd \
                or (abs(self.EnvInfo.State.StartPntStep[0]) > 15 or abs(self.EnvInfo.State.StartPntStep[1]) > 10)
        # info = (self.EnvInfo.StepCnt, \
        #         'ActOvlp' if self.EnvInfo.ActionVehOvlp else 'ActNotOvlp', \
        #         'PathFnd' if self.EnvInfo.PathFnd else 'PathNotFnd' \
        #         'VehOutMap' if (abs(self.EnvInfo.State.StartPntStep[0]) > 20 or abs(self.EnvInfo.State.StartPntStep[1]) > 20) else 'VehinMap')
        info = "StepCnt{}, {}, {}, {}".format(self.EnvInfo.StepCnt, \
                        'ActOvlp' if self.EnvInfo.ActionVehOvlp else 'ActNotOvlp', \
                            'PathFnd' if self.EnvInfo.PathFnd else 'PathNotFnd', \
                                'VehOutMap' if (abs(self.EnvInfo.State.StartPntStep[0]) > 15 or abs(self.EnvInfo.State.StartPntStep[1]) > 10) else 'VehinMap')

        next_state = utils.EnvDRL_StateMapping(self.EnvInfo.State)
        reward = self.EnvInfo.Reward_z
        
        return next_state, reward, done, info
    
    def show(self):
        """绘制所有obj,slot,host veh和other veh"""
        if not self.EnvInfo.State.ObjRect and self.EnvInfo.SlotRectInit is None:
            print("No obj to show. Please call reset() first.")
            return

        fig, ax = plt.subplots()
        for corners in self.EnvInfo.State.ObjRect:
            polygon = patches.Polygon(corners, closed=True, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)

        for corners in self.EnvInfo.State.OthVehRect:
            polygon = patches.Polygon(corners, closed=True, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)

        if self.EnvInfo.SlotRectInit is not None:
            slot_polygon = patches.Polygon(self.EnvInfo.SlotRectInit, closed=True, edgecolor='b', facecolor='none')
            ax.add_patch(slot_polygon)
        
        host_veh_polygon = patches.Polygon(self.EnvInfo.VehRectInit, closed=True, edgecolor='g', facecolor='none')
        ax.add_patch(host_veh_polygon)
        
        # 绘制SP和TP
        SP_circle = plt.Circle(self.EnvInfo.VehPntInit, 0.05, color='g')  # 以绿色表示SP
        ax.add_artist(SP_circle)
        TP_circle = plt.Circle(self.EnvInfo.SlotPntInit, 0.05, color='b')  # 以蓝色表示TP
        ax.add_artist(TP_circle)

        SP_Step_circle = plt.Circle(self.EnvInfo.State.StartPntStep, 0.03, color='r')  # 以黄色表示规划过程
        ax.add_artist(SP_Step_circle)

        buffer = 1
        all_corners = np.vstack(self.EnvInfo.State.ObjRect + [self.EnvInfo.SlotRectInit])
        all_corners = np.vstack([all_corners, self.EnvInfo.VehRectInit])
        
        # xlim = (min(all_corners[:, 0]) - buffer, max(all_corners[:, 0]) + buffer)
        # ylim = (min(all_corners[:, 1]) - buffer, max(all_corners[:, 1]) + buffer)
        xlim = (ParaCfg.HASParam.xmin, ParaCfg.HASParam.xmax)
        ylim = (ParaCfg.HASParam.ymin, ParaCfg.HASParam.ymax)
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