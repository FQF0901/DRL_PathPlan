import numpy as np
import ParaCfg
import math
import reeds_shepp as rs

def get_rectangle_corners(x, y, yaw, length, width):
    """根据中心点、旋转角度、长度和宽度计算矩形的四个角点。"""
    yaw_rad = yaw
    rectangle_corners = np.array([
        [length / 2, width / 2],
        [-length / 2, width / 2],
        [-length / 2, -width / 2],
        [length / 2, -width / 2]
    ])
    rotation_matrix = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad)],
        [np.sin(yaw_rad), np.cos(yaw_rad)]
    ])
    rotated_corners = np.dot(rectangle_corners, rotation_matrix.T) + np.array([x, y])
    return rotated_corners

def get_TP(slot, yaw):
    """通过矩形的四个角点坐标和偏移量Xm计算A点的坐标"""
    # 计算矩形的几何中心
    x_coordinates = [point[0] for point in slot]
    y_coordinates = [point[1] for point in slot]
    center_x = sum(x_coordinates) / len(x_coordinates)
    center_y = sum(y_coordinates) / len(y_coordinates)

    # 确定矩形的长度方向
    length_vector = (slot[1][0] - slot[0][0], slot[1][1] - slot[0][1])
    norm = math.sqrt(pow(slot[1][0] - slot[0][0], 2) + pow(slot[1][1] - slot[0][1], 2))

    # 根据矩形几何中心和长度方向计算A点坐标
    A_x = center_x + length_vector[0] * (ParaCfg.VehPara.length / 2 - ParaCfg.VehPara.rear_to_back) / norm
    A_y = center_y + length_vector[1] * (ParaCfg.VehPara.length / 2 - ParaCfg.VehPara.rear_to_back) / norm

    TP = np.array([A_x, A_y, yaw])

    return TP   

def get_Veh_corners(x, y, yaw, lat, lgt):
    """根据后轴中心、长度和宽度计算车辆的四个角点。"""
    # 计算车辆后轴中心到车头的距离
    rear_to_front = ParaCfg.VehPara.length - ParaCfg.VehPara.rear_to_back

    # 车辆四个角点相对于车辆后轴中心的局部坐标
    local_corners = np.array([
        [-ParaCfg.VehPara.rear_to_back - lgt, -ParaCfg.VehPara.width / 2 - lat],
        [-ParaCfg.VehPara.rear_to_back - lgt, ParaCfg.VehPara.width / 2 + lat],
        [rear_to_front + lgt, ParaCfg.VehPara.width / 2 + lat],
        [rear_to_front + lgt, -ParaCfg.VehPara.width / 2 - lat]
    ])

    # 根据车辆的航向角（yaw）进行旋转
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    rotated_corners = np.dot(local_corners, rotation_matrix.T)

    # 平移车辆四个角点至全局坐标系
    global_corners = rotated_corners + np.array([x, y])

    return global_corners

def get_projection(corners, axis):
    """计算多边形顶点在指定轴上的投影范围（最小值和最大值）。"""
    projections = [np.dot(corner, axis) for corner in corners]
    return min(projections), max(projections)

def overlap(proj1, proj2):
    """检查两个投影是否重叠。"""
    return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])

def is_overlap_Rect(rect1, rect2):
    """使用分离轴定理检查两个矩形是否重叠。"""
    axes = []
    for rect in [rect1, rect2]:
        for i in range(len(rect)):
            edge = rect[i] - rect[i-1]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal / np.linalg.norm(normal))
    
    for axis in axes:
        proj1 = get_projection(rect1, axis)
        proj2 = get_projection(rect2, axis)
        if not overlap(proj1, proj2):
            return False  # 如果找到分离轴，则不重叠
    return True  # 所有轴上的投影都重叠，说明矩形重叠

def is_overlap_node(node, Rects, lat, lgt):
    host_veh = get_Veh_corners(node.x, node.y, node.theta, lat, lgt)

    for rect in Rects:
        if is_overlap_Rect(rect, host_veh):
            return True
    return False

def cal_VechPose(x, y, yaw, steer, gear, dist):

    SteerAng = math.radians(28.78) * steer    # deg
    Dist = dist * gear
    
    if SteerAng != 0:
        Turning_Angle = math.tan(SteerAng) * Dist / ParaCfg.VehPara.wheelbase
        Turn_Radius = Dist / Turning_Angle
        NextYaw = math.fmod(yaw + Turning_Angle + math.pi, 2*math.pi) - math.pi
        NextX = x - Turn_Radius * math.sin(yaw) + Turn_Radius * math.sin(NextYaw)
        NextY = y + Turn_Radius * math.cos(yaw) - Turn_Radius * math.cos(NextYaw)

    else:
        Turning_Angle = 0
        Turn_Radius = dist
        NextX = x + Dist * math.cos(yaw)
        NextY = y + Dist * math.sin(yaw)
        NextYaw = yaw

    return NextX, NextY, NextYaw

def cal_validRS(current_node, goal_node, obstacles):
    RSpath = rs.calc_optimal_path(current_node, goal_node)

    for i in range(0, len(RSpath.x)):
        RSpathx = RSpath.x[i]
        RSpathy = RSpath.y[i]
        RSpathyaw = RSpath.yaw[i]

        RSnode = ParaCfg.Node(RSpathx, RSpathy, RSpathyaw, 0, 0)
        if is_overlap_node(RSnode, obstacles, 0.1, 0.1):
            return False, [], []
        elif i == len(RSpath.x) - 1:    # 全部RS校验完成都没有碰撞
            path = []
            while current_node:
                path.append(current_node)
                current_node = current_node.parent
            return True, path[::-1], RSpath

def EnvNextState(action, EnvInfo):
    steer = action[0]
    gear = action[1]

    x = EnvInfo.State.StartPntStep[0]
    y = EnvInfo.State.StartPntStep[1]
    yaw = EnvInfo.State.StartPntStep[2]

    nextX, nextY, nextYaw = cal_VechPose(x, y, yaw, steer, gear, ParaCfg.HASParam.step_size)

    EnvInfo.State.StartPntStep = np.array([nextX, nextY, nextYaw])

    return EnvInfo

def EnvReward(action, EnvInfo):
    x = EnvInfo.State.StartPntStep[0] # new state 已经产生，因此这里是执行action后的state
    y = EnvInfo.State.StartPntStep[1]
    yaw = EnvInfo.State.StartPntStep[2]
    Curt_node = ParaCfg.Node(x, y, yaw, 0, 0)
    x = EnvInfo.State.TgtPntStep[0] # new state 已经产生，因此这里是执行action后的state
    y = EnvInfo.State.TgtPntStep[1]
    yaw = EnvInfo.State.TgtPntStep[2]
    Tgt_node = ParaCfg.Node(x, y, yaw, 0, 0)
    obstacles = EnvInfo.State.ObjRect + EnvInfo.State.OthVehRect

    # Cost
    ExpansionCost = -1
    SteerCost = abs(action[0] - EnvInfo.action_z[0]) * -1 + 0.2
    GearCost = 0.2 if action[1] == EnvInfo.action_z[0] else -5
    
    CloseObjCost = 0
    if is_overlap_node(Curt_node, obstacles, 0.25, 0.15):
        CloseObjCost = -0.5

    CollisionCost = 0
    if is_overlap_node(Curt_node, obstacles, 0.1, 0.1):
        CollisionCost = -5  # 碰撞不应由DNN保证，因此不应因碰撞大幅惩罚DNN参数
        EnvInfo.ActionVehOvlp = True

    PathNotFndCost = 0
    PlanFnd, _, _ = cal_validRS(Curt_node, Tgt_node, obstacles)
    if EnvInfo.StepCnt >= ParaCfg.HASParam.maxEpsd and (not PlanFnd):
        PathNotFndCost = -200  

    TolCost = ExpansionCost + SteerCost + GearCost + CloseObjCost + CollisionCost + PathNotFndCost

    # Reward
    SpcUseReward = 0
    if is_overlap_node(Curt_node, obstacles, 0.0, ParaCfg.HASParam.step_size - 0.01) and \
        (not is_overlap_node(Curt_node, obstacles, 0.0, 0.1)):
        SpcUseReward = 3

    PathFoundReward = 0
    if PlanFnd:
        PathFoundReward = 10000

    TolReward = SpcUseReward + PathFoundReward

    EnvInfo.action_z = action
    EnvInfo.Reward_z = TolCost + TolReward

    return EnvInfo

def EnvDRL_StateMapping(EnvInfoState):
    if not len(EnvInfoState.ObjRect) == 0:
        ObjRect_array = np.concatenate(EnvInfoState.ObjRect).ravel()
        pad_width = ((0, 10 * 4 * 2 - ObjRect_array.size))
        ObjRect_Array = np.pad(ObjRect_array, pad_width, mode='constant', constant_values=0)
    else:
        ObjRect_Array = np.zeros(10 * 4 * 2)

    if not len(EnvInfoState.OthVehRect) == 0:
        OthVehRect_array = np.concatenate(EnvInfoState.OthVehRect).ravel()
        pad_width = ((0, 5 * 4 * 2 - OthVehRect_array.size))
        OthVehRect_Array = np.pad(OthVehRect_array, pad_width, mode='constant', constant_values=0)
    else:
        OthVehRect_Array = np.zeros(5 * 4 * 2)
    
    StartPntStep_Array = np.pad(EnvInfoState.StartPntStep, ((0, 1)), mode='constant', constant_values=0)
    TgtPntStep_Array = np.pad(EnvInfoState.TgtPntStep, ((0, 1)), mode='constant', constant_values=0)
    DRLstate = np.concatenate((ObjRect_Array, OthVehRect_Array, StartPntStep_Array, TgtPntStep_Array))
    
    return DRLstate

def EnvDRL_ActionMapping(DRLaction):
    if DRLaction == 1:
        EnvAction = [-1, 1]
    elif DRLaction == 2:
        EnvAction = [0, 1]
    elif DRLaction == 3:
        EnvAction = [1, 1]
    elif DRLaction == 4:
        EnvAction = [-1, -1]
    elif DRLaction == 5:
        EnvAction = [0, -1]
    elif DRLaction == 6:
        EnvAction = [1, -1]
    else:
        EnvAction = [0, 1]

    return EnvAction
    