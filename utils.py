import numpy as np
import ParaCfg
import math

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

def get_Veh_corners(x, y, yaw):
    """根据后轴中心、长度和宽度计算车辆的四个角点。"""
    # 计算车辆后轴中心到车头的距离
    rear_to_front = ParaCfg.VehPara.length - ParaCfg.VehPara.rear_to_back

    # 车辆四个角点相对于车辆后轴中心的局部坐标
    local_corners = np.array([
        [-ParaCfg.VehPara.rear_to_back, -ParaCfg.VehPara.width / 2],
        [-ParaCfg.VehPara.rear_to_back, ParaCfg.VehPara.width / 2],
        [rear_to_front, ParaCfg.VehPara.width / 2],
        [rear_to_front, -ParaCfg.VehPara.width / 2]
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

def check_overlap(rect1, rect2):
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
