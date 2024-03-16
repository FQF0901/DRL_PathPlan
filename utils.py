import numpy as np
import ParaCfg

def get_rectangle_corners(x, y, yaw, length, width):
    """根据中心点、旋转角度、长度和宽度计算矩形的四个角点。"""
    yaw_rad = np.radians(yaw)
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
        [np.cos(np.radians(yaw)), -np.sin(np.radians(yaw))],
        [np.sin(np.radians(yaw)), np.cos(np.radians(yaw))]
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
