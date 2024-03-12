import numpy as np

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
