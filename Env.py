import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Env:
    def __init__(self):
        self.rectangles = []  # 存储随机数量矩形的角点
        self.slot = None  # 存储特定矩形（slot）的角点
    
    def reset(self):
        self.rectangles.clear()  # 清空之前的矩形数据
        n_rectangles = np.random.randint(1, 65)  # 随机确定矩形的数量（1~64）
        
        for _ in range(n_rectangles):
            x, y = np.random.uniform(-8, 8, 2)  # 中心点坐标在-8到8之间
            yaw = np.random.uniform(0, 360)  # 旋转角度在0到360度之间
            length, width = np.random.uniform(0.1, 0.5, 2)  # 长度和宽度在0.1到0.5之间
            
            corners = self._get_rectangle_corners(x, y, yaw, length, width)
            self.rectangles.append(corners)
        
        # 生成特定范围内的矩形（slot）
        self.slot = self._generate_slot()
        
        # 检查并移除与slot存在重叠的矩形
        self.rectangles = [rect for rect in self.rectangles if not self._check_overlap(rect, self.slot)]
        
        return self.rectangles, self.slot
    
    def _get_rectangle_corners(self, x, y, yaw, length, width):
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
    
    def _generate_slot(self):
        """生成一个特殊的矩形作为slot。"""
        x, y = np.random.uniform(-1, 1, 2)
        yaw = np.random.uniform(0, 360)
        length = np.random.uniform(5, 5.6)
        width = np.random.uniform(1.8, 2.4)
        return self._get_rectangle_corners(x, y, yaw, length, width)
    
    def _get_projection(self, corners, axis):
        """计算多边形顶点在指定轴上的投影范围（最小值和最大值）。"""
        projections = [np.dot(corner, axis) for corner in corners]
        return min(projections), max(projections)

    def _overlap(self, proj1, proj2):
        """检查两个投影是否重叠。"""
        return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])

    def _check_overlap(self, rect1, rect2):
        """使用分离轴定理检查两个矩形是否重叠。"""
        axes = []
        for rect in [rect1, rect2]:
            for i in range(len(rect)):
                edge = rect[i] - rect[i-1]
                normal = np.array([-edge[1], edge[0]])
                axes.append(normal / np.linalg.norm(normal))
        
        for axis in axes:
            proj1 = self._get_projection(rect1, axis)
            proj2 = self._get_projection(rect2, axis)
            if not self._overlap(proj1, proj2):
                return False  # 如果找到分离轴，则不重叠
        return True  # 所有轴上的投影都重叠，说明矩形重叠
    
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
