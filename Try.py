def Path_show(path, obstacles, start, goal):
    fig, ax = plt.subplots()

    # 绘制障碍物
    for obstacle in obstacles:
        rect = patches.Polygon(obstacle, closed=True, linewidth=1, edgecolor='r', facecolor='r')
        ax.add_patch(rect)

    # 绘制路径
    path_x = [node.x for node in path]
    path_y = [node.y for node in path]
    ax.plot(path_x, path_y, 'b-')

    # 标记起点和终点
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'ro', markersize=10, label='Goal')

    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.show()