import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots()

# Set the size of the figure in inches (512x256 pixels)
fig.set_size_inches(512 / 100, 384 / 100)

# Set the limits of the plot
ax.set_xlim(-16, 16)
ax.set_ylim(-9, 9)

# Draw grid lines
ax.grid(True, which='both', linestyle='--', linewidth=0.2)

# Draw a square at (0, 0) with size 0.1x0.1
square = plt.Rectangle((4.35, 4.35), 0.1, 0.1, linewidth=0.5, edgecolor='r', facecolor='none')
ax.add_patch(square)
square = plt.Rectangle((-4.35, 4.35), 0.2, 0.2, linewidth=0.5, edgecolor='r', facecolor='none')
ax.add_patch(square)

# 设置坐标轴的位置
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Set aspect ratio to be equal, so the plot is not distorted
ax.set_aspect('equal')

# Save the figure as a PNG image without extra whitespace around it
plt.savefig('coordinate_grid.png', bbox_inches='tight', pad_inches=0)

# Show the plot (optional)
# plt.show()
