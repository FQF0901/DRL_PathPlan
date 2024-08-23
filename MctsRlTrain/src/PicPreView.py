# import cv2
# import numpy as np

# # 1. 生成3*224*384的白色背景图片
# height, width = 224, 384
# channels = 3
# image = np.ones((height, width, channels), dtype=np.uint8) * 255

# # 2. 计算物理范围的像素位置
# physical_width, physical_height = 20, 20/384*224
# pixel_width = width / physical_width
# pixel_height = height / physical_height

# # 3. 计算1.85*4.43的红色空心方框的像素尺寸
# box_width = int(4.43 * pixel_width)
# box_height = int(1.85 * pixel_height)

# top_left_x = int((width - box_width) / 2)
# top_left_y = int((height - box_height) / 2)
# bottom_right_x = top_left_x + box_width
# bottom_right_y = top_left_y + box_height

# # 4. 绘制红色空心方框
# color = (0, 0, 255)  # 红色
# thickness = 1  # 方框的厚度
# cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)

# # 5. 使用cv2显示图片
# cv2.imshow('Image', image)

# # 等待用户按键后关闭窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 6. 保存图片到本地
# cv2.imwrite('output_image.png', image)

# ===============================================

from PIL import Image

# 示例用法
image_path = r'C:\01_Project\10_Git\PECU_DRL\mf4_time_slice_data_part3_rowidx15_PcptGeo.png'  # 替换为你的图像文件路径

with Image.open(image_path) as img:
    # 获取图像尺寸
    width, height = img.size

    print(width, height)
