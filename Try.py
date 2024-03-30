import numpy as np

# 假设 Cc_values_array 是一个列表
Cc_values_array = [0, -1, 0, 0, -1, 0]

# 将列表转换为 NumPy 数组
Cc_values_array = np.array(Cc_values_array)

# 找到所有非 -1 的元素的索引
non_negative_indices = np.where(Cc_values_array != -1)[0]

# 从非 -1 的元素中随机选择一个
selected_index = np.random.choice(non_negative_indices)

# 获取选中元素的值和索引
selected_element = Cc_values_array[selected_index]

print("选中的元素:", selected_element)
print("选中元素的索引:", selected_index)
