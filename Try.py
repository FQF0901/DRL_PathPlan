import numpy as np

def update_q_values_and_find_max(q_values_array, cc_values_array):
    idx1 = np.where(cc_values_array == -1)[0]
    q_values_array[idx1] = np.min(q_values_array) - 1
    idx_max = np.argmax(q_values_array)
    return idx_max

# Example
cc_values_array = np.array([0, -1, 0, 0, -1, -1])
q_values_array = np.array([1, 2, 3, 4, 5, 6])

idx_max = update_q_values_and_find_max(q_values_array, cc_values_array)
print("Index of max value in q_values_array (excluding -1 elements):", idx_max)
