import os

def count_mf4_files(folder):
    count = 0
    for root, dirs, files in os.walk(folder):
        count += sum(1 for file in files if file.endswith('.mf4'))
    return count

'''
Statistics for mf4 files
'''
# 替换下面的路径为实际的路径
BX_mf4_count = count_mf4_files(r'D:\DataSet\Mf4RawFile\BXdemo')
A2_mf4_count = count_mf4_files(r'D:\DataSet\Mf4RawFile\A2')

print(f'Number of Total.mf4 files: {BX_mf4_count + A2_mf4_count}')
print(f'Number of BX.mf4 files: {BX_mf4_count}')
print(f'Number of A2.mf4 files: {A2_mf4_count}')
