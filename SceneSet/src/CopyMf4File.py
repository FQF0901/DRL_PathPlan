import os
import shutil

def is_min_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def process_folders(src_folder, dest_folder, min_num, max_num):
    # 步骤1: 得到指定文件夹A下的所有子文件夹
    subfolders = [f.path for f in os.scandir(src_folder) if f.is_dir()]
    
    # 步骤2: 找到符合条件的子文件夹，文件夹名字转换为数字后，大于min_num
    list_folder = []
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        if is_min_number(folder_name):
            if float(folder_name) >= min_num and float(folder_name) <= max_num:
                list_folder.append(folder_name)
    
    # 步骤3: 在指定位置B处新建List_Folder的元素相同名字的文件夹
    for folder_name in list_folder:
        new_folder_path = os.path.join(dest_folder, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        
        # 步骤4: 将List_Folder的元素文件夹内后缀为.mf4的文件拷贝到B处新文件夹内
        src_subfolder = os.path.join(src_folder, folder_name)
        for root, dirs, files in os.walk(src_subfolder):
            for file in files:
                if not 'TDA4DDS'in file and not 'BEV'in file and file.endswith('.mf4'):
                    src_file = os.path.join(root, file)
                    shutil.copy(src_file, new_folder_path)
                    print(f'Copied: {src_file} to {new_folder_path}')

'''
Copy mf4 files
'''
# 替换下面的路径和数字为实际的值
source_folder = r'Z:\PR62383\canape'
destination_folder = r'D:\DataSet\Mf4RawFile\A2'
min_num = 20240819  # 替换为实际的数字
max_num = 20240823

process_folders(source_folder, destination_folder, min_num, max_num)
print('===== Copy done ! =====')