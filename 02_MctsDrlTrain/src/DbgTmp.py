import os
import pandas as pd

def store_in_pkl(new_df, store_folder_path, max_file_size_M):
    # Get a list of all pkl files in the store folder
    pkl_files = [f for f in os.listdir(store_folder_path) if f.endswith(".pkl")]

    # Sort the pkl files by suffix (part number)
    pkl_files.sort(key=lambda x: int(x.split("_part")[1].split(".pkl")[0]) if "_part" in x else 0)

    # Get the path of the pkl file with the largest suffix
    if pkl_files:
        largest_suffix_pkl = os.path.join(store_folder_path, pkl_files[-1])
        existing_df = pd.read_pickle(largest_suffix_pkl)
        sum_sf = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        sum_sf = new_df

    # Check if the combined DataFrame size exceeds max_file_size_M
    if sum_sf.memory_usage(deep=True).sum() > max_file_size_M * 1024 * 1024:
        # Create a new part+1.pkl file
        next_part = len(pkl_files) + 1
        pkl_file_path = os.path.join(store_folder_path, f"mf4_time_slice_data_part{next_part}.pkl")
        new_df.to_pickle(pkl_file_path)
    else:
        # Write to the existing pkl file
        pkl_file_path = os.path.join(store_folder_path, 'mf4_time_slice_data.pkl')
        sum_sf.to_pickle(pkl_file_path)

    return os.path.basename(pkl_file_path)

# Example usage
new_df = pd.DataFrame({'A': [i for i in range(1, 20000)]})
store_folder_path = r'C:\01_Project\10_Git\PECU_DRL\01_DataSetGen\DataSet'
max_file_size_M = 0.5
result = store_in_pkl(new_df, store_folder_path, max_file_size_M)
print(f"Saved to {result}")
