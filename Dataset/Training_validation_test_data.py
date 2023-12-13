import os
import shutil
import random
from tqdm import tqdm  # Import the tqdm library for progress bars

if __name__ == '__main__':
    # Define the paths
    original_data_folder = '/media/imaginarium/a0c299ce-8eea-4f25-92a4-b572d215821b/merge/'
    train_data_folder = '/media/imaginarium/12T_2/train'
    valid_data_folder = '/media/imaginarium/12T_2/valid'
    test_data_folder = '/media/imaginarium/12T_2/test'

    # Create the train, valid, and test folders if they don't exist
    os.makedirs(train_data_folder, exist_ok=True)
    os.makedirs(valid_data_folder, exist_ok=True)
    os.makedirs(test_data_folder, exist_ok=True)

    # Set the percentage split
    train_split = 0.7
    valid_split = 0.2
    test_split = 0.1

    # List all files in the 'data' folder
    data_files = os.listdir(os.path.join(original_data_folder, 'data'))

    # Shuffle the files randomly
    random.shuffle(data_files)

    # Calculate split points
    num_files = len(data_files)
    train_end = int(num_files * train_split)
    valid_end = train_end + int(num_files * valid_split)
    print('num_files',num_files,'train_end',train_end,'valid_end',valid_end)

    # Split the files into train, valid, and test sets
    train_files = data_files[:train_end]
    valid_files = data_files[train_end:valid_end]
    test_files = data_files[valid_end:]

    # Split the files into train, valid, and test sets with tqdm progress bars
    for folder, file_list in [(train_data_folder, data_files[:train_end]),
                              (valid_data_folder, data_files[train_end:valid_end]),
                              (test_data_folder, data_files[valid_end:])]:
        for file in tqdm(file_list, desc=f'Copying files to {folder}'):
            # Copy data files
            data_file_path = os.path.join(original_data_folder, 'data', file)
            shutil.copy(data_file_path, os.path.join(folder, 'data', file))

            # Copy label files with the same name but different suffix
            label_file_name, _ = os.path.splitext(file)
            label_file_path = os.path.join(original_data_folder, 'GT', label_file_name + '.npy')
            shutil.copy(label_file_path, os.path.join(folder, 'GT', label_file_name + '.npy'))