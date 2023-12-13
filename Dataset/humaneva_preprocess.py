from scipy.io import loadmat
import cv2
import os
import glob
import numpy as np

def split_video_into_frames(video_path, output_folder,ground_truth):
    output_temp_GT = {}
    ouput_GT = []
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the video name without the extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]


    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    # Read each frame and save it as an image
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        output_filename = f"{video_name}_frame_{frame_count:04d}.png"
        output_path = os.path.join(output_folder, output_filename)

        cv2.imwrite(output_path, frame)
        output_temp_GT['image'] = video_name + '/' + output_filename
        # output_temp_GT['joints_3d'] = ground_truth[frame_count-1]
        ouput_GT.append(output_temp_GT)
        output_temp_GT = {}

    cap.release()

    print(f"Video '{video_name}' has been split into {frame_count} frames.")

    return ouput_GT


if __name__ == '__main__':
    root_path = '/media/imaginarium/2T/dataset/'
    output_folder = root_path + 'HumanEva-I/images/'
    Seqences = ['S1', 'S2', 'S3']
    GT = []
    GT_path = os.path.join(root_path, 'HumanEva-I/converted_15j/data_3d_humaneva15.npz')
    ground_turth = np.load(GT_path,allow_pickle=True)['positions_3d'].item()



    for Seqence in Seqences:
        input_folder = os.path.join(root_path, 'HumanEva-I/',Seqence,'Image_Data')
        video_files = glob.glob(os.path.join(input_folder, "*.avi"))

        for video_file in video_files:
            gt_path = os.path.join(root_path,'HumanEva-I/converted_15j/Validate',Seqence)
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            gt_path = os.path.join(gt_path, video_name[:-5])
            ground_truth = loadmat(gt_path)['poses_3d']

            tempGT = split_video_into_frames(video_file, os.path.join(output_folder, video_name),ground_truth)
            GT = GT + tempGT