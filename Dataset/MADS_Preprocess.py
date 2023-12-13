import numpy as np
from PIL import Image
import cv2
import os
import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle


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
        output_temp_GT['joints_3d'] = ground_truth[frame_count-1]
        ouput_GT.append(output_temp_GT)
        output_temp_GT = {}

    cap.release()

    print(f"Video '{video_name}' has been split into {frame_count} frames.")

    return ouput_GT


if __name__ == '__main__':
    root_path = '/media/imaginarium/2T/dataset/'
    output_folder = root_path + 'MADS/images/'
    acts = ['Taichi', 'Kata', 'Jazz', 'HipHop', 'Sports']
    GT = []
    for act in acts:
        input_folder = os.path.join(root_path,'MADS/multi_view_data/',act)
        video_files = glob.glob(os.path.join(input_folder, "*.avi"))
        for video_file in video_files:
            # print(video_file)
            video_name = os.path.splitext(os.path.basename(video_file))[0]

            gt_path = video_file[:-6] + 'GT.mat'
            ground_truth = loadmat(gt_path)['GTpose2'][0]
            if ground_truth[0].shape[0] < 17:
                continue
            else:
                ouput_GT = split_video_into_frames(video_file, os.path.join(output_folder, video_name),ground_truth)
                GT = GT + ouput_GT

    with open(os.path.join(root_path,'MADS/annot',"MADS.pkl"), "wb") as f:
        pickle.dump(GT, f)