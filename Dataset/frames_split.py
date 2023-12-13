import os
import cv2
import numpy as np
from scipy.io import loadmat


def process_video(video_path, gt_path, frames_output_folder, gt_output_folder):
    video = cv2.VideoCapture(video_path)
    # gt_path = 'D:\\X_Pose\\MADS_prepocess\\original\\Kata_P3_GT.mat'

    ground_truth = loadmat(gt_path)
    # name=''
    # split_names = name.split('_') [:-2]

    i = 0
    count = 0
    ground_truth_bundle = []
    frame_bundle = []
    frame_id = 0
    set_id = 0

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        if np.isnan(ground_truth['GTpose2'][0][frame_id]).any():
            count = 0
            ground_truth_bundle = []
            frame_bundle = []
            print('find NaN:',clip_name,'set_id:', set_id,'frame_id:',frame_id)
            continue
        else:
            ground_truth_bundle.append(ground_truth['GTpose2'][0][frame_id])
            frame_bundle.append(frame)
            count += 1

        frame_id += 1

        if count == 16:
            ground_truth_bundle = np.array(ground_truth_bundle)
            if ground_truth_bundle.shape[1] < 19:
                print(clip_name, set_id, ground_truth_bundle.shape)
                count = 0
                ground_truth_bundle = []
                frame_bundle = []


                continue
            np.save(f'{gt_output_folder}/{clip_name}_frameset_{set_id}.npy', ground_truth_bundle)

            saveIMG_path = f'{frames_output_folder}/{clip_name}_frameset_{set_id}/'
            if not os.path.exists(saveIMG_path):
                os.makedirs(saveIMG_path)

            frame_id = frame_id - 16
            for j, frame in enumerate(frame_bundle):
                cv2.imwrite(saveIMG_path + f'frame{frame_id}.jpg', frame)
                frame_id += 1

            ground_truth_bundle = []
            frame_bundle = []
            i += 1
            set_id += 1
            count = 0

    video.release()

if __name__ == '__main__':
    root_folder = 'D:\\X_Pose\\MADS_prepocess\\original\\'
    frames_output_folder = 'D:\\X_Pose\\MADS_prepocess\\Clips\\frames\\'
    gt_output_folder = 'D:\\X_Pose\\MADS_prepocess\\Clips\\ground_truth\\'

    # Create the frames and ground_truth directories if they don't exist
    if not os.path.exists(frames_output_folder):
        print('create path',frames_output_folder)
        os.makedirs(frames_output_folder)
    if not os.path.exists(gt_output_folder):
        print('create path', gt_output_folder)
        os.makedirs(gt_output_folder)

    # Iterate over all subdirectories in the root folder
    for folder_name, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.avi'):

                print(filename)
                # if filename[0:3] == 'Hip' or filename[0:3] == 'Jaz':
                #     continue
                video_path = os.path.join(folder_name, filename)
                # remove the suffix C0.avi
                video_name = video_path[:-6]

                gt_path = video_name + 'GT.mat'
                clip_name = os.path.splitext(filename)[0]
                process_video(video_path, gt_path, frames_output_folder, gt_output_folder)



