
import cv2
import os
import glob
from scipy.io import loadmat
import pickle
import multiprocessing
from functools import partial

def split_video_into_frames(video_path, output_folder):
    output_temp_GT = {}
    ouput_GT = []
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_folder, video_name)

    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    GTnames = video_name.split('_')
    GTnames[2] = 'cAll'
    GTname = '_'.join(GTnames)
    GT_path = os.path.join(root_path, 'AIST_Dance/keypoints3d', GTname + '.pkl')
    with open(GT_path, 'rb') as file:
        AIST = pickle.load(file)
    ground_truth = AIST['keypoints3d_optim']
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

        # Resize the frame to the desired dimensions
        resized_frame = cv2.resize(frame, (384, 384))

        cv2.imwrite(output_path, resized_frame)
        output_temp_GT['image'] = video_name + '/' + output_filename
        output_temp_GT['joints_3d'] = ground_truth[frame_count-1]
        ouput_GT.append(output_temp_GT)
        output_temp_GT = {}

    cap.release()

    print(f"Video '{video_name}' has been split into {frame_count} frames.")

    return ouput_GT

if __name__ == '__main__':
    root_path = '/home/imaginarium/Documents/dataset/'
    output_folder = '/media/imaginarium/2T/dataset/AIST_Dance/images/'
    GT = []

    video_folder = os.path.join(root_path, 'AIST_Dance/aist_dance_videos')
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))

    # Use functools.partial to create a new function with fixed additional parameters
    partial_process_video = partial(split_video_into_frames, output_folder=output_folder)

    # Use multiprocessing to process videos in parallel
    num_processes = 28
    pool = multiprocessing.Pool(processes=num_processes)

    # Apply the function asynchronously and get the result
    results = [pool.apply_async(partial_process_video, (video,)) for video in video_files]

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Get the output from the result objects
    for video, result in zip(video_files, results):
        output = result.get()
        GT = GT + output
        # print(f"Output for {video}: {output}")

    print("Video frames extraction and additional processing completed.")
    # for video_file in video_files:
    #     # print(video_file)
    #     video_name = os.path.splitext(os.path.basename(video_file))[0]
    #     GTnames = video_name.split('_')
    #     GTnames[2] = 'cAll'
    #     GTname = '_'.join(GTnames)
    #     GT_path = os.path.join(root_path, 'AIST_Dance/keypoints3d', GTname + '.pkl')
    #     with open(GT_path, 'rb') as file:
    #         AIST = pickle.load(file)
    #     ouput_GT = split_video_into_frames(video_file, os.path.join(output_folder, video_name), video_name,
    #                                        AIST['keypoints3d_optim'])
    #     GT = GT + ouput_GT



    with open(os.path.join(root_path,'/media/imaginarium/2T/dataset/AIST_Dance/annot',"AIST_Dance.pkl"), "wb") as f:
        pickle.dump(GT, f)