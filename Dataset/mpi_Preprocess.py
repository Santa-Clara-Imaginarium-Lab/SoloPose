import numpy as np
from PIL import Image
import cv2
import os
import glob
from scipy.io import loadmat
import pickle
import multiprocessing
from functools import partial

def split_video_into_frames(video_path, output_folder,ground_truth):
    output_temp_GT = {}
    ouput_GT = []

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = output_folder + video_name
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ViewID = os.path.splitext(os.path.basename(video_path))[0].split('_')[-1]
    ViewID = int(ViewID)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    ground_truth = ground_truth[ViewID][0].reshape(-1,28,3)

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
        output_temp_GT['image'] = output_folder.split('/')[-1] + '/' + output_filename
        output_temp_GT['joints_3d'] = ground_truth[frame_count-1]
        ouput_GT.append(output_temp_GT)
        output_temp_GT = {}

    cap.release()

    print(f"Video '{video_name}' has been split into {frame_count} frames.")

    return ouput_GT

if __name__ == '__main__':
    root_path = '/home/imaginarium/Documents/dataset/'
    output_folder = root_path + 'mpi_inf_3dhp_dataset/images/'
    # Define the number of processes you want to use (adjust as needed)
    num_processes = multiprocessing.cpu_count()

    Seqences = ['S1', 'S2', 'S3', 'S4', 'S5','S6','S7','S8']
    GT = []
    for Seqence in Seqences:
        input_folder = os.path.join(root_path, 'mpi_inf_3dhp_dataset/',Seqence)
        for subSeq in ['Seq1','Seq2']:
            video_folder = os.path.join(input_folder, subSeq,'imageSequence')
            video_files = glob.glob(os.path.join(video_folder, "*.avi"))
            GT_path = os.path.join(input_folder, subSeq, 'annot.mat')
            ground_truth = loadmat(GT_path)['univ_annot3']

            video_name = Seqence + '_' + subSeq + '_'
            # Use functools.partial to create a new function with fixed additional parameters
            partial_process_video = partial(split_video_into_frames, output_folder=os.path.join(output_folder, video_name),ground_truth=ground_truth)

            # Use multiprocessing to process videos in parallel
            num_processes = multiprocessing.cpu_count()
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
            #     video_name = Seqence + '_' + subSeq+'_' + os.path.splitext(os.path.basename(video_file))[0]
            #     ouput_GT = split_video_into_frames(video_file, os.path.join(output_folder, video_name), video_name,ground_truth)
            #     GT = GT + ouput_GT

    with open(os.path.join(root_path,'mpi_inf_3dhp_dataset/annot',"mpi_inf_3dhp_dataset.pkl"), "wb") as f:
        pickle.dump(GT, f)