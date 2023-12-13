import numpy as np
from PIL import Image
import cv2
import os
import glob
from scipy.io import loadmat
import pickle
import multiprocessing
from functools import partial
from transformers import CLIPProcessor, CLIPVisionModel
from pytorchvideo.data.encoded_video import EncodedVideo
import torch

def rigid_transform_3D(A, B, scale):
    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    if scale:
        H = np.transpose(BB) * AA / N
    else:
        H = np.transpose(BB) * AA

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T

    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R * centroid_B.T + centroid_A.T

    return c, R, t

# different dataset's index ID is different, so u cannot use the same function on the all dataset
def uniform_coordinate_system(skeleton_data,s, ret_R, ret_t):
    if s == 0:
        # left shoulder id is 11, right shoulder id is 14
        # left hip id is 1,right hip id is 4
        standard = [[0.05,0,-5],[0,-1,0],[0,1,0]]
        keypoints = [skeleton_data[1],skeleton_data[14],skeleton_data[10]]



        n = skeleton_data.shape[0]
        s, ret_R, ret_t = rigid_transform_3D(np.matrix(standard), np.matrix(keypoints), scale=False)
        output = (ret_R * skeleton_data.T) + np.tile(ret_t, (1, n))
        output = output.T
    else:
        n = skeleton_data.shape[0]
        output = (ret_R * skeleton_data.T) + np.tile(ret_t, (1, n))
        output = output.T
    return output,s, ret_R, ret_t


def uniform_key_points_index(skeleton_data):
    # 0:1;
    # 1:6;
    # 2:7;
    # 3:8
    # 4:2
    # 5:3
    # 6:4
    # 7:
    # 8:0
    # 9:
    # 10:18
    # 11:14
    # 12:15
    # 13:16
    # 14:10
    # 15:11
    # 16:12
    output = np.empty((15, 3))
    output[0] = skeleton_data[1]
    output[1] = skeleton_data[6]
    output[2] = skeleton_data[7]
    output[3] = skeleton_data[8]
    output[4] = skeleton_data[2]
    output[5] = skeleton_data[3]
    output[6] = skeleton_data[4]
    output[7] = skeleton_data[0]
    output[8] = skeleton_data[18]
    output[9] = skeleton_data[14]
    output[10] = skeleton_data[15]
    output[11] = skeleton_data[16]
    output[12] = skeleton_data[10]
    output[13] = skeleton_data[11]
    output[14] = skeleton_data[12]
    # output[15] = skeleton_data[1]
    # output[16] = skeleton_data[1]
    # output[14] = skeleton_data[1]

    return output

if __name__ == '__main__':
    root_path = '/home/imaginarium/Documents/dataset/MADS/'
    # output_folder = '/media/imaginarium/a0c299ce-8eea-4f25-92a4-b572d215821b/MergeDataset/'
    output_folder = '/home/imaginarium/Documents/dataset/merge/'
    GT_path = root_path + 'annot/MADS.pkl'

    with open(GT_path, 'rb') as file:
        GT_file = pickle.load(file)

    # print(GT_file)
    index = 0
    MAX_NUM = 30
    SKIP_NUM = 25
    save_frames = []
    save_GTs = []
    save_tensor = []
    s, ret_R, ret_t = 0,0,0
    current_frame_name = ''
    saveID = 0

    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to('cuda')

    for i in range(len(GT_file)):
        if len(save_frames) > 0 and current_frame_name != GT_file[i]['image'].split('/')[0]:
            current_frame_name = GT_file[i]['image'].split('/')[0]
            save_frames = []
            save_GTs = []
            save_tensor = []
            s, ret_R, ret_t = 0, 0, 0
            print('deal with ',current_frame_name)
        else:
            current_frame_name = GT_file[i]['image'].split('/')[0]
            GT_joints = GT_file[i]['joints_3d']
            if np.isnan(GT_joints).any() or np.isinf(GT_joints).any():
                save_frames = []
                save_GTs = []
                save_tensor = []
                s, ret_R, ret_t = 0, 0, 0
                print('Find the error data')
            else:
                # valid joints without NAN and infinite values
                GT_joints, s, ret_R, ret_t = uniform_coordinate_system(GT_joints, s, ret_R, ret_t)
                GT_joints = uniform_key_points_index(GT_joints)


                frame_path = root_path + 'images/' + GT_file[i]['image']
                frame = Image.open(frame_path)
                CLIP_inputs = processor(images=frame, return_tensors="pt")
                outputs = model(**CLIP_inputs.to('cuda'))
                last_hidden_state = outputs.last_hidden_state
                if len(save_frames) < MAX_NUM:
                    save_GTs.append(GT_joints)
                    frame = frame.resize((384,384))
                    save_frames.append(frame)
                    save_tensor.append(last_hidden_state)
                    # save_tensor.append(frame)

                else:
                    # folder_name = output_folder + 'data/MADA' + '_' + str(saveID)
                    # # Check if the folder exists
                    # if not os.path.exists(folder_name):
                    #     # If it doesn't exist, create the folder
                    #     os.mkdir(folder_name)
                    #     print(f"Folder '{folder_name}' created successfully.")
                    #
                    # # Iterate through the list of images and save them
                    # for i, image in enumerate(save_frames):
                    #     # image = Image.fromarray(image)
                    #     filename = os.path.join(folder_name, f"image_{i + 1}.png")
                    #
                    #     # Save the image to the specified filename
                    #     image.save(filename)

                    savePath = output_folder + 'data/MADA' + '_' + str(saveID) + '.pt'
                    output = torch.cat(save_tensor, dim=0)
                    torch.save(output.detach(), savePath)

                    GT_savePath = output_folder + 'GT/MADA' + '_' + str(saveID) + '.npy'
                    np.save(GT_savePath,save_GTs)
                    print('saved ',str(saveID),'size:',len(save_GTs))
                    saveID = saveID + 1

                    # remove the first element and add the new one
                    del save_GTs[0:SKIP_NUM]
                    del save_frames[0:SKIP_NUM]
                    del save_tensor[0:SKIP_NUM]

                    save_GTs.append(GT_joints)
                    save_tensor.append(last_hidden_state)
                    save_frames.append(frame)




