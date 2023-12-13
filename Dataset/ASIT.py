# import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_skeleton_3d(skeleton_data,num_person, output, min_x, max_x,min_y, max_y,min_z, max_z):
    def update_frame(frame_idx):
        # Update the 3D plot for the current frame
        ax.cla()
        for i in range(num_person):
            draw_list = [0,5,6,7,8,9,10,11,12,13,14,15,16]
            ax.scatter(skeleton_data[i][frame_idx][draw_list, 0], skeleton_data[i][frame_idx][draw_list, 1],
                       skeleton_data[i][frame_idx][draw_list, 2], c='b', marker='o')

            for connection in joint_connections:
                joint1, joint2 = connection
                ax.plot([skeleton_data[i][frame_idx][joint1, 0], skeleton_data[i][frame_idx][joint2, 0]],
                        [skeleton_data[i][frame_idx][joint1, 1], skeleton_data[i][frame_idx][joint2, 1]],
                        [skeleton_data[i][frame_idx][joint1, 2], skeleton_data[i][frame_idx][joint2, 2]], c='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.plot([5300, 5300], [0, 0], [-1250, 500], c='g')

        ax.set_xlim3d(min_x, max_x)
        ax.set_ylim3d(min_y, max_y)
        ax.set_zlim3d(min_z, max_z)
        # ax.view_init(10, -30)
        # ax.invert_zaxis()

        # Set the frame's title (optional)
        ax.set_title(f'Frame {frame_idx}')
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # List of joint connections to draw lines between joints
    # joint_connections = [(0, 1), (1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),
    #                      (8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]
    # 0:;
    # 1:11;
    # 2:13;
    # 3:15
    # 4:12
    # 5:14
    # 6:16
    # 7:
    # 8:
    # 9:
    # 10:0
    # 11:5
    # 12:7
    # 13:9
    # 14:6
    # 15:8
    # 16:10
    joint_connections = [(11,13),(13,15),(11,12),(12,14),(14,16),(5,7),(7,9),(5,6),(6,8),(8,10)]


    ani = FuncAnimation(fig, update_frame, frames=len(skeleton_data[0]), interval=41)
    # ani.save(output, writer='ffmpeg')
    plt.show()

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
def uniform_coordinate_system(skeleton_data,s, ret_R, ret_t):
    if s == 0:
        # left shoulder id is 11, right shoulder id is 14
        # left hip id is 1,right hip id is 4
        standard = [[0.05,0,-5],[0,-1,0],[0,1,0]]
        keypoints = [(skeleton_data[11]+skeleton_data[12])//2,skeleton_data[5],skeleton_data[6]]

        n = skeleton_data.shape[0]
        s, ret_R, ret_t = rigid_transform_3D(np.matrix(standard), np.matrix(keypoints), scale=False)
        output = (ret_R * skeleton_data.T) + np.tile(ret_t, (1, n))
        output = output.T
    else:
        n = skeleton_data.shape[0]
        output = (ret_R * skeleton_data.T) + np.tile(ret_t, (1, n))
        output = output.T
    return output,s, ret_R, ret_t

def test_uniform_system(root_path,file_path,showNum):
    with open(file_path, 'rb') as file:
        MADS = pickle.load(file)

    # record the range of datasets
    min_x = 999999
    min_y = 999999
    min_z = 999999
    max_x = -999999
    max_y = -999999
    max_z = -999999

    skeletons = []
    temp_skeletons = []

    video_name = 0
    s, ret_R, ret_t = 0, 0, 0
    for data in MADS:
        if video_name != 0 and data['image'].split('/')[0] != video_name:
            # generate multi-view video from the folder
            # frame_folder = multi_view_images_path

            # save_path = os.path.join(multi_view_video_path, video_name[:-3])
            # if not os.path.exists(save_path):
            #     os.mkdir(save_path)
            # save_multi_view_video_path = os.path.join(save_path, video_name[:-3] + '.mp4')
            # frames_to_video(frame_folder, save_multi_view_video_path, fps=24)
            s, ret_R, ret_t = 0, 0, 0
            skeletons.append(temp_skeletons)
            temp_skeletons = []
            print('renew the cooridante system')
            if len(skeletons) == showNum:
                break

        video_name = data['image'].split('/')[0]

        print(data['image'])

        # there is non-wrong shape joint
        skeleton_data = data['joints_3d']
        # skeleton_data = np.matrix(skeleton_data)
        skeleton_data, s, ret_R, ret_t = uniform_coordinate_system(skeleton_data, s, ret_R, ret_t)

        min_x = min(min_x, np.min(skeleton_data, 0)[0, 0])
        min_y = min(min_y, np.min(skeleton_data, 0)[0, 1])
        min_z = min(min_z, np.min(skeleton_data, 0)[0, 2])
        max_x = max(max_x, np.max(skeleton_data, 0)[0, 0])
        max_y = max(max_y, np.max(skeleton_data, 0)[0, 1])
        max_z = max(max_z, np.max(skeleton_data, 0)[0, 2])

        temp_skeletons.append(skeleton_data)

    print(len(skeletons))
    skeleton_savePath = root_path + 'human3.6M/human3.6mtoolbox处理后的/s_01_act_02_subact_01_ca_after.mp4'
    plot_skeleton_3d(skeletons, len(skeletons), skeleton_savePath, min_x, max_x, min_y, max_y, min_z, max_z)

    print('The range of x:', min_x, max_x)
    print('The range of y:', min_y, max_y)
    print('The range of z:', min_z, max_z)

if __name__ == '__main__':
    root_path = '/media/imaginarium/2T/dataset/'
    file_path = root_path + 'AIST_Dance/annot/AIST_Dance.pkl'

    test_uniform_system(root_path, file_path, 2)