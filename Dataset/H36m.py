import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from moviepy.editor import VideoFileClip, clips_array
from math import sqrt
from bvh_skeleton import openpose_skeleton,h36m_skeleton,cmu_skeleton,smartbody_skeleton
from scipy.stats import multivariate_normal


def frames_to_video(frame_folder, output_path, fps=30):
    # Check if the output file extension is .avi
    if not output_path.lower().endswith('.mp4'):
        print("Error: Output file should have .mp4 extension.")
        return

    # Get the list of frame files in the folder
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')])

    if not frame_files:
        print("Error: No image frames found in the folder.")
        return

    # Get the first frame to retrieve its dimensions
    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

    print(f"Video created and saved as {output_path}")

def group_images(image_paths, output_path):
    # Check if the number of images provided is exactly four
    if len(image_paths) != 4:
        print("Error: Exactly four images are required to group.")
        return

    # Open and resize the images to a common size
    images = [Image.open(path).resize((1000, 1000)) for path in image_paths]

    # Create a blank canvas for the final image
    width, height = images[0].size
    final_image = Image.new('RGB', (2 * width, 2 * height))

    # Paste the images on the canvas in a 2x2 grid
    final_image.paste(images[0], (0, 0))
    final_image.paste(images[1], (width, 0))
    final_image.paste(images[2], (0, height))
    final_image.paste(images[3], (width, height))

    # Save the final grouped image
    # final_image.show()
    final_image.save(output_path)
    print(f"Images grouped successfully and saved as {output_path}!")

def load_heatmap_from_loction(heatmap,x,y,z,loction3D,x_step,covariance,adjust):
    # input a heatmap matrix and a key point's loction
    # x,y,z is the mesh gird
    # return a new heatmap

    min_x = np.min(x)
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)
    min_z = np.min(z)
    max_z = np.max(z)

    x_points = int((max_x - min_x) / x_step) + 1
    y_points = int((max_y - min_y) / x_step) + 1
    z_points = int((max_z - min_z) / x_step) + 1

    x_index = (loction3D[0] - min_x) / ((max_x - min_x) / (x_points - 1))
    y_index = (loction3D[1] - min_y) / ((max_y - min_y) / (y_points - 1))
    z_index = (loction3D[2] - min_z) / ((max_z - min_z) / (z_points - 1))

    for i in range(0,1):
        mean_x = min_x + (round(x_index) + i) * ((max_x - min_x) / (x_points - 1))
        for j in range(0,1):
            mean_y = min_y + (round(y_index) + j) * ((max_y - min_y) / (y_points - 1))
            for k in range(0,1):
                mean_z = min_z + (round(z_index) + k) * ((max_z - min_z) / (z_points - 1))
                # Define the mean vector and covariance matrix
                mean = [mean_x, mean_y, mean_z]
                # covariance = np.array([[x_step*x_step//2, 0, 0],
                #                        [0, x_step*x_step//2, 0],
                #                        [0, 0, x_step*x_step//2]])
                # # Create a grid of x, y, and z values
                # x, y, z = np.meshgrid(np.linspace(-4, 3, x_step),
                #                       np.linspace(-3, 3, x_step),
                #                       np.linspace(-3, 3, x_step))

                xyz = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

                # Calculate the probability density at each point in the grid
                pdf_values = multivariate_normal(mean, covariance, allow_singular=True).pdf(xyz)
                if i == 0 and j == 0 and k == 0:
                    # print(i,j,k,'100')
                    pdf_values = pdf_values.reshape(heatmap.shape) * 0.0003 * adjust
                else:
                    # print(i, j, k)
                    pdf_values = pdf_values.reshape(heatmap.shape) * 0
                heatmap = heatmap + pdf_values

    # Find the indices where values are greater than 1
    indices = np.where(heatmap > 1)
    # if len(indices[0]) > 0:
    #     print('find the value larger than 1',indices)
    # Change the values at those indices to 1
    heatmap[indices] = 1

    return heatmap

def plot_skeleton_3d(skeleton_data,num_person, output, min_x, max_x,min_y, max_y,min_z, max_z):

    def update_frame(frame_idx):
        # Update the 3D plot for the current frame
        ax.cla()
        for i in range(num_person):
            new_heatmap = heatmap
            sigma = x_step * x_step / 0.51
            covariance = np.array([[sigma, 0, 0],
                                   [0, sigma, 0],
                                   [0, 0, sigma]])

            ax.scatter(skeleton_data[i][frame_idx][:, 0], skeleton_data[i][frame_idx][:, 1],
                       skeleton_data[i][frame_idx][:, 2], c='b', marker='o')

            for connection in joint_connections:
                joint1, joint2 = connection
                ax.plot([skeleton_data[i][frame_idx][joint1, 0], skeleton_data[i][frame_idx][joint2, 0]],
                        [skeleton_data[i][frame_idx][joint1, 1], skeleton_data[i][frame_idx][joint2, 1]],
                        [skeleton_data[i][frame_idx][joint1, 2], skeleton_data[i][frame_idx][joint2, 2]], c='r')

                point1 = skeleton_data[i][frame_idx][joint1]
                point2 = skeleton_data[i][frame_idx][joint2]

                # middle_point = (point1 + point2) / 2
                distance = np.linalg.norm(point1 - point2)
                number_middle_points = int(distance / (x_step * 1))
                # print(joint1, joint2, number_middle_points)

                for middle_index in range(1,number_middle_points):
                    vector = point2 - point1
                    divided_vectors = vector * middle_index / number_middle_points
                    middle_point = point1 + divided_vectors

                    # middle_point = (point1 + point2) * middle_index / number_middle_points
                    new_heatmap = load_heatmap_from_loction(new_heatmap, x, y, z, middle_point,x_step, covariance,adjust=1)


            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.set_xlim3d(min_x, max_x)
            ax.set_ylim3d(min_y, max_y)
            ax.set_zlim3d(min_z, max_z)


            for j in range(len(skeleton_data[i][frame_idx])):
                new_heatmap = load_heatmap_from_loction(new_heatmap, x, y, z, skeleton_data[i][frame_idx][j],x_step,covariance,adjust=10)

            values = new_heatmap.ravel()
            threshold = 0.1
            # Filter data to include only points where values are above or equal to the threshold
            filtered_x = x[values >= threshold]
            filtered_y = y[values >= threshold]
            filtered_z = z[values >= threshold]
            filtered_values = values[values >= threshold]

            sc = ax.scatter(filtered_x, filtered_y, filtered_z, c=filtered_values, cmap='rainbow', alpha=0.1)


        # ax.plot([5300, 5300], [0, 0], [-1250, 500], c='g')

        # Set the frame's title (optional)
        ax.set_title(f'Frame {frame_idx}')

    # # Create a colormap that goes from blue to red
    # cmap = plt.get_cmap('coolwarm')

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # # Normalize the heatmap values to fit within the colormap
    # norm = plt.Normalize(0, 1)


    # ax.view_init(10, -30)
    # ax.invert_zaxis()

    # Define the desired step size for each axis
    x_step = 0.025
    y_step = x_step
    z_step = x_step

    # Calculate the number of points based on the step size
    x_points = int((max_x - min_x) / x_step) + 1
    y_points = int((max_y - min_y) / y_step) + 1
    z_points = int((max_z - min_z) / z_step) + 1

    # Define the range for 'x,' 'y,' and 'z'
    x_range = (min_x, max_x)
    y_range = (min_y, max_y)
    z_range = (min_z, max_z)

    # Initialize heatmap with zeros and the same size as x
    heatmap = np.zeros((x_points,y_points,z_points))

    # Create 3D heatmap
    x, y, z = np.indices(heatmap.shape)

    # Scale 'x,' 'y,' and 'z' values to the desired range
    x = x_range[0] + (x / (heatmap.shape[0] - 1)) * (x_range[1] - x_range[0])
    y = y_range[0] + (y / (heatmap.shape[1] - 1)) * (y_range[1] - y_range[0])
    z = z_range[0] + (z / (heatmap.shape[2] - 1)) * (z_range[1] - z_range[0])

    # Flatten the indices and value_matrix arrays
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    # Initialize the scatter plot with empty data
    sc = ax.scatter([], [], [], c=[], cmap='rainbow', alpha=0.3)


    # List of joint connections to draw lines between joints
    joint_connections = [(0, 1), (1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),
                         (8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]

    ani = FuncAnimation(fig, update_frame, frames=len(skeleton_data[0]), interval=41)
    # Add a colorbar for reference
    cbar = plt.colorbar(sc)

    # Label the colorbar if needed
    cbar.set_label('HeatMap')
    # ani.save(output, writer='ffmpeg')
    plt.show()

def combine_videos(video1_path, video2_path, output_path):
    # Load the video clips
    video1 = VideoFileClip(video1_path)
    video2 = VideoFileClip(video2_path)
    # video3 = VideoFileClip(video3_path)

    # Make sure all videos have the same height (resize if needed)
    target_height = max(video1.h, video2.h) # , video3.h
    if video1.h != target_height:
        video1 = video1.resize(height=target_height)
    if video2.h != target_height:
        video2 = video2.resize(height=target_height)
    # if video3.h != target_height:
    #     video3 = video3.resize(height=target_height)

    # Combine the videos side by side
    combined_video = clips_array([[video1, video2]])# , video3

    # Write the combined video to the output file
    combined_video.write_videofile(output_path, codec='libx264')

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
        keypoints = [skeleton_data[0],skeleton_data[11],skeleton_data[14]]
        # standard = [[0, 0, 0], [0.72, -0.28, 0], [1.06, -3.2, -0.34], [1.59,-7.84,-0.76],[-0.73,-0.29,0],
        #             [-1.05,-3.22,-0.36],[-1.58,-7.85,-0.75],[0,5,0],[0,6.5,0.23],[1.55,4.68,0],
        #             [3.53,4,-0.65],[5.64,3.13,-0.56],[-1.64,4.68,-0.08],[-3.51,4.06,-0.58],[-6,3,-0.54]]
        # keypoints = [skeleton_data[0],skeleton_data[1],skeleton_data[2],skeleton_data[3]
        #              ,skeleton_data[4],skeleton_data[5],skeleton_data[6],skeleton_data[8],skeleton_data[10]
        #              ,skeleton_data[11],skeleton_data[12],skeleton_data[13],skeleton_data[14],skeleton_data[15]
        #              ,skeleton_data[16]]

        n = skeleton_data.shape[0]
        s, ret_R, ret_t = rigid_transform_3D(np.matrix(standard), np.matrix(keypoints), scale=False)
        output = (ret_R * skeleton_data.T) + np.tile(ret_t, (1, n))
        output = output.T
    else:
        n = skeleton_data.shape[0]
        output = (ret_R * skeleton_data.T) + np.tile(ret_t, (1, n))
        output = output.T
    return output,s, ret_R, ret_t

def generate_multi_view(root_path,file_path,folder_name):
    # s_01_act_02_subact_01_ca

    with open(file_path, 'rb') as file:
        H36M = pickle.load(file)

    # record the range of datasets
    min_x = 999999
    min_y = 999999
    min_z = 999999
    max_x = -999999
    max_y = -999999
    max_z = -999999

    skeletons1 = []
    skeletons2 = []
    skeletons3 = []
    skeletons4 = []
    video_name = 0
    s, ret_R, ret_t = 0, 0, 0
    for data in H36M:
        # this function is to find the specific video folder
        if data['image'].split('/')[0][:-3] == folder_name:
            print('find!')
        else:
            # print(data['image'].split('/')[0])
            continue

        if video_name != 0 and data['image'].split('/')[0] != video_name:
            # generate multi-view video from the folder
            # frame_folder = multi_view_images_path

            # save_path = os.path.join(multi_view_video_path, video_name[:-3])
            # if not os.path.exists(save_path):
            #     os.mkdir(save_path)
            # save_multi_view_video_path = os.path.join(save_path, video_name[:-3] + '.mp4')
            # frames_to_video(frame_folder, save_multi_view_video_path, fps=24)
            s, ret_R, ret_t = 0, 0, 0
            print('renew the cooridante system')

            # break

        video_name = data['image'].split('/')[0]

        print(data['image'])

        # find the other view of camera
        video_name_c2 = video_name[:-1] + '2'
        video_name_c3 = video_name[:-1] + '3'
        video_name_c4 = video_name[:-1] + '4'

        frameName = data['image'].split('/')[-1].split('_')[-1]
        imagePath1 = os.path.join(images_path, data['image'])
        imagePath2 = os.path.join(images_path, video_name_c2, video_name_c2 + '_' + frameName)
        imagePath3 = os.path.join(images_path, video_name_c3, video_name_c3 + '_' + frameName)
        imagePath4 = os.path.join(images_path, video_name_c4, video_name_c4 + '_' + frameName)

        save_images_path = os.path.join(multi_view_images_path, video_name[:-3])
        if not os.path.exists(save_images_path):
            os.mkdir(save_images_path)
        save_multi_view_images_path = os.path.join(save_images_path, frameName)

        # group_images([imagePath1,imagePath2,imagePath3,imagePath4],save_multi_view_images_path)
        # break

        # there is non-invisible joint
        if not np.all(data['joints_vis'] == 1):
            print('There is the invisible point :', data['image'])

        # there is non-wrong shape joint
        skeleton_data = data['joints_3d']
        # skeleton_data = np.matrix(skeleton_data)
        skeleton_data, s, ret_R, ret_t = uniform_coordinate_system(skeleton_data, s, ret_R, ret_t)
        if skeleton_data.shape != (17, 3):
            print('There is the missing key points :', data['image'], skeleton_data.shape)

        min_x = min(min_x, np.min(skeleton_data, 0)[0, 0])
        min_y = min(min_y, np.min(skeleton_data, 0)[0, 1])
        min_z = min(min_z, np.min(skeleton_data, 0)[0, 2])
        max_x = max(max_x, np.max(skeleton_data, 0)[0, 0])
        max_y = max(max_y, np.max(skeleton_data, 0)[0, 1])
        max_z = max(max_z, np.max(skeleton_data, 0)[0, 2])

        if video_name[-1] == '1':
            skeletons1.append(skeleton_data)
        elif video_name[-1] == '2':
            skeletons2.append(skeleton_data)
        elif video_name[-1] == '3':
            skeletons3.append(skeleton_data)
        elif video_name[-1] == '4':
            skeletons4.append(skeleton_data)

    print(len(skeletons1))
    skeleton_savePath = root_path + 'human3.6M/human3.6mtoolbox处理后的/s_01_act_02_subact_01_ca_after.mp4'
    plot_skeleton_3d([skeletons1, skeletons2, skeletons3, skeletons4], 4, skeleton_savePath, min_x, max_x, min_y, max_y,
                     min_z, max_z)

    # root = '/Volumes/4T/dataset/human3.6M/human3.6mtoolbox处理后的/'
    # video1_path = root + 'multi_view_video/s_01_act_02_subact_01_ca.mp4'
    # video2_path = root + 's_01_act_02_subact_01_ca.mp4'
    # # video3_path = root + 's_01_act_02_subact_01_ca_1.mp4'
    # #
    # # Provide the output path for the combined video
    # output_path = root + "output_combined_video.mp4"
    #
    # combine_videos(video1_path, video2_path, output_path)

def test_uniform_system(root_path,file_path,showNum):
    with open(file_path, 'rb') as file:
        H36M = pickle.load(file)

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
    for data in H36M:
        # this function is to find the specific video folder
        # if data['image'].split('/')[0][:-3] == 's_01_act_02_subact_01_ca':
        #     print('find!')
        # else:
        #     # print(data['image'].split('/')[0])
        #     continue

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

        # find the other view of camera
        # video_name_c2 = video_name[:-1] + '2'
        # video_name_c3 = video_name[:-1] + '3'
        # video_name_c4 = video_name[:-1] + '4'
        #
        # frameName = data['image'].split('/')[-1].split('_')[-1]
        # imagePath1 = os.path.join(images_path, data['image'])
        # imagePath2 = os.path.join(images_path, video_name_c2, video_name_c2+'_'+frameName)
        # imagePath3 = os.path.join(images_path, video_name_c3, video_name_c3+'_'+frameName)
        # imagePath4 = os.path.join(images_path, video_name_c4, video_name_c4+'_'+frameName)
        #
        # save_images_path = os.path.join(multi_view_images_path,video_name[:-3])
        # if not os.path.exists(save_images_path):
        #     os.mkdir(save_images_path)
        # save_multi_view_images_path = os.path.join(save_images_path, frameName)

        # group_images([imagePath1,imagePath2,imagePath3,imagePath4],save_multi_view_images_path)
        # break

        # there is non-invisible joint
        if not np.all(data['joints_vis'] == 1):
            print('There is the invisible point :', data['image'])

        # there is non-wrong shape joint
        skeleton_data = data['joints_3d']
        skeleton_data, s, ret_R, ret_t = uniform_coordinate_system(skeleton_data, s, ret_R, ret_t)
        if skeleton_data.shape != (17, 3):
            print('There is the missing key points :', data['image'], skeleton_data.shape)

        min_x = min(min_x, np.min(skeleton_data, 0)[0, 0])
        min_y = min(min_y, np.min(skeleton_data, 0)[0, 1])
        min_z = min(min_z, np.min(skeleton_data, 0)[0, 2])
        max_x = max(max_x, np.max(skeleton_data, 0)[0, 0])
        max_y = max(max_y, np.max(skeleton_data, 0)[0, 1])
        max_z = max(max_z, np.max(skeleton_data, 0)[0, 2])

        temp_skeletons.append(skeleton_data)

    print(len(skeletons))
    # write_standard_bvh('./test.bvh', np.array(skeletons[0]))

    skeleton_savePath = root_path + 'human3.6M/human3.6mtoolbox处理后的/s_01_act_02_subact_01_ca_after.mp4'
    new_skeletons = np.array(skeletons)
    max_distance = max(abs(max_x),abs(min_x), abs(max_y),abs(min_y), abs(max_z),abs(min_z))
    new_skeletons = new_skeletons / max_distance
    # min_x, max_x, min_y, max_y, min_z, max_z = min_x/ max_distance, max_x/ max_distance, min_y/ max_distance, max_y/ max_distance, min_z/ max_distance, max_z/ max_distance

    plot_skeleton_3d(new_skeletons, len(skeletons), skeleton_savePath, -1, 1, -1, 1, -1, 1)


    print('The range of x:', min_x, max_x)
    print('The range of y:', min_y, max_y)
    print('The range of z:', min_z, max_z)

# 将3dpoint转换为标准的bvh格式并输出到outputs/outputvideo/alpha_pose_视频名/bvh下
def write_standard_bvh(outbvhfilepath, prediction3dpoint):
    '''
    :param outbvhfilepath: 输出bvh动作文件路径
    :param prediction3dpoint: 预测的三维关节点
    :return:
    '''

    # 将预测的点放大100倍
    # for frame in prediction3dpoint:
    #     for point3d in frame:
    #         point3d[0] *= 100
    #         point3d[1] *= 100
    #         point3d[2] *= 100
    #
    #         # 交换Y和Z的坐标
    #         # X = point3d[0]
    #         # Y = point3d[1]
    #         # Z = point3d[2]
    #
    #         # point3d[0] = -X
    #         # point3d[1] = Z
    #         # point3d[2] = Y

    # dir_name = os.path.dirname(outbvhfilepath)
    # basename = os.path.basename(outbvhfilepath)
    # video_name = basename[:basename.rfind('.')]
    # bvhfileDirectory = os.path.join(dir_name, video_name, "bvh")
    # if not os.path.exists(bvhfileDirectory):
    #     os.makedirs(bvhfileDirectory)
    # bvhfileName = os.path.join(dir_name, video_name, "bvh", "{}.bvh".format(video_name))
    human36m_skeleton = h36m_skeleton.H36mSkeleton()
    human36m_skeleton.poses2bvh(prediction3dpoint, output_file=outbvhfilepath)

if __name__ == '__main__':
    # the label path
    root_path = './data/'
    file_path = root_path + 'Huam3.6M/h36m_train.pkl'
    images_path = root_path +  'human3.6M/human3.6mtoolbox处理后的/images/'
    multi_view_video_path = root_path +  'human3.6M/human3.6mtoolbox处理后的/multi_view_video/'
    multi_view_images_path = root_path +  'human3.6M/human3.6mtoolbox处理后的/multi_view_images/'

    test_uniform_system(root_path, file_path,1)
    # generate_multi_view(root_path, file_path, 's_01_act_03_subact_01_ca')

    # root = '/Volumes/4T/dataset/human3.6M/human3.6mtoolbox处理后的/'
    # video1_path = root + 'multi_view_video/s_01_act_02_subact_01_ca.mp4'
    # video2_path = root + 's_01_act_02_subact_01_ca.mp4'
    # # video3_path = root + 's_01_act_02_subact_01_ca_1.mp4'
    # #
    # # Provide the output path for the combined video
    # output_path = root + "output_combined_video.mp4"
    #
    # combine_videos(video1_path, video2_path, output_path)


    # plot_skeleton_3d(skeleton_data)

    # print(H36M)