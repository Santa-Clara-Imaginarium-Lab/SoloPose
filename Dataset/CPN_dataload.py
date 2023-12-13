import numpy as np

if __name__ == '__main__':
    path = '/home/imaginarium/Downloads/positions_2d.npy'

    keypoint_2d = np.load(path,allow_pickle=True)
    print('')