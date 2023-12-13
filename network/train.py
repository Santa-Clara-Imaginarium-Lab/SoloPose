import os
import numpy as np
import torch
import argparse
import random
from torch.utils.data import Dataset
# from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import sys
from model import EmotionNetwork
from PIL import Image
# import pandas as pd
from torchvision import transforms
from scipy.stats import multivariate_normal
import skimage.io
import skimage.transform
import skimage.color
import skimage


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def configure_optimizers(net, args):

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    # parser.add_argument(
    #     "-cd", "--contextDataset", type=str,
    #     default='D:/Tianma/dataset/Pre_process/train_tensor/',
    #     help="Training dataset"
    # )

    parser.add_argument(
        "-td", "--testing_Data", type=str, default='/media/imaginarium/2T/valid/',help="testing dataset"
    )
    # /media/imaginarium/2T   '/media/imaginarium/12T_2/train/
    parser.add_argument(
        "-d", "--Training_Data", type=str,default='/media/imaginarium/2T/train/',help="Training dataset"
    )
    parser.add_argument("-e","--epochs",default=1000000,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument(
        "-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n","--num-workers",type=int,default=16,help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",type=int,default=16,help="Test batch size (default: %(default)s)",
    )
    parser.add_argument("--cuda",  default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save_path", type=str, default="./save/", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s")

    parser.add_argument("--checkpoint",
     default="",  # ./train0008/18.ckpt
     type=str, help="Path to a checkpoint")

    args = parser.parse_args(argv)
    return args

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, min_side=256, max_side=256):
        # image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 256 - rows
        pad_h = 256 - cols

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        # annots *= scale

        return  torch.from_numpy(new_image), scale

def load_heatmap_from_loction(heatmap,x,y,z,loction3D,x_step,covariance):
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
                    pdf_values = pdf_values.reshape(heatmap.shape) * 0.001
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


class myDataset(Dataset):
    def __init__(self, root,transform):
        # self.df = pd.read_csv(root)
        self.clipTensor = []
        folders = os.listdir(root+'data/')
        for folder in folders:
            # or folder.startswith('MADA')
            # if folder.startswith('MADA') :
            #     self.clipTensor.append(os.path.join(root + 'data/', folder))
            self.clipTensor.append(os.path.join(root+'data/', folder))

        self.transform = transform

    def __getitem__(self, index):
        Spatial_feature_map_path = self.clipTensor[index]
        label_file_name = Spatial_feature_map_path.split('/')[-1][:-3]# [:-3]
        paths_name = Spatial_feature_map_path.split('/')[:-2]
        original_data_folder = '/'
        for path_name in paths_name:
            original_data_folder = os.path.join(original_data_folder,path_name)
        GT_path = os.path.join(original_data_folder, 'GT', label_file_name + '.npy')

        Spatial_feature_map = torch.load(Spatial_feature_map_path, map_location=lambda storage, loc: storage)
        Spatial_feature_map = Spatial_feature_map.view(30,200,192)
        # Images = []
        # imagesPaths = os.listdir(Spatial_feature_map_path)
        # for imagePath in imagesPaths:
        #     imagePath = os.path.join(Spatial_feature_map_path, imagePath)
        #     img = Image.open(imagePath).convert("RGB")
        #     img, img_scale = self.transform(np.array(img))
        #     img = img.permute(2, 0, 1)
        #     Images.append(img)
        # Images = np.concatenate(Images, axis=0)
        # Spatial_feature_map = Images

        # GT_npy = np.array(np.load(GT_path), dtype='f')
        GT_npy = torch.from_numpy(np.array(np.load(GT_path), dtype='f'))
        # GT_npy = GT_npy[0:30,:,:]
        # Spatial_feature_map = Spatial_feature_map[0:30*3, :, :]

        # scales = torch.max(torch.abs(GT_npy))
        # restrict the range in [-1,1]
        # GT_npy size = 81 x 15 x 3
        GT_npy = GT_npy * 1 / (100 * 1)
        heatmaps = GT_npy


        # min_x, max_x, min_y, max_y, min_z, max_z = -1,1,-1,1,-1,1
        #
        # # Define the desired step size for each axis
        # x_step = 0.025
        # y_step = x_step
        # z_step = x_step
        #
        # sigma = x_step * x_step / 0.51
        # covariance = np.array([[sigma, 0, 0],
        #                        [0, sigma, 0],
        #                        [0, 0, sigma]])
        #
        # # Calculate the number of points based on the step size
        # x_points = int((max_x - min_x) / x_step) + 1
        # y_points = int((max_y - min_y) / y_step) + 1
        # z_points = int((max_z - min_z) / z_step) + 1
        #
        # # Define the range for 'x,' 'y,' and 'z'
        # x_range = (min_x, max_x)
        # y_range = (min_y, max_y)
        # z_range = (min_z, max_z)
        #
        # # Initialize heatmap with zeros and the same size as x
        # heatmap = np.zeros((x_points, y_points, z_points))
        #
        # # Create 3D heatmap
        # x, y, z = np.indices(heatmap.shape)
        #
        # # Scale 'x,' 'y,' and 'z' values to the desired range
        # x = x_range[0] + (x / (heatmap.shape[0] - 1)) * (x_range[1] - x_range[0])
        # y = y_range[0] + (y / (heatmap.shape[1] - 1)) * (y_range[1] - y_range[0])
        # z = z_range[0] + (z / (heatmap.shape[2] - 1)) * (z_range[1] - z_range[0])
        #
        # # Flatten the indices and value_matrix arrays
        # # x = x.ravel()
        # # y = y.ravel()
        # # z = z.ravel()
        #
        # # List of joint connections to draw lines between joints
        # joint_connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
        #                      (7, 9), (9, 10), (10, 11), (7, 12), (12, 13), (13, 14)]
        #
        # heatmaps = []
        # for i in range(0, GT_npy.shape[0]):
        #     for j in range(0,15):
        #         loction3D = GT_npy[i][j]
        #
        #         heatmap = load_heatmap_from_loction(heatmap,x,y,z,loction3D,x_step,covariance)
        #
        #     for connection in joint_connections:
        #         joint1, joint2 = connection
        #         point1 = GT_npy[i][joint1]
        #         point2 = GT_npy[i][joint2]
        #
        #         # middle_point = (point1 + point2) / 2
        #         distance = np.linalg.norm(point1 - point2)
        #         number_middle_points = int(distance / (x_step * 1))
        #         # print(joint1, joint2, number_middle_points)
        #
        #         for middle_index in range(1,number_middle_points):
        #             vector = point2 - point1
        #             divided_vectors = vector * middle_index / number_middle_points
        #             middle_point = point1 + divided_vectors
        #
        #             # middle_point = (point1 + point2) * middle_index / number_middle_points
        #             heatmap = load_heatmap_from_loction(heatmap, x, y, z, middle_point,x_step, covariance)
        #
        #     heatmaps.append(heatmap)
        #     heatmap = np.zeros((x_points, y_points, z_points))


        # 81 x 15 x 3
        return Spatial_feature_map,heatmaps

    def __len__(self):
        return len(self.clipTensor)



def train_one_epoch(model, train_dataloader, optimizer, epoch, clip_max_norm):
    model.train()
    device = next(model.parameters()).device
    start = time.time()
    # accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    sample_num = 0

    for i, d in enumerate(train_dataloader):

        Images,GT_npy = d
        Images = Images.to(device)
        GT_npy = GT_npy.to(device)
        optimizer.zero_grad()
        sample_num += Images.shape[0]

        out_net = model(Images)
        # pred_classes = torch.max(out_net, dim=1)[1]
        # accu_num += torch.eq(pred_classes, label.to(device)).sum()

        loss_function = torch.nn.MSELoss(reduction='mean')
        # print(out_net.shape,GT_npy.shape)
        out_criterion = loss_function(out_net, GT_npy)
        out_criterion.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 3000 == 0:
            enc_time = time.time() - start
            start = time.time()
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(Images)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion.item():.4f} |'
                # f'\tacc: {accu_num.item() / sample_num:.4f} |'
                f"\ttime: {enc_time:.1f}"
            )

def test_epoch(epoch, test_dataloader, model):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    loss_function = torch.nn.MSELoss(reduction='mean')
    # accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    sample_num = 0

    with torch.no_grad():
        for d in test_dataloader:
            Images,GT_npy = d
            sample_num += Images.shape[0]
            out_net = model(Images.to(device))
            # pred_classes = torch.max(out_net, dim=1)[1]
            # accu_num += torch.eq(pred_classes, label.to(device)).sum()

            out_criterion = loss_function(out_net, GT_npy.to(device))

            loss.update(out_criterion)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        # f'\tacc: {accu_num.item() / sample_num:.4f} |'
    )
    return loss.avg

def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose([Resizer()])

    # test_transforms = transforms.Compose([Resizer()])

    train_dataset = myDataset(args.Training_Data,train_transforms)
    test_dataset = myDataset(args.testing_Data,train_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        # pin_memory=(device == "cuda"),
    )

    net = EmotionNetwork()
    net = net.to(device)

    # print('GPU:',torch.cuda.device_count())
    #
    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
    # criterion = RateDistortionLoss(lmbda=args.lmbda)
    #
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        new_state_dict = checkpoint["state_dict"]
        # new_state_dict = OrderedDict()

        # for k, v in checkpoint["state_dict"].items():
        #     # if 'gaussian_conditional' in k:
        #     #     new_state_dict[k]=v
        #     #     print(k)
        #     #     continue
        #     # if 'module' not in k:
        #     k = k[7:]
        #     # else:
        #     #     k = k.replace('features.module.', 'module.features.')
        #     new_state_dict[k]=v

        net.load_state_dict(new_state_dict)


        optimizer.load_state_dict(checkpoint["optimizer"])
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(net,train_dataloader,optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if is_best and epoch % 1 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                args.save_path +str(epoch)+'.ckpt'
            )

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])











