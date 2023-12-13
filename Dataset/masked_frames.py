import cv2
import os
import numpy as np
import random
import re  # Regular expressions
import shutil  # To copy files
import torch
import torchvision
import torchvision.transforms as T


def get_human_bounding_box(frame):
    # Preprocess the frame for the model
    transform = T.Compose([T.ToTensor()])
    input_frame = transform(frame)

    # Run the model inference
    model.eval()
    model.cuda()
    with torch.no_grad():
        predictions = model([input_frame.to('cuda')])

    # Retrieve the bounding box coordinates for human class
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    for box, label, score in zip(boxes, labels, scores):
        if label == 1:  # Assuming class 1 is for humans
            if score > 0.5:  # Filter based on confidence threshold
                # Return the bounding box coordinates
                return tuple(box.int().tolist())

    return None


def draw_rectangle(image, mask):
    mask_xmin, mask_ymin, mask_xmax, mask_ymax = mask
    cv2.rectangle(image, (mask_xmin, mask_ymin), (mask_xmax, mask_ymax), (0, 0, 0), -1)


def draw_circle(image, mask):
    mask_xmin, mask_ymin, mask_xmax, mask_ymax = mask
    radius = min(mask_xmax - mask_xmin, mask_ymax - mask_ymin) // 2
    center = (mask_xmin + radius, mask_ymin + radius)
    cv2.circle(image, center, radius, (0, 0, 0), -1)


def draw_triangle(image, mask):
    mask_xmin, mask_ymin, mask_xmax, mask_ymax = mask
    points = np.array([[mask_xmin, mask_ymax], [(mask_xmin + mask_xmax) // 2, mask_ymin], [mask_xmax, mask_ymax]], dtype=np.int32)
    cv2.fillPoly(image, [points], (0, 0, 0))


def generate_mask(box, shapes=['rectangle', 'circle', 'triangle'], min_scale=1 / 16, max_scale=1 / 10):
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin

    # Select a random scale for the mask
    scale = random.uniform(min_scale, max_scale)
    mask_area = width * height * scale

    # Select a random shape for the mask
    shape = random.choice(shapes)

    # Determine the side lengths of the mask
    mask_width = int(np.sqrt(mask_area))
    mask_height = int(mask_area / mask_width)

    # Determine the top left corner of the mask
    mask_xmin = random.randint(xmin, xmax - mask_width)
    mask_ymin = random.randint(ymin, ymax - mask_height)

    return shape, (mask_xmin, mask_ymin, mask_xmin + mask_width, mask_ymin + mask_height)


def select_random_files(files, lower_bound=3, upper_bound=12, num_files=3):
    # Sort files by the numerical ID at the end
    sorted_files = sorted(files, key=lambda x: int(re.findall(r'\d+$', x.split('.')[0])[0]))
    # Select random indices
    selected_indices = set(random.sample(range(lower_bound, upper_bound), num_files))
    return sorted_files, selected_indices


def process_frames(index, filename, selected_indices):
    frame_path = os.path.join(subfolder_path, filename)

    # Create path in masked_frames folder
    masked_subfolder_path = os.path.join(masked_folder_path, subfolder)
    os.makedirs(masked_subfolder_path, exist_ok=True)
    masked_frame_path = os.path.join(masked_subfolder_path, filename)

    if index in selected_indices:  # If the frame is selected for masking
        # Read the frame
        frame = cv2.imread(frame_path)
        bounding_box = get_human_bounding_box(frame)

        shape, mask = generate_mask(bounding_box)
        if shape == 'rectangle':
            draw_rectangle(frame, mask)
        elif shape == 'circle':
            draw_circle(frame, mask)
        elif shape == 'triangle':
            draw_triangle(frame, mask)

        # Save the masked frame
        cv2.imwrite(masked_frame_path, frame)

    else:  # If the frame is not selected for masking, copy it over without changes
        shutil.copy2(frame_path, masked_frame_path)


if __name__ == "__main__":
    # Folder path containing the frames
    folder_path = 'D:\\X_Pose\\MADS_prepocess\\Clips\\frames\\'  # Replace with the actual folder path
    # Create a new folder to store the masked frames
    masked_folder_path = 'D:\\X_Pose\\MADS_prepocess\\Clips\\masked_frames\\'  # Replace with the desired folder path
    os.makedirs(masked_folder_path, exist_ok=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            files, selected_indices = select_random_files(os.listdir(subfolder_path))

            for index, filename in enumerate(files):
                if filename.endswith('.jpg'):  # Adjust the file extensions as needed
                    process_frames(index, filename, selected_indices)
