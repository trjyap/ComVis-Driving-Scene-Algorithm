import os
import cv2
import numpy as np

# Paths
kitti_img_dir = '../KITTI Dataset/training/image_2'
kitti_label_dir = '../KITTI Dataset/training/label_2'
output_mask_dir = './kitti_masks'  # Where to save PNG masks

os.makedirs(output_mask_dir, exist_ok=True)

# Class mapping
class_map = {
    'Car': 1,
    'Pedestrian': 2,
    'Cyclist': 3,
    # Add more if needed
}

for label_file in os.listdir(kitti_label_dir):
    if not label_file.endswith('.txt'):
        continue
    img_name = label_file.replace('.txt', '.png')
    img_path = os.path.join(kitti_img_dir, img_name)
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(os.path.join(kitti_label_dir, label_file), 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = parts[0]
            if cls not in class_map:
                continue
            x1, y1, x2, y2 = map(float, parts[4:8])
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            mask[y1:y2, x1:x2] = class_map[cls]

    out_mask_path = os.path.join(output_mask_dir, img_name)
    cv2.imwrite(out_mask_path, mask)
    print(f"Saved mask: {out_mask_path}")