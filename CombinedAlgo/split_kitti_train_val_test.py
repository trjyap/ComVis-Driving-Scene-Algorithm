import os
import shutil
import random

# Paths
img_dir = '../KITTI Dataset/training/image_2'  # Update if needed
mask_dir = './kitti_masks'
output_root = '../KITTI Dataset'  # Where to create images/masks/train,val,test

# Output folders
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(output_root, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'masks', split), exist_ok=True)

# Get all mask files (assuming mask and image names match)
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
random.shuffle(mask_files)

n_total = len(mask_files)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

splits_counts = {
    'train': n_train,
    'val': n_val,
    'test': n_test
}

split_files = {
    'train': mask_files[:n_train],
    'val': mask_files[n_train:n_train+n_val],
    'test': mask_files[n_train+n_val:]
}

for split in splits:
    for fname in split_files[split]:
        # Copy mask
        src_mask = os.path.join(mask_dir, fname)
        dst_mask = os.path.join(output_root, 'masks', split, fname)
        shutil.copy(src_mask, dst_mask)
        # Copy image
        src_img = os.path.join(img_dir, fname)
        dst_img = os.path.join(output_root, 'images', split, fname)
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        else:
            print(f"Warning: Image {src_img} not found for mask {fname}")

print("Data split complete!") 