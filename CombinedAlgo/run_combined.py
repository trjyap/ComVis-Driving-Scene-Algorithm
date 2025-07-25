import sys
import cv2
import numpy as np
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import time
import os
from glob import glob

# Add both project paths
sys.path.append('../pytorch-deeplab-xception')
sys.path.append('../YOLOv8-3D')

# Import DeepLab
from modeling.deeplab import DeepLab
# Import YOLOv8
from ultralytics import YOLO

# --------- CONFIG ---------
KITTI_IMAGE_PATH = '../KITTI Dataset/training/image_2/000000.png'  # Change as needed
DEEPLAB_BACKBONE = 'mobilenet'  # or 'resnet', etc.
DEEPLAB_NUM_CLASSES = 21        # or 19 for cityscapes, etc.
DEEPLAB_WEIGHTS = '../pytorch-deeplab-xception/deeplab-mobilenet.pth/deeplab-mobilenet.pth'  # Updated to use newly trained model
#'../pytorch-deeplab-xception/run/kitti/deeplab-mobilenet/model_best.pth.tar'
YOLO_WEIGHTS = '../YOLOv8-3D/yolov8n-seg.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# --------------------------

# --------- Load Image ---------
# --------- Load Video ---------
VIDEO_PATH = '../YOLOv8-3D/assets/2011_10_03_drive_0034_sync_video_trimmed.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

# Preprocess for DeepLab
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DeepLab inference
deeplab_model = DeepLab(backbone=DEEPLAB_BACKBONE, output_stride=16, num_classes=DEEPLAB_NUM_CLASSES)
# Load checkpoint and extract state_dict
checkpoint = torch.load(DEEPLAB_WEIGHTS, map_location=DEVICE, weights_only=False)
deeplab_model.load_state_dict(checkpoint['state_dict'])
deeplab_model.eval()
deeplab_model.to(DEVICE)

# YOLOv8 inference
yolo_model = YOLO(YOLO_WEIGHTS)

# --------- Visualization ---------
# Overlay DeepLab segmentation
from dataloaders.utils import decode_segmap

# Create output directories for masks
os.makedirs('deeplab_masks', exist_ok=True)
os.makedirs('combined_masks', exist_ok=True)

# Initialize FPS calculation
frameId = 0
start_time = time.time()
fps = ""

while True:
    frameId += 1
    ret, img_bgr = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (513, 513))  # DeepLab expects 513x513

    # DeepLab inference
    input_tensor = preprocess(img_resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = deeplab_model(input_tensor)
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Resize DeepLab mask to original image size
    pred_resized = cv2.resize(pred.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Save DeepLab mask
    deeplab_mask_path = os.path.join('deeplab_masks', f'{frameId:06d}.png')
    cv2.imwrite(deeplab_mask_path, pred_resized)

    # YOLOv8 inference
    yolo_results = yolo_model(img_bgr)

    # Create YOLO mask (binary, same size as image)
    yolo_mask = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)
    for r in yolo_results:
        if r.boxes is not None:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                yolo_mask[y1:y2, x1:x2] = 1

    # Combine DeepLab and YOLO masks (union)
    combined_pred = np.where((pred_resized > 0) | (yolo_mask > 0), 1, 0).astype(np.uint8)
    combined_mask_path = os.path.join('combined_masks', f'{frameId:06d}.png')
    cv2.imwrite(combined_mask_path, combined_pred * 255)

    # Overlay DeepLab segmentation
    segmap = decode_segmap(pred, dataset='coco')  # or 'pascal', 'cityscapes' as appropriate
    segmap = (segmap * 255).astype(np.uint8)
    segmap = cv2.resize(segmap, (img_bgr.shape[1], img_bgr.shape[0]))
    overlay = cv2.addWeighted(img_bgr, 0.5, segmap, 0.5, 0)

    # Draw YOLOv8 detections
    for r in yolo_results:
        if r.boxes is not None:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)

    # Calculate FPS every 20 frames
    if frameId % 20 == 0:
        elapsed_time = time.time() - start_time
        fps_current = frameId / elapsed_time
        fps = f'FPS: {fps_current:.2f}'

    # Overlay FPS on the output frame
    cv2.putText(overlay, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show or save the result
    cv2.imshow('Combined Output', overlay)
    if out is None:
        out = cv2.VideoWriter('combined_output1.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (overlay.shape[1], overlay.shape[0]))
    out.write(overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

# ----------------- mIoU Calculation -----------------
import os
from glob import glob

def compute_miou(pred_dir, gt_dir):
    pred_files = sorted(glob(os.path.join(pred_dir, '*.png')))
    if not pred_files:
        print(f"No predicted masks found in {pred_dir}")
        return
    intersection = 0
    union = 0
    count = 0
    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, filename)
        if not os.path.exists(gt_path):
            print(f"Ground truth mask not found for {filename}, skipping.")
            continue
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred_mask is None or gt_mask is None:
            print(f"Error reading mask for {filename}, skipping.")
            continue
        # Binarize masks (assume threshold at 128)
        pred_bin = (pred_mask >= 128).astype(np.uint8)
        gt_bin = (gt_mask >= 128).astype(np.uint8)
        inter = np.logical_and(pred_bin, gt_bin).sum()
        uni = np.logical_or(pred_bin, gt_bin).sum()
        if uni == 0:
            continue  # Avoid division by zero
        intersection += inter
        union += uni
        count += 1
    if count == 0 or union == 0:
        print("No valid mask pairs found for mIoU calculation.")
        return
    miou = intersection / union
    print(f"\nPost-processing mIoU over {count} mask pairs: {miou:.4f}")

# Calculate mIoU for DeepLab masks
print("\nCalculating mIoU for DeepLab predictions:")
compute_miou(
    pred_dir=os.path.join(os.path.dirname(__file__), 'deeplab_masks'),
    gt_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../KITTI Dataset/data_semantics/training/semantic'))
)

# Calculate mIoU for Combined model masks
print("\nCalculating mIoU for Combined model predictions:")
compute_miou(
    pred_dir=os.path.join(os.path.dirname(__file__), 'combined_masks'),
    gt_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../KITTI Dataset/data_semantics/training/semantic'))
)