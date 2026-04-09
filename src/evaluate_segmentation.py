import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from glob import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_IMG_DIR = os.path.join(BASE_DIR, 'data', 'autolabel', 'images')
VAL_LBL_DIR = os.path.join(BASE_DIR, 'data', 'autolabel', 'labels')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 1.0
    return intersection / union

def compute_dice(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    if mask1.sum() + mask2.sum() == 0: return 1.0
    return 2 * intersection / (mask1.sum() + mask2.sum())

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found. Ensure YOLOv12-seg is fine-tuned first.")
        # Fallback for testing purposes
        MODEL_PATH_LOAD = 'yolov12n-seg.pt'
    else:
        MODEL_PATH_LOAD = MODEL_PATH

    try:
        model = YOLO(MODEL_PATH_LOAD)
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return

    images = glob(os.path.join(VAL_IMG_DIR, '*.jpg'))
    results = []
    
    for img_path in images:
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        lbl_path = os.path.join(VAL_LBL_DIR, f"{name}.txt")
        
        if not os.path.exists(lbl_path): continue
        
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        
        gt_mask = np.zeros((H, W), dtype=np.uint8)
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                points = [float(p) for p in parts[1:]]
                poly = np.array(points).reshape(-1, 2)
                poly[:, 0] *= W
                poly[:, 1] *= H
                cv2.fillPoly(gt_mask, [poly.astype(np.int32)], 1)
                
        preds = model(img_path, device='mps', verbose=False)[0]
        pred_mask = np.zeros((H, W), dtype=np.uint8)
        if preds.masks is not None:
            for m in preds.masks.data.cpu().numpy():
                m_resized = cv2.resize(m, (W, H))
                pred_mask = np.logical_or(pred_mask, m_resized > 0.5).astype(np.uint8)
        
        iou = compute_iou(gt_mask, pred_mask)
        dice = compute_dice(gt_mask, pred_mask)
        results.append({'filename': filename, 'iou': iou, 'dice': dice})
        
    if not results:
        print("No validation data found to evaluate.")
        return
        
    df = pd.DataFrame(results)
    print("\n=== YOLOv12 Segmentation Evaluation Report ===")
    print(f"Average IoU: {df['iou'].mean():.4f}")
    print(f"Average Dice Coefficient: {df['dice'].mean():.4f}\n")
    
    print("Top 5 Worst Performing Images:")
    worst = df.sort_values('iou').head(5)
    for _, row in worst.iterrows():
        print(f"- {row['filename']}: IoU = {row['iou']:.4f}")

if __name__ == '__main__':
    main()
