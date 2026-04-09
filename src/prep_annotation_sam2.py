import os
import cv2
import numpy as np
import random
from glob import glob
from ultralytics import SAM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, 'data', 'autolabel', 'images')
OUTPUT_LBL_DIR = os.path.join(BASE_DIR, 'data', 'autolabel', 'labels')

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

def main():
    all_images = glob(os.path.join(INPUT_DIR, '*.jpg')) + glob(os.path.join(INPUT_DIR, '*.JPG'))
    if not all_images:
        print("No raw images found.")
        return
        
    sample_images = random.sample(all_images, min(100, len(all_images)))

    print(f"Loading Grounding DINO and SAM 2.1 for high-precision auto-labeling {len(sample_images)} images...")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from PIL import Image
    
    dino_id = 'IDEA-Research/grounding-dino-base'
    processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)
    sam_model = SAM('sam2.1_b.pt')

    for img_path in sample_images:
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)

        img_cv = cv2.imread(img_path)
        H, W = img_cv.shape[:2]
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, filename), img_cv)

        try:
            img_pil = Image.open(img_path).convert('RGB')
            inputs = processor(images=img_pil, text='bird feather.', return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = dino_model(**inputs)
            
            det_results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, target_sizes=[img_pil.size[::-1]]
            )[0]
            
            boxes = [b.tolist() for s, b in zip(det_results['scores'], det_results['boxes']) if s > 0.45]
            
            if boxes:
                sam_results = sam_model(img_path, bboxes=boxes, device=device, verbose=False)[0]
                label_path = os.path.join(OUTPUT_LBL_DIR, f"{name}.txt")
                
                with open(label_path, 'w') as f:
                    if sam_results.masks is not None:
                        for m in sam_results.masks.data.cpu().numpy():
                            m = cv2.resize(m.astype(np.float32), (W, H))
                            binary_mask = (m > 0.5).astype(np.uint8) * 255
                            
                            cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if cnts:
                                c = max(cnts, key=cv2.contourArea)
                                if cv2.contourArea(c) > 5000:
                                    x, y, w, h = cv2.boundingRect(c)
                                    y_max = y + h - int(h * 0.12) # Trim tape automatically!
                                    
                                    trimmed_mask = np.zeros_like(binary_mask)
                                    cv2.drawContours(trimmed_mask, [c], -1, 255, -1)
                                    trimmed_mask[y_max:, :] = 0
                                    
                                    t_cnts, _ = cv2.findContours(trimmed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    if t_cnts:
                                        tc = max(t_cnts, key=cv2.contourArea)
                                        poly = [f"{pt[0][0]/W:.5f} {pt[0][1]/H:.5f}" for pt in tc]
                                        f.write("0 " + " ".join(poly) + "\n")
        except Exception as e:
            print(f"DINO+SAM inference failed for {filename}: {e}")
                            
    print(f"Auto-labeling complete. Review in Roboflow/CVAT: {os.path.join(BASE_DIR, 'data', 'autolabel')}")

if __name__ == '__main__':
    main()
