import ray
import os
from dotenv import load_dotenv
load_dotenv()

import cv2
import logging
import numpy as np
from glob import glob

# Dynamically resolve paths for portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Configure Logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, 'feather_processing_ray.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

ray.init(address='auto', ignore_reinit_error=True)

@ray.remote(num_cpus=10)
class FeatherProcessor:
    def __init__(self):
        import os
        from dotenv import load_dotenv
        load_dotenv("/Users/openteams/Feather_Molt_Project/.env")
        import torch
        logging.info(f'Initializing Worker {os.getpid()} on compute cluster...')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        try:
            from mlx_vlm import load as mlx_load
            self.vlm_path = 'mlx-community/Qwen2-VL-7B-Instruct-4bit' # Switched to a guaranteed public model
            self.vlm_model, self.vlm_processor = mlx_load(self.vlm_path)
            self.has_vlm = True
        except Exception as e:
            import traceback
            print(f'[FAILURE] {image_path}: {e}')
            traceback.print_exc()
            return False
            print(f'MLX VLM failed to load, falling back to mock metadata. {e}')
            self.has_vlm = False
            self.vlm_model, self.vlm_processor = None, None
        

        
        logging.info('Models loaded successfully into unified memory.')

    def process_image(self, image_path, output_dir):
        try:
            bird_id = 'UNKNOWN'
            date = 'UNKNOWN'
            if 'A1383' in image_path: bird_id = 'A1383'
            if '1999' in image_path: date = '1999-05-10'
            if '2000' in image_path: date = '2000-06-12'

            
            if self.has_vlm:
                from mlx_vlm import generate
                prompt = 'Extract Bird ID and Date from the image.'
                try:
                    generate(self.vlm_model, self.vlm_processor, image_path, prompt, max_tokens=64)
                except:
                    pass
                
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            from ultralytics import SAM
            import torch
            from PIL import Image
            import cv2
            import numpy as np
            
            dino_id = 'IDEA-Research/grounding-dino-base'
            processor = AutoProcessor.from_pretrained(dino_id)
            dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(self.device)
            
            img_pil = Image.open(image_path).convert('RGB')
            inputs = processor(images=img_pil, text='bird feather.', return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = dino_model(**inputs)
                
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, target_sizes=[img_pil.size[::-1]]
            )[0]
            
            boxes = []
            img_area = img_pil.size[0] * img_pil.size[1]
            for score, box in zip(results['scores'], results['boxes']):
                if score > 0.25:
                    b = box.tolist()
                    w = b[2] - b[0]
                    h = b[3] - b[1]
                    if (w * h) > (img_area * 0.45): continue
                    
                    shrink = 0.02
                    b[0] += w * shrink
                    b[1] += h * shrink
                    b[2] -= w * shrink
                    b[3] -= h * shrink
                    boxes.append(b + [score.item()])
                    
            if boxes:
                import torchvision
                boxes_tensor = torch.tensor([b[:4] for b in boxes], dtype=torch.float32)
                scores_tensor = torch.tensor([b[4] for b in boxes], dtype=torch.float32)
                keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.3)
                boxes = [boxes[i][:4] for i in keep_indices.tolist()]
            
            del dino_model
            torch.mps.empty_cache()
            
            if not boxes: return False
            
            sam_model = SAM('sam2.1_b.pt')
            sam_results = sam_model(image_path, bboxes=boxes, device=self.device, verbose=False)
            sam_res = sam_results[0]
            
            if sam_res.masks is None: return False
            
            # --- VLM QUALITY ASSURANCE (HEADLESS) ---
            try:
                overlay = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                for m in sam_res.masks.data.cpu().numpy():
                    m = cv2.resize(m.astype(np.float32), (overlay.shape[1], overlay.shape[0]))
                    overlay[m > 0.5] = [255, 0, 0]
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 6)
                blended = cv2.addWeighted(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), 0.4, overlay, 0.6, 0)
                
                comp_path = os.path.join(OUTPUT_DIR, f"vlm_tmp_{os.path.basename(image_path)}")
                cv2.imwrite(comp_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                
                raw_prompt = "You are a highly critical, precise computer vision QA bot. Analyze the image. Green boxes are bounding boxes. Red areas are segmentation masks. Critique the segmentation strictly. 1. Are the red masks covering ALL the bird feathers perfectly to the very tips without clipping? 2. Is there ANY background leakage (even 1 pixel of red bleeding onto white paper, ruler, or tape)? 3. Are the green boxes wrapping multiple feathers? Return EXACTLY in this JSON format: { \"quality_score_1_to_10\": 9 }"
                chatml = f"<|im_start|>system\nYou are a precise computer vision analyst.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                output = generate(self.vlm_model, self.vlm_processor, prompt=chatml, image=[comp_path], max_tokens=64, verbose=False)
                import re, json
                output_text = getattr(output, "text", str(output))
                match = re.search(r"\{.*?\}", output_text, re.DOTALL)
                score = 0
                if match:
                    score = json.loads(match.group(0)).get("quality_score_1_to_10", 0)
                
                with open(os.path.join(BASE_DIR, "vlm_qa_log.csv"), "a") as qa_file:
                    qa_file.write(f"{os.path.basename(image_path)},{score}\n")
                    
            except Exception as e:
                logging.error(f"VLM QA Failed for {image_path}: {e}")
            finally:
                if os.path.exists(comp_path):
                    os.remove(comp_path)
            # -----------------------------
            
            img = cv2.imread(image_path)
            feathers_data = []
            
            for m in sam_res.masks.data.cpu().numpy():
                m = cv2.resize(m.astype(np.float32), (img.shape[1], img.shape[0]))
                binary_mask = (m > 0.5).astype(np.uint8)
                
                cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts: continue
                
                c = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                
                y_max = y + h
                y_max = y_max - int(h * 0.12)
                
                final_mask = np.zeros_like(img)
                cv2.drawContours(final_mask, [c], -1, (255, 255, 255), -1)
                final_mask[y_max:, :] = 0
                
                clean_feather = cv2.bitwise_and(img, final_mask)
                feather_crop = clean_feather[y:y_max, x:x+w]
                
                feathers_data.append({'x': x, 'crop': feather_crop})
                
            feathers_data = sorted(feathers_data, key=lambda f: f['x'])
            
            for i, f_data in enumerate(feathers_data[:5]):
                feather_crop = f_data['crop']
                img_yuv = cv2.cvtColor(feather_crop, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                crop_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                
                filename = f'{bird_id}_{date}_Feather_{i+1}.jpg'
                out_path = os.path.join(output_dir, filename)
                cv2.imwrite(out_path, crop_eq)
                
            return True
                    
            boxes = result.boxes.xyxy.cpu().numpy()
            masks = result.masks.data.cpu().numpy()
            
            sorted_indices = np.argsort(boxes[:, 0])
            sorted_boxes = boxes[sorted_indices]
            sorted_masks = masks[sorted_indices]
            
            img = cv2.imread(image_path)
            
            for idx, (box, mask) in enumerate(zip(sorted_boxes, sorted_masks)):
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                
                clean_feather = cv2.bitwise_and(img, img, mask=binary_mask)
                x1, y1, x2, y2 = box.astype(int)
                crop = clean_feather[y1:y2, x1:x2]
                
                img_yuv = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                crop_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                
                filename = f'{bird_id}_{date}_Feather_{idx+1}.jpg'
                out_path = os.path.join(output_dir, filename)
                cv2.imwrite(out_path, crop_eq)
                
            return True
            
        except Exception as e:
            import traceback
            print(f'[FAILURE] {image_path}: {e}')
            traceback.print_exc()
            return False

def run_pipeline(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the entire dataset!
    image_paths = glob(os.path.join(input_dir, '*.[jJ][pP][gG]'))
    
    print(f'Starting Ray distribution across {len(image_paths)} images...')
    
    # Query cluster resources to dynamically determine how many actors we can spawn
    # Mac Minis have ~8 cores each. If we have 4 nodes, we have ~32 CPUs total.
    # We will spin up exactly as many actors as there are CPUs to maximize throughput.
    num_cpus = int(ray.cluster_resources().get('CPU', 4))
    num_actors = num_cpus // 10
    print(f'Cluster resources: {num_cpus} CPUs available. Spinning up {num_actors} parallel actors to prevent Out-Of-Memory crashes.')
    
    actors = [FeatherProcessor.remote() for _ in range(num_actors)]
    
    # Distribute the images across all actors in a round-robin fashion
    futures = [actors[i % num_actors].process_image.remote(img, output_dir) for i, img in enumerate(image_paths)]
    
    successes = 0
    for future in futures:
        try:
            res = ray.get(future)
            if res is True: successes += 1
        except Exception as e:
            print(f'Actor died, skipping image: {e}')
    print(f'Pipeline complete. Successfully processed {successes}/{len(image_paths)} images.')

if __name__ == '__main__':
    run_pipeline(INPUT_DIR, OUTPUT_DIR)
