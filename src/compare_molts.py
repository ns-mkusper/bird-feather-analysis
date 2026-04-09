import os
import pandas as pd
import torch
import open_clip
from PIL import Image
from torch.nn.functional import cosine_similarity
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(BASE_DIR, 'data', 'metadata_manifest.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, 'molt_analysis_results.csv')

def main():
    if not os.path.exists(MANIFEST_PATH):
        print("Manifest not found. Run extract_all_metadata.py first.")
        return
        
    df = pd.read_csv(MANIFEST_PATH)
    
    print("Loading BioCLIP for Siamese Network Comparison...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2.5-vith14')
    model.to(device).eval()

    results = []
    
    # Identify longitudinal pairs
    for bird_id, group in df.groupby('bird_id'):
        if len(group) > 1 and bird_id not in ["UNKNOWN", "ERROR"]:
            group = group.sort_values('date')
            dates = group['date'].tolist()
            
            # Compare Gen 1 to Gen 2 (Feather 1)
            crop1 = os.path.join(PROCESSED_DIR, f"{bird_id}_{dates[0]}_Feather_1.jpg")
            crop2 = os.path.join(PROCESSED_DIR, f"{bird_id}_{dates[1]}_Feather_1.jpg")
            
            # Simulated Identical Control: "Zero-Distance" baseline
            crop_control1 = crop1
            crop_control2 = crop1
            
            if os.path.exists(crop1) and os.path.exists(crop2):
                print(f"Comparing molts for Bird ID: {bird_id}")
                img1 = preprocess(Image.open(crop1)).unsqueeze(0).to(device)
                img2 = preprocess(Image.open(crop2)).unsqueeze(0).to(device)
                
                img_ctrl = preprocess(Image.open(crop1)).unsqueeze(0).to(device)
                
                with torch.no_grad(), torch.autocast("mps"):
                    emb_c = model.encode_image(img_ctrl)
                    emb_c /= emb_c.norm(dim=-1, keepdim=True)
                    ctrl_sim = cosine_similarity(emb_c, emb_c).item()
                    print(f"[{bird_id}] ZERO-DISTANCE CONTROL TEST -> Similarity: {ctrl_sim:.5f} | Change Score: {1.0 - ctrl_sim:.5f}")

                    emb1 = model.encode_image(img1)
                    emb2 = model.encode_image(img2)
                    emb1 /= emb1.norm(dim=-1, keepdim=True)
                    emb2 /= emb2.norm(dim=-1, keepdim=True)
                    
                    cos_sim = cosine_similarity(emb1, emb2).item()
                    # Calculate Euclidean Distance
                    euc_dist = torch.cdist(emb1, emb2).item()
                    
                results.append({
                    "bird_id": bird_id,
                    "gen1_date": dates[0],
                    "gen2_date": dates[1],
                    "cosine_similarity": round(cos_sim, 5),
                    "euclidean_distance": round(euc_dist, 5)
                })
                
                print(f"[{bird_id}] Change Score: {1.0 - cos_sim:.5f}")
                
                # R Handoff: If the similarity is low (drastic change), generate advanced transition plot
                if cos_sim < 0.98: # Threshold for drift detection
                    print(f"Significant phenotypic drift detected! Launching Advanced R-Visualizer for {bird_id}...")
                    pdf_out = os.path.join(RESULTS_DIR, f"{bird_id}_molt_transition.pdf")
                    r_script_path = os.path.join(BASE_DIR, 'src', 'visualize_transition.R')
                    
                    env = os.environ.copy()
                    env['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
                    env['PATH'] = env['R_HOME'] + '/bin:' + env['PATH']
                    
                    subprocess.run(['Rscript', r_script_path, crop1, crop2, pdf_out, bird_id], env=env)

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_CSV, index=False)
    print(f"Phase 2 Comparison complete! Results saved to {RESULTS_CSV}")

if __name__ == "__main__":
    main()
