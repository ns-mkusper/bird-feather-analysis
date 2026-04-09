# 🪶 How to Teach the AI to Find Feathers (The Fine-Tuning Guide)

Right now, the base AI is a blank slate. It knows what dogs and cars are, but it doesn't know exactly what your specific taped feathers look like. We need to give it an "Answer Key" (called Ground Truth) for a small batch of images. 

Because drawing outlines around 500 feathers by hand sounds miserable, we built a script to do 90% of the work for you. Once you teach it on just ~100 images, the AI will lock in and process the other 1,900 images automatically with >99% accuracy!

Here is the exact, step-by-step guide to get your custom AI up and running:

### Step 1: Generate the Rough Drafts (Auto-Labeling)
Instead of starting from scratch, we will use a massive AI (Meta's SAM 2) to take a really good guess at the feather outlines.
1. Open your Jupyter Lab browser window (`http://10.0.0.148:8889/lab`).
2. At the bottom of the Launcher screen, click **Terminal** to open a black command box.
3. Type this exact command and hit Enter: 
   `python3 src/prep_annotation_sam2.py`
4. Wait a few minutes. It will grab 100 random photos and generate rough outlines for them, saving them in the `data/autolabel/` folder.

### Step 2: Upload to the Visual Editor (Roboflow)
Now you just need to visually check the AI's math.
1. Go to **[Roboflow.com](https://roboflow.com)** and create a free account.
2. Click **Create New Project**. Set the Project Type to **Instance Segmentation** and name what you are detecting (e.g., "Feather").
3. It will ask you to upload data. Drag and drop your newly generated `data/autolabel/images` folder AND the `data/autolabel/labels` folder directly into the browser. 
4. Roboflow will automatically marry the AI's rough outlines to the photos!

### Step 3: Grade the AI's Homework (The Manual Step)
1. Click through the uploaded images in Roboflow.
2. You will see polygons drawn over the feathers. If the AI accidentally included a piece of tape, or missed a fluffy piece of the feather, just click the dots on the outline and drag them to snap tightly around the feather. 
3. *Goal:* Make sure all 5 feathers are perfectly traced, ignoring the tape and background. (Put on a podcast, this is the only manual part of the whole project!)

### Step 4: Export the "Answer Key"
1. Once your images are approved, click **Generate Version** on the left sidebar. 
2. Click **Export Dataset** and select **YOLOv11 / YOLOv12** format. 
3. Download the `.zip` file it gives you.
4. Extract that zip file and place its contents back onto the compute cluster into your `data/autolabel/` folder. *(You will know you did it right if there is a file called `data.yaml` or `dataset.yaml` in there).*

### Step 5: Train Your Custom AI
Now the compute cluster takes over to actually "learn" from your answer key.
1. Go back to Jupyter Lab.
2. Open the `notebooks/train_yolov12_seg.ipynb` file.
3. Click the `▶` **Run** button at the top (or click *Run -> Run All Cells*).
4. The compute cluster will now spend an hour or two studying your perfect examples. When it finishes, it saves a new, custom AI brain called **`best.pt`** in the `models/` folder.

### Step 6: Run the Final Pipeline!
1. Open the main `minimal_slice_native.ipynb` notebook. 
2. Click **Run All Cells**.
3. *Magic:* The notebook will immediately see your custom `best.pt` file, realize it doesn't need to use the generic fallback AI anymore, and perfectly slice your feathers out of the image!
