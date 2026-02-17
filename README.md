# 🐾 Human & Animal Detection – Offline Vision System

## 📌 Project Overview

This project implements a fully offline **Human & Animal Detection system** using:

- ✅ Custom Convolutional Neural Network (CNN)
- ✅ Haar Cascade (Human face detection)
- ✅ Selective Search (Animal region proposals)
- ✅ Streamlit Web Application
- ✅ Standalone Inference Script (`main.py`)
- ❌ No YOLO
- ❌ No COCO / ImageNet training datasets
- ❌ No cloud-based APIs

The system detects and classifies Humans and Animals in images using classical computer vision combined with deep learning.

---

```bash
project/
├── datasets/
│   └── train/
│       ├── human/
│       └── animal/
├── models/
│   └── classifier.pth
├── test_videos/
├── outputs/
│   ├── output_image.jpg
│   └── sample_output.json
├── main.py
├── app.py
├── requirements.txt
└── README.md
```
---

# 🧠 Dataset Justification

### Why NOT COCO?
- COCO is a large-scale object detection dataset with 80+ classes.
- Requires bounding box annotations.
- Heavy and unnecessary for binary classification.
- Violates assignment constraint.

### Why NOT ImageNet?
- Generic multi-class dataset.
- Heavy pretrained dependency.
- Not optimized for Human vs Animal binary task.

### Why Custom Dataset?
- Focused binary classification (Human vs Animal).
- Lightweight and controlled.
- Balanced classes.
- Fully offline training.
- Faster experimentation and debugging.

Dataset Format:
datasets/train/human/
datasets/train/animal/


---

# 🧠 Model Selection Justification

Selected Model: **Custom CNN**

Reasons:
- Lightweight architecture
- Fast training
- Low memory usage
- CPU compatible
- No internet dependency
- Suitable for binary classification

Architecture:
- 3 Convolution layers
- ReLU activation
- MaxPooling
- Fully connected layers
- Sigmoid output

Loss Function:
Binary Cross Entropy (BCELoss)


Optimizer:
Adam (lr = 0.001)


Epochs:
10


YOLO was avoided because:
- Prohibited in assignment
- Heavy detection model
- Unnecessary complexity

---

# ⚙️ Training Process

1. Images resized to 128x128
2. Normalization applied
3. Loaded using PyTorch ImageFolder
4. Batch size = 16
5. Trained for 10 epochs
6. Model saved as:

models/classifier.pth


---

# 🔍 Inference Pipeline (Step-by-Step)

Implemented in both:

- `main.py` → Command Line Inference
- `app.py` → Streamlit Web UI

### Step 1: Human Detection
- Uses OpenCV Haar Cascade
- Detects faces in image

### Step 2: Animal Region Proposals
- Uses Selective Search
- Generates candidate regions

### Step 3: Classification
Each detected region:
- Resized to 128x128
- Normalized
- Passed through CNN
- Threshold applied:
  - prob > 0.7 → Human
  - prob < 0.3 → Animal

### Step 4: Output Generation
- Bounding boxes drawn
- Image saved in `/outputs/`
- JSON structured output generated

---

# 📊 Evaluation Metrics

For classification:

- Accuracy
- Precision
- Recall
- F1 Score

For detection (if extended):

- mAP@0.5 (Mean Average Precision)
- IoU threshold = 0.5

mAP is mentioned for completeness as standard detection evaluation metric.

---

# 🔒 Offline Compliance

- No YOLO
- No COCO dataset
- No ImageNet dependency
- No Cloud APIs
- No internet required
- No runtime model downloads

All models load locally from:
models/classifier.pth


---

# 💻 Hardware Constraints

Minimum Requirements:

- 8GB RAM
- Intel i5 or equivalent
- CPU supported
- GPU optional (auto-detected)

Optimizations:

- Image resizing (128x128)
- Lightweight CNN
- Limited region proposals
- Batch size control

System is designed to work in low-resource offline environments.

---

# ⚙️ Installation Instructions

## Step 1: Create Virtual Environment (Recommended)

python -m venv venv


Activate:

Windows:
venv\Scripts\activate


Mac/Linux:
source venv/bin/activate


---

## Step 2: Install Dependencies

pip install -r requirements.txt


---

## Step 3: Ensure Dataset Exists

datasets/train/human/
datasets/train/animal/


---

# 🚀 Running the Application

## Option 1: Run Streamlit App

streamlit run app.py


Upload an image and click **Run Detection**.

Output saved to:
outputs/output_image.jpg


---

## Option 2: Run Inference via CLI

python main.py


Outputs will be saved in:
outputs/


---

# 📦 Sample Output (JSON Format)

Example `sample_output.json`:

```json
{
  "file_name": "sample_image.jpg",
  "detections": [
    {
      "label": "Human",
      "confidence": 0.91
    },
    {
      "label": "Animal",
      "confidence": 0.87
    }
  ]
}
⚠️ Important Notes
Requires opencv-contrib-python for Selective Search.

First run may train model if classifier.pth does not exist.
However the classifier.pth model 🔗link is provided in 
models folder📁 --> classifier_drive_path.pth 📝

Training time: 2–5 minutes on CPU.

🧩 Challenges Faced
False positives from selective search

Threshold tuning for binary classification

Avoiding heavy detection frameworks

Maintaining full offline functionality

CPU performance optimization

🔮 Possible Improvements
Add Non-Maximum Suppression (NMS)

Add bounding box evaluation with mAP

Add video frame-by-frame detection

Improve dataset diversity

Add confusion matrix visualization

👩‍💻 Author
Human & Animal Detection – Offline Vision System
Assignment Submission


---

If you want, I can also generate:

- 📄 Professional Documentation (DOC/PDF content)
- 📦 Final Submission Checklist Summary
- ⭐ GitHub-ready version with badges and formatting

Just tell me.
