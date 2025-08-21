# 🐶🐱 Dog vs Cat Image Classification

This project demonstrates a simple **image classification** task using a model trained on **Teachable Machine** with two classes: **Dog** and **Cat**.  
The dataset (~6000 images) was collected Roboflow

---
# How the Model Was Trained (Teachable Machine)

Base Model: Transfer learning with a pre-trained CNN backbone (MobileNetV2 / EfficientNetLite) trained on ImageNet.

Training Strategy:

- The feature extractor layers of the base model are frozen (not updated).

- A new dense classification head (fully connected layer + softmax) was added and trained with the dataset.

Hyperparameters:

- Epochs: 25 → The model was trained for 25 full passes over the dataset, enough for convergence without heavy overfitting.

- Batch size: 16 → Each gradient update used 16 images at once, balancing between stability and speed on limited hardware.

- Learning rate: 0.001 → A small learning rate ensured stable training, preventing the optimizer from overshooting minima.

- Optimizer: Adam (adaptive learning rate optimization, widely used for deep learning).

- Loss Function: Binary cross-entropy (since there are 2 classes: Dog vs Cat).

Dataset: ~6000 labeled images (balanced between Dogs and Cats).

Export: The trained model was exported in TensorFlow SavedModel format, compatible with Python, TensorFlow.js, and TensorFlow Lite.

## 📂 Project Structure
```bash
├── dataset/ # Training & validation data
├── savedmode/ # Saved TensorFlow model (exported from Teachable Machine)
│ ├── model.savedmodel/ # Model folder
│ └── labels.txt # Class labels
├── evaluate.py # Evaluate model performance
├── load_savemode.py # Classify new images by path
├── requirements.txt # Python dependencies
└── README.md # Documentation
```

## 🚀 Setup & Installation
### 1. Clone this repository and move into the project directory:
```bash
   git clone <your-repo-url>
   cd GO_code_test1_Vision_internAI
```
### 2. (Optional but recommended) Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate      
```
### 3. Install dependencies:
```bash
pip install -r requirements.txt
```
## 📊 Evaluate the Model
To evaluate the model on your dataset:
```bash
python evaluate.py
```
<img width="566" height="406" alt="Screenshot 2025-08-21 105349" src="https://github.com/user-attachments/assets/fadef4b0-74c1-4e35-bdca-e57a26f800b2" />

This script will load the trained model from savedmode/ and print evaluation metrics (accuracy, loss, etc.) on the validation set.
## 🖼️ Classify a New Image
To classify a single image (Dog or Cat) using the saved model:
```bash
# Preprocess the image
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open("Image_Path").convert("RGB") #Replace imagepath to classify
```
Replace with your image path to classify the image



