# 🐶🐱 Dog vs Cat Image Classification

This project demonstrates a simple **image classification** task using a model trained on **Teachable Machine** with two classes: **Dog** and **Cat**.  
The dataset (~6000 images) was collected Roboflow

---

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
python load_savemode.py -imagepath Image_Path_Here
```
Replace with your image path to classify the image
## Report:
[Dog and Cat Classification](https://github.com/nguyentiendat12032003/GO_code_test1_Vision_internAI/blob/main/Report_Ex1.md)

[TTS_Vietnamese](https://github.com/nguyentiendat12032003/GO_code_test1_Vision_internAI/blob/main/Propose_TTS.md)
## [Demo](https://drive.google.com/file/d/1DuWaFRqTTSq9d9w27Z20Q9serDlm_Pdv/view?usp=sharing) 







