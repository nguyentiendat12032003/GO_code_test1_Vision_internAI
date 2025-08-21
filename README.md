# 🐶🐱 Dog vs Cat Image Classification

This project demonstrates a simple **image classification** task using a model trained on **Teachable Machine** with two classes: **Dog** and **Cat**.  
The dataset (~6000 images) was collected Roboflow

---

## 📂 Project Structure
│── dataset/ # Training & validation data (from Roboflow)
│── model/ # Save TensorFlow h5
│── savedmode/ # Saved TensorFlow SavedModel format
│── evaluate.py # Script to evaluate the trained model
│── load_savemode.py # Script to classify new images by path
│── requirements.txt # Python dependencies
│── README.md # Project documentation

## 🚀 Setup & Installation

###1. Clone this repository and move into the project directory:
```bash
   git clone <your-repo-url>
   cd GO_code_test1_Vision_internAI
```
###2. (Optional but recommended) Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate      
```
###3. Install dependencies:
```bash
pip install -r requirements.txt
```
##📊 Evaluate the Model
To evaluate the model on your dataset:
```bash
python evaluate.py
```
<img width="566" height="406" alt="Screenshot 2025-08-21 105349" src="https://github.com/user-attachments/assets/fadef4b0-74c1-4e35-bdca-e57a26f800b2" />

This script will load the trained model from savedmode/ and print evaluation metrics (accuracy, loss, etc.) on the validation set.
##🖼️ Classify a New Image
To classify a single image (Dog or Cat) using the saved model:
```bash
python load_savemode.py --image <path-to-image>
```



