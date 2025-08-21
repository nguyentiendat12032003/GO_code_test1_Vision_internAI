from keras.layers import TFSMLayer
from PIL import Image, ImageOps
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load model
model = TFSMLayer("saved_model/model.savedmodel", call_endpoint="serving_default")

# Load labels
class_names = [line.strip() for line in open("savedmode/labels.txt", "r").readlines()]
print("Classes:", class_names)

# Validation dataset path
val_dir = "dataset/valid"

# Parameters
img_size = (224, 224)

y_true = []
y_pred = []

# Duyệt qua từng class folder
for idx, class_name in enumerate(class_names):
    class_dir = os.path.join(val_dir, class_name)
    for file in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file)
        try:
            # Load and preprocess image
            image = Image.open(file_path).convert("RGB")
            image = ImageOps.fit(image, img_size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.expand_dims(normalized_image_array, axis=0)  # shape (1,224,224,3)

            # Predict
            prediction = model(data)
            if isinstance(prediction, dict):  # nếu output là dict
                prediction = list(prediction.values())[0]
            prediction = prediction.numpy()

            # Predicted label
            pred_index = np.argmax(prediction)
            y_pred.append(pred_index)
            y_true.append(idx)
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {file_path}: {e}")

# Chuyển về numpy
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Evaluation report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Accuracy
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted")  # hoặc "macro"
rec = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\n--- Evaluation Metrics ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")