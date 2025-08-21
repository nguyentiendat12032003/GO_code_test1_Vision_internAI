from keras.layers import TFSMLayer
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model from SavedModel using TFSMLayer
model = TFSMLayer("savedmode/model.savedmodel", call_endpoint="serving_default")

# Load the labels
class_names = open("model/labels.txt", "r").readlines()

# Preprocess the image
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open("dataset/valid/cat/cat-1_jpg.rf.dfdaf1fded89a64424977ee56ea087eb.jpg").convert("RGB")

# Resize and crop to 224x224
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Convert to numpy array
image_array = np.asarray(image)

# Normalize (Teachable Machine expects [-1,1])
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load into batch
data[0] = normalized_image_array

# Run prediction
prediction = model(data)

# If model returns a dict, get the first value
if isinstance(prediction, dict):
    print("Available output keys:", prediction.keys())  # Debug
    prediction = list(prediction.values())[0]

# Convert tensor to numpy
prediction = prediction.numpy()

# Get the predicted class
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name.strip())
print("Confidence Score:", confidence_score)
