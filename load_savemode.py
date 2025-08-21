import argparse
from keras.layers import TFSMLayer
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np

# Disable scientific notation
np.set_printoptions(suppress=True)

# CLI arguments
parser = argparse.ArgumentParser(description="Image classification with SavedModel")
parser.add_argument("-imagepath", type=str, required=True, help="Path to the input image")
args = parser.parse_args()

# Load model
model = TFSMLayer("saved_model/model.savedmodel", call_endpoint="serving_default")

# Load labels
class_names = open("saved_model/labels.txt", "r").readlines()

# Preprocess image
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open(args.imagepath).convert("RGB")

# Resize
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Convert to numpy array
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data[0] = normalized_image_array

# Predict
prediction = model(data)
if isinstance(prediction, dict):
    prediction = list(prediction.values())[0]

prediction = prediction.numpy()

# Get class + score
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]

print("Class:", class_name)
print("Confidence Score:", confidence_score)

# === Draw result on image ===
draw = ImageDraw.Draw(image)

# Text content
text = f"{class_name} ({confidence_score*100:.2f}%)"

# Optional: font
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# Draw black rectangle behind text
draw.rectangle([(0, 0), (223, 30)], fill=(0, 0, 0))
draw.text((5, 5), text, fill=(255, 255, 255), font=font)

# Save & show
image.save("result.jpg")
image.show()
