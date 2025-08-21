import streamlit as st
from keras.layers import TFSMLayer
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import io

# Disable scientific notation
np.set_printoptions(suppress=True)

# Load model once
@st.cache_resource
def load_model():
    model = TFSMLayer("saved_model/model.savedmodel", call_endpoint="serving_default")
    return model

# Load labels once
@st.cache_resource
def load_labels():
    return [line.strip() for line in open("model/labels.txt", "r").readlines()]

model = load_model()
class_names = load_labels()

# Streamlit UI
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image and the model will classify it as a **cat** or **dog**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict
    prediction = model(data)
    if isinstance(prediction, dict):
        prediction = list(prediction.values())[0]

    prediction = prediction.numpy()
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display result
    st.success(f"Prediction: **{class_name}** ({confidence_score*100:.2f}%)")

    # Annotate image
    draw = ImageDraw.Draw(image_resized)
    text = f"{class_name} ({confidence_score*100:.2f}%)"

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw.rectangle([(0, 0), (223, 30)], fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255), font=font)

    # Show annotated image
    st.image(image_resized, caption="Result", use_column_width=True)

    # Option to download result
    buf = io.BytesIO()
    image_resized.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    st.download_button(
        label="Download Result Image",
        data=byte_im,
        file_name="result.jpg",
        mime="image/jpeg"
    )
