import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
model = load_model("hand_gesture_cnn.h5")

# Define image size (same as training)
img_size = (64, 64)

# Correct class labels (from train_generator.class_indices)
class_labels = ['fist', 'five', 'none', 'okay', 'peace', 'rad', 'straight', 'thumbs']

st.title("🖐 Hand Gesture Recognition")
st.write("Upload a hand gesture image and let the model predict it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess image (must match training preprocessing)
    img = load_img(uploaded_file, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    
    threshold = 80

    if confidence < threshold:
        st.subheader("❓ Unknown Gesture")
        st.write(f"Confidence too low ({confidence:.2f}%)")
    else:
        st.subheader(f"✅ Predicted Gesture: {class_labels[class_index]}")
        st.write(f"Confidence: {confidence:.2f}%")
