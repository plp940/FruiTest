# 🍎 Fruit Freshness Classifier Streamlit App (Follows User Rules)

import streamlit as st
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
from utils_gradcam import make_gradcam_heatmap, overlay_heatmap_on_image
from utils_predict import interpret_predictions_with_rules
import matplotlib.pyplot as plt
import cv2

st.set_page_config(page_title="Fruit Freshness Classifier", layout="wide")

# ✅ Load model once
@st.cache_resource
def load_model_resnet():
    return load_model("final_resnet.h5")

model = load_model_resnet()

# ✅ Class names mapping
class_names = {
    0: 'FreshApple', 1: 'FreshBanana', 2: 'FreshGrape', 3: 'FreshGuava',
    4: 'FreshJujube', 5: 'FreshOrange', 6: 'FreshPomegranate', 7: 'FreshStrawberry',
    8: 'RottenApple', 9: 'RottenBanana', 10: 'RottenGrape', 11: 'RottenGuava',
    12: 'RottenJujube', 13: 'RottenOrange', 14: 'RottenPomegranate', 15: 'RottenStrawberry'
}

def get_category(label):
    return 'Fresh' if 'Fresh' in label else 'Rotten'

def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict_category(image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    top_indices = preds.argsort()[-2:][::-1]
    top_preds = [(class_names[i], preds[i]) for i in top_indices]
    return top_preds

def generate_message(top_preds):
    label1, conf1 = top_preds[0]
    label2, conf2 = top_preds[1]
    cat1, cat2 = get_category(label1), get_category(label2)

    conf1_pct = conf1 * 100
    conf2_pct = conf2 * 100

    if conf1_pct >= 80:
        msg = f"Prediction: **{cat1}** ({conf1_pct:.2f}%)\n"
        msg += "✅ Seems good to eat!" if cat1 == "Fresh" else "⚠️ Do not eat!"
        return msg

    elif cat1 != cat2:
        msg = f"Top 2 Predictions:\n- {cat1}: {conf1_pct:.2f}%\n- {cat2}: {conf2_pct:.2f}%\n"
        if cat1 == "Rotten":
            return msg + "⚠️ Rotten detected! Better not eat."
        else:
            return msg + "💡 Seems to be fresh, but handle with caution."

    elif cat1 == cat2:
        msg = f"Prediction: **{cat1}** ({conf1_pct:.2f}%)\n"
        if cat1 == "Rotten":
            return msg + "⚠️ Looks Rotten. Avoid eating."
        else:
            return msg + "💡 Seems to be fresh, but confidence is low."

# ✅ Streamlit Tabs
tabs = st.tabs(["📖 About", "🖼️ Predict", "📷 Live Camera"])

# 📖 Tab 1: About
with tabs[0]:
    st.title("🍏 Fruit Freshness Classifier")
    st.markdown("""
    This app predicts whether your fruit is **Fresh** or **Rotten** using a trained deep learning model (ResNet50).

    #### 🔍 How It Works:
    - Upload or capture fruit images
    - App shows predictions based on strict rules to ensure clarity
    - Works with individual or multiple photos
                
    #### Tab 2: Predict
    - Upload one or more images to get freshness predictions
    - Displays top 2 predictions with confidence scores
    - Uses Grad-CAM to visualize model focus areas
    - Provides clear, actionable messages based on predictions
    #### Tab 3: camera
    - Use your device's camera to take a photo
    - Predictions related to the photo taken
                
    #### 📱 How to Use as PWA:
    - Open in mobile browser
    - Tap 3 dots → *Add to Home Screen*
    - Use as app like experience!
    """)

# 🖼️ Tab 2: Predict (Upload)
with tabs[1]:
    st.header("Upload Fruit Images for Prediction")
    st.markdown("""
    #### Features:
    - **Upload one or more images of a single fruit** to predict freshness.
    - **Upload multiple images of different fruits** to get batch predictions.
    """)
       
    files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files:
        st.write("### Results")
        for file in files:
            image = Image.open(file).convert("RGB")
            st.image(image, caption=file.name, width=300)

            preds = predict_category(image)
            st.markdown(generate_message(preds))

           # heatmap = make_gradcam_heatmap(preprocess_image, model, last_conv_layer_name="conv5_block3_out")
            #overlay = overlay_heatmap_on_image(np.array(image.resize((224, 224))), heatmap)
            #st.image(overlay, caption="🔍 Grad-CAM: Model Focus", width=300)


with tabs[2]:
    st.header("📷 Take a Photo with Camera")
    st.markdown("""
    Use your device's camera to take a photo **only when you're ready**.

    ✅ Predictions follow the strict 5-rule freshness logic.
    """)

    # Session state init
    if "show_camera" not in st.session_state:
        st.session_state.show_camera = False
    if "captured" not in st.session_state:
        st.session_state.captured = False

    # Start camera button
    if not st.session_state.show_camera:
        if st.button("📷 Start Camera"):
            st.session_state.show_camera = True
            st.session_state.captured = False
            st.rerun()

    # Show camera
    if st.session_state.show_camera and not st.session_state.captured:
        captured_photo = st.camera_input("📸 Capture a fruit photo")

        if captured_photo:
            st.session_state.captured = True
            st.image(captured_photo, caption="📸 Captured Image", width=300)

            # Preprocess and predict
            image = Image.open(captured_photo).convert("RGB")
            processed_image = preprocess_image(image)
            preds = model.predict(processed_image)[0]

            # Get top 2 predictions
            top_indices = preds.argsort()[-2:][::-1]
            top_preds = [(class_names[i], preds[i]) for i in top_indices]

            # Apply 5-rule logic
            def get_category(label):
                return 'Fresh' if 'Fresh' in label else 'Rotten'

            label1, conf1 = top_preds[0]
            label2, conf2 = top_preds[1]
            cat1, cat2 = get_category(label1), get_category(label2)

            conf1_pct = conf1 * 100
            conf2_pct = conf2 * 100

            if conf1_pct >= 80:
                msg = f"Prediction: **{cat1}** ({conf1_pct:.2f}%)\n"
                msg += "✅ Seems good to eat!" if cat1 == "Fresh" else "⚠️ Do not eat!"
            elif cat1 != cat2:
                msg = f"Top 2 Predictions:\n- {cat1}: {conf1_pct:.2f}%\n- {cat2}: {conf2_pct:.2f}%\n"
                msg += "⚠️ Rotten detected! Better not eat." if cat1 == "Rotten" else "💡 Seems to be fresh, but handle with caution."
            elif cat1 == cat2:
                msg = f"Prediction: **{cat1}** ({conf1_pct:.2f}%)\n"
                msg += "⚠️ Looks Rotten. Avoid eating." if cat1 == "Rotten" else "💡 Seems to be fresh, but confidence is low."

            st.markdown(msg)

            # ✅ Optional Grad-CAM
            heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name="conv5_block3_out")
            img_array = np.array(image.resize((224, 224)))
            gradcam_img = overlay_heatmap_on_image(img_array, heatmap)
            st.image(gradcam_img, caption="🔥 Grad-CAM (Model Focus Area)", width=300)

    # Stop/Retake
    if st.session_state.show_camera:
        if st.button("🛑 Stop / Retake"):
            st.session_state.show_camera = False
            st.session_state.captured = False
            st.rerun()