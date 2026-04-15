import streamlit as st
import numpy as np
import cv2
import tf_keras as keras
from PIL import Image
import gdown
import os

# Page config
st.set_page_config(
    page_title="Lung Cancer Detector",
    page_icon="🫁",
    layout="centered"
)

# CSS styling
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .title { text-align: center; font-size: 2.5rem; font-weight: bold; color: #1a1a2e; }
    .subtitle { text-align: center; color: #555; margin-bottom: 2rem; }
    .result-box { padding: 1.5rem; border-radius: 12px; text-align: center; font-size: 1.3rem; font-weight: bold; margin-top: 1rem; }
    .cancer { background-color: #ffe0e0; color: #c0392b; border: 2px solid #c0392b; }
    .normal { background-color: #e0ffe0; color: #27ae60; border: 2px solid #27ae60; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🫁 Lung Cancer Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a lung histopathology image to detect cancer</div>', unsafe_allow_html=True)

# -----------------------------------------------
# Model load karo — Google Drive se
# -----------------------------------------------
MODEL_PATH = "lung_cancer_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... please wait ⏳"):
            # Apna Google Drive file ID yahan daalo
            file_id = "https://drive.google.com/file/d/1tiazW0ro3XRMWq2bKNC775ZMu5VfRwEZ/view?usp=sharing"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, MODEL_PATH, quiet=False)
    model = keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

classes = ['lung_aca', 'lung_n', 'lung_scc']
class_labels = {
    'lung_aca': '🔴 Lung Adenocarcinoma (Cancer)',
    'lung_n':   '🟢 Normal Lung Tissue',
    'lung_scc': '🔴 Lung Squamous Cell Carcinoma (Cancer)'
}

IMG_SIZE = 128

# -----------------------------------------------
# Upload section
# -----------------------------------------------
st.markdown("---")
uploaded_file = st.file_uploader("📁 Upload Histopathology Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image dikhao
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # Predict
    with st.spinner("Analyzing image... 🔍"):
        pred = model.predict(np.expand_dims(img_resized, axis=0))
        predicted_class = classes[np.argmax(pred)]
        confidence = np.max(pred) * 100

    # Result dikhao
    label = class_labels[predicted_class]
    box_class = "normal" if predicted_class == "lung_n" else "cancer"

    st.markdown(f"""
        <div class="result-box {box_class}">
            {label}<br>
            <span style="font-size:1rem; font-weight:normal;">Confidence: {confidence:.2f}%</span>
        </div>
    """, unsafe_allow_html=True)

    # Probabilities
    st.markdown("### 📊 All Class Probabilities:")
    for i, cls in enumerate(classes):
        st.progress(float(pred[0][i]), text=f"{class_labels[cls]}: {pred[0][i]*100:.2f}%")

    st.markdown("---")
    st.warning("⚠️ This tool is for educational purposes only. Not a substitute for medical diagnosis.")
