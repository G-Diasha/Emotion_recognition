import streamlit as st
import numpy as np
import cv2
from PIL import Image
from src.prediction import predict_emotion

st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="centered"
)
st.title("😊 Emotion Recognition App")
st.markdown(
    "<h2 style='color:purple; text-align:center;'>Upload an image and the model will predict your true emotion.</h3>",
    unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg","jpeg","png"]
)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    bytes_value = uploaded_file.getvalue()
    np_array = np.frombuffer(bytes_value, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    emotion =predict_emotion(img)
    st.success(f"Predicted Emotion: {emotion}")

