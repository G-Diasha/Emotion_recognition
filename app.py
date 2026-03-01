import streamlit as st
import numpy as np
import cv2
from src.prediction import predict_emotion

st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="centered"
)
st.title("😊 Emotion Recognition Model")

'''
uploaded_file = st.file_uploader(
    "Upload images", accept_multiple_files="directory", type=["jpg", "png"]
)
for uploaded_file in uploaded_file:
    st.image(uploaded_file)
'''
enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)
if picture:
    st.image(picture)
    bytes_value = picture.getvalue()
    np_array = np.frombuffer(bytes_value, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    st.image(img, channels='BGR')
    emotion =predict_emotion(img)
    st.markdown(emotion)

