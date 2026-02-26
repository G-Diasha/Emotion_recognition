import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="centered"
)
st.title("😊 Emotion Recognition Model")

uploaded_file = st.file_uploader(
    "Upload images", accept_multiple_files="directory", type=["jpg", "png"]
)
for uploaded_file in uploaded_file:
    st.image(uploaded_file)
enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)
if picture:
    st.image(picture)
    bytes_data = picture.getvalue()
    img_tensor = tf.io.decode_image(bytes_data, channels = 3)
    