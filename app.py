import streamlit as st
from src.prediction import predict_emotion

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
    emotion =predict_emotion(picture)
    st.subheader("This is your emotion right now", emotion)

