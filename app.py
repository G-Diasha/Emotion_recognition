import streamlit as st

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