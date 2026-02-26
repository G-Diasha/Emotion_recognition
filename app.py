import streamlit as st

uploaded_file = st.file_uploader(
    "Upload images", accept_multiple_files="directory", type=["jpg", "png"]
)
for uploaded_file in uploaded_file:
    st.image(uploaded_file)