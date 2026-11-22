
import streamlit as st
import subprocess
import tempfile
import os

st.set_page_config(page_title="AI Auto Clipper")

st.title("🎬 AI Auto Clipper")
st.write("Turn Long Videos Into Short Clips")

video = st.file_uploader(
    "Upload Your Video",
    type=["mp4", "mov", "mkv"]
)

duration = st.number_input(
    "Clip Duration (Seconds)",
    min_value=10,
    max_value=60,
    value=30
)

if video and st.button("Create Clips"):
    st.info("Video uploaded successfully!")
    st.write("Selected clip duration:", duration, "seconds")
