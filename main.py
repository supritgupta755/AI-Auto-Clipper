import streamlit as st
import subprocess
import tempfile
import os
import zipfile
import imageio_ffmpeg

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

    with tempfile.TemporaryDirectory() as folder:
        input_path = os.path.join(folder, "input.mp4")

        with open(input_path, "wb") as f:
            f.write(video.getbuffer())

        output_path = os.path.join(folder, "clip_%03d.mp4")

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg,
            "-i", input_path,
            "-map", "0",
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(duration),
            "-reset_timestamps", "1",
            output_path
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True
            )

            clips = [
                f for f in os.listdir(folder)
                if f.startswith("clip_")
            ]

            if clips:
                zip_path = os.path.join(folder, "clips.zip")

                with zipfile.ZipFile(zip_path, "w") as z:
                    for clip in clips:
                        z.write(
                            os.path.join(folder, clip),
                            clip
                        )

                with open(zip_path, "rb") as f:
                    st.download_button(
                        "Download All Clips",
                        f,
                        file_name="AI_Clips.zip"
                    )
            else:
                st.error("No clips created.")

        except Exception as e:
            st.error(f"Error: {e}")
