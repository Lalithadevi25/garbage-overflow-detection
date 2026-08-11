```python
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import subprocess
import imageio_ffmpeg


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Garbage Overflow Detection",
    page_icon="🗑️",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🗑️ Garbage Overflow Detection System")
st.write("YOLOv8-based Normal and Overflow Detection")


# -----------------------------
# Load Model
# -----------------------------

MODEL_PATH = "best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# -----------------------------
# Upload Video
# -----------------------------

uploaded_file = st.file_uploader(
    "📤 Upload a garbage video",
    type=["mp4", "avi", "mov"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    st.success("Video uploaded successfully!")

    # Save uploaded video
    temp_input = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_input.write(uploaded_file.read())
    temp_input.close()

    st.subheader("🎥 Input Video")
    st.video(temp_input.name)

    # Predict button
    if st.button("🔍 Detect Garbage Overflow"):

        st.info("Processing video... Please wait.")

        output_dir = tempfile.mkdtemp()

        # YOLO prediction
        results = model.predict(
            source=temp_input.name,
            save=True,
            conf=0.5,
            project=output_dir,
            name="prediction"
        )

        # Find predicted video
        predicted_files = []

        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv")
                ):
                    predicted_files.append(
                        os.path.join(root, file)
                    )

        if predicted_files:

            output_video = predicted_files[0]

            # -----------------------------
            # Convert output to browser-friendly MP4
            # -----------------------------

            converted_video = os.path.join(
                output_dir,
                "garbage_detection_result.mp4"
            )

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

            command = [
                ffmpeg_path,
                "-y",
                "-i",
                output_video,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                converted_video
            ]

            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            st.success("✅ Prediction completed!")

            # -----------------------------
            # Detection Result
            # -----------------------------

            st.subheader("📊 Detection Result")

            with open(converted_video, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(
                video_bytes,
                format="video/mp4"
            )

            # Download button
            st.download_button(
                label="⬇️ Download Detection Result",
                data=video_bytes,
                file_name="garbage_detection_result.mp4",
                mime="video/mp4"
            )

            # -----------------------------
            # Check Overflow
            # -----------------------------

            overflow_found = False

            for result in results:

                if result.boxes is not None:

                    for cls in result.boxes.cls:

                        class_name = model.names[int(cls)]

                        if class_name.lower() == "overflow":
                            overflow_found = True
                            break

                if overflow_found:
                    break

            # -----------------------------
            # Final Result
            # -----------------------------

            if overflow_found:

                st.error("🚨 OVERFLOW DETECTED!")

                st.warning(
                    "Garbage overflow has been detected. "
                    "Immediate collection/cleaning is recommended."
                )

            else:

                st.success("🟢 No Overflow Detected")

        else:

            st.error(
                "❌ Prediction video could not be generated."
            )
```
