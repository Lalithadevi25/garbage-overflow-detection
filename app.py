import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2
import gc

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

    st.success("✅ Video uploaded successfully!")

    # Save uploaded video directly to disk
    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    while True:
        chunk = uploaded_file.read(1024 * 1024)

        if not chunk:
            break

        input_file.write(chunk)

    input_file.close()

    st.subheader("🎥 Input Video")
    st.video(input_file.name)

    # -----------------------------
    # Predict Button
    # -----------------------------

    if st.button("🔍 Detect Garbage Overflow"):

        st.info("Processing video... Please wait.")

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_file.name
        output_file.close()

        # -----------------------------
        # Read video
        # -----------------------------

        cap = cv2.VideoCapture(input_file.name)

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Reduce resolution for efficiency
        max_width = 720

        if width > max_width:
            new_width = max_width
            new_height = int(height * max_width / width)
        else:
            new_width = width
            new_height = height

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (new_width, new_height)
        )

        # -----------------------------
        # Detection variables
        # -----------------------------

        overflow_found = False

        consecutive_overflow = 0

        required_consecutive_frames = 3

        frame_count = 0

        processed_count = 0

        # -----------------------------
        # Progress bar
        # -----------------------------

        progress = st.progress(0)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # -----------------------------
        # Process video frame-by-frame
        # -----------------------------

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            # Process every 2nd frame
            if frame_count % 2 != 0:
                writer.write(
                    cv2.resize(
                        frame,
                        (new_width, new_height)
                    )
                )
                continue

            processed_count += 1

            # Resize frame
            frame = cv2.resize(
                frame,
                (new_width, new_height)
            )

            # YOLO prediction
            result = model.predict(
                source=frame,
                conf=0.5,
                verbose=False
            )[0]

            frame_has_overflow = False

            if result.boxes is not None:

                for cls in result.boxes.cls:

                    class_name = model.names[int(cls)]

                    if class_name.lower() == "overflow":

                        frame_has_overflow = True
                        break

            # -----------------------------
            # Consecutive frame logic
            # -----------------------------

            if frame_has_overflow:

                consecutive_overflow += 1

            else:

                consecutive_overflow = 0

            if consecutive_overflow >= required_consecutive_frames:

                overflow_found = True

            # -----------------------------
            # Draw detections
            # -----------------------------

            annotated_frame = result.plot()

            writer.write(annotated_frame)

            # -----------------------------
            # Progress
            # -----------------------------

            if total_frames > 0:

                progress_value = min(
                    frame_count / total_frames,
                    1.0
                )

                progress.progress(progress_value)

        # -----------------------------
        # Release resources
        # -----------------------------

        cap.release()
        writer.release()

        progress.progress(1.0)

        gc.collect()

        st.success("✅ Prediction completed!")

        # -----------------------------
        # Detection Result
        # -----------------------------

        st.subheader("📊 Detection Result")

        st.video(output_path)

        # -----------------------------
        # Download Result
        # -----------------------------

        with open(output_path, "rb") as file:

            st.download_button(
                label="⬇️ Download Detection Result",
                data=file,
                file_name="garbage_detection_result.mp4",
                mime="video/mp4"
            )

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

            st.success(
                "🟢 No Overflow Detected"
            )

        # -----------------------------
        # Cleanup
        # -----------------------------

        try:

            os.remove(input_file.name)

            os.remove(output_path)

        except Exception:

            pass
