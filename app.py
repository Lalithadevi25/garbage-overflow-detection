import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2
import imageio_ffmpeg
import subprocess
import gc

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Garbage Overflow Detection",
    page_icon="🗑️",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🗑️ Garbage Overflow Detection System")

st.write(
    "YOLOv8-based Normal and Overflow Detection"
)

# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# =========================================================
# VIDEO UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload a garbage video",
    type=["mp4", "avi", "mov"]
)

# =========================================================
# IF VIDEO IS UPLOADED
# =========================================================

if uploaded_file is not None:

    st.success("✅ Video uploaded successfully!")

    # =====================================================
    # SAVE INPUT VIDEO
    # =====================================================

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

    # =====================================================
    # DISPLAY INPUT VIDEO
    # =====================================================

    st.subheader("🎥 Input Video")

    st.video(input_file.name)

    # =====================================================
    # DETECT BUTTON
    # =====================================================

    if st.button("🔍 Detect Garbage Overflow"):

        st.info(
            "Processing video... Please wait."
        )

        # =================================================
        # OPEN INPUT VIDEO
        # =================================================

        cap = cv2.VideoCapture(
            input_file.name
        )

        if not cap.isOpened():

            st.error(
                "❌ Unable to open the uploaded video."
            )

            st.stop()

        # =================================================
        # GET VIDEO PROPERTIES
        # =================================================

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25

        original_width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        original_height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        # =================================================
        # PRESERVE ORIGINAL ASPECT RATIO
        # =================================================

        max_width = 720

        if original_width > max_width:

            output_width = max_width

            output_height = int(
                original_height
                * output_width
                / original_width
            )

        else:

            output_width = original_width
            output_height = original_height

        # -------------------------------------------------
        # Make dimensions even for H.264 compatibility
        # -------------------------------------------------

        output_width = output_width - (
            output_width % 2
        )

        output_height = output_height - (
            output_height % 2
        )

        # =================================================
        # CREATE TEMPORARY OUTPUT
        # =================================================

        temp_output = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = temp_output.name

        temp_output.close()

        # =================================================
        # VIDEO WRITER
        # =================================================

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (
                output_width,
                output_height
            )
        )

        if not writer.isOpened():

            cap.release()

            st.error(
                "❌ Unable to create output video."
            )

            st.stop()

        # =================================================
        # DETECTION VARIABLES
        # =================================================

        overflow_found = False

        consecutive_overflow = 0

        required_consecutive_frames = 3

        frame_count = 0

        # =================================================
        # PROGRESS BAR
        # =================================================

        progress = st.progress(0)

        # =================================================
        # PROCESS VIDEO
        # =================================================

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            # =============================================
            # RESIZE WHILE PRESERVING ASPECT RATIO
            # =============================================

            frame = cv2.resize(
                frame,
                (
                    output_width,
                    output_height
                ),
                interpolation=cv2.INTER_AREA
            )

            # =============================================
            # YOLO PREDICTION
            # =============================================

            result = model.predict(
                source=frame,
                conf=0.5,
                verbose=False
            )[0]

            # =============================================
            # CHECK OVERFLOW
            # =============================================

            frame_has_overflow = False

            if result.boxes is not None:

                for cls in result.boxes.cls:

                    class_name = model.names[
                        int(cls)
                    ]

                    if (
                        class_name.lower()
                        == "overflow"
                    ):

                        frame_has_overflow = True

                        break

            # =============================================
            # CONSECUTIVE FRAME LOGIC
            # =============================================

            if frame_has_overflow:

                consecutive_overflow += 1

            else:

                consecutive_overflow = 0

            if (
                consecutive_overflow
                >= required_consecutive_frames
            ):

                overflow_found = True

            # =============================================
            # DRAW YOLO PREDICTIONS
            # =============================================

            annotated_frame = result.plot()

            # =============================================
            # SAFETY: MAINTAIN OUTPUT SIZE
            # =============================================

            if (
                annotated_frame.shape[1]
                != output_width
                or
                annotated_frame.shape[0]
                != output_height
            ):

                annotated_frame = cv2.resize(
                    annotated_frame,
                    (
                        output_width,
                        output_height
                    ),
                    interpolation=cv2.INTER_AREA
                )

            # =============================================
            # WRITE FRAME
            # =============================================

            writer.write(
                annotated_frame
            )

            # =============================================
            # UPDATE PROGRESS
            # =============================================

            if total_frames > 0:

                progress_value = (
                    frame_count
                    / total_frames
                )

                progress.progress(
                    min(
                        progress_value,
                        1.0
                    )
                )

        # =================================================
        # RELEASE RESOURCES
        # =================================================

        cap.release()

        writer.release()

        progress.progress(1.0)

        gc.collect()

        # =================================================
        # CONVERT OUTPUT TO H.264 MP4
        # =================================================

        st.info(
            "Preparing browser-compatible video..."
        )

        browser_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        browser_video_path = (
            browser_video.name
        )

        browser_video.close()

        ffmpeg_path = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )

        # =================================================
        # FFMPEG COMMAND
        # =================================================

        command = [

            ffmpeg_path,

            "-y",

            "-i",
            output_path,

            # Video codec
            "-c:v",
            "libx264",

            # Fast encoding
            "-preset",
            "veryfast",

            # Quality
            "-crf",
            "23",

            # Browser compatible pixel format
            "-pix_fmt",
            "yuv420p",

            # Preserve original display aspect ratio
            "-aspect",
            f"{output_width}:{output_height}",

            # Better browser playback
            "-movflags",
            "+faststart",

            # Remove audio
            "-an",

            browser_video_path
        ]

        try:

            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

        except subprocess.CalledProcessError:

            st.error(
                "❌ Video conversion failed."
            )

            cap.release()

            if os.path.exists(output_path):
                os.remove(output_path)

            st.stop()

        # =================================================
        # SUCCESS
        # =================================================

        st.success(
            "✅ Prediction completed!"
        )

        # =================================================
        # DETECTION RESULT
        # =================================================

        st.subheader(
            "📊 Detection Result"
        )

        # =================================================
        # READ OUTPUT VIDEO
        # =================================================

        with open(
            browser_video_path,
            "rb"
        ) as video_file:

            video_bytes = video_file.read()

        # =================================================
        # DISPLAY OUTPUT VIDEO
        # =================================================

        st.video(
            video_bytes,
            format="video/mp4"
        )

        # =================================================
        # DOWNLOAD BUTTON
        # =================================================

        st.download_button(

            label="⬇️ Download Detection Result",

            data=video_bytes,

            file_name=
            "garbage_detection_result.mp4",

            mime="video/mp4"
        )

        # =================================================
        # FINAL RESULT
        # =================================================

        st.subheader(
            "📢 Detection Status"
        )

        if overflow_found:

            st.error(
                "🚨 OVERFLOW DETECTED!"
            )

            st.warning(
                "Garbage overflow has been detected. "
                "Immediate collection/cleaning is recommended."
            )

        else:

            st.success(
                "🟢 No Overflow Detected"
            )

        # =================================================
        # CLEAN TEMPORARY FILES
        # =================================================

        try:

            os.remove(
                input_file.name
            )

            os.remove(
                output_path
            )

            # Don't delete browser_video_path
            # immediately because Streamlit needs it
            # for the displayed/downloaded video.

        except Exception:

            pass
