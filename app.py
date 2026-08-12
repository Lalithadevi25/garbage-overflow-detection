import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2
import gc
import imageio_ffmpeg
import subprocess


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Garbage Overflow Detection",
    page_icon="🗑️",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🗑️ Garbage Overflow Detection System")

st.write(
    "YOLOv8-based Normal and Overflow Detection"
)


# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# ==========================================
# VIDEO UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📤 Upload a garbage video",
    type=["mp4", "avi", "mov"]
)


# ==========================================
# IF VIDEO IS UPLOADED
# ==========================================

if uploaded_file is not None:

    st.success("✅ Video uploaded successfully!")

    # --------------------------------------
    # Save uploaded video
    # --------------------------------------

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

    # --------------------------------------
    # Display Input Video
    # --------------------------------------

    st.subheader("🎥 Input Video")

    st.video(input_file.name)


    # ======================================
    # DETECT BUTTON
    # ======================================

    if st.button("🔍 Detect Garbage Overflow"):

        st.info(
            "Processing video... Please wait."
        )


        # ----------------------------------
        # Temporary output file
        # ----------------------------------

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_file.name

        output_file.close()


        # ==================================
        # OPEN INPUT VIDEO
        # ==================================

        cap = cv2.VideoCapture(
            input_file.name
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25


        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )


        # ==================================
        # RESIZE VIDEO
        # ==================================

        max_width = 720

        if width > max_width:

            new_width = max_width

            new_height = int(
                height * max_width / width
            )

        else:

            new_width = width
            new_height = height


        # ==================================
        # VIDEO WRITER
        # ==================================

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (new_width, new_height)
        )


        # ==================================
        # DETECTION VARIABLES
        # ==================================

        overflow_found = False

        consecutive_overflow = 0

        required_consecutive_frames = 3

        frame_count = 0


        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        # ==================================
        # PROGRESS BAR
        # ==================================

        progress = st.progress(0)


        # ==================================
        # PROCESS VIDEO FRAME BY FRAME
        # ==================================

        while True:

            success, frame = cap.read()

            if not success:
                break


            frame_count += 1


            # --------------------------------
            # Process every 2nd frame
            # --------------------------------

            if frame_count % 2 != 0:

                frame = cv2.resize(
                    frame,
                    (new_width, new_height)
                )

                writer.write(frame)

                continue


            # --------------------------------
            # Resize frame
            # --------------------------------

            frame = cv2.resize(
                frame,
                (new_width, new_height)
            )


            # --------------------------------
            # YOLO prediction
            # --------------------------------

            result = model.predict(
                source=frame,
                conf=0.5,
                verbose=False
            )[0]


            frame_has_overflow = False


            # =================================
            # CHECK OVERFLOW
            # =================================

            if result.boxes is not None:

                for cls in result.boxes.cls:

                    class_name = model.names[
                        int(cls)
                    ]

                    if class_name.lower() == "overflow":

                        frame_has_overflow = True

                        break


            # =================================
            # CONSECUTIVE FRAME LOGIC
            # =================================

            if frame_has_overflow:

                consecutive_overflow += 1

            else:

                consecutive_overflow = 0


            if (
                consecutive_overflow
                >= required_consecutive_frames
            ):

                overflow_found = True


            # =================================
            # DRAW PREDICTIONS
            # =================================

            annotated_frame = result.plot()

            writer.write(
                annotated_frame
            )


            # =================================
            # UPDATE PROGRESS
            # =================================

            if total_frames > 0:

                progress_value = min(
                    frame_count / total_frames,
                    1.0
                )

                progress.progress(
                    progress_value
                )


        # ==================================
        # RELEASE VIDEO RESOURCES
        # ==================================

        cap.release()

        writer.release()

        progress.progress(1.0)

        gc.collect()


        # ==================================
        # CONVERT TO H264 MP4
        # ==================================

        st.info(
            "Preparing browser-compatible video..."
        )


        browser_video = output_path.replace(
            ".mp4",
            "_browser.mp4"
        )


        ffmpeg_path = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )


        command = [

            ffmpeg_path,

            "-y",

            "-i",
            output_path,

            "-c:v",
            "libx264",

            "-preset",
            "veryfast",

            "-crf",
            "28",

            "-pix_fmt",
            "yuv420p",

            "-movflags",
            "+faststart",

            "-an",

            browser_video
        ]


        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )


        st.success(
            "✅ Prediction completed!"
        )


        # ==================================
        # DETECTION RESULT
        # ==================================

        st.subheader(
            "📊 Detection Result"
        )


        # ----------------------------------
        # Read browser-compatible video
        # ----------------------------------

        with open(
            browser_video,
            "rb"
        ) as video_file:

            video_bytes = video_file.read()


        # ----------------------------------
        # Display output video
        # ----------------------------------

        st.video(
            video_bytes,
            format="video/mp4"
        )


        # ==================================
        # DOWNLOAD BUTTON
        # ==================================

        st.download_button(
            label="⬇️ Download Detection Result",

            data=video_bytes,

            file_name=
            "garbage_detection_result.mp4",

            mime="video/mp4"
        )


        # ==================================
        # FINAL OVERFLOW RESULT
        # ==================================

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


        # ==================================
        # CLEAN TEMPORARY FILES
        # ==================================

        try:

            os.remove(
                input_file.name
            )

            os.remove(
                output_path
            )

            os.remove(
                browser_video
            )

        except Exception:

            pass
