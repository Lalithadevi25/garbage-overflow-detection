import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EcoBin AI",
    page_icon="🗑️",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best.pt"

OVERFLOW_CLASS = "overflow"
NORMAL_CLASS = "normal"


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "location" not in st.session_state:
    st.session_state.location = "Ramachandrapuram Municipal area"

if "alert" not in st.session_state:
    st.session_state.alert = None


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-bottom: 2rem;
    }


    /* ========================================================
       PAGE 1
       ======================================================== */

    .home-box {
        width: 100%;
        border: 3px solid black;
        background: white;
        color: black;
    }

    .home-header {
        height: 125px;
        border-bottom: 3px solid black;

        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;

        text-align: center;
    }

    .home-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .home-subtitle {
        font-size: 20px;
        font-weight: 600;
        margin-top: 8px;
    }

    .home-main {
        display: grid;
        grid-template-columns: 37% 63%;
        min-height: 680px;
    }

    .home-left {
        border-right: 3px solid black;
        text-align: center;
        padding-top: 110px;
    }

    .aicw {
        font-size: 28px;
        font-weight: 700;
        line-height: 1.5;
    }

    .capstone {
        font-size: 23px;
        font-weight: 600;
        margin-top: 35px;
    }

    .home-right {
        display: grid;
        grid-template-rows: 90px 340px 1fr;
    }

    .title-section {
        border-bottom: 3px solid black;
        padding: 20px 25px;
    }

    .title-section h2 {
        margin: 0;
        font-size: 27px;
    }

    .description-section {
        border-bottom: 3px solid black;
        padding: 20px 25px;
    }

    .description-section h2 {
        margin: 0 0 15px 0;
        font-size: 24px;
    }

    .description {
        font-size: 16px;
        line-height: 1.65;
        text-align: justify;
    }

    .bottom-section {
        display: grid;
        grid-template-columns: 60% 40%;
    }

    .team-section {
        border-right: 3px solid black;
        padding: 25px;
    }

    .guide-section {
        padding: 25px;
    }

    .section-heading {
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .member {
        font-size: 16px;
        margin-bottom: 14px;
    }

    .guide-name {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .guide-designation {
        font-size: 16px;
    }


    /* ========================================================
       PAGE 2
       ======================================================== */

    .page2-header {
        width: 100%;
        border: 3px solid black;
        text-align: center;
        padding: 18px;
        box-sizing: border-box;
    }

    .page2-header h1 {
        margin: 0;
        font-size: 36px;
        font-weight: 800;
    }

    .page2-header p {
        margin: 8px 0 0 0;
        font-size: 19px;
        font-weight: 600;
    }


    .box {
        border: 3px solid black;
        padding: 15px;
        min-height: 300px;
        background: white;
        color: black;
    }

    .box-heading {
        text-align: center;
        font-size: 22px;
        font-weight: 800;
        border-bottom: 2px solid black;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }


    /* ========================================================
       ALERT
       ======================================================== */

    .alert-message {
        border: 3px solid #b00020;
        background: #fff4f4;
        color: #8b0000;
        padding: 20px;
        margin-top: 20px;
        margin-bottom: 20px;
        font-size: 17px;
        font-weight: 700;
    }


    /* ========================================================
       NORMAL
       ======================================================== */

    .normal-message {
        border: 3px solid green;
        background: #f3fff3;
        color: green;
        padding: 20px;
        margin-top: 20px;
        margin-bottom: 20px;
        font-size: 17px;
        font-weight: 700;
    }


    @media (max-width: 800px) {

        .home-main {
            grid-template-columns: 1fr;
        }

        .home-left {
            border-right: none;
            border-bottom: 3px solid black;
            padding-bottom: 40px;
        }

        .home-right {
            grid-template-rows: auto auto auto;
        }

        .bottom-section {
            grid-template-columns: 1fr;
        }

        .team-section {
            border-right: none;
            border-bottom: 3px solid black;
        }

        .home-title {
            font-size: 32px;
        }

        .home-subtitle {
            font-size: 16px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TIME
# ============================================================

def current_time():

    return datetime.now().strftime(
        "%d-%m-%Y %I:%M:%S %p"
    )


# ============================================================
# OVERFLOW CHECK
# ============================================================

def is_overflow(result):

    if result.boxes is None:
        return False

    if len(result.boxes) == 0:
        return False

    for box in result.boxes:

        class_id = int(box.cls[0])

        class_name = str(
            model.names[class_id]
        ).lower().strip()

        if class_name == OVERFLOW_CLASS:
            return True

    return False


# ============================================================
# ALERT
# ============================================================

def set_alert():

    st.session_state.alert = {
        "location": st.session_state.location,
        "time": current_time(),
        "status": "Garbage Overflow Detected"
    }


def clear_alert():

    st.session_state.alert = None


# ============================================================
# SHOW ALERT
# ============================================================

def show_alert():

    if st.session_state.alert is None:
        return

    alert = st.session_state.alert

    st.markdown(
        f"""
        <div class="alert-message">

            🚨 ALERT MESSAGE

            <br><br>

            <b>Garbage Overflow Detected!</b>

            <br><br>

            📍 Location:
            {alert["location"]}

            <br><br>

            🕒 Date & Time:
            {alert["time"]}

            <br><br>

            ⚠️ Status:
            {alert["status"]}

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# IMAGE DETECTION
# ============================================================

def detect_image(image):

    if model is None:

        st.error(
            "best.pt file not found. "
            "Please keep best.pt in the same folder as app.py."
        )

        return None, False

    image_array = np.array(image)

    results = model.predict(
        source=image_array,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    output = result.plot()

    overflow = is_overflow(result)

    return output, overflow


# ============================================================
# VIDEO PROCESSING
# ============================================================

def detect_video(video_file):

    if model is None:

        st.error(
            "best.pt file not found."
        )

        return None, False

    input_path = None
    output_path = None

    try:

        extension = os.path.splitext(
            video_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp:

            temp.write(
                video_file.read()
            )

            input_path = temp.name


        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():

            st.error(
                "Could not open video."
            )

            return None, False


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

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        output_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_temp.name

        output_temp.close()


        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )


        overflow_detected = False

        frame_count = 0

        progress = st.progress(0)


        while True:

            success, frame = cap.read()

            if not success:
                break


            results = model.predict(
                source=frame,
                conf=0.25,
                verbose=False
            )

            result = results[0]


            if is_overflow(result):

                overflow_detected = True


            output_frame = result.plot()

            writer.write(
                output_frame
            )


            frame_count += 1

            if total_frames > 0:

                progress.progress(
                    min(
                        frame_count / total_frames,
                        1.0
                    )
                )


        cap.release()

        writer.release()

        progress.empty()


        return output_path, overflow_detected


    except Exception as e:

        st.error(
            f"Video processing error: {e}"
        )

        return None, False


    finally:

        if (
            input_path
            and os.path.exists(input_path)
        ):

            try:
                os.remove(input_path)
            except:
                pass


# ============================================================
# PAGE 1
# ============================================================

def home_page():

    st.markdown(
        """
        <div class="home-box">

            <div class="home-header">

                <div class="home-title">
                    ECOBIN AI
                </div>

                <div class="home-subtitle">
                    Smart Garbage Overflow Detection System
                </div>

            </div>


            <div class="home-main">


                <div class="home-left">

                    <div class="aicw">

                        AI Career for Women
                        <br>
                        (AICW)

                    </div>


                    <div class="capstone">
                        Capstone Project
                    </div>

                </div>


                <div class="home-right">


                    <div class="title-section">

                        <h2>
                            TITLE
                        </h2>

                    </div>


                    <div class="description-section">

                        <h2>
                            DESCRIPTION
                        </h2>

                        <div class="description">

                            EcoBin AI is an intelligent Smart Garbage
                            Overflow Detection System designed to identify
                            overflowing garbage bins automatically using
                            Artificial Intelligence and computer vision.
                            The system uses a trained YOLOv8 deep learning
                            model to analyze camera images, uploaded images
                            and videos and detect garbage overflow conditions.
                            When an overflow violation is detected, EcoBin AI
                            generates an alert containing the detection
                            status, location and date and time. This system
                            can help municipalities and sanitation teams
                            monitor waste collection points, respond quickly
                            to overflowing bins and improve cleanliness.
                            The solution supports automated monitoring and
                            helps create cleaner and smarter communities.

                        </div>

                    </div>


                    <div class="bottom-section">


                        <div class="team-section">

                            <div class="section-heading">
                                TEAM MEMBERS
                            </div>

                            <div class="member">
                                1. Member Name — member1@email.com
                            </div>

                            <div class="member">
                                2. Member Name — member2@email.com
                            </div>

                            <div class="member">
                                3. Member Name — member3@email.com
                            </div>

                            <div class="member">
                                4. Member Name — member4@email.com
                            </div>

                        </div>


                        <div class="guide-section">

                            <div class="section-heading">
                                GUIDE
                            </div>

                            <div class="guide-name">
                                Guide Name
                            </div>

                            <div class="guide-designation">
                                Guide Designation
                            </div>

                        </div>


                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    # PREDICT button
    # Positioned below the left side approximately like the reference

    left, middle, right = st.columns(
        [37, 15, 48]
    )

    with left:

        if st.button(
            "PREDICT",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = 2

            st.rerun()


# ============================================================
# PAGE 2
# ============================================================

def detection_page():

    st.markdown(
        """
        <div class="page2-header">

            <h1>
                ECOBIN AI
            </h1>

            <p>
                Smart Garbage Overflow Detection System
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    if st.button(
        "⬅️ Back to Home"
    ):

        clear_alert()

        st.session_state.page = 1

        st.rerun()


    # ========================================================
    # LOCATION
    # ========================================================

    st.subheader(
        "📍 Detection Location"
    )


    st.session_state.location = st.text_input(
        "Enter the garbage-bin / camera location",
        value=st.session_state.location
    )


    st.divider()


    # ========================================================
    # CAMERA
    # ========================================================

    st.header("📷 Camera")


    camera1, camera2, camera3 = st.columns(
        3
    )


    with camera1:

        st.markdown(
            '<div class="box"><div class="box-heading">Camera</div>',
            unsafe_allow_html=True
        )

        camera_photo = st.camera_input(
            "Take Photo",
            key="camera_input"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with camera2:

        st.markdown(
            '<div class="box"><div class="box-heading">Input</div>',
            unsafe_allow_html=True
        )

        if camera_photo:

            camera_image = Image.open(
                camera_photo
            )

            st.image(
                camera_image,
                use_container_width=True
            )

        else:

            st.info(
                "Take a photo using the camera."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with camera3:

        st.markdown(
            '<div class="box"><div class="box-heading">Output</div>',
            unsafe_allow_html=True
        )

        if camera_photo:

            camera_image = Image.open(
                camera_photo
            )

            camera_output, camera_overflow = detect_image(
                camera_image
            )

            if camera_output is not None:

                st.image(
                    camera_output,
                    channels="RGB",
                    use_container_width=True
                )

            if camera_overflow:

                set_alert()

            else:

                st.session_state.alert = None

                st.success(
                    "✅ No Garbage Overflow Detected"
                )

        else:

            st.info(
                "Detection output will appear here."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    if camera_photo and st.session_state.alert:

        show_alert()


    # ========================================================
    # IMAGE
    # ========================================================

    st.divider()

    st.header("🖼️ Image")


    image1, image2, image3 = st.columns(
        3
    )


    with image1:

        st.markdown(
            '<div class="box"><div class="box-heading">Upload Image</div>',
            unsafe_allow_html=True
        )

        uploaded_image = st.file_uploader(
            "Choose Image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],
            key="image_upload"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with image2:

        st.markdown(
            '<div class="box"><div class="box-heading">Input</div>',
            unsafe_allow_html=True
        )

        if uploaded_image:

            image = Image.open(
                uploaded_image
            )

            st.image(
                image,
                use_container_width=True
            )

        else:

            st.info(
                "Upload an image."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with image3:

        st.markdown(
            '<div class="box"><div class="box-heading">Output</div>',
            unsafe_allow_html=True
        )

        if uploaded_image:

            image = Image.open(
                uploaded_image
            )

            image_output, image_overflow = detect_image(
                image
            )

            if image_output is not None:

                st.image(
                    image_output,
                    channels="RGB",
                    use_container_width=True
                )

            if image_overflow:

                set_alert()

            else:

                st.session_state.alert = None

                st.success(
                    "✅ No Garbage Overflow Detected"
                )

        else:

            st.info(
                "Detection output will appear here."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    if uploaded_image and st.session_state.alert:

        show_alert()


    # ========================================================
    # VIDEO
    # ========================================================

    st.divider()

    st.header("🎥 Video")


    video1, video2, video3 = st.columns(
        3
    )


    with video1:

        st.markdown(
            '<div class="box"><div class="box-heading">Upload Video</div>',
            unsafe_allow_html=True
        )

        uploaded_video = st.file_uploader(
            "Choose Video",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv"
            ],
            key="video_upload"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with video2:

        st.markdown(
            '<div class="box"><div class="box-heading">Input</div>',
            unsafe_allow_html=True
        )

        if uploaded_video:

            st.video(
                uploaded_video
            )

        else:

            st.info(
                "Upload a video."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with video3:

        st.markdown(
            '<div class="box"><div class="box-heading">Output</div>',
            unsafe_allow_html=True
        )

        if uploaded_video:

            if st.button(
                "🔍 Detect Overflow",
                key="detect_video"
            ):

                with st.spinner(
                    "Processing video..."
                ):

                    result_video, video_overflow = detect_video(
                        uploaded_video
                    )


                if result_video:

                    with open(
                        result_video,
                        "rb"
                    ) as f:

                        video_bytes = f.read()


                    st.video(
                        video_bytes
                    )


                    st.download_button(
                        "⬇️ Download Result",
                        data=video_bytes,
                        file_name="ecobin_result.mp4",
                        mime="video/mp4"
                    )


                    if video_overflow:

                        set_alert()

                    else:

                        st.session_state.alert = None

                        st.success(
                            "✅ No Garbage Overflow Detected"
                        )

        else:

            st.info(
                "Detection output will appear here."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    if st.session_state.alert:

        show_alert()


# ============================================================
# APP ROUTING
# ============================================================

if st.session_state.page == 1:

    home_page()

else:

    detection_page()
