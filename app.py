import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EcoBin AI",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="collapsed"
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
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GENERAL STREAMLIT
       ====================================================== */

    .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ======================================================
       PAGE 1 - HOME
       ====================================================== */

    .home-container {
        width: 100%;
        border: 3px solid #111111;
        background: white;
        color: #111111;
        box-sizing: border-box;
    }


    .home-header {
        width: 100%;
        min-height: 125px;
        border-bottom: 3px solid #111111;

        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;

        text-align: center;
        box-sizing: border-box;
    }


    .home-header-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: 2px;
    }


    .home-header-subtitle {
        font-size: 20px;
        font-weight: 600;
        margin-top: 8px;
    }


    .home-body {
        display: grid;
        grid-template-columns: 37% 63%;
        width: 100%;
        min-height: 650px;
    }


    .home-left {
        border-right: 3px solid #111111;
        padding: 50px 30px;
        text-align: center;
        box-sizing: border-box;
    }


    .aicw-text {
        font-size: 28px;
        font-weight: 700;
        line-height: 1.4;
        margin-top: 80px;
    }


    .project-text {
        font-size: 23px;
        font-weight: 600;
        margin-top: 35px;
    }


    .home-right {
        display: grid;
        grid-template-rows: 100px 300px 1fr;
        min-width: 0;
    }


    .title-area {
        border-bottom: 3px solid #111111;
        padding: 25px;
        box-sizing: border-box;
    }


    .title-area h2 {
        margin: 0;
        font-size: 28px;
    }


    .description-area {
        border-bottom: 3px solid #111111;
        padding: 25px;
        box-sizing: border-box;
    }


    .description-area h2 {
        margin-top: 0;
        font-size: 24px;
    }


    .description-text {
        font-size: 16px;
        line-height: 1.7;
        text-align: justify;
    }


    .bottom-area {
        display: grid;
        grid-template-columns: 60% 40%;
        min-height: 250px;
    }


    .team-area {
        border-right: 3px solid #111111;
        padding: 25px;
        box-sizing: border-box;
    }


    .guide-area {
        padding: 25px;
        box-sizing: border-box;
    }


    .section-heading {
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 20px;
    }


    .team-member {
        font-size: 16px;
        margin-bottom: 14px;
    }


    .guide-name {
        font-size: 18px;
        font-weight: 700;
        margin-top: 15px;
    }


    .guide-designation {
        font-size: 16px;
        margin-top: 8px;
    }


    /* ======================================================
       PAGE 2 HEADER
       ====================================================== */

    .page2-header {
        width: 100%;
        border: 3px solid #111111;
        padding: 18px;
        text-align: center;
        box-sizing: border-box;
        margin-bottom: 15px;
    }


    .page2-header h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 800;
    }


    .page2-header p {
        margin: 7px 0 0 0;
        font-size: 18px;
        font-weight: 600;
    }


    /* ======================================================
       DETECTION BOXES
       ====================================================== */

    .detection-box {
        border: 3px solid #111111;
        min-height: 300px;
        padding: 15px;
        box-sizing: border-box;
        background: white;
        color: #111111;
    }


    .detection-heading {
        text-align: center;
        font-size: 22px;
        font-weight: 800;

        border-bottom: 2px solid #111111;

        padding-bottom: 10px;
        margin-bottom: 15px;
    }


    /* ======================================================
       ALERT
       ====================================================== */

    .alert-box {
        border: 3px solid #b00020;

        padding: 18px;
        margin-top: 20px;
        margin-bottom: 20px;

        background: #fff5f5;
        color: #8b0000;

        font-size: 16px;
        font-weight: 700;
    }


    /* ======================================================
       SAFE MESSAGE
       ====================================================== */

    .safe-box {
        border: 3px solid #168516;

        padding: 18px;
        margin-top: 20px;
        margin-bottom: 20px;

        background: #f3fff3;
        color: #146b14;

        font-size: 16px;
        font-weight: 700;
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 800px) {

        .home-body {
            grid-template-columns: 1fr;
        }

        .home-left {
            border-right: none;
            border-bottom: 3px solid #111111;
        }

        .home-right {
            grid-template-rows: auto auto auto;
        }

        .bottom-area {
            grid-template-columns: 1fr;
        }

        .team-area {
            border-right: none;
            border-bottom: 3px solid #111111;
        }

        .home-header-title {
            font-size: 32px;
        }

        .home-header-subtitle {
            font-size: 16px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CURRENT DATE & TIME
# ============================================================

def get_current_datetime():

    return datetime.now().strftime(
        "%d-%m-%Y %I:%M:%S %p"
    )


# ============================================================
# CREATE ALERT
# ============================================================

def create_alert():

    st.session_state.alert = {
        "location": st.session_state.location,
        "time": get_current_datetime(),
        "status": "Garbage Overflow Detected"
    }


# ============================================================
# CLEAR ALERT
# ============================================================

def clear_alert():

    st.session_state.alert = None


# ============================================================
# CHECK OVERFLOW
# ============================================================

def check_overflow(result):

    if model is None:
        return False

    if result.boxes is None:
        return False

    if len(result.boxes) == 0:
        return False

    for box in result.boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        class_name = str(class_name).lower().strip()

        if class_name == OVERFLOW_CLASS:

            return True

    return False


# ============================================================
# IMAGE DETECTION
# ============================================================

def detect_image(image):

    if model is None:

        st.error(
            "best.pt was not found. "
            "Place best.pt in the same folder as app.py."
        )

        return None, False


    image_array = np.array(image)


    results = model.predict(
        source=image_array,
        conf=0.25,
        verbose=False
    )


    result = results[0]


    annotated_image = result.plot()


    violation = check_overflow(result)


    return annotated_image, violation


# ============================================================
# VIDEO DETECTION
# ============================================================

def process_video(video_file):

    if model is None:

        st.error(
            "best.pt was not found."
        )

        return None, False


    input_path = None
    output_path = None


    try:

        extension = os.path.splitext(
            video_file.name
        )[1]


        # ----------------------------------------------------
        # SAVE INPUT VIDEO
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_input:

            temp_input.write(
                video_file.read()
            )

            input_path = temp_input.name


        # ----------------------------------------------------
        # OPEN VIDEO
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            input_path
        )


        if not cap.isOpened():

            st.error(
                "Unable to open the uploaded video."
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


        # ----------------------------------------------------
        # OUTPUT FILE
        # ----------------------------------------------------

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )


        output_path = output_file.name


        output_file.close()


        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )


        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )


        violation_detected = False


        frame_number = 0


        progress = st.progress(0)


        # ----------------------------------------------------
        # FRAME BY FRAME DETECTION
        # ----------------------------------------------------

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


            if check_overflow(result):

                violation_detected = True


            annotated_frame = result.plot()


            writer.write(
                annotated_frame
            )


            frame_number += 1


            if total_frames > 0:

                progress.progress(
                    min(
                        frame_number / total_frames,
                        1.0
                    )
                )


        cap.release()

        writer.release()

        progress.empty()


        return (
            output_path,
            violation_detected
        )


    except Exception as error:

        st.error(
            f"Video processing error: {error}"
        )

        return None, False


    finally:

        if (
            input_path
            and os.path.exists(input_path)
        ):

            try:

                os.remove(
                    input_path
                )

            except Exception:
                pass


# ============================================================
# DISPLAY ALERT
# ============================================================

def display_alert():

    if st.session_state.alert is None:
        return


    alert = st.session_state.alert


    st.markdown(
        f"""
        <div class="alert-box">

            🚨 ALERT MESSAGE

            <br><br>

            <b>Garbage Overflow Detected!</b>

            <br><br>

            📍 Location:
            {alert["location"]}

            <br>

            🕒 Date & Time:
            {alert["time"]}

            <br>

            ⚠️ Status:
            {alert["status"]}

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 1
# ============================================================

def page_one():

    # --------------------------------------------------------
    # EXACT HOME UI
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="home-container">

            <!-- HEADER -->

            <div class="home-header">

                <div class="home-header-title">
                    ECOBIN AI
                </div>

                <div class="home-header-subtitle">
                    Smart Garbage Overflow Detection System
                </div>

            </div>


            <!-- MAIN BODY -->

            <div class="home-body">


                <!-- LEFT SIDE -->

                <div class="home-left">

                    <div class="aicw-text">

                        AI Career for Women
                        <br>

                        (AICW)

                    </div>


                    <div class="project-text">

                        Capstone Project

                    </div>


                </div>


                <!-- RIGHT SIDE -->

                <div class="home-right">


                    <!-- TITLE -->

                    <div class="title-area">

                        <h2>
                            TITLE
                        </h2>

                    </div>


                    <!-- DESCRIPTION -->

                    <div class="description-area">

                        <h2>
                            DESCRIPTION
                        </h2>


                        <div class="description-text">

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


                    <!-- TEAM + GUIDE -->

                    <div class="bottom-area">


                        <!-- TEAM MEMBERS -->

                        <div class="team-area">

                            <div class="section-heading">
                                TEAM MEMBERS
                            </div>


                            <div class="team-member">
                                1. Member Name — member1@email.com
                            </div>


                            <div class="team-member">
                                2. Member Name — member2@email.com
                            </div>


                            <div class="team-member">
                                3. Member Name — member3@email.com
                            </div>


                            <div class="team-member">
                                4. Member Name — member4@email.com
                            </div>

                        </div>


                        <!-- GUIDE -->

                        <div class="guide-area">

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


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(
        [37, 15, 48]
    )


    with col1:

        if st.button(
            "PREDICT",
            key="home_predict",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = 2

            st.rerun()


# ============================================================
# PAGE 2
# ============================================================

def page_two():

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # BACK TO HOME
    # --------------------------------------------------------

    if st.button(
        "⬅️ Back to Home",
        key="back_home"
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


    location = st.text_input(
        "Enter the garbage-bin / camera location",
        value=st.session_state.location,
        key="location_text"
    )


    st.session_state.location = location


    st.markdown("---")


    # ========================================================
    # CAMERA
    # ========================================================

    st.markdown(
        "## 📷 Camera"
    )


    camera_col1, camera_col2, camera_col3 = st.columns(
        3,
        gap="medium"
    )


    # --------------------------------------------------------
    # CAMERA - TAKE PHOTO
    # --------------------------------------------------------

    with camera_col1:

        st.markdown(
            """
            <div class="detection-box">

                <div class="detection-heading">
                    Camera
                </div>

            """,
            unsafe_allow_html=True
        )


        camera_photo = st.camera_input(
            "Take Photo",
            key="take_photo"
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # CAMERA - INPUT
    # --------------------------------------------------------

    camera_image = None


    if camera_photo is not None:

        camera_image = Image.open(
            camera_photo
        )


    with camera_col2:

        st.markdown(
            """
            <div class="detection-box">

                <div class="detection-heading">
                    Input
                </div>

            """,
            unsafe_allow_html=True
        )


        if camera_image is not None:

            st.image(
                camera_image,
                use_container_width=True
            )

        else:

            st.info(
                "Take a photo using the camera."
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # CAMERA - OUTPUT
    # --------------------------------------------------------

    camera_violation = False


    with camera_col3:

        st.markdown(
            """
            <div class="detection-box">

                <div class="detection-heading">
                    Output
                </div>

            """,
            unsafe_allow_html=True
        )


        if camera_image is not None:

            camera_output, camera_violation = detect_image(
                camera_image
            )


            if camera_output is not None:

                st.image(
                    camera_output,
                    channels="RGB",
                    use_container_width=True
                )


            if camera_violation:

                create_alert()


        else:

            st.info(
                "Detection output will appear here."
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    if camera_violation:

        display_alert()


    # ========================================================
    # IMAGE
    # ========================================================

    st.markdown("---")


    st.markdown(
        "## 🖼️ Image"
    )


    image_col1, image_col2, image_col3 = st.columns(
        3,
        gap="medium"
    )


    # --------------------------------------------------------
    # IMAGE - UPLOAD
    # --------------------------------------------------------

    with image_col1:

        st.markdown(
            """
            <div class="detection-box">

                <div class="detection-heading">
                    Upload Image
                </div>

            """,
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
            key="upload_image"
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # IMAGE - INPUT
    # --------------------------------------------------------

    image_input = None


    if uploaded_image is not None:

        image_input = Image.open(
            uploaded_image
        )


    with image_col2:

        st.markdown(
            """
            <div class="detection-box">

                <div class="detection-heading">
                    Input
                </div>

            """,
            unsafe_allow_html=True
        )


        if image_input is not None:

            st.image(
                image_input,
                use_container_width=True
            )

        else:

            st.info(
                "Upload an image."
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # IMAGE - OUTPUT
    # --------------------------------------------------------

    image_violation = False


    with image_col3:

        st.markdown(
            """
            <div class="detection-box">

                <div class="detection-heading">
                    Output
                </div>

            """,
            unsafe_allow_html=True
        )


        if image_input is not None:

            image_output, image_violation = detect_image(
                image_input
            )


            if image_output is not None:

                st.image(
                    image_output,
                    channels="RGB",
                    use_container_width=True
                )


            if image_violation:

                create_alert()


        else:

            st.info(
                "Detection output will appear here."
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    if image_violation:

        display_alert()


    # ========================================================
    # VIDEO
    # ========================================================

    st.markdown("---")


    st.markdown(
        "## 🎥 Video"
    )


    video_col1, video_col2, video_col3 = st.columns(
        3,
        gap="medium"
    )


    # --------------------------------------------------------
    # VIDEO - UPLOAD
    # --------------------------------------------------------

    with video_col1:

        st.markdown(
            """
            <div class="detection-box">

                <div class="detection-heading">
                    Upload Video
                </div>

            """,
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
            key="upload_video"
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # VIDEO - INPUT
    # --------------------------------------------------------

    with video_col2:

        st.markdown(
            """
            <div class="detection-box">

                <div class="detection-heading">
                    Input
                </div>

            """,
            unsafe_allow_html=True
        )


        if uploaded_video is not None:

            st.video(
                uploaded_video
            )

        else:

            st.info(
                "Upload a video."
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # VIDEO - OUTPUT
    # --------------------------------------------------------

    with video_col3:

        st.markdown(
            """
            <div class="detection-box">

                <div class="detection-heading">
                    Output
                </div>

            """,
            unsafe_allow_html=True
        )


        if uploaded_video is not None:

            if st.button(
                "▶️ Detect Overflow",
                key="video_detect_button"
            ):

                with st.spinner(
                    "Processing video with YOLOv8..."
                ):

                    output_video, video_violation = process_video(
                        uploaded_video
                    )


                if output_video is not None:

                    with open(
                        output_video,
                        "rb"
                    ) as video_file:

                        video_bytes = video_file.read()


                    st.video(
                        video_bytes
                    )


                    st.download_button(
                        "⬇️ Download Result",
                        data=video_bytes,
                        file_name="ecobin_detection.mp4",
                        mime="video/mp4",
                        key="download_result"
                    )


                    if video_violation:

                        create_alert()


                    else:

                        st.markdown(
                            """
                            <div class="safe-box">

                                ✅ No Garbage Overflow Detected

                                <br><br>

                                Status:
                                Normal

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


        else:

            st.info(
                "Processed video will appear here."
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # FINAL ALERT
    # ========================================================

    if st.session_state.alert is not None:

        display_alert()


        if st.button(
            "Clear Alert",
            key="clear_alert_button"
        ):

            clear_alert()

            st.rerun()


# ============================================================
# APPLICATION ROUTING
# ============================================================

if st.session_state.page == 1:

    page_one()

else:

    page_two()
