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
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "image_result" not in st.session_state:
    st.session_state.image_result = None

if "image_detections" not in st.session_state:
    st.session_state.image_detections = []

if "image_alert" not in st.session_state:
    st.session_state.image_alert = False

if "camera_result" not in st.session_state:
    st.session_state.camera_result = None

if "camera_detections" not in st.session_state:
    st.session_state.camera_detections = []

if "camera_alert" not in st.session_state:
    st.session_state.camera_alert = False

if "video_output" not in st.session_state:
    st.session_state.video_output = None

if "video_alert" not in st.session_state:
    st.session_state.video_alert = False


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f4f6fa;
}

.block-container {
    max-width: 1400px;
    padding-top: 25px;
    padding-bottom: 30px;
    padding-left: 7%;
    padding-right: 7%;
}

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ================================
   HOME TITLE
   ================================ */

.main-title {
    color: #17345f !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 32px;
}


/* ================================
   HOME TEXT
   ================================ */

.aicw-text {
    color: #17345f !important;
    font-size: 25px !important;
    font-weight: 800 !important;
    line-height: 1.55;
}

.capstone-text {
    color: #334155 !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    margin-top: 42px;
}

.description-title {
    color: #17345f !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    margin-bottom: 12px;
}

.description-box {
    background: #ffffff;
    border: 1px solid #dfe4ec;
    border-radius: 14px;
    padding: 22px;
    color: #374151 !important;
    font-size: 15px;
    line-height: 1.7;
}


/* ================================
   CARD
   ================================ */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #dfe4ec !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 10px rgba(30, 41, 59, 0.06);
    padding: 8px !important;
}

.card-heading {
    color: #26364d !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    margin-bottom: 16px;
}

.card-text {
    color: #4b5563 !important;
    font-size: 14px !important;
    line-height: 2.2 !important;
}


/* ================================
   BUTTON
   ================================ */

div.stButton > button {
    width: 100%;
    height: 45px;
    background: #ffffff !important;
    color: #334155 !important;
    border: 1px solid #d5dce6 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

div.stButton > button:hover {
    border-color: #17345f !important;
    color: #17345f !important;
    background: #f8fafc !important;
}


/* ================================
   DETECTION PAGE
   ================================ */

.detect-title {
    color: #17345f !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 5px;
}

.detect-subtitle {
    color: #64748b !important;
    text-align: center;
    font-size: 15px !important;
    margin-bottom: 25px;
}


/* ================================
   DETECTION BOX
   ================================ */

.detect-box {
    background: white;
    border: 1px solid #dfe4ec;
    border-radius: 14px;
    padding: 18px;
    min-height: 180px;
    box-shadow: 0 2px 8px rgba(30, 41, 59, 0.05);
}

.box-title {
    color: #17345f !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 12px;
}


/* ================================
   ALERT
   ================================ */

.alert-box {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
}

.no-alert-box {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
}

.alert-title {
    color: #c2410c !important;
    font-size: 18px !important;
    font-weight: 800 !important;
}

.no-alert-title {
    color: #15803d !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}


/* ================================
   MOBILE
   ================================ */

@media(max-width: 900px) {

    .block-container {
        padding-left: 5%;
        padding-right: 5%;
    }

    .main-title {
        font-size: 24px !important;
    }

    .detect-title {
        font-size: 25px !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(__file__),
        "best.pt"
    )

    return YOLO(model_path)


# ============================================================
# DETECTION HELPER
# ============================================================

def run_detection(image, model):

    result = model.predict(
        image,
        conf=0.30,
        verbose=False
    )[0]

    annotated = result.plot()

    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    detections = []
    overflow_detected = False

    # Possible class names
    overflow_classes = {
        "overflow",
        "garbage_overflow",
        "garbage-overflow",
        "overflowing_bin",
        "overflow_bin",
        "full_bin",
        "overfilled_bin",
        "waste_overflow",
        "garbage"
    }

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            name = str(
                result.names[class_id]
            )

            detections.append(
                {
                    "name": name,
                    "confidence": confidence
                }
            )

            # Check overflow class
            if name.lower().strip() in overflow_classes:
                overflow_detected = True

    return annotated, detections, overflow_detected


# ============================================================
# ALERT FUNCTION
# ============================================================

def show_alert(location, detected=True):

    current_time = datetime.now().strftime(
        "%d-%b-%Y %I:%M:%S %p"
    )

    if detected:

        st.markdown(
            f"""
            <div class="alert-box">

                <div class="alert-title">
                    🚨 Garbage Overflow Detected!
                </div>

                <br>

                📍 <b>Location:</b> {location}<br>

                🕒 <b>Date & Time:</b> {current_time}<br>

                ⚠️ <b>Status:</b> Violation Detected

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="no-alert-box">

                <div class="no-alert-title">
                    ✅ No Garbage Overflow Detected
                </div>

                <br>

                Status: Normal

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    st.markdown(
        '<div class="main-title">'
        '♻️ EcoBin AI – Smart Garbage Overflow Detection'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # TOP SECTION
    # ========================================================

    left_col, right_col = st.columns(
        [0.38, 0.62],
        gap="large"
    )


    # ========================================================
    # LEFT
    # ========================================================

    with left_col:

        st.markdown(
            '<div class="aicw-text">'
            'AI Career for Women'
            '<br>'
            '(AICW)'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="capstone-text">'
            'Capstone Project'
            '</div>',
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "🔍  PREDICT",
            key="predict",
            use_container_width=True
        ):

            st.session_state.page = "predict"

            st.rerun()


    # ========================================================
    # RIGHT
    # ========================================================

    with right_col:

        st.markdown(
            '<div class="description-title">'
            'Project Description'
            '</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):

            st.markdown(
                """
                EcoBin AI is an AI-powered Smart Garbage Overflow
                Detection System designed to automatically detect
                overflowing garbage bins from CCTV camera feeds,
                images, and videos.

                The system uses a trained YOLOv8 object detection
                model to identify garbage overflow conditions and
                visually mark the detected area using bounding boxes
                and confidence scores.

                When a garbage overflow violation is detected,
                EcoBin AI generates an alert containing the
                detection location, date and time, and violation
                status. This helps reduce manual monitoring and
                enables faster response to overflowing garbage bins.
                """
            )


    st.write("")
    st.write("")


    # ========================================================
    # BOTTOM CARDS
    # ========================================================

    team_col, gmail_col, guide_col = st.columns(
        [1.25, 1.25, 0.75],
        gap="large"
    )


    # ========================================================
    # TEAM MEMBERS
    # ========================================================

    with team_col:

        with st.container(border=True):

            st.markdown(
                '<div class="card-heading">'
                'TEAM MEMBERS'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="card-text">

                1. K.Lalitha Devi<br>
                2. Y.Haasini<br>
                3. G.Sri Divya<br>
                4. N.Sushma sri

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # GMAIL
    # ========================================================

    with gmail_col:

        with st.container(border=True):

            st.markdown(
                '<div class="card-heading">'
                'GMAIL'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="card-text">

                lalithadevi825@gmail.com<br>
                haasiniyanamadala@gmail.com<br>
                galidivya534@gmail.com<br>
                nadimpallisushmasri29@gmail.com

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # GUIDE
    # ========================================================

    with guide_col:

        with st.container(border=True):

            st.markdown(
                '<div class="card-heading">'
                'GUIDE NAME'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="card-text">'
                'MD.Abdul Aziz'
                '</div>',
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown(
                '<div class="card-heading">'
                'Designation'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="card-text">'
                'Trainer, Co-Lead-AICW'
                '</div>',
                unsafe_allow_html=True
            )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        '<div style="text-align:center;'
        'color:#6b7280;'
        'font-size:14px;'
        'margin-top:32px;">'
        'EcoBin AI – Smart Garbage Overflow Detection'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

else:

    # ========================================================
    # TITLE
    # ========================================================

    st.markdown(
        '<div class="detect-title">'
        '♻️ EcoBin AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="detect-subtitle">'
        'Smart Garbage Overflow Detection and Alert System'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # BACK
    # ========================================================

    if st.button(
        "← Back to Home",
        key="back"
    ):

        st.session_state.page = "home"

        st.rerun()


    st.write("")


    # ========================================================
    # LOAD MODEL
    # ========================================================

    try:

        model = load_model()

    except Exception as e:

        st.error(
            "❌ best.pt model load avvaledu."
        )

        st.info(
            "Make sure best.pt is in the same folder as app.py."
        )

        st.stop()


    # ========================================================
    # LOCATION
    # ========================================================

    location = st.text_input(
        "Detection Location",
        placeholder="Enter camera / detection location"
    )

    if not location:
        location = "User Configured Location"


    # ========================================================
    # ROW 1 — CAMERA
    # ========================================================

    st.markdown("### Camera")

    camera_col1, camera_col2, camera_col3 = st.columns(
        3,
        gap="medium"
    )


    # --------------------------------------------------------
    # CAMERA SOURCE
    # --------------------------------------------------------

    with camera_col1:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">Camera</div>',
                unsafe_allow_html=True
            )

            st.info(
                "Use the camera to capture a garbage-bin image."
            )

            camera_image = st.camera_input(
                "Camera",
                label_visibility="collapsed"
            )


    # --------------------------------------------------------
    # CAMERA INPUT
    # --------------------------------------------------------

    with camera_col2:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">Input</div>',
                unsafe_allow_html=True
            )

            if camera_image:

                camera_pil = Image.open(
                    camera_image
                ).convert("RGB")

                st.image(
                    camera_pil,
                    caption="Camera Input",
                    use_container_width=True
                )

            else:

                st.info(
                    "Camera input will appear here."
                )


    # --------------------------------------------------------
    # CAMERA OUTPUT
    # --------------------------------------------------------

    with camera_col3:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">Output</div>',
                unsafe_allow_html=True
            )

            if camera_image:

                if st.button(
                    "Detect Camera",
                    key="camera_detect",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Detecting garbage overflow..."
                    ):

                        camera_array = np.array(
                            camera_pil
                        )

                        annotated, detections, overflow = (
                            run_detection(
                                camera_array,
                                model
                            )
                        )

                        st.session_state.camera_result = annotated

                        st.session_state.camera_detections = detections

                        st.session_state.camera_alert = overflow


                if st.session_state.camera_result is not None:

                    st.image(
                        st.session_state.camera_result,
                        caption="YOLO Detection Output",
                        use_container_width=True
                    )

                    if st.session_state.camera_detections:

                        for detection in st.session_state.camera_detections:

                            st.write(
                                f"**{detection['name']}** — "
                                f"{detection['confidence'] * 100:.1f}%"
                            )

                    if st.session_state.camera_alert:

                        show_alert(
                            location,
                            True
                        )

                    else:

                        show_alert(
                            location,
                            False
                        )

            else:

                st.info(
                    "Detection output will appear here."
                )


    st.write("")
    st.write("")


    # ========================================================
    # ROW 2 — IMAGE
    # ========================================================

    st.markdown("### Image Alert")

    image_col1, image_col2, image_col3 = st.columns(
        3,
        gap="medium"
    )


    # --------------------------------------------------------
    # UPLOAD IMAGE
    # --------------------------------------------------------

    with image_col1:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">Upload Image</div>',
                unsafe_allow_html=True
            )

            uploaded_image = st.file_uploader(
                "Upload Image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                key="garbage_image",
                label_visibility="collapsed"
            )


    # --------------------------------------------------------
    # IMAGE INPUT
    # --------------------------------------------------------

    with image_col2:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">Input</div>',
                unsafe_allow_html=True
            )

            if uploaded_image:

                input_image = Image.open(
                    uploaded_image
                ).convert("RGB")

                st.image(
                    input_image,
                    caption="Uploaded Image",
                    use_container_width=True
                )

            else:

                st.info(
                    "Uploaded image will appear here."
                )


    # --------------------------------------------------------
    # IMAGE OUTPUT
    # --------------------------------------------------------

    with image_col3:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">Output</div>',
                unsafe_allow_html=True
            )

            if uploaded_image:

                if st.button(
                    "Detect Garbage Overflow",
                    key="image_detect",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Running YOLOv8 detection..."
                    ):

                        image_array = np.array(
                            input_image
                        )

                        annotated, detections, overflow = (
                            run_detection(
                                image_array,
                                model
                            )
                        )

                        st.session_state.image_result = annotated

                        st.session_state.image_detections = detections

                        st.session_state.image_alert = overflow


                if st.session_state.image_result is not None:

                    st.image(
                        st.session_state.image_result,
                        caption="YOLO Detection Output",
                        use_container_width=True
                    )

                    if st.session_state.image_detections:

                        for detection in st.session_state.image_detections:

                            st.write(
                                f"**{detection['name']}** — "
                                f"{detection['confidence'] * 100:.1f}%"
                            )

                    if st.session_state.image_alert:

                        show_alert(
                            location,
                            True
                        )

                    else:

                        show_alert(
                            location,
                            False
                        )

            else:

                st.info(
                    "Detection output will appear here."
                )


    # --------------------------------------------------------
    # IMAGE ALERT
    # --------------------------------------------------------

    st.markdown("### Alert")

    if (
        uploaded_image
        and st.session_state.image_alert
    ):

        show_alert(
            location,
            True
        )

    else:

        st.info(
            "No alert until garbage overflow is detected."
        )


    st.write("")
    st.write("")


    # ========================================================
    # ROW 3 — VIDEO
    # ========================================================

    st.markdown("### Video Alert")

    video_col1, video_col2, video_col3 = st.columns(
        3,
        gap="medium"
    )


    # --------------------------------------------------------
    # UPLOAD VIDEO
    # --------------------------------------------------------

    with video_col1:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">Upload Video</div>',
                unsafe_allow_html=True
            )

            uploaded_video = st.file_uploader(
                "Upload Video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv"
                ],
                key="garbage_video",
                label_visibility="collapsed"
            )


    # --------------------------------------------------------
    # VIDEO INPUT
    # --------------------------------------------------------

    with video_col2:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">Input</div>',
                unsafe_allow_html=True
            )

            if uploaded_video:

                st.video(
                    uploaded_video
                )

            else:

                st.info(
                    "Uploaded video will appear here."
                )


    # --------------------------------------------------------
    # VIDEO OUTPUT
    # --------------------------------------------------------

    with video_col3:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">Output</div>',
                unsafe_allow_html=True
            )

            if uploaded_video:

                if st.button(
                    "Detect Garbage in Video",
                    key="video_detect",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Processing video with YOLOv8..."
                    ):

                        temp_input = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        temp_input.write(
                            uploaded_video.getbuffer()
                        )

                        temp_input.close()


                        cap = cv2.VideoCapture(
                            temp_input.name
                        )

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

                        fps = cap.get(
                            cv2.CAP_PROP_FPS
                        )

                        if fps <= 0:
                            fps = 20


                        temp_output = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        temp_output.close()


                        fourcc = cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )

                        writer = cv2.VideoWriter(
                            temp_output.name,
                            fourcc,
                            fps,
                            (width, height)
                        )


                        overflow_detected = False

                        frame_count = 0

                        confirmation_count = 0

                        max_confirmation = 3


                        while True:

                            ret, frame = cap.read()

                            if not ret:
                                break


                            frame_count += 1


                            # Process every 2nd frame
                            if frame_count % 2 != 0:

                                writer.write(frame)

                                continue


                            result = model.predict(
                                frame,
                                conf=0.30,
                                verbose=False
                            )[0]


                            annotated_frame = result.plot()


                            frame_overflow = False


                            if result.boxes is not None:

                                for box in result.boxes:

                                    class_id = int(
                                        box.cls[0]
                                    )

                                    class_name = str(
                                        result.names[class_id]
                                    ).lower().strip()


                                    if class_name in {
                                        "overflow",
                                        "garbage_overflow",
                                        "garbage-overflow",
                                        "overflowing_bin",
                                        "overflow_bin",
                                        "full_bin",
                                        "overfilled_bin",
                                        "waste_overflow",
                                        "garbage"
                                    }:

                                        frame_overflow = True


                            # Confirmation mechanism
                            if frame_overflow:

                                confirmation_count += 1

                            else:

                                confirmation_count = 0


                            if (
                                confirmation_count
                                >= max_confirmation
                            ):

                                overflow_detected = True


                            writer.write(
                                annotated_frame
                            )


                        cap.release()

                        writer.release()


                        os.unlink(
                            temp_input.name
                        )


                        st.session_state.video_output = (
                            temp_output.name
                        )

                        st.session_state.video_alert = (
                            overflow_detected
                        )


                if st.session_state.video_output:

                    st.video(
                        st.session_state.video_output
                    )

                    if st.session_state.video_alert:

                        show_alert(
                            location,
                            True
                        )

                    else:

                        show_alert(
                            location,
                            False
                        )

            else:

                st.info(
                    "Processed video will appear here."
                )


    # --------------------------------------------------------
    # VIDEO ALERT
    # --------------------------------------------------------

    st.markdown("### Alert")

    if (
        uploaded_video
        and st.session_state.video_alert
    ):

        show_alert(
            location,
            True
        )

    else:

        st.info(
            "No alert until garbage overflow is detected."
        )
