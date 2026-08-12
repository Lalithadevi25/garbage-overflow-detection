import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
from zoneinfo import ZoneInfo


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

if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = None


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

/* Hide Streamlit default UI */

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ============================================================
   MAIN TITLE
   ============================================================ */

.main-title {
    color: #17345f !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 32px;
}


/* ============================================================
   AICW
   ============================================================ */

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


/* ============================================================
   DESCRIPTION
   ============================================================ */

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


/* ============================================================
   STREAMLIT TEXT
   ============================================================ */

.stMarkdown,
.stMarkdown p,
.stMarkdown li {
    color: #374151;
}


/* ============================================================
   CARD CONTAINERS
   ============================================================ */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #dfe4ec !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 10px rgba(30, 41, 59, 0.06);
    padding: 8px !important;
}


/* ============================================================
   CARD HEADINGS
   ============================================================ */

.card-heading {
    color: #26364d !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    margin-bottom: 16px;
}


/* ============================================================
   CARD TEXT
   ============================================================ */

.card-text {
    color: #4b5563 !important;
    font-size: 14px !important;
    line-height: 2.2 !important;
}


/* ============================================================
   BUTTON
   ============================================================ */

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


/* ============================================================
   INPUT LABELS
   ============================================================ */

label {
    color: #334155 !important;
}


/* ============================================================
   DETECTION PAGE
   ============================================================ */

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
    font-size: 15px;
    margin-bottom: 25px;
}


/* ============================================================
   DETECTION BOX
   ============================================================ */

.detection-box-title {
    color: #17345f !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    margin-bottom: 12px;
}


/* ============================================================
   ALERT
   ============================================================ */

.alert-box {
    background: #fff1f2;
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 18px;
    margin-top: 10px;
    color: #991b1b !important;
}

.normal-box {
    background: #f0fdf4;
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 18px;
    margin-top: 10px;
    color: #166534 !important;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media(max-width: 900px) {

    .block-container {
        padding-left: 5%;
        padding-right: 5%;
    }

    .main-title {
        font-size: 24px !important;
    }

    .aicw-text {
        font-size: 21px !important;
    }

    .capstone-text {
        font-size: 19px !important;
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
# GET LOCATION
# ============================================================

def get_location():

    try:
        location = st.secrets["LOCATION"]

        if location:
            return location

    except Exception:
        pass

    return "Location not configured"


# ============================================================
# CURRENT DATE & TIME
# ============================================================

def get_current_time():

    india_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    return india_time.strftime(
        "%d-%b-%Y %I:%M:%S %p"
    )


# ============================================================
# CHECK OVERFLOW
# ============================================================

def is_overflow_detected(result):

    if result.boxes is None:
        return False

    if len(result.boxes) == 0:
        return False

    for box in result.boxes:

        class_id = int(box.cls[0])

        class_name = str(
            result.names[class_id]
        ).lower().strip()

        confidence = float(
            box.conf[0]
        )

        # Only overclass should trigger alert
        if (
            class_name == "overclass"
            and confidence >= 0.30
        ):
            return True

    return False


# ============================================================
# SEND EMAIL ALERT
# ============================================================

def send_email_alert():

    try:

        sender_email = st.secrets["EMAIL_SENDER"]
        sender_password = st.secrets["EMAIL_PASSWORD"]
        receiver_email = st.secrets["EMAIL_RECEIVER"]

        location = get_location()
        current_time = get_current_time()

        message = EmailMessage()

        message["Subject"] = "🚨 EcoBin AI - Garbage Overflow Alert"

        message["From"] = sender_email

        message["To"] = receiver_email

        message.set_content(
            f"""
🚨 GARBAGE OVERFLOW DETECTED!

EcoBin AI – Smart Garbage Overflow Detection System

📍 Location:
{location}

🕒 Date & Time:
{current_time}

⚠️ Status:
Violation Detected

🤖 Detection Class:
overclass

This alert was automatically generated by EcoBin AI.
"""
        )

        with smtplib.SMTP(
            "smtp.gmail.com",
            587
        ) as server:

            server.starttls()

            server.login(
                sender_email,
                sender_password
            )

            server.send_message(
                message
            )

        return True

    except Exception as e:

        st.error(
            f"Email alert failed: {e}"
        )

        return False


# ============================================================
# GENERATE ALERT
# ============================================================

def generate_alert():

    current_time = get_current_time()
    location = get_location()

    # Avoid repeated alerts during one continuous event
    if st.session_state.last_alert_time is not None:

        previous = st.session_state.last_alert_time

        current_dt = datetime.now(
            ZoneInfo("Asia/Kolkata")
        )

        difference = (
            current_dt - previous
        ).total_seconds()

        # 5 minute cooldown
        if difference < 300:

            st.warning(
                "⚠️ Overflow detected. "
                "Alert cooldown active."
            )

            return

    st.session_state.last_alert_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    st.markdown(
        f"""
        <div class="alert-box">

        <h3>🚨 Garbage Overflow Detected!</h3>

        <b>📍 Location:</b> {location}<br><br>

        <b>🕒 Date & Time:</b> {current_time}<br><br>

        <b>⚠️ Status:</b> Violation Detected

        </div>
        """,
        unsafe_allow_html=True
    )

    # Send email
    email_sent = send_email_alert()

    if email_sent:

        st.success(
            "📧 Alert email sent successfully to the user."
        )


# ============================================================
# DETECTION DETAILS
# ============================================================

def get_detection_details(result):

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        class_name = result.names[
            class_id
        ]

        detections.append(
            (
                class_name,
                confidence
            )
        )

    return detections


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    # ========================================================
    # TITLE
    # ========================================================

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
                Detection System designed to automatically identify
                overflowing garbage bins using computer vision and
                YOLOv8 object detection.

                The system analyzes images, camera-captured photos,
                and CCTV/video files to identify garbage overflow
                conditions. The trained YOLOv8 model classifies the
                detected garbage condition into two classes:
                <b>overclass</b> and <b>normal</b>.

                When an overflow condition is detected, EcoBin AI
                automatically generates an alert containing the
                location, date and time, and violation status.
                The alert is also sent to the configured user's
                email address.

                This system helps reduce manual monitoring effort,
                support faster waste-management response, and
                improve cleanliness in public and residential areas.
                """,
                unsafe_allow_html=True
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
        '<div class="footer-text">'
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
        'AI-Powered Smart Garbage Overflow Detection System'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # BACK BUTTON
    # ========================================================

    if st.button(
        "← Back to Home",
        key="back"
    ):

        st.session_state.page = "home"

        st.rerun()


    st.write("")


    # ========================================================
    # LOAD YOLO
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
    # ROW 1 - CAMERA
    # ========================================================

    camera_col, camera_input_col, camera_output_col = st.columns(
        3,
        gap="medium"
    )


    # --------------------------------------------------------
    # CAMERA BOX
    # --------------------------------------------------------

    with camera_col:

        with st.container(border=True):

            st.markdown(
                '<div class="detection-box-title">'
                'Camera'
                '</div>',
                unsafe_allow_html=True
            )

            camera_image = st.camera_input(
                "Capture Image",
                key="camera"
            )


    # --------------------------------------------------------
    # CAMERA INPUT
    # --------------------------------------------------------

    with camera_input_col:

        with st.container(border=True):

            st.markdown(
                '<div class="detection-box-title">'
                'Input'
                '</div>',
                unsafe_allow_html=True
            )

            if camera_image:

                camera_pil = Image.open(
                    camera_image
                ).convert("RGB")

                st.image(
                    camera_pil,
                    use_container_width=True
                )

            else:

                st.info(
                    "Camera input will appear here."
                )


    # --------------------------------------------------------
    # CAMERA OUTPUT
    # --------------------------------------------------------

    with camera_output_col:

        with st.container(border=True):

            st.markdown(
                '<div class="detection-box-title">'
                'Output'
                '</div>',
                unsafe_allow_html=True
            )

            if camera_image:

                if st.button(
                    "🔍 Detect",
                    key="camera_detect"
                ):

                    with st.spinner(
                        "Detecting garbage..."
                    ):

                        camera_result = model.predict(
                            np.array(camera_pil),
                            conf=0.30,
                            verbose=False
                        )[0]

                    camera_annotated = camera_result.plot()

                    camera_annotated = cv2.cvtColor(
                        camera_annotated,
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        camera_annotated,
                        use_container_width=True
                    )

                    if is_overflow_detected(
                        camera_result
                    ):

                        generate_alert()

                    else:

                        st.markdown(
                            """
                            <div class="normal-box">
                            ✅ No Garbage Overflow Detected<br>
                            Status: Normal
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


            else:

                st.info(
                    "Detection result will appear here."
                )


    # ========================================================
    # ROW 2 - IMAGE ALERT
    # ========================================================

    st.write("")
    st.write("")


    image_upload_col, image_input_col, image_output_col = st.columns(
        3,
        gap="medium"
    )


    # --------------------------------------------------------
    # UPLOAD IMAGE
    # --------------------------------------------------------

    with image_upload_col:

        with st.container(border=True):

            st.markdown(
                '<div class="detection-box-title">'
                'Upload Image'
                '</div>',
                unsafe_allow_html=True
            )

            uploaded_image = st.file_uploader(
                "Choose image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                key="image_upload"
            )


    # --------------------------------------------------------
    # IMAGE INPUT
    # --------------------------------------------------------

    with image_input_col:

        with st.container(border=True):

            st.markdown(
                '<div class="detection-box-title">'
                'Input'
                '</div>',
                unsafe_allow_html=True
            )

            if uploaded_image:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")

                st.image(
                    image,
                    use_container_width=True
                )

            else:

                st.info(
                    "Uploaded image will appear here."
                )


    # --------------------------------------------------------
    # IMAGE OUTPUT
    # --------------------------------------------------------

    with image_output_col:

        with st.container(border=True):

            st.markdown(
                '<div class="detection-box-title">'
                'Output'
                '</div>',
                unsafe_allow_html=True
            )

            if uploaded_image:

                if st.button(
                    "🔍 Detect",
                    key="image_detect"
                ):

                    with st.spinner(
                        "Detecting garbage..."
                    ):

                        image_result = model.predict(
                            np.array(image),
                            conf=0.30,
                            verbose=False
                        )[0]

                    image_annotated = image_result.plot()

                    image_annotated = cv2.cvtColor(
                        image_annotated,
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        image_annotated,
                        use_container_width=True
                    )

                    if is_overflow_detected(
                        image_result
                    ):

                        generate_alert()

                    else:

                        st.markdown(
                            """
                            <div class="normal-box">
                            ✅ No Garbage Overflow Detected<br>
                            Status: Normal
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            else:

                st.info(
                    "Detection result will appear here."
                )


    # ========================================================
    # IMAGE ALERT HEADING
    # ========================================================

    st.markdown(
        "### Alert:"
    )


    # ========================================================
    # ROW 3 - VIDEO ALERT
    # ========================================================

    st.write("")
    st.write("")


    video_upload_col, video_input_col, video_output_col = st.columns(
        3,
        gap="medium"
    )


    # --------------------------------------------------------
    # UPLOAD VIDEO
    # --------------------------------------------------------

    with video_upload_col:

        with st.container(border=True):

            st.markdown(
                '<div class="detection-box-title">'
                'Upload Video'
                '</div>',
                unsafe_allow_html=True
            )

            uploaded_video = st.file_uploader(
                "Choose video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv"
                ],
                key="video_upload"
            )


    # --------------------------------------------------------
    # VIDEO INPUT
    # --------------------------------------------------------

    with video_input_col:

        with st.container(border=True):

            st.markdown(
                '<div class="detection-box-title">'
                'Input'
                '</div>',
                unsafe_allow_html=True
            )

            if uploaded_video:

                video_bytes = uploaded_video.getvalue()

                st.video(
                    video_bytes
                )

            else:

                st.info(
                    "Uploaded video will appear here."
                )


    # --------------------------------------------------------
    # VIDEO OUTPUT
    # --------------------------------------------------------

    with video_output_col:

        with st.container(border=True):

            st.markdown(
                '<div class="detection-box-title">'
                'Output'
                '</div>',
                unsafe_allow_html=True
            )

            if uploaded_video:

                if st.button(
                    "🎥 Detect",
                    key="video_detect"
                ):

                    with st.spinner(
                        "Processing video..."
                    ):

                        input_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        input_file.write(
                            uploaded_video.getbuffer()
                        )

                        input_file.close()


                        cap = cv2.VideoCapture(
                            input_file.name
                        )


                        fps = cap.get(
                            cv2.CAP_PROP_FPS
                        )

                        if fps <= 0:
                            fps = 20


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


                        overflow_found = False

                        frame_count = 0


                        while True:

                            ret, frame = cap.read()

                            if not ret:
                                break


                            frame_count += 1


                            result = model.predict(
                                frame,
                                conf=0.30,
                                verbose=False
                            )[0]


                            if is_overflow_detected(
                                result
                            ):

                                overflow_found = True


                            annotated = result.plot()


                            writer.write(
                                annotated
                            )


                        cap.release()

                        writer.release()

                        try:
                            os.remove(
                                input_file.name
                            )
                        except:
                            pass


                    st.success(
                        "✅ Video processing completed."
                    )


                    st.video(
                        output_path
                    )


                    # ------------------------------------------------
                    # VIDEO ALERT
                    # ------------------------------------------------

                    if overflow_found:

                        generate_alert()

                    else:

                        st.markdown(
                            """
                            <div class="normal-box">
                            ✅ No Garbage Overflow Detected<br>
                            Status: Normal
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            else:

                st.info(
                    "Processed video will appear here."
                )


    # ========================================================
    # VIDEO ALERT HEADING
    # ========================================================

    st.markdown(
        "### Alert:"
    )
