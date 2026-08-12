import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


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

if "email_sent" not in st.session_state:
    st.session_state.email_sent = False


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


/* ============================================================
   HOME PAGE
   ============================================================ */

.main-title {
    color: #17345f !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 32px;
}

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


/* ============================================================
   STREAMLIT CARDS
   ============================================================ */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #dfe4ec !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 10px rgba(30, 41, 59, 0.06);
    padding: 8px !important;
}


/* ============================================================
   BUTTONS
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
    font-size: 15px !important;
    margin-bottom: 25px;
}

.box-title {
    color: #17345f !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 12px;
}


/* ============================================================
   ALERT
   ============================================================ */

.alert-box {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
}

.alert-title {
    color: #c2410c !important;
    font-size: 18px !important;
    font-weight: 800 !important;
}

.normal-box {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
}

.normal-title {
    color: #15803d !important;
    font-size: 16px !important;
    font-weight: 700 !important;
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
# EMAIL ALERT FUNCTION
# ============================================================

def send_email_alert(location):

    try:

        sender_email = st.secrets["EMAIL_ADDRESS"]
        app_password = st.secrets["EMAIL_APP_PASSWORD"]

        receiver_email = "lalithadevi825@gmail.com"

        current_time = datetime.now().strftime(
            "%d-%b-%Y %I:%M:%S %p"
        )

        subject = (
            "🚨 EcoBin AI - Garbage Overflow Detected"
        )

        message = f"""
🚨 Garbage Overflow Detected!

EcoBin AI has detected a garbage overflow violation.

📍 Location:
{location}

🕒 Date & Time:
{current_time}

⚠️ Status:
Violation Detected

This alert was automatically generated by:

EcoBin AI
Smart Garbage Overflow Detection System
"""

        email_message = MIMEMultipart()

        email_message["From"] = sender_email
        email_message["To"] = receiver_email
        email_message["Subject"] = subject

        email_message.attach(
            MIMEText(
                message,
                "plain"
            )
        )

        with smtplib.SMTP(
            "smtp.gmail.com",
            587
        ) as server:

            server.starttls()

            server.login(
                sender_email,
                app_password
            )

            server.sendmail(
                sender_email,
                receiver_email,
                email_message.as_string()
            )

        return True

    except Exception as e:

        st.error(
            f"Email alert failed: {e}"
        )

        return False


# ============================================================
# DETECTION FUNCTION
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

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = str(
                result.names[class_id]
            )

            detections.append(
                {
                    "name": class_name,
                    "confidence": confidence
                }
            )

            # ==================================================
            # IMPORTANT:
            # ONLY overclass means violation
            # ==================================================

            if class_name.lower().strip() == "overclass":

                overflow_detected = True

    return (
        annotated,
        detections,
        overflow_detected
    )


# ============================================================
# ALERT DISPLAY FUNCTION
# ============================================================

def display_alert(location):

    current_time = datetime.now().strftime(
        "%d-%b-%Y %I:%M:%S %p"
    )

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


# ============================================================
# PROCESS ALERT
# ============================================================

def process_alert(overflow_detected, location):

    if overflow_detected:

        # Display alert
        display_alert(location)

        # Send email only once
        if not st.session_state.email_sent:

            success = send_email_alert(
                location
            )

            if success:

                st.session_state.email_sent = True

                st.success(
                    "📧 Alert email sent to "
                    "lalithadevi825@gmail.com"
                )

    else:

        st.markdown(
            """
            <div class="normal-box">

                <div class="normal-title">
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
    # LEFT SECTION
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
    # RIGHT SECTION
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
                overflowing garbage bins from camera images,
                uploaded images, and videos.

                The system uses a trained YOLOv8 object detection
                model to identify garbage overflow conditions and
                visually mark the detected area using bounding boxes
                and confidence scores.

                When a garbage overflow violation is detected,
                EcoBin AI generates an alert containing the
                detection location, date and time, and violation
                status. An email notification is also sent to the
                configured user for faster response.
                """
            )


    st.write("")
    st.write("")


    # ========================================================
    # TEAM / GMAIL / GUIDE
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
        """
        <div style="
            text-align:center;
            color:#6b7280;
            font-size:14px;
            margin-top:32px;
        ">
        EcoBin AI – Smart Garbage Overflow Detection
        </div>
        """,
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
    # LOAD MODEL
    # ========================================================

    try:

        model = load_model()

    except Exception:

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

        location = "Location Not Configured"


    # ========================================================
    # ROW 1 — CAMERA
    # ========================================================

    camera_col1, camera_col2, camera_col3 = st.columns(
        3,
        gap="medium"
    )


    # ========================================================
    # CAMERA
    # ========================================================

    with camera_col1:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">'
                'Camera'
                '</div>',
                unsafe_allow_html=True
            )

            camera_image = st.camera_input(
                "Take a photo",
                label_visibility="collapsed"
            )


    # ========================================================
    # CAMERA INPUT
    # ========================================================

    with camera_col2:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">'
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
                    caption="Camera Input",
                    use_container_width=True
                )

            else:

                st.info(
                    "Camera input will appear here."
                )


    # ========================================================
    # CAMERA OUTPUT
    # ========================================================

    with camera_col3:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">'
                'Output'
                '</div>',
                unsafe_allow_html=True
            )

            if camera_image:

                if st.button(
                    "Detect Camera",
                    key="camera_detect",
                    use_container_width=True
                ):

                    # Allow email for new detection
                    st.session_state.email_sent = False

                    with st.spinner(
                        "Detecting garbage overflow..."
                    ):

                        image_array = np.array(
                            camera_pil
                        )

                        (
                            annotated,
                            detections,
                            overflow
                        ) = run_detection(
                            image_array,
                            model
                        )

                        st.session_state.camera_result = annotated

                        st.session_state.camera_detections = detections

                        st.session_state.camera_alert = overflow


                if st.session_state.camera_result is not None:

                    st.image(
                        st.session_state.camera_result,
                        caption="YOLOv8 Detection Output",
                        use_container_width=True
                    )

                    for detection in st.session_state.camera_detections:

                        st.write(
                            f"**{detection['name']}** — "
                            f"{detection['confidence'] * 100:.1f}%"
                        )

                    if st.session_state.camera_alert:

                        process_alert(
                            True,
                            location
                        )

                    else:

                        process_alert(
                            False,
                            location
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

    image_col1, image_col2, image_col3 = st.columns(
        3,
        gap="medium"
    )


    # ========================================================
    # UPLOAD IMAGE
    # ========================================================

    with image_col1:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">'
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
                key="garbage_image",
                label_visibility="collapsed"
            )


    # ========================================================
    # IMAGE INPUT
    # ========================================================

    with image_col2:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">'
                'Input'
                '</div>',
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


    # ========================================================
    # IMAGE OUTPUT
    # ========================================================

    with image_col3:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">'
                'Output'
                '</div>',
                unsafe_allow_html=True
            )

            if uploaded_image:

                if st.button(
                    "Detect Garbage Overflow",
                    key="image_detect",
                    use_container_width=True
                ):

                    # Allow email for new detection
                    st.session_state.email_sent = False

                    with st.spinner(
                        "Running YOLOv8 detection..."
                    ):

                        image_array = np.array(
                            input_image
                        )

                        (
                            annotated,
                            detections,
                            overflow
                        ) = run_detection(
                            image_array,
                            model
                        )

                        st.session_state.image_result = annotated

                        st.session_state.image_detections = detections

                        st.session_state.image_alert = overflow


                if st.session_state.image_result is not None:

                    st.image(
                        st.session_state.image_result,
                        caption="YOLOv8 Detection Output",
                        use_container_width=True
                    )

                    for detection in st.session_state.image_detections:

                        st.write(
                            f"**{detection['name']}** — "
                            f"{detection['confidence'] * 100:.1f}%"
                        )

                    if st.session_state.image_alert:

                        process_alert(
                            True,
                            location
                        )

                    else:

                        process_alert(
                            False,
                            location
                        )

            else:

                st.info(
                    "Detection output will appear here."
                )


    # ========================================================
    # IMAGE ALERT
    # ========================================================

    st.markdown("### Alert")

    if uploaded_image:

        if st.session_state.image_alert:

            display_alert(
                location
            )

        else:

            st.success(
                "✅ No garbage overflow detected. "
                "No alert generated."
            )

    else:

        st.info(
            "Alert will appear here when overclass is detected."
        )


    st.write("")
    st.write("")


    # ========================================================
    # ROW 3 — VIDEO
    # ========================================================

    video_col1, video_col2, video_col3 = st.columns(
        3,
        gap="medium"
    )


    # ========================================================
    # UPLOAD VIDEO
    # ========================================================

    with video_col1:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">'
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
                key="garbage_video",
                label_visibility="collapsed"
            )


    # ========================================================
    # VIDEO INPUT
    # ========================================================

    with video_col2:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">'
                'Input'
                '</div>',
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


    # ========================================================
    # VIDEO OUTPUT
    # ========================================================

    with video_col3:

        with st.container(border=True):

            st.markdown(
                '<div class="box-title">'
                'Output'
                '</div>',
                unsafe_allow_html=True
            )

            if uploaded_video:

                if st.button(
                    "Detect Garbage in Video",
                    key="video_detect",
                    use_container_width=True
                ):

                    # New video detection
                    st.session_state.email_sent = False

                    with st.spinner(
                        "Processing video with YOLOv8..."
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


                        output_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        output_file.close()


                        fourcc = cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )

                        writer = cv2.VideoWriter(
                            output_file.name,
                            fourcc,
                            fps,
                            (width, height)
                        )


                        overflow_detected = False

                        confirmation_count = 0

                        required_confirmations = 3

                        frame_count = 0


                        while True:

                            ret, frame = cap.read()

                            if not ret:

                                break


                            frame_count += 1


                            # Process every 2nd frame
                            if frame_count % 2 != 0:

                                writer.write(
                                    frame
                                )

                                continue


                            result = model.predict(
                                frame,
                                conf=0.30,
                                verbose=False
                            )[0]


                            annotated_frame = result.plot()


                            frame_has_overflow = False


                            if result.boxes is not None:

                                for box in result.boxes:

                                    class_id = int(
                                        box.cls[0]
                                    )

                                    class_name = str(
                                        result.names[class_id]
                                    ).lower().strip()


                                    if class_name == "overclass":

                                        frame_has_overflow = True


                            # =================================
                            # CONFIRMATION MECHANISM
                            # =================================

                            if frame_has_overflow:

                                confirmation_count += 1

                            else:

                                confirmation_count = 0


                            if (
                                confirmation_count
                                >= required_confirmations
                            ):

                                overflow_detected = True


                            writer.write(
                                annotated_frame
                            )


                        cap.release()

                        writer.release()


                        os.remove(
                            input_file.name
                        )


                        st.session_state.video_output = (
                            output_file.name
                        )

                        st.session_state.video_alert = (
                            overflow_detected
                        )


                if st.session_state.video_output:

                    st.video(
                        st.session_state.video_output
                    )

                    if st.session_state.video_alert:

                        process_alert(
                            True,
                            location
                        )

                    else:

                        process_alert(
                            False,
                            location
                        )

            else:

                st.info(
                    "Processed video will appear here."
                )


    # ========================================================
    # VIDEO ALERT
    # ========================================================

    st.markdown("### Alert")

    if uploaded_video:

        if st.session_state.video_alert:

            display_alert(
                location
            )

        else:

            st.success(
                "✅ No garbage overflow detected. "
                "No alert generated."
            )

    else:

        st.info(
            "Alert will appear here when overclass is detected."
        )
