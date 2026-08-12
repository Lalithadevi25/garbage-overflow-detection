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
# SETTINGS
# ============================================================

# YOLO will look for detections from this confidence
MODEL_CONFIDENCE = 0.10

# Minimum confidence required for overflow decision
OVERFLOW_CONFIDENCE = 0.20

# Email cooldown = 5 minutes
ALERT_COOLDOWN = 300


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

st.markdown(
    """
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


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    .main-title {
        color: #17345f !important;
        font-size: 32px !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 32px;
    }


    /* ========================================================
       HOME TEXT
       ======================================================== */

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


    /* ========================================================
       DESCRIPTION
       ======================================================== */

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


    /* ========================================================
       TEXT
       ======================================================== */

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li {
        color: #374151;
    }


    /* ========================================================
       CARDS
       ======================================================== */

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


    /* ========================================================
       BUTTON
       ======================================================== */

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


    /* ========================================================
       INPUT LABELS
       ======================================================== */

    label {
        color: #334155 !important;
    }


    /* ========================================================
       DETECTION PAGE
       ======================================================== */

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


    /* ========================================================
       DETECTION BOX TITLE
       ======================================================== */

    .detection-box-title {
        color: #17345f !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        margin-bottom: 12px;
    }


    /* ========================================================
       OVERFLOW BOX
       ======================================================== */

    .alert-box {
        background: #fff1f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 18px;
        margin-top: 10px;
        color: #991b1b !important;
    }


    /* ========================================================
       NORMAL BOX
       ======================================================== */

    .normal-box {
        background: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 18px;
        margin-top: 10px;
        color: #166534 !important;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

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
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(__file__),
        "best.pt"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "best.pt file not found."
        )

    return YOLO(model_path)


# ============================================================
# LOCATION
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
# NORMALIZE CLASS NAME
# ============================================================

def normalize_class_name(class_name):

    name = str(
        class_name
    ).lower().strip()

    name = name.replace(
        "_",
        ""
    )

    name = name.replace(
        "-",
        ""
    )

    name = name.replace(
        " ",
        ""
    )

    # Overflow class variations
    if name in [
        "overclass",
        "overflow",
        "garbageoverflow",
        "overflowclass"
    ]:

        return "overclass"

    # Normal class variations
    if name in [
        "normal",
        "normalclass",
        "normalbin"
    ]:

        return "normal"

    return name


# ============================================================
# EXTRACT DETECTIONS
# ============================================================

def extract_detections(result):

    detections = []

    if result.boxes is None:
        return detections

    if len(result.boxes) == 0:
        return detections

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        original_name = str(
            result.names[class_id]
        )

        class_name = normalize_class_name(
            original_name
        )

        detections.append(
            {
                "class": class_name,
                "original_class": original_name,
                "confidence": confidence
            }
        )

    return detections


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image):

    image_np = np.array(
        image
    )

    results = model.predict(
        source=image_np,
        conf=MODEL_CONFIDENCE,
        iou=0.45,
        verbose=False
    )

    result = results[0]

    detections = extract_detections(
        result
    )

    return result, detections


# ============================================================
# FINAL PREDICTION
# ============================================================

def get_final_prediction(
    detections
):

    if len(detections) == 0:

        return (
            "NO CLEAR DETECTION",
            0.0
        )


    # --------------------------------------------------------
    # Find overclass detections
    # --------------------------------------------------------

    over_detections = [
        d
        for d in detections
        if d["class"] == "overclass"
    ]


    # --------------------------------------------------------
    # Find normal detections
    # --------------------------------------------------------

    normal_detections = [
        d
        for d in detections
        if d["class"] == "normal"
    ]


    # --------------------------------------------------------
    # BEST OVERFLOW
    # --------------------------------------------------------

    best_overflow = None

    if len(over_detections) > 0:

        best_overflow = max(
            over_detections,
            key=lambda x: x["confidence"]
        )


    # --------------------------------------------------------
    # OVERFLOW DECISION
    # --------------------------------------------------------

    if (
        best_overflow is not None
        and
        best_overflow["confidence"]
        >= OVERFLOW_CONFIDENCE
    ):

        return (
            "GARBAGE OVERFLOW",
            best_overflow["confidence"]
        )


    # --------------------------------------------------------
    # NORMAL DECISION
    # --------------------------------------------------------

    if len(normal_detections) > 0:

        best_normal = max(
            normal_detections,
            key=lambda x: x["confidence"]
        )

        return (
            "NORMAL",
            best_normal["confidence"]
        )


    # --------------------------------------------------------
    # OTHER / UNKNOWN
    # --------------------------------------------------------

    return (
        "NO CLEAR DETECTION",
        0.0
    )


# ============================================================
# SEND EMAIL ALERT
# ============================================================

def send_email_alert():

    try:

        sender_email = st.secrets[
            "EMAIL_SENDER"
        ]

        sender_password = st.secrets[
            "EMAIL_PASSWORD"
        ]

        receiver_email = st.secrets[
            "EMAIL_RECEIVER"
        ]

        location = get_location()

        current_time = get_current_time()

        message = EmailMessage()

        message["Subject"] = (
            "🚨 EcoBin AI - Garbage Overflow Alert"
        )

        message["From"] = sender_email

        message["To"] = receiver_email

        message.set_content(
            f"""
🚨 GARBAGE OVERFLOW DETECTED!

EcoBin AI – Smart Garbage Overflow Detection System

Location:
{location}

Date & Time:
{current_time}

Status:
Violation Detected

Detection Class:
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


    # --------------------------------------------------------
    # CHECK COOLDOWN
    # --------------------------------------------------------

    if (
        st.session_state.last_alert_time
        is not None
    ):

        previous_time = (
            st.session_state.last_alert_time
        )

        current_dt = datetime.now(
            ZoneInfo("Asia/Kolkata")
        )

        difference = (
            current_dt - previous_time
        ).total_seconds()

        if difference < ALERT_COOLDOWN:

            st.info(
                "Email alert cooldown is active."
            )

            return


    # --------------------------------------------------------
    # SAVE ALERT TIME
    # --------------------------------------------------------

    st.session_state.last_alert_time = (
        datetime.now(
            ZoneInfo("Asia/Kolkata")
        )
    )


    # --------------------------------------------------------
    # EMAIL
    # --------------------------------------------------------

    email_sent = send_email_alert()

    if email_sent:

        st.success(
            "📧 Alert email sent successfully."
        )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

def display_prediction(
    result,
    detections,
    title="Prediction Result"
):

    st.markdown(
        f"""
        <div class="detection-box-title">
        {title}
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # ANNOTATED IMAGE
    # --------------------------------------------------------

    annotated = result.plot()

    st.image(
        annotated,
        use_container_width=True
    )


    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    status, confidence = (
        get_final_prediction(
            detections
        )
    )


    st.write("")


    # ========================================================
    # OVERFLOW
    # ========================================================

    if status == "GARBAGE OVERFLOW":

        st.markdown(
            f"""
            <div class="alert-box">

            <h3>🚨 Garbage Overflow Detected</h3>

            <b>Detection:</b>
            overclass
            <br><br>

            <b>Confidence:</b>
            {confidence * 100:.2f}%
            <br><br>

            <b>📍 Location:</b>
            {get_location()}
            <br><br>

            <b>🕒 Date & Time:</b>
            {get_current_time()}
            <br><br>

            <b>⚠️ Status:</b>
            Violation Detected

            </div>
            """,
            unsafe_allow_html=True
        )

        generate_alert()


    # ========================================================
    # NORMAL
    # ========================================================

    elif status == "NORMAL":

        st.markdown(
            f"""
            <div class="normal-box">

            <h3>✅ No Garbage Overflow Detected</h3>

            <b>Detection:</b>
            normal
            <br><br>

            <b>Confidence:</b>
            {confidence * 100:.2f}%
            <br><br>

            <b>Status:</b>
            Normal

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # NO DETECTION
    # ========================================================

    else:

        st.warning(
            "⚠️ No clear garbage condition detected."
        )


    # ========================================================
    # DETECTION DETAILS
    # ========================================================

    if len(detections) > 0:

        st.write("")

        st.markdown(
            "**Detection Details**"
        )

        for detection in detections:

            class_name = detection[
                "class"
            ]

            original_class = detection[
                "original_class"
            ]

            confidence = detection[
                "confidence"
            ]

            st.write(
                f"• {original_class} → "
                f"{class_name} — "
                f"{confidence * 100:.2f}%"
            )


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        return (
            "NO CLEAR DETECTION",
            None,
            0
        )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    # --------------------------------------------------------
    # Approximately 1 frame per second
    # --------------------------------------------------------

    frame_skip = max(
        int(fps),
        1
    )


    frame_number = 0

    overflow_count = 0

    normal_count = 0

    detection_count = 0


    best_frame = None

    best_overflow_confidence = 0.0


    progress = st.progress(
        0
    )


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # ----------------------------------------------------
        # Process selected frames
        # ----------------------------------------------------

        if (
            frame_number
            % frame_skip
            == 0
        ):

            results = model.predict(
                source=frame,
                conf=MODEL_CONFIDENCE,
                iou=0.45,
                verbose=False
            )

            result = results[0]


            detections = (
                extract_detections(
                    result
                )
            )


            status, confidence = (
                get_fi
