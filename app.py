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
    page_title="EcoBin AI – Smart Garbage Overflow Prediction",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f5f7fb;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.5rem;
    max-width: 1450px;
}

/* ==========================================================
   PAGE 1
   ========================================================== */

.home-title {
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 4px;
}

.home-subtitle {
    text-align: center;
    font-size: 19px;
    color: #475569;
    margin-bottom: 25px;
}

.aicw-title {
    font-size: 30px;
    font-weight: 800;
    color: #172554;
    margin-top: 35px;
}

.capstone {
    font-size: 24px;
    font-weight: 700;
    color: #334155;
    margin-top: 35px;
    margin-bottom: 35px;
}

.description-title {
    font-size: 25px;
    font-weight: 800;
    color: #172554;
}

.description {
    font-size: 16px;
    line-height: 1.65;
    color: #475569;
    margin-top: 12px;
}

/* Team cards */
.team-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 18px;
    min-height: 250px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
}

.team-heading {
    font-size: 17px;
    font-weight: 800;
    color: #334155;
    margin-bottom: 20px;
}

.team-line {
    font-size: 15px;
    color: #334155;
    margin-bottom: 18px;
}

.guide-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 18px;
    min-height: 250px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
}

.guide-heading {
    font-size: 17px;
    font-weight: 800;
    color: #334155;
    margin-bottom: 22px;
}

.guide-name {
    font-size: 16px;
    color: #334155;
    margin-bottom: 30px;
}

.guide-designation-title {
    font-size: 15px;
    font-weight: 800;
    color: #334155;
    margin-bottom: 12px;
}

.guide-designation {
    font-size: 15px;
    color: #475569;
}


/* ==========================================================
   PAGE 2
   ========================================================== */

.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 4px;
}

.main-subtitle {
    text-align: center;
    font-size: 18px;
    color: #475569;
    margin-bottom: 20px;
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 12px;
}

.detection-heading {
    font-size: 20px;
    font-weight: 800;
    color: #172554;
    margin-top: 15px;
    margin-bottom: 8px;
}

.waiting {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 14px;
    padding: 55px 20px;
    text-align: center;
}

.waiting h3 {
    color: #64748b;
}

.normal-result {
    background: #ecfdf5;
    border: 2px solid #86efac;
    border-radius: 15px;
    padding: 28px;
    text-align: center;
}

.normal-result h2 {
    color: #15803d;
}

.overflow-result {
    background: #fef2f2;
    border: 2px solid #fca5a5;
    border-radius: 15px;
    padding: 28px;
    text-align: center;
}

.overflow-result h2 {
    color: #dc2626;
}

.alert-box {
    background: #fff7ed;
    border: 2px solid #fb923c;
    border-radius: 14px;
    padding: 20px;
    margin-top: 18px;
}

.alert-title {
    color: #dc2626;
    font-size: 22px;
    font-weight: 800;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "result_ready" not in st.session_state:
    st.session_state.result_ready = False

if "result_type" not in st.session_state:
    st.session_state.result_type = None

if "result_data" not in st.session_state:
    st.session_state.result_data = None


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best.pt"
)

CONF_THRESHOLD = 0.50


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error("❌ YOLO model could not be loaded.")
    st.write("Make sure `best.pt` is present beside `app.py`.")
    st.code(str(e))
    st.stop()


# ============================================================
# HELPER FUNCTION
# ============================================================

def process_image(image):

    result = model.predict(
        source=np.array(image),
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = result.names[class_id].lower().strip()

        detections.append({
            "class": class_name,
            "confidence": confidence
        })

    annotated = result.plot()

    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # YOUR TWO CLASSES ARE:
    # normal
    # overflow
    # --------------------------------------------------------

    overflow_detections = [
        d for d in detections
        if d["class"] == "overflow"
    ]

    normal_detections = [
        d for d in detections
        if d["class"] == "normal"
    ]

    if overflow_detections:

        best = max(
            overflow_detections,
            key=lambda x: x["confidence"]
        )

        return {
            "type": "overflow",
            "class": "overflow",
            "confidence": best["confidence"],
            "image": annotated
        }

    elif normal_detections:

        best = max(
            normal_detections,
            key=lambda x: x["confidence"]
        )

        return {
            "type": "normal",
            "class": "normal",
            "confidence": best["confidence"],
            "image": annotated
        }

    else:

        return {
            "type": "normal",
            "class": "normal",
            "confidence": 0,
            "image": annotated
        }


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="home-title">
            🗑️ EcoBin AI
        </div>

        <div class="home-subtitle">
            Smart Garbage Overflow Detection System
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MAIN AREA
    # --------------------------------------------------------

    left, right = st.columns(
        [1, 2],
        gap="large"
    )


    # ========================================================
    # LEFT SIDE
    # ========================================================

    with left:

        st.markdown(
            """
            <div class="aicw-title">
                AI Career for Women
                <br>
                (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="capstone">
                Capstone Project
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "🔍 PREDICT",
            use_container_width=True
        ):

            st.session_state.page = 2
            st.session_state.result_ready = False
            st.session_state.result_type = None
            st.session_state.result_data = None

            st.rerun()


    # ========================================================
    # RIGHT SIDE
    # ========================================================

    with right:

        st.markdown(
            """
            <div class="description-title">
                EcoBin AI – Smart Garbage Overflow Prediction
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="description">

            EcoBin AI is an intelligent Smart Garbage Overflow
            Detection System developed using Artificial Intelligence
            and computer vision. The system uses a trained YOLOv8
            deep learning model to identify whether a garbage bin is
            in a normal condition or overflowing. It supports camera
            input, image upload and video upload for real-time and
            offline monitoring. When garbage overflow is detected,
            the system generates an alert containing the detection
            status, location and date and time. EcoBin AI can help
            municipalities and sanitation teams monitor garbage
            collection points, respond quickly to overflowing bins,
            improve waste collection efficiency and support cleaner
            and smarter communities.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # TEAM + GUIDE
    # ========================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    team_col, guide_col = st.columns(
        [2, 1],
        gap="medium"
    )


    # --------------------------------------------------------
    # TEAM MEMBERS
    # --------------------------------------------------------

    with team_col:

        st.markdown(
            """
            <div class="team-box">

                <div class="team-heading">
                    TEAM MEMBERS
                </div>

                <div class="team-line">
                    1. K.Lalitha Devi
                    &nbsp;&nbsp;&nbsp;
                    — lalithadevi825@gmail.com
                </div>

                <div class="team-line">
                    2. Y.Haasini
                    &nbsp;&nbsp;&nbsp;
                    — haasiniyanamadala@gmail.com
                </div>

                <div class="team-line">
                    3. G.Sri Divya
                    &nbsp;&nbsp;&nbsp;
                    — galidivya534@gmail.com
                </div>

                <div class="team-line">
                    4. N.Sushma sri
                    &nbsp;&nbsp;&nbsp;
                    — nadimpallisushmasri29@gmail.com
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # GUIDE
    # --------------------------------------------------------

    with guide_col:

        st.markdown(
            """
            <div class="guide-box">

                <div class="guide-heading">
                    GUIDE
                </div>

                <div class="guide-name">
                    Md.Abdul Aziz
                </div>

                <div class="guide-designation-title">
                    Designation
                </div>

                <div class="guide-designation">
                    Co Lead & Trainer AICW
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        """
        <div class="footer">
            EcoBin AI – Smart Garbage Overflow Detection System
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2
# ============================================================

else:

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        """
        <div class="main-title">
            🗑️ EcoBin AI
        </div>

        <div class="main-subtitle">
            Smart Garbage Overflow Detection System
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # BACK BUTTON
    # ========================================================

    if st.button("⬅️ Back to Home"):

        st.session_state.page = 1
        st.session_state.result_ready = False
        st.session_state.result_type = None
        st.session_state.result_data = None

        st.rerun()


    # ========================================================
    # LOCATION
    # ========================================================

    st.markdown(
        '<div class="section-title">📍 Detection Location</div>',
        unsafe_allow_html=True
    )

    location = st.text_input(
        "Enter the garbage-bin / camera location",
        value="Ramachandrapuram Municipal area"
    )


    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # INPUT / OUTPUT COLUMNS
    # ========================================================

    input_col, output_col = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # INPUT
    # ========================================================

    with input_col:

        st.markdown(
            '<div class="section-title">📥 INPUT</div>',
            unsafe_allow_html=True
        )


        input_type = st.radio(
            "Select Detection Method",
            [
                "📷 Camera",
                "🖼️ Image",
                "🎥 Video"
            ],
            horizontal=True
        )


        # ====================================================
        # CAMERA
        # ====================================================

        if input_type == "📷 Camera":

            st.markdown(
                '<div class="detection-heading">📷 Camera</div>',
                unsafe_allow_html=True
            )

            camera_image = st.camera_input(
                "Take Photo"
            )

            if camera_image:

                image = Image.open(
                    camera_image
                ).convert("RGB")

                st.image(
                    image,
                    caption="Captured Image",
                    use_container_width=True
                )

                if st.button(
                    "🔍 Detect Garbage Overflow",
                    use_container_width=True,
                    key="camera_detect"
                ):

                    with st.spinner(
                        "Analyzing captured image..."
                    ):

                        result = process_image(image)

                    st.session_state.result_ready = True
                    st.session_state.result_type = result["type"]
                    st.session_state.result_data = result

                    st.rerun()


        # ====================================================
        # IMAGE
        # ====================================================

        elif input_type == "🖼️ Image":

            st.markdown(
                '<div class="detection-heading">🖼️ Upload Image</div>',
                unsafe_allow_html=True
            )

            uploaded_image = st.file_uploader(
                "Choose Image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                key="garbage_image"
            )

            if uploaded_image:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")

                st.image(
                    image,
                    caption="Input Image",
                    use_container_width=True
                )

                if st.button(
                    "🔍 Detect Garbage Overflow",
                    use_container_width=True,
                    key="image_detect"
                ):

                    with st.spinner(
                        "Analyzing image..."
                    ):

                        result = process_image(image)

                    st.session_state.result_ready = True
                    st.session_state.result_type = result["type"]
                    st.session_state.result_data = result

                    st.rerun()


        # ====================================================
        # VIDEO
        # ====================================================

        elif input_type == "🎥 Video":

            st.markdown(
                '<div class="detection-heading">🎥 Upload Video</div>',
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
                key="garbage_video"
            )

            if uploaded_video:

                st.video(
                    uploaded_video
                )

                if st.button(
                    "🔍 Detect Garbage Overflow",
                    use_container_width=True,
                    key="video_detect"
                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        # ------------------------------------
                        # INPUT VIDEO
                        # ------------------------------------

                        input_temp = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        input_temp.write(
                            uploaded_video.getbuffer()
                        )

                        input_temp.close()


                        cap = cv2.VideoCapture(
                            input_temp.name
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

                        total_frames = int(
                            cap.get(
                                cv2.CAP_PROP_FRAME_COUNT
                            )
                        )


                        # ------------------------------------
                        # OUTPUT VIDEO
                        # ------------------------------------

                        output_temp = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        output_temp.close()


                        fourcc = cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )

                        writer = cv2.VideoWriter(
                            output_temp.name,
                            fourcc,
                            fps,
                            (width, height)
                        )


                        any_overflow = False
                        max_overflow_confidence = 0.0

                        progress = st.progress(0)

                        frame_count = 0


                        # ------------------------------------
                        # FRAME PROCESSING
                        # ------------------------------------

                        while True:

                            ret, frame = cap.read()

                            if not ret:
                                break


                            result = model.predict(
                                source=frame,
                                conf=CONF_THRESHOLD,
                                verbose=False
                            )[0]


                            # Check detections
                            for box in result.boxes:

                                class_id = int(
                                    box.cls[0]
                                )

                                confidence = float(
                                    box.conf[0]
                                )

                                class_name = result.names[
                                    class_id
                                ].lower().strip()


                                if class_name == "overflow":

                                    any_overflow = True

                                    max_overflow_confidence = max(
                                        max_overflow_confidence,
                                        confidence
                                    )


                            # Draw boxes
                            annotated = result.plot()

                            writer.write(
                                annotated
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


                        # Remove input temp
                        try:
                            os.remove(
                                input_temp.name
                            )
                        except:
                            pass


                        # ------------------------------------
                        # STORE VIDEO RESULT
                        # ------------------------------------

                        if any_overflow:

                            st.session_state.result_ready = True

                            st.session_state.result_type = "overflow_video"

                            st.session_state.result_data = {
                                "confidence":
                                    max_overflow_confidence,

                                "video":
                                    output_temp.name
                            }

                        else:

                            st.session_state.result_ready = True

                            st.session_state.result_type = "normal_video"

                            st.session_state.result_data = {
                                "video":
                                    output_temp.name
                            }


                        st.rerun()


    # ========================================================
    # OUTPUT
    # ========================================================

    with output_col:

        st.markdown(
            '<div class="section-title">📤 OUTPUT</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # WAITING
        # ----------------------------------------------------

        if not st.session_state.result_ready:

            st.markdown(
                """
                <div class="waiting">

                    <h3>
                        ⏳ WAITING FOR DETECTION
                    </h3>

                    <p>
                        Upload an image/video or take a photo
                        using the camera.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # NORMAL IMAGE / CAMERA
        # ----------------------------------------------------

        elif st.session_state.result_type == "normal":

            data = st.session_state.result_data

            st.markdown(
                """
                <div class="normal-result">

                    <h2>
                        🟢 NORMAL
                    </h2>

                    <p>
                        No garbage overflow detected.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            if data and "image" in data:

                st.image(
                    data["image"],
                    caption="Detection Output",
                    use_container_width=True
                )


        # ----------------------------------------------------
        # OVERFLOW IMAGE / CAMERA
        # ----------------------------------------------------

        elif st.session_state.result_type == "overflow":

            data = st.session_state.result_data

            st.markdown(
                """
                <div class="overflow-result">

                    <h2>
                        🔴 GARBAGE OVERFLOW DETECTED
                    </h2>

                    <p>
                        Overflow violation detected.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            if data and "image" in data:

                st.image(
                    data["image"],
                    caption="Detection Output",
                    use_container_width=True
                )

            # Alert
            current_time = datetime.now().strftime(
                "%d-%m-%Y %I:%M:%S %p"
            )

            confidence = data.get(
                "confidence",
                0
            )

            st.markdown(
                f"""
                <div class="alert-box">

                    <div class="alert-title">
                        🚨 ALERT MESSAGE
                    </div>

                    <br>

                    <b>Garbage Overflow Detected!</b>

                    <br><br>

                    📍 <b>Location:</b>
                    {location}

                    <br><br>

                    🕒 <b>Date & Time:</b>
                    {current_time}

                    <br><br>

                    ⚠️ <b>Status:</b>
                    Garbage Overflow Detected

                    <br><br>

                    🎯 <b>Confidence:</b>
                    {confidence * 100:.2f}%

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # NORMAL VIDEO
        # ----------------------------------------------------

        elif st.session_state.result_type == "normal_video":

            data = st.session_state.result_data

            st.markdown(
                """
                <div class="normal-result">

                    <h2>
                        🟢 NORMAL
                    </h2>

                    <p>
                        No garbage overflow detected in the video.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            video_path = data["video"]

            if os.path.exists(video_path):

                with open(
                    video_path,
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()

                st.video(
                    video_bytes
                )


        # ----------------------------------------------------
        # OVERFLOW VIDEO
        # ----------------------------------------------------

        elif st.session_state.result_type == "overflow_video":

            data = st.session_state.result_data

            st.markdown(
                """
                <div class="overflow-result">

                    <h2>
                        🔴 GARBAGE OVERFLOW DETECTED
                    </h2>

                    <p>
                        Overflow violation detected in video.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            video_path = data["video"]


            if os.path.exists(video_path):

                with open(
                    video_path,
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()

                st.video(
                    video_bytes
                )


            current_time = datetime.now().strftime(
                "%d-%m-%Y %I:%M:%S %p"
            )

            confidence = data.get(
                "confidence",
                0
            )


            # Alert
            st.markdown(
                f"""
                <div class="alert-box">

                    <div class="alert-title">
                        🚨 ALERT MESSAGE
                    </div>

                    <br>

                    <b>Garbage Overflow Detected!</b>

                    <br><br>

                    📍 <b>Location:</b>
                    {location}

                    <br><br>

                    🕒 <b>Date & Time:</b>
                    {current_time}

                    <br><br>

                    ⚠️ <b>Status:</b>
                    Garbage Overflow Detected

                    <br><br>

                    🎯 <b>Confidence:</b>
                    {confidence * 100:.2f}%

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        """
        <div class="footer">
            EcoBin AI – Smart Garbage Overflow Detection System
        </div>
        """,
        unsafe_allow_html=True
    )
