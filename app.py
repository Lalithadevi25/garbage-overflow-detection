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
    page_title="EcoBin AI – Smart Garbage Overflow Detection",
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
    padding-top: 1rem;
    padding-bottom: 1.5rem;
    max-width: 1250px;
}


/* ============================================================
   MAIN TITLE
   ============================================================ */

.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 18px;
}


/* ============================================================
   PAGE 1
   ============================================================ */

.aicw-title {
    font-size: 28px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 10px;
}

.capstone {
    font-size: 22px;
    font-weight: 700;
    color: #334155;
    margin-top: 25px;
    margin-bottom: 25px;
}

.description-title {
    font-size: 23px;
    font-weight: 700;
    color: #172554;
}

.description {
    font-size: 15px;
    line-height: 1.6;
    color: #475569;
}

.team-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 13px;
    margin-top: 18px;
    font-size: 14px;
    line-height: 1.4;
}


/* ============================================================
   INPUT / RESULT CARDS
   ============================================================ */

.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px;
    min-height: 330px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.section-title {
    font-size: 21px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 10px;
}


/* ============================================================
   WAITING RESULT
   ============================================================ */

.waiting {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    padding: 30px 15px;
    text-align: center;
    margin-top: 10px;
}

.waiting h3 {
    color: #64748b;
    font-size: 20px;
}


/* ============================================================
   NORMAL RESULT
   ============================================================ */

.good-result {
    background: #ecfdf5;
    border: 2px solid #86efac;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-top: 10px;
}

.good-result h2 {
    color: #15803d;
    font-size: 25px;
}


/* ============================================================
   OVERFLOW RESULT
   ============================================================ */

.bad-result {
    background: #fef2f2;
    border: 2px solid #fca5a5;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    margin-top: 10px;
}

.bad-result h2 {
    color: #dc2626;
    font-size: 25px;
}


/* ============================================================
   ALERT
   ============================================================ */

.alert-box {
    background: #fff1f2;
    border: 2px solid #ef4444;
    border-left: 6px solid #dc2626;
    border-radius: 10px;
    padding: 16px;
    margin-top: 15px;
    color: #991b1b;
    font-size: 15px;
}


/* ============================================================
   NORMAL MESSAGE
   ============================================================ */

.normal-box {
    background: #ecfdf5;
    border: 2px solid #86efac;
    border-left: 6px solid #16a34a;
    border-radius: 10px;
    padding: 15px;
    margin-top: 15px;
    color: #166534;
    font-size: 15px;
}


/* ============================================================
   DETECTION INFORMATION
   ============================================================ */

.detection-box {
    background: #fff7ed;
    border-left: 5px solid #f97316;
    padding: 14px;
    border-radius: 8px;
    margin-top: 12px;
}

.confidence {
    font-size: 16px;
    font-weight: 700;
    color: #334155;
}


/* ============================================================
   LOCATION
   ============================================================ */

.location-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 20px;
    font-size: 13px;
}


/* ============================================================
   REDUCE STREAMLIT IMAGE DISPLAY SIZE
   ============================================================ */

[data-testid="stImage"] img {
    max-height: 300px;
    object-fit: contain;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton > button {
    border-radius: 8px;
    font-weight: 700;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 800px) {

    .main-title {
        font-size: 27px;
    }

    .aicw-title {
        font-size: 23px;
    }

    .description-title {
        font-size: 21px;
    }

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

if "location" not in st.session_state:
    st.session_state.location = "Ramachandrapuram Municipal area"

if "alert" not in st.session_state:
    st.session_state.alert = False

if "alert_time" not in st.session_state:
    st.session_state.alert_time = None


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

    st.error(
        "❌ Trained model could not be loaded."
    )

    st.write(
        "Make sure `best.pt` is present beside `app.py`."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_class_name(result, class_id):

    name = result.names[class_id]

    return str(name).lower().strip()


def detect_overflow(result):

    """
    Returns:
        True  -> overflow detected
        False -> normal / no overflow
    """

    if result.boxes is None:
        return False

    if len(result.boxes) == 0:
        return False

    for box in result.boxes:

        class_id = int(box.cls[0])

        class_name = get_class_name(
            result,
            class_id
        )

        if class_name == "overflow":

            return True

    return False


def get_best_detection(result):

    if result.boxes is None:
        return None

    if len(result.boxes) == 0:
        return None

    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = get_class_name(
            result,
            class_id
        )

        detections.append(
            (
                class_name,
                confidence
            )
        )

    if not detections:
        return None

    return max(
        detections,
        key=lambda x: x[1]
    )


def create_alert():

    st.session_state.alert = True

    st.session_state.alert_time = (
        datetime.now().strftime(
            "%d-%m-%Y %I:%M:%S %p"
        )
    )


def clear_alert():

    st.session_state.alert = False

    st.session_state.alert_time = None


def show_alert():

    if not st.session_state.alert:
        return

    location = st.session_state.location

    alert_time = st.session_state.alert_time

    st.markdown(
        f"""
        <div class="alert-box">

            🚨 <b>ALERT MESSAGE</b>

            <br><br>

            <b>Garbage Overflow Detected!</b>

            <br><br>

            📍 <b>Location:</b>
            {location}

            <br><br>

            🕒 <b>Date & Time:</b>
            {alert_time}

            <br><br>

            ⚠️ <b>Status:</b>
            Garbage Overflow Detected

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:34px;
            font-weight:800;
            color:#172554;
            margin-bottom:20px;
        ">
        🗑️ EcoBin AI – Smart Garbage Overflow Detection
        </div>
        """,
        unsafe_allow_html=True
    )


    left, right = st.columns(
        [1, 2],
        gap="large"
    )


    # ========================================================
    # LEFT
    # ========================================================

    with left:

        st.markdown(
            """
            <div class="aicw-title">
                AI Career for Women (AICW)
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

            clear_alert()

            st.rerun()


    # ========================================================
    # RIGHT
    # ========================================================

    with right:

        st.markdown(
            """
            <div class="description-title">
                Project Description
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <div class="description">

            EcoBin AI is an intelligent Smart Garbage Overflow
            Detection System designed to automatically identify
            overflowing garbage bins using Artificial Intelligence
            and computer vision. The system uses a trained YOLOv8
            deep learning model to analyze images, camera input
            and videos. It detects two conditions: normal and
            overflow. When garbage overflow is detected, the
            system generates an alert containing the detection
            status, location and date and time. This solution can
            help municipalities and sanitation teams monitor
            garbage collection points, respond quickly to overflow
            situations and improve cleanliness in smart cities.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # TEAM
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    team_col, gmail_col, guide_col = st.columns(
        [1.4, 1.4, 1],
        gap="medium"
    )


    with team_col:

        st.markdown(
            """
            <div class="team-box">

            <b>TEAM MEMBERS</b>

            <br><br>

            1. Member Name

            <br><br>

            2. Member Name

            <br><br>

            3. Member Name

            <br><br>

            4. Member Name

            </div>
            """,
            unsafe_allow_html=True
        )


    with gmail_col:

        st.markdown(
            """
            <div class="team-box">

            <b>GMAIL</b>

            <br><br>

            member1@email.com

            <br><br>

            member2@email.com

            <br><br>

            member3@email.com

            <br><br>

            member4@email.com

            </div>
            """,
            unsafe_allow_html=True
        )


    with guide_col:

        st.markdown(
            """
            <div class="team-box">

            <b>GUIDE NAME</b>

            <br><br>

            Guide Name

            <br><br>

            <b>Designation</b>

            <br><br>

            Co Lead & Trainer AICW

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
    # TITLE
    # ========================================================

    st.markdown(
        """
        <div class="main-title">
            🗑️ EcoBin AI – Smart Garbage Overflow Detection
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # BACK
    # ========================================================

    if st.button(
        "← Back to Project"
    ):

        st.session_state.page = 1

        st.session_state.result_ready = False

        st.session_state.result_type = None

        st.session_state.result_data = None

        clear_alert()

        st.rerun()


    # ========================================================
    # LOCATION
    # ========================================================

    st.markdown(
        """
        <div class="location-box">

        <b>📍 Detection Location</b>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.session_state.location = st.text_input(
        "Enter the garbage-bin / camera location",
        value=st.session_state.location
    )


    st.divider()


    # ========================================================
    # INPUT / RESULT
    # ========================================================

    input_col, result_col = st.columns(
        [1, 1],
        gap="medium"
    )


    # ========================================================
    # INPUT COLUMN
    # ========================================================

    with input_col:

        st.markdown(
            '<div class="section-title">📥 INPUT</div>',
            unsafe_allow_html=True
        )


        input_type = st.radio(
            "Select Input Type:",
            [
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video"
            ],
            horizontal=True
        )


        st.write("")


        # ====================================================
        # IMAGE
        # ====================================================

        if input_type == "🖼️ Image":

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


            if uploaded_image:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")


                st.image(
                    image,
                    caption="Input Image",
                    width=420
                )


                analyze_image = st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                )


                if analyze_image:

                    with st.spinner(
                        "Detecting garbage overflow..."
                    ):

                        result = model.predict(
                            source=np.array(image),
                            conf=CONF_THRESHOLD,
                            verbose=False
                        )[0]


                    overflow = detect_overflow(
                        result
                    )


                    best = get_best_detection(
                        result
                    )


                    annotated = result.plot()


                    annotated = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )


                    st.session_state.result_ready = True


                    if overflow:

                        st.session_state.result_type = "overflow"

                        st.session_state.result_data = {
                            "image": annotated,
                            "detection": best
                        }

                        create_alert()

                    else:

                        st.session_state.result_type = "normal"

                        st.session_state.result_data = {
                            "image": annotated,
                            "detection": best
                        }

                        clear_alert()


                    st.rerun()


        # ====================================================
        # CAMERA
        # ====================================================

        elif input_type == "📷 Camera":

            camera_image = st.camera_input(
                "Take Photo",
                key="camera_input"
            )


            if camera_image:

                image = Image.open(
                    camera_image
                ).convert("RGB")


                st.image(
                    image,
                    caption="Captured Image",
                    width=420
                )


                analyze_camera = st.button(
                    "🔍 Analyze Photo",
                    use_container_width=True
                )


                if analyze_camera:

                    with st.spinner(
                        "Analyzing camera image..."
                    ):

                        result = model.predict(
                            source=np.array(image),
                            conf=CONF_THRESHOLD,
                            verbose=False
                        )[0]


                    overflow = detect_overflow(
                        result
                    )


                    best = get_best_detection(
                        result
                    )


                    annotated = result.plot()


                    annotated = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )


                    st.session_state.result_ready = True


                    if overflow:

                        st.session_state.result_type = "overflow"

                        st.session_state.result_data = {
                            "image": annotated,
                            "detection": best
                        }

                        create_alert()

                    else:

                        st.session_state.result_type = "normal"

                        st.session_state.result_data = {
                            "image": annotated,
                            "detection": best
                        }

                        clear_alert()


                    st.rerun()


        # ====================================================
        # VIDEO
        # ====================================================

        elif input_type == "🎥 Video":

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


            if uploaded_video:

                st.video(
                    uploaded_video
                )


                analyze_video = st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                )


                if analyze_video:

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
                        # OUTPUT
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


                        overflow_detected = False

                        highest_confidence = 0.0

                        best_class = "normal"


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


                            if detect_overflow(result):

                                overflow_detected = True


                            best = get_best_detection(
                                result
                            )


                            if best is not None:

                                class_name, confidence = best


                                if confidence > highest_confidence:

                                    highest_confidence = confidence

                                    best_class = class_name


                            annotated = result.plot()


                            writer.write(
                                annotated
                            )


                            frame_count += 1


                            if total_frames > 0:

                                progress.progress(
                                    min(
                                        frame_count /
                                        total_frames,
                                        1.0
                                    )
                                )


                        cap.release()

                        writer.release()

                        progress.empty()


                        # ------------------------------------
                        # SAVE RESULT
                        # ------------------------------------

                        st.session_state.result_ready = True


                        if overflow_detected:

                            st.session_state.result_type = "overflow_video"

                            st.session_state.result_data = {
                                "video":
                                    output_temp.name,
                                "confidence":
                                    highest_confidence,
                                "class":
                                    "overflow"
                            }

                            create_alert()

                        else:

                            st.session_state.result_type = "normal_video"

                            st.session_state.result_data = {
                                "video":
                                    output_temp.name,
                                "confidence":
                                    highest_confidence,
                                "class":
                                    "normal"
                            }

                            clear_alert()


                        try:

                            os.remove(
                                input_temp.name
                            )

                        except:

                            pass


                        st.rerun()


    # ========================================================
    # RESULT COLUMN
    # ========================================================

    with result_col:

        st.markdown(
            """
            <div class="section-title">
                🤖 DETECTION RESULT
            </div>
            """,
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
                        Upload an image, take a photo,
                        or upload a video to start detection.
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
                <div class="good-result">

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


            if data["image"] is not None:

                st.image(
                    data["image"],
                    caption="Detection Result",
                    width=420
                )


            best = data["detection"]


            if best is not None:

                class_name, confidence = best


                st.info(
                    f"Detected Class: {class_name} | "
                    f"Confidence: {confidence * 100:.2f}%"
                )


        # ----------------------------------------------------
        # OVERFLOW IMAGE / CAMERA
        # ----------------------------------------------------

        elif st.session_state.result_type == "overflow":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="bad-result">

                    <h2>
                        🔴 OVERFLOW
                    </h2>

                    <p>
                        Garbage Overflow Detected
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            best = data["detection"]


            if best is not None:

                class_name, confidence = best


                st.markdown(
                    f"""
                    <div class="detection-box">

                        <b>Detected Class:</b>
                        {class_name}

                        <br><br>

                        <span class="confidence">

                        Confidence:
                        {confidence * 100:.2f}%

                        </span>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.image(
                data["image"],
                caption="Garbage Overflow Detection",
                width=420
            )


        # ----------------------------------------------------
        # NORMAL VIDEO
        # ----------------------------------------------------

        elif st.session_state.result_type == "normal_video":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="good-result">

                    <h2>
                        🟢 NORMAL
                    </h2>

                    <p>
                        No garbage overflow detected
                        in the video.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            if os.path.exists(
                data["video"]
            ):

                with open(
                    data["video"],
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()


                st.video(
                    video_bytes
                )


                st.download_button(
                    "⬇️ Download Detection Video",
                    data=video_bytes,
                    file_name="ecobin_normal_result.mp4",
                    mime="video/mp4"
                )


        # ----------------------------------------------------
        # OVERFLOW VIDEO
        # ----------------------------------------------------

        elif st.session_state.result_type == "overflow_video":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="bad-result">

                    <h2>
                        🔴 OVERFLOW
                    </h2>

                    <p>
                        Garbage Overflow Detected
                        in the video.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="detection-box">

                    <b>Detected Class:</b>
                    overflow

                    <br><br>

                    <span class="confidence">

                    Highest Confidence:
                    {data["confidence"] * 100:.2f}%

                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )


            if os.path.exists(
                data["video"]
            ):

                with open(
                    data["video"],
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()


                st.video(
                    video_bytes
                )


                st.download_button(
                    "⬇️ Download Detection Video",
                    data=video_bytes,
                    file_name="ecobin_overflow_result.mp4",
                    mime="video/mp4"
                )


    # ========================================================
    # ALERT MESSAGE
    # ========================================================

    if st.session_state.alert:

        show_alert()


    # ========================================================
    # CLEAR ALERT
    # ========================================================

    if st.session_state.alert:

        if st.button(
            "Clear Alert"
        ):

            clear_alert()

            st.rerun()


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
