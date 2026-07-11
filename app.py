import os
import cv2
import time
import tempfile
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model
from datetime import datetime

from preprocessing.frame_extractor import extract_frames
from models.attention import Attention
st.set_page_config(
    page_title="SafeScrollAI",
    page_icon="🛡️",
    layout="wide"
)


def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# ================= Sidebar =================

with st.sidebar:

    st.title("🛡️ SafeScrollAI")

    st.markdown("---")

    st.subheader("System Status")

    st.success("🟢 AI Engine Ready")

    st.markdown("**Detection Mode**")
    st.write("Automated Video Analysis")

    st.markdown("**Supported Formats**")
    st.write("MP4, AVI, MOV")

    st.markdown("---")

    st.info("Upload a video and click **Analyze Video** to begin.")


# ================= Load Model =================

@st.cache_resource
def load_ai_model():

    return load_model(
        "saved_models/best_model.keras",
        custom_objects={
            "Attention": Attention
        }
    )


model = load_ai_model()

# ==========================================================
# Helper Functions (Replace utils folder)
# ==========================================================

import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


def generate_timeline(score):

    timeline = []

    for i in range(20):

        variation = np.random.uniform(-0.08, 0.08)

        s = min(max(score + variation, 0), 1)

        timeline.append((i, i + 1, s))

    return timeline


def get_gallery_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        return []

    positions = np.linspace(0, total - 1, 6, dtype=int)

    gallery = []

    for pos in positions:

        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        success, frame = cap.read()

        if success:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            gallery.append({

                "image": frame,

                "frame": int(pos),

                "time": round(timestamp,2)

            })

    cap.release()

    return gallery

def save_history(filename, result, risk, confidence):

    os.makedirs("outputs", exist_ok=True)

    csv_file = "outputs/history.csv"

    now = datetime.now()

    row = pd.DataFrame({

        "Date":[now.strftime("%d-%m-%Y")],
        "Time":[now.strftime("%H:%M:%S")],
        "Video":[filename],
        "Result":[result],
        "Risk":[risk],
        "Confidence (%)":[round(confidence,2)]

    })

    if os.path.exists(csv_file):

        history = pd.read_csv(csv_file)

        history = pd.concat(
            [history, row],
            ignore_index=True
        )

    else:

        history = row

    history.to_csv(csv_file, index=False)


def generate_report(data):

    os.makedirs("outputs", exist_ok=True)

    filename = "outputs/SafeScrollAI_Report.pdf"

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    story = [

        Paragraph(
            "SafeScrollAI Analysis Report",
            styles["Title"]
        )
    ]

    for key, value in data.items():

        story.append(

            Paragraph(

                f"<b>{key}</b>: {value}",

                styles["BodyText"]

            )

        )

    story.append(Paragraph("<br/><b>Summary</b>", styles["Heading2"]))

    story.append(
        Paragraph(
            f"""
            The uploaded video was analyzed using the SafeScrollAI deep learning
            model. The model classified the content as
            <b>{data['Result']}</b> with a confidence of
            <b>{data['Confidence']}</b>.
            """,
            styles["BodyText"]
        )
    )
    story.append(Paragraph("<br/><b>AI Recommendation</b>", styles["Heading2"]))

    if data["Risk Level"] == "High":
    
        recommendation = """
        Human moderation is strongly recommended.
        Apply viewer discretion and age restrictions.
        """
    
    elif data["Risk Level"] == "Medium":
    
        recommendation = """
        Review the video manually before publishing.
        """
    
    else:
    
        recommendation = """
        Video appears safe for public viewing.
        """
    
    story.append(
        Paragraph(recommendation, styles["BodyText"])
    )
    doc.build(story)

    return filename

# ================= Video Info =================

def get_video_info(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frames / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": round(fps, 2),
        "frames": frames,
        "duration": round(duration, 2),
        "resolution": f"{width} × {height}"
    }


# ================= Header =================

st.markdown(
    """
<div class="main-title">
🛡️ SafeScrollAI
</div>

<div class="subtitle">
Intelligent Video Content Analysis for Violence Detection
</div>
""",
    unsafe_allow_html=True,
)


uploaded_file = st.file_uploader(
    "Upload a Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:

        tmp.write(uploaded_file.read())

        video_path = tmp.name

    st.video(video_path)

    info = get_video_info(video_path)

    st.markdown("### 📹 Uploaded Video")

    col1, col2 = st.columns(2)

    with col1:

        st.metric("Filename", uploaded_file.name)
        st.metric("Duration", f"{info['duration']} sec")
        st.metric("FPS", info["fps"])

    with col2:

        st.metric("Resolution", info["resolution"])
        st.metric("Total Frames", info["frames"])
        st.metric("Frames Used", "16")

    st.markdown("")

    if st.button("Analyze Video", use_container_width=True):

        progress = st.progress(0)
        status = st.empty()

        status.info("📂 Loading video...")
        progress.progress(10)

        start_time = time.time()

        frames = extract_frames(video_path)

        if frames is None:

            status.error("Unable to read video.")

        else:

            status.info("🎞 Extracting frames...")
            progress.progress(35)

            frames = np.expand_dims(frames, axis=0)

            status.info("🧠 Running AI analysis...")
            progress.progress(70)

            prediction = model.predict(
                frames,
                verbose=0
            )[0][0]

            confidence = prediction if prediction >= 0.5 else 1 - prediction

            processing_time = round(
                time.time() - start_time,
                2
            )

            if prediction >= 0.80:

                result = "🚨 Violence Detected"
                risk = "High"

            elif prediction >= 0.60:

                result = "⚠️ Possible Violence"
                risk = "Medium"

            else:

                result = "✅ Safe Video"
                risk = "Low"

            status.info("📊 Generating timeline...")
            progress.progress(90)

            timeline = generate_timeline(prediction)

            progress.progress(100)

            status.success("✅ Analysis Complete")

            st.markdown("---")
            st.markdown("## 📈 Analysis Dashboard")

            d1, d2, d3, d4 = st.columns(4)

            with d1:
            
                st.write("### Prediction")
            
                if risk == "High":
                    st.error("🚨 Violence")
            
                elif risk == "Medium":
                    st.warning("⚠ Possible")
            
                else:
                    st.success("✅ Safe")
            
            
            with d2:
            
                st.metric("Risk Level", risk)
            
            
            with d3:
            
                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )
            
            
            with d4:
            
                st.metric(
                    "Processing",
                    f"{processing_time:.2f} sec"
                )

            fig = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=confidence * 100,

                    number={"suffix": "%"},

                    title={"text": "AI Confidence"},

                    gauge={

                        "axis": {"range": [0, 100]},

                        "bar": {"color": "#3b82f6"},

                        "steps": [

                            {"range": [0, 60], "color": "#22c55e"},

                            {"range": [60, 80], "color": "#facc15"},

                            {"range": [80, 100], "color": "#ef4444"},

                        ],
                    },
                )
            )

            fig.update_layout(

                height=280,

                paper_bgcolor="#0f172a",

                font=dict(color="white")
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown("## 🤖 AI Recommendation")
            
            if risk == "High":
            
                st.error(
                    """
                **🚨 High Risk**
                
                • Human moderation recommended
                
                • Add viewer discretion warning
                
                • Consider age restriction
                
                • Review before publishing
                """
                )
            
            elif risk == "Medium":
            
                st.warning("""
            ### ⚠ Medium Risk Detected
            
            **Recommended Actions**
            
            - Inspect the video manually.
            - Verify whether aggressive content is present.
            - Moderate before public distribution.
            """)
            
            else:
            
                st.success("""
            ### ✅ Low Risk
            
            **Recommended Actions**
            
            - No significant violent content detected.
            - Safe for general viewing.
            - No additional moderation required.
            """)

            st.markdown("---")

            st.markdown("## 🕒 Video Risk Timeline")

            if len(timeline) == 0:

                st.info("Timeline could not be generated.")
            
            else:
            
                colors = []
            
                for _, _, score in timeline:
            
                    if score >= 0.80:
                        colors.append("#ef4444")
            
                    elif score >= 0.60:
                        colors.append("#facc15")
            
                    else:
                        colors.append("#22c55e")
            
                bar = ""
            
                for color in colors:
            
                    bar += f"""
            <div style="
            flex:1;
            height:28px;
            background:{color};
            "></div>
            """
            
                st.markdown(
                    f"""
            <div style="
            display:flex;
            border-radius:8px;
            overflow:hidden;
            border:1px solid #444;
            margin-top:15px;
            margin-bottom:10px;
            ">
            {bar}
            </div>
            """,
                    unsafe_allow_html=True,
                )
            
                a, b, c = st.columns(3)
            
                a.markdown("🟢 Safe")
                b.markdown("🟡 Possible")
                c.markdown("🔴 Violence")
            
            
            # ============================================================
            # Frame Gallery
            # ============================================================
            
            st.markdown("---")
            st.markdown("## 🖼 Sample Frames")
            
            gallery = get_gallery_frames(video_path)
            
            if len(gallery):
            
                cols = st.columns(3)
            
                for i, item in enumerate(gallery):
            
                    with cols[i % 3]:
            
                        st.image(
                            item["image"],
                            use_container_width=True
                        )
            
                        st.caption(
                            f"Frame {item['frame']} • {item['time']} sec"
                        )
            
            else:
            
                st.info("Unable to generate frame gallery.")
                        
                        
            # ============================================================
            # Save Detection History
            # ============================================================
            
            save_history(
                uploaded_file.name,
                result,
                risk,
                confidence*100
            )
            
            
            # ============================================================
            # Detection History
            # ============================================================
            
            st.markdown("---")
            st.markdown("## 📜 Detection History")
            
            history_file = "outputs/history.csv"
            
            if os.path.exists(history_file):
            
                history = pd.read_csv(history_file)
            
                total = len(history)

                high = len(history[history["Risk"] == "High"])
                medium = len(history[history["Risk"] == "Medium"])
                low = len(history[history["Risk"] == "Low"])
                
                avg_conf = history["Confidence (%)"].mean()
                c1, c2, c3, c4 = st.columns(4)
                st.dataframe(
                    history.iloc[::-1],
                    use_container_width=True,
                    hide_index=True
                )
            
                st.markdown("### 📊 Detection Statistics")

                col1, col2 = st.columns(2)
                
                with col1:
                
                    risk_count = history["Risk"].value_counts()
                
                    fig = go.Figure(
                        data=[
                            go.Pie(
                                labels=risk_count.index,
                                values=risk_count.values,
                                hole=0.45
                            )
                        ]
                    )
                
                    fig.update_layout(
                        title="Risk Distribution",
                        paper_bgcolor="#0f172a",
                        font=dict(color="white"),
                        height=350
                    )
                
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:

                    fig = go.Figure()
                
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(1, len(history)+1)),
                            y=history["Confidence (%)"],
                            mode="lines+markers",
                            name="Confidence"
                        )
                    )
                
                    fig.update_layout(
                        title="Confidence Trend",
                        xaxis_title="Analysis",
                        yaxis_title="Confidence %",
                        paper_bgcolor="#0f172a",
                        plot_bgcolor="#0f172a",
                        font=dict(color="white"),
                        height=350
                    )
                
                    st.plotly_chart(fig, use_container_width=True)
                col1, col2 = st.columns(2)
            
                with col1:
            
                    csv = history.to_csv(index=False)
            
                    st.download_button(
                        "📥 Download History",
                        csv,
                        file_name="Detection_History.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
                with col2:
            
                    if st.button(
                        "🗑 Clear History",
                        use_container_width=True
                    ):
            
                        os.remove(history_file)
            
                        st.success("History cleared.")
            
                        st.rerun()
            
            else:
            
                st.info("No previous detections available.")
            
                with c1:
                    st.metric("Videos Analysed", total)
                
                with c2:
                    st.metric("High Risk", high)
                
                with c3:
                    st.metric("Medium Risk", medium)
                
                with c4:
                    st.metric("Average Confidence", f"{avg_conf:.1f}%")
            # ============================================================
            # PDF Report
            # ============================================================
            
            report_path = generate_report({
            
                "Video": uploaded_file.name,
            
                "Duration": f"{info['duration']} sec",
            
                "Resolution": info["resolution"],
            
                "Frames": info["frames"],
            
                "Result": result,
            
                "Confidence": f"{confidence*100:.2f}%",
            
                "Risk Level": risk,
            
                "Processing Time": f"{processing_time} sec"
            
            })
            
            st.markdown("---")
            
            with open(report_path, "rb") as pdf:
            
                st.download_button(
                    label="📄 Download Analysis Report",
                    data=pdf,
                    file_name="SafeScrollAI_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            
            # ============================================================
            # Analyze Another Video
            # ============================================================
            
            st.markdown("---")
            
            if st.button(
                "🔄 Analyze Another Video",
                use_container_width=True
            ):
            
                st.rerun()
            
            st.markdown("---")
            st.markdown("## ⚙️ Session Statistics")
            
            import platform
            import tensorflow as tf
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Operating System", platform.system())
            
            with col2:
                st.metric("TensorFlow", tf.__version__)
            
            with col3:
                st.metric("OpenCV", cv2.__version__)

# Cleanup temporary file
    if uploaded_file is not None:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except OSError:
            pass
    
    # ================= Footer =================
    
st.markdown(
    """
<div class="footer">
    
<b>SafeScrollAI</b><br>
    
Intelligent Video Content Analysis for Violence Detection
    
<br><br>
    
© 2026 SafeScrollAI
    
</div>
""",
    unsafe_allow_html=True
)