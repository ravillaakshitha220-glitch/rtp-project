 import streamlit as st
import time
import pandas as pd
import cv2
import av
import speech_recognition as sr
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ---------------------------------------------------------------------
# AI Answer Evaluation (simple logic)
# ---------------------------------------------------------------------
def analyze_answer_text(question, transcribed_text):
    if not transcribed_text:
        return {"score": 0, "feedback": "No audio detected."}

    words = len(transcribed_text.split())

    if words < 5:
        return {"score": 40, "feedback": "Too short answer."}
    elif words > 20:
        return {"score": 90, "feedback": "Detailed answer."}
    else:
        return {"score": 75, "feedback": "Good but can be improved."}

# ---------------------------------------------------------------------
# Emotion Detection (WebRTC)
# ---------------------------------------------------------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence_score = 0
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % 15 == 0:
            try:
                result = DeepFace.analyze(
                    img,
                    actions=['emotion'],
                    enforce_detection=False
                )

                emotion = result[0]['dominant_emotion']

                emotion_map = {
                    "happy": 95,
                    "neutral": 85,
                    "surprise": 70,
                    "sad": 40,
                    "fear": 30,
                    "angry": 50,
                    "disgust": 30
                }

                self.confidence_score = emotion_map.get(emotion, 50)

                cv2.putText(
                    img,
                    f"Emotion: {emotion}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            except:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------------------------------------------------
# Speech to Text
# ---------------------------------------------------------------------
def record_and_transcribe():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("🎙️ Speak your answer...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

    try:
        st.info("⚙️ Transcribing...")
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return "Speech API error"

# ---------------------------------------------------------------------
# Streamlit Setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="AI Interviewer", layout="wide")

if "current_question" not in st.session_state:
    st.session_state.current_question = 0

if "interview_log" not in st.session_state:
    st.session_state.interview_log = []

if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False

# ---------------------------------------------------------------------
# ✅ 10 QUESTIONS ONLY
# ---------------------------------------------------------------------
QUESTIONS = [
    "Tell me about yourself.",
    "What are your strengths?",
    "What are your weaknesses?",
    "Explain a project you worked on.",
    "Why should we hire you?",
    "What is machine learning?",
    "How do you handle deadlines?",
    "Explain teamwork experience.",
    "Where do you see yourself in 5 years?",
    "Do you have any questions for us?"
]

st.title("🎙️ AI Interview System (10 Questions)")

# ---------------------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------------------
if st.session_state.interview_complete:
    st.header("📊 Final Report")

    df = pd.DataFrame(st.session_state.interview_log)

    avg_tech = df["tech_score"].mean()
    avg_conf = df["confidence_score"].mean()
    overall = (avg_tech * 0.6) + (avg_conf * 0.4)

    col1, col2, col3 = st.columns(3)
    col1.metric("Technical Score", f"{avg_tech:.1f}")
    col2.metric("Confidence Score", f"{avg_conf:.1f}")
    col3.metric("Overall", f"{overall:.1f}%")

    if overall >= 80:
        verdict = "Highly Recommended"
        color = "green"
    elif overall >= 65:
        verdict = "Needs Review"
        color = "orange"
    else:
        verdict = "Not Suitable"
        color = "red"

    st.markdown(f"### Verdict: :{color}[{verdict}]")

    st.dataframe(df)

    if st.button("Restart Interview"):
        st.session_state.clear()
        st.rerun()

# ---------------------------------------------------------------------
# INTERVIEW FLOW
# ---------------------------------------------------------------------
else:
    q_index = st.session_state.current_question
    question = QUESTIONS[q_index]

    st.progress(q_index / len(QUESTIONS))
    st.subheader(f"Question {q_index + 1} of {len(QUESTIONS)}")
    st.info(question)

    col1, col2 = st.columns(2)

    with col1:
        webrtc_ctx = webrtc_streamer(
            key="camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        if st.button("🎤 Record Answer"):
            if not webrtc_ctx.state.playing:
                st.warning("Please start camera first.")
            else:
                text = record_and_transcribe()

                confidence = (
                    webrtc_ctx.video_processor.confidence_score
                    if webrtc_ctx.video_processor
                    else 50
                )

                result = analyze_answer_text(question, text)

                st.session_state.interview_log.append({
                    "question": question,
                    "transcription": text,
                    "tech_score": result["score"],
                    "confidence_score": confidence,
                    "feedback": result["feedback"]
                })

                time.sleep(1)

                if q_index < len(QUESTIONS) - 1:
                    st.session_state.current_question += 1
                else:
                    st.session_state.interview_complete = True

                st.rerun()