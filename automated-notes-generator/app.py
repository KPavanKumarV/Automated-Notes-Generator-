# app.py

import streamlit as st
import openai
import tempfile
import os
from pytube import YouTube
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
from utils.summarizer import summarize_text, extract_keywords

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Lecture Notes Generator", layout="centered")
st.title("🎓 Automated Notes Generator from Lecture Videos")

option = st.radio("Choose input type:", ["📁 Upload Video File", "🌐 Enter YouTube URL"])

video_path = None

if option == "📁 Upload Video File":
    uploaded_file = st.file_uploader("Upload a lecture video", type=["mp4", "mov", "mkv", "wav"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

elif option == "🌐 Enter YouTube URL":
    url = st.text_input("Enter YouTube Video URL")
    if url:
        try:
            st.info("Downloading video...")
            yt = YouTube(url)
            stream = yt.streams.filter(only_audio=True).first()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                stream.download(filename=tmp.name)
                video_path = tmp.name
            st.success("Download complete.")
        except Exception as e:
            st.error(f"Error downloading video: {e}")

# Transcription & Summary
if video_path:
    st.info("Extracting audio & transcribing...")

    try:
        # Extract audio only for faster Whisper transcription
        audio_path = video_path.replace(".mp4", ".wav")
        VideoFileClip(video_path).audio.write_audiofile(audio_path)

        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)["text"]

        st.success("Transcription complete!")
        st.subheader("📝 Transcript")
        st.write(transcript)

        st.info("Generating summary and keywords...")
        summary = summarize_text(transcript)
        keywords = extract_keywords(transcript)

        st.subheader("📌 Summary")
        for bullet in summary.split("\n"):
            st.markdown(f"- {bullet.strip()}")

        st.subheader("🔑 Keywords")
        st.markdown(", ".join(keywords))

    except Exception as e:
        st.error(f"Error during processing: {e}")
