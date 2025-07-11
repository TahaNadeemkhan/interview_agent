import streamlit as st
import PyPDF2
from docx import Document
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
import asyncio
import json
import os
import pandas as pd
import requests
from dotenv import load_dotenv
import time
import nest_asyncio
import datetime
from gtts import gTTS
import io
import base64
from pydub import AudioSegment
import wave
import pyaudio
import speech_recognition as sr
import logging
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORDING_FILE = "temp_recording.wav"

nest_asyncio.apply()
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class AudioRecorder:
    def __init__(self):
        try:
            self.audio = pyaudio.PyAudio()
            self.frames = []
            self.is_recording = False
            self.stream = None
            self.initialized = True
        except Exception as e:
            st.error(f"Failed to initialize audio recorder: {str(e)}")
            self.initialized = False

    def start_recording(self):
        if not self.initialized:
            st.error("Audio recorder not properly initialized")
            return False
            
        try:
            self.frames = []
            self.is_recording = True
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self.callback
            )
            self.stream.start_stream()
            return True
        except Exception as e:
            st.error(f"Error starting recording: {str(e)}")
            self.is_recording = False
            return False

    def callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.save_recording()

    def save_recording(self):
        wf = wave.open(RECORDING_FILE, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def transcribe_recording(self):
        r = sr.Recognizer()
        with sr.AudioFile(RECORDING_FILE) as source:
            audio_data = r.record(source)
            try:
                text = r.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError:
                return "API unavailable"

def record_audio():
    if 'recorder' not in st.session_state or not hasattr(st.session_state.recorder, 'initialized'):
        st.session_state.recorder = AudioRecorder()
    
    if not st.session_state.recorder.initialized:
        st.error("Audio recording is not available. Please check your microphone settings.")
        return ""
    
    for var in ['recording', 'audio_file', 'transcribed_text']:
        if var not in st.session_state:
            st.session_state[var] = False if var == 'recording' else None

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Start Recording", disabled=st.session_state.recording):
            if st.session_state.recorder.start_recording():
                st.session_state.recording = True
                st.session_state.audio_file = None
                st.session_state.transcribed_text = ""
                st.toast("Recording started... Speak now!")
            else:
                st.session_state.recording = False
    
    with col2:
        if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording):
            st.session_state.recorder.stop_recording()
            st.session_state.recording = False
            st.session_state.audio_file = RECORDING_FILE
            st.toast("Recording stopped!")
    
    if st.session_state.audio_file:
        try:
            st.audio(st.session_state.audio_file, format='audio/wav')
            
            if st.button("📝 Transcribe Recording"):
                with st.spinner("Transcribing audio..."):
                    st.session_state.transcribed_text = st.session_state.recorder.transcribe_recording()
                    if st.session_state.transcribed_text:
                        st.toast("Transcription complete!")
                    else:
                        st.error("Failed to transcribe audio")
        except Exception as e:
            st.error(f"Error playing audio: {str(e)}")
    
    if st.session_state.transcribed_text:
        return st.text_area("Transcribed Text", 
                          value=st.session_state.transcribed_text, 
                          height=150,
                          key=f"transcribed_{st.session_state.current_question_index}")
    
    return ""

def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        return None
    
def autoplay_audio(audio_bytes):
    audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
    audio_html = f"""
    <audio controls autoplay>
    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    Your browser does not support the audio element.
    </audio>
    """
    st.components.v1.html(audio_html, height=50)

def extract_text_from_document(uploaded_file):
    file_name = uploaded_file.name.lower()
    text = ""
    try:
        if file_name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif file_name.endswith((".doc", ".docx")):
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file_name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
        else:
            st.warning("Unsupported file type. Please upload PDF, DOCX, or TXT.")
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {str(e)}")
        return ""
    return text

def load_shortlisted_candidates_from_excel(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        if 'Name' not in df.columns:
            st.error("Excel file must contain a column named 'Name' for candidate names.")
            return None
        if 'Job Description' not in df.columns:
            st.warning("Excel file does not contain a 'Job Description' column.")
            df['Job Description'] = ""
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None

def format_transcript_for_download(interview_data):
    candidate_name = interview_data.get("candidate_name", "N/A Candidate")
    timestamp = interview_data.get("timestamp", "N/A Date")
    total_score = interview_data.get("total_score", "N/A")
    jd = interview_data.get("jd", "Not provided.")
    resume_text = interview_data.get("verification_text", "Not provided.")

    transcript_lines = [
        f"--- AI Interview Transcript for {candidate_name} ---",
        f"Interview Date: {timestamp}",
        f"Overall Score: {total_score}/30",
        f"--------------------------------------------------",
        f"\nJob Description:",
        f"-------------------",
        jd,
        f"-------------------",
        f"\nCandidate's Resume/Verification Text (Excerpt):",
        f"-------------------",
        f"{resume_text[:1000]}...",
        f"-------------------",
        f"\nDetailed Questions & Answers:"
    ]

    for i, qa in enumerate(interview_data.get("qa", []), 1):
        question = qa.get("question", "N/A Question")
        answer = qa.get("answer", "N/A Answer")
        score = qa.get("score", "N/A")
        feedback = qa.get("feedback", "No specific feedback provided.")
        transcript_lines.extend([
            f"\nQuestion {i}: {question}",
            f"Candidate Answer: {answer}",
            f"Score: {score}/10",
            f"Feedback: {feedback}",
            f"---"
        ])
    transcript_lines.append(f"\n--- End of Interview Transcript ---")
    return "\n".join(transcript_lines)

async def generate_interview_questions(jd):
    try:
        provider = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY
        )
        model = OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
        set_tracing_disabled(disabled=True)

        agent = Agent(
            name="Question Generator",
            instructions=f"""
            Generate 3 interview questions based on this Job Description:
            "{jd}"

            Output format:
            {{
                "questions": [
                    "Question 1",
                    "Question 2", 
                    "Question 3"
                ]
            }}
            """,
            model=model,
        )

        result = Runner.run_streamed(starting_agent=agent, input="Generate questions")
        full_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response += event.data.delta

        response_json = json.loads(full_response.strip().strip('```json').strip('```'))
        return response_json["questions"]
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return ["Tell me about yourself.", "Describe your experience.", "Why are you interested in this role?"]

async def conduct_interview(questions, resume_text):
    try:
        provider = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY
        )
        model = OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
        set_tracing_disabled(disabled=True)

        agent = Agent(
            name="AI Interview Evaluator",
            instructions=f"""
            Evaluate interview responses based on:
            - Resume: "{resume_text}"
            - Job Description: "{st.session_state.interview_data['jd']}"
            
            For each question, provide:
            - score (1-10)
            - constructive feedback
            
            Output format:
            {{
                "questions": [
                    {{
                        "question": "...",
                        "answer": "...",
                        "score": 0-10,
                        "feedback": "..."
                    }}
                ],
                "total_score": 0-30
            }}
            """,
            model=model,
        )

        qa_data = st.session_state.interview_data["qa"]
        interview_input = "\n\n".join([f"Question {i+1}: {q['question']}\nAnswer: {q['answer']}" for i, q in enumerate(qa_data)])

        st.info("Analyzing responses...")
        result = Runner.run_streamed(starting_agent=agent, input=interview_input)
        full_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response += event.data.delta

        response_json = json.loads(full_response.strip().strip('```json').strip('```'))
        
        for i, qa in enumerate(st.session_state.interview_data["qa"]):
            if i < len(response_json["questions"]):
                ai_qa = response_json["questions"][i]
                qa["score"] = int(ai_qa.get("score", 0))
                qa["feedback"] = ai_qa.get("feedback", "No feedback")
        
        st.session_state.interview_data["total_score"] = int(response_json.get("total_score", 0))
        st.session_state.interviews[st.session_state.interview_data["candidate_name"]] = {
            "timestamp": st.session_state.interview_data["timestamp"],
            "score": st.session_state.interview_data["total_score"],
            "qa": st.session_state.interview_data["qa"],
            "jd": st.session_state.interview_data["jd"],
            "resume_text": st.session_state.interview_data["verification_text"]
        }
        st.session_state.interview_processed_successfully = True
    except Exception as e:
        st.error(f"Error in evaluation: {str(e)}")
        st.session_state.interview_started_processing = False
        st.session_state.interview_processed_successfully = False

def recruiter_login_logic():
    st.subheader("🔑 Recruiter Login")
    password = st.text_input("Password", type="password", key="recruiter_password")
    if st.button("Login", key="recruiter_login_btn"):
        if password == os.getenv("RECRUITER_PASSWORD"):
            st.session_state.current_page = "recruiter_dashboard"
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")

# Initialize session state
for key, default_value in {
    'current_page': "verification",
    'interview_data': {},
    'shortlisted_df': None,
    'interviews': {},
    'current_question_index': 0,
    'interview_started_processing': False,
    'interview_processed_successfully': False,
    'authenticated': False,
    'dynamic_questions': [],
    'audio_question_played': False,
    'recorder': None,
    'recording': False,
    'audio_file': None,
    'transcribed_text': ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Main App
st.title("AI Interview Portal 🚀")
st.markdown("---")

if st.session_state.current_page == "verification":
    st.header("📝 Candidate Verification")
    with st.expander("Candidate Information", expanded=True):
        full_name = st.text_input("Full Name")
        uploaded_file = st.file_uploader("Upload Resume/CV", type=["pdf", "doc", "docx", "txt"])

    st.markdown("---")
    st.subheader("Recruiter Actions")
    uploaded_excel = st.file_uploader("Upload Shortlisted Candidates (Excel)", type=["xlsx"])
    
    if uploaded_excel and st.button("Load Shortlisted List"):
        st.session_state.shortlisted_df = load_shortlisted_candidates_from_excel(uploaded_excel)
        if st.session_state.shortlisted_df is not None:
            st.success("Candidates loaded!")
            st.dataframe(st.session_state.shortlisted_df.head())

    st.markdown("---")
    if st.button("Start Interview", type="primary"):
        if not full_name.strip():
            st.error("Please provide your name")
        elif st.session_state.shortlisted_df is None:
            st.error("Shortlist not loaded")
        else:
            candidate_row = st.session_state.shortlisted_df[st.session_state.shortlisted_df['Name'].str.strip().str.lower() == full_name.strip().lower()]
            
            if candidate_row.empty:
                st.error("Name not found in shortlist")
            else:
                if uploaded_file:
                    verification_text = extract_text_from_document(uploaded_file)
                    if verification_text.strip():
                        candidate_jd = candidate_row['Job Description'].iloc[0] if 'Job Description' in candidate_row.columns else ""
                        
                        st.session_state.dynamic_questions = asyncio.run(generate_interview_questions(candidate_jd))
                        if not st.session_state.dynamic_questions:
                            st.session_state.dynamic_questions = [
                                "Tell me about yourself.",
                                "Describe your experience.",
                                "Why are you interested in this role?"
                            ]

                        st.session_state.current_page = "interview"
                        st.session_state.interview_data = {
                            "candidate_name": full_name.strip(),
                            "jd": candidate_jd.strip(),
                            "verification_text": verification_text.strip(),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.interview_data["qa"] = []
                        st.rerun()
                    else:
                        st.error("Invalid resume file")
                else:
                    st.error("Please upload your resume")

elif st.session_state.current_page == "interview":
    candidate_name = st.session_state.interview_data["candidate_name"]
    st.header(f"Interview: {candidate_name}")
    st.markdown("---")

    if st.session_state.interview_processed_successfully:
        st.subheader("✅ Interview Completed")
        st.markdown(f"**Score:** {st.session_state.interview_data['total_score']}/30")
        
        for i, qa in enumerate(st.session_state.interview_data["qa"], 1):
            with st.expander(f"Question {i}"):
                st.write(f"**Q:** {qa['question']}")
                st.write(f"**A:** {qa['answer']}")
                if qa.get('audio_file'):
                    st.audio(qa['audio_file'])
                st.markdown(f"**Score:** {qa.get('score', 'N/A')}/10")
                st.markdown(f"**Feedback:** {qa.get('feedback', 'None')}")

        if st.button("Back to Start"):
            st.session_state.current_page = "verification"
            st.rerun()

    else:
        if st.session_state.current_question_index < len(st.session_state.dynamic_questions):
            current_question = st.session_state.dynamic_questions[st.session_state.current_question_index]
            st.subheader(f"Question {st.session_state.current_question_index + 1}/{len(st.session_state.dynamic_questions)}")
            
            if not st.session_state.audio_question_played:
                audio_bytes = text_to_speech(current_question)
                if audio_bytes:
                    st.write(f"**{current_question}**")
                    autoplay_audio(audio_bytes)
                st.session_state.audio_question_played = True
            else:
                st.write(f"**{current_question}**")
            
            st.write("Record your answer:")
            transcribed_text = record_audio()
            
            answer = st.text_area("Or type your answer", key=f"answer_{st.session_state.current_question_index}")
            
            final_answer = transcribed_text if transcribed_text else answer

            if st.button("Submit Answer"):
                if final_answer.strip():
                    st.session_state.interview_data["qa"].append({
                        "question": current_question,
                        "answer": final_answer.strip(),
                        "audio_file": st.session_state.audio_file
                    })
                    st.session_state.current_question_index += 1
                    st.session_state.audio_question_played = False
                    st.rerun()
                else:
                    st.warning("Please provide an answer")

        elif not st.session_state.interview_started_processing:
            st.info("All questions answered. Processing results...")
            st.session_state.interview_started_processing = True
            asyncio.run(conduct_interview(st.session_state.dynamic_questions, st.session_state.interview_data["verification_text"]))
            st.rerun()

elif st.session_state.current_page == "recruiter_dashboard":
    if not st.session_state.authenticated:
        recruiter_login_logic()
    else:
        st.header("📊 Recruiter Dashboard")
        
        if st.session_state.interviews:
            st.subheader("Completed Interviews")
            for candidate, data in st.session_state.interviews.items():
                with st.expander(f"{candidate} - {data['score']}/30"):
                    st.write(f"Date: {data['timestamp']}")
                    st.download_button(
                        label="Download Transcript",
                        data=format_transcript_for_download(data),
                        file_name=f"{candidate}_interview.txt"
                    )
        else:
            st.info("No interviews completed yet")
            
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.current_page = "verification"
            st.rerun()