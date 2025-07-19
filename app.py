from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
import io
import openai
import os
os.environ["SF_OCSP_FAIL_OPEN"] = "true"
os.environ["SF_OCSP_RESPONSE_CACHE_SERVER_ENABLED"] = "false"
from datetime import datetime, timezone
import base64
from gtts import gTTS
import tempfile

# from pydub import AudioSegment
import cv2
import time
import numpy as np
import re
import json
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import PyPDF2
import docx
from io import BytesIO
import hashlib
import wave
from werkzeug.security import generate_password_hash
import uuid
import random
import string
from werkzeug.security import check_password_hash
import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv
import webrtcvad  # WebRTC VAD library
from collections import Counter

load_dotenv()

OUTLOOK_EMAIL = os.getenv("OUTLOOK_EMAIL")
OUTLOOK_PASSWORD = os.getenv("OUTLOOK_PASSWORD")
LOGIN_URL = os.getenv("LOGIN_URL", "http://localhost:5000")


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['PERMANENT_SESSION_LIFETIME'] = 4600
app.config['SESSION_COOKIE_MAX_SIZE'] = 4093  # Set max cookie size

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('interview_app.log', maxBytes=10000000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

from openai import OpenAI


# OpenAI API Configuration
# openai.api_key = "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"
openai.api_base = "https://api.openai.com/v1"



# If you're using environment variable, make sure it's set
api_key = os.getenv("OPENAI_API_KEY") or "your-openai-api-key-here"

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Snowflake credentials
import psycopg2

# # PostgreSQL credentials (use environment variables in production)
# PG_HOST = "localhost"
# PG_PORT = "5432"
# PG_DB = "ibot_final"
# PG_USER = "postgres"
# PG_PASSWORD = "root"




PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")




def get_postgres_connection():
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD
        )
        return conn
    except Exception as e:
        logger.error(f"PostgreSQL connection error: {str(e)}")
        return None

# Interview Bot Configuration
MAX_FRAME_SIZE = 500
FRAME_CAPTURE_INTERVAL = 5
INTERVIEW_DURATION = 3600  # 1 hour in seconds
PAUSE_THRESHOLD = 40
VAD_SAMPLING_RATE = 16000  # WebRTC VAD only supports 8kHz, 16kHz, 32kHz, 48kHz
VAD_FRAME_DURATION = 30  # ms, supported values are 10, 20, or 30
VAD_MODE = 2  # Aggressiveness mode (0-3)

# Initialize WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

def get_postgres_connection():
    try:
        logger.debug("Attempting to establish PostgreSQL connection")
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD
        )
        logger.debug("Successfully connected to PostgreSQL")
        return conn
    except Exception as e:
        logger.error(f"PostgreSQL connection error: {str(e)}")
        return None



def init_interview_data():
    logger.debug("Initializing new interview data structure")
    return {
        "questions": [],
        "answers": [],
        "ratings": [],
        "current_question": 0,
        "interview_started": False,
        "conversation_history": [],
        "jd_text": "",
        "difficulty_level": "medium",
        "student_info": {
            'name': '',
            'roll_no': '',
            'batch_no': '',
            'center': '',
            'course': '',
            'eval_date': ''
        },
        "start_time": None,
        "end_time": None,
        "visual_feedback": [],
        "last_frame_time": 0,
        "last_activity_time": None,
        "current_context": "",
        "last_speech_time": None,
        "speech_detected": False,
        "current_answer": "",
        "speech_start_time": None,
        "is_processing_answer": False,
        "interview_time_used": 0,
        "visual_feedback_data": [],
        "waiting_for_answer": False,
        "report_generated": False
    }

def extract_text_from_file(file):
    try:
        logger.debug(f"Attempting to extract text from file: {file.filename}")
        if file.filename.lower().endswith('.pdf'):
            logger.debug("Processing PDF file")
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            logger.debug(f"Extracted {len(text)} characters from PDF")
            return text
        elif file.filename.lower().endswith(('.doc', '.docx')):
            logger.debug("Processing Word document")
            doc = docx.Document(BytesIO(file.read()))
            text = "\n".join([para.text for para in doc.paragraphs])
            logger.debug(f"Extracted {len(text)} characters from Word document")
            return text
        elif file.filename.lower().endswith('.txt'):
            logger.debug("Processing text file")
            text = file.read().decode('utf-8')
            logger.debug(f"Extracted {len(text)} characters from text file")
            return text
        else:
            logger.warning(f"Unsupported file format: {file.filename}")
            return None
    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}")
        return None

def convert_audio_to_wav(audio_bytes):
    try:
        logger.debug("Starting audio conversion to WAV format")
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
            tmp_mp3.write(audio_bytes)
            tmp_mp3_filename = tmp_mp3.name
        
        try:
            logger.debug("Attempting to read audio file")
            # audio = AudioSegment.from_mp3(tmp_mp3_filename)
        except:
            logger.debug("MP3 read failed, trying generic file reading")
            # audio = AudioSegment.from_file(tmp_mp3_filename)
        
        logger.debug("Setting audio frame rate and channels")
        audio = audio.set_frame_rate(VAD_SAMPLING_RATE).set_channels(1)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            logger.debug("Exporting to WAV format")
            audio.export(tmp_wav.name, format="wav")
            with open(tmp_wav.name, 'rb') as f:
                wav_data = f.read()
        
        logger.debug("Cleaning up temporary files")
        os.unlink(tmp_mp3_filename)
        os.unlink(tmp_wav.name)
        logger.debug("Audio conversion completed successfully")
        return wav_data
    except Exception as e:
        logger.error(f"Error converting audio to WAV: {str(e)}")
        return None




def process_audio_with_vad(audio_bytes):
    """
    Process audio using WebRTC VAD for voice activity detection
    """
    try:
        logger.debug("Starting VAD processing")
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Calculate number of frames
        frame_size = int(VAD_SAMPLING_RATE * VAD_FRAME_DURATION / 1000)
        frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size)]
        logger.debug(f"Processing {len(frames)} audio frames")
        
        # Process each frame
        speech_frames = 0
        for frame in frames:
            # WebRTC VAD requires 16-bit mono PCM audio
            if len(frame) < frame_size:
                # Pad with zeros if last frame is too short
                frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')
            
            # Convert to bytes
            frame_bytes = frame.tobytes()
            
            # Check if frame contains speech
            if vad.is_speech(frame_bytes, VAD_SAMPLING_RATE):
                speech_frames += 1
        
        # Calculate speech ratio
        speech_ratio = speech_frames / len(frames) if frames else 0
        has_speech = speech_ratio > 0.5
        logger.debug(f"VAD processing complete - Speech detected: {has_speech} (ratio: {speech_ratio:.2f})")
        return has_speech, speech_ratio
    except Exception as e:
        logger.error(f"Error in VAD processing: {e}", exc_info=True)
        return False, 0

def process_audio_from_base64(audio_data_base64):
    try:
        logger.debug("Processing audio from base64")
        audio_bytes = base64.b64decode(audio_data_base64.split(',')[1])
        logger.debug(f"Decoded audio data length: {len(audio_bytes)} bytes")
        
        wav_data = convert_audio_to_wav(audio_bytes)
        if not wav_data:
            logger.warning("Audio conversion failed")
            return False, 0
        
        # Process with WebRTC VAD
        return process_audio_with_vad(wav_data)
    except Exception as e:
        logger.error(f"Error processing audio from base64: {str(e)}", exc_info=True)
        return False, 0

def save_conversation_to_file(conversation_data, roll_no=None):
    try:
        filename = f"interview_conversation_{roll_no}.txt" if roll_no else "interview_conversation.txt"
        logger.debug(f"Saving conversation to file: {filename}")
        
        with open(filename, "a") as f:
            for item in conversation_data:
                if 'speaker' in item:
                    f.write(f"{item['speaker']}: {item['text']}\n")
                elif 'question' in item:
                    f.write(f"Question: {item['question']}\n")
        logger.debug(f"Conversation saved to file {filename}")
    except Exception as e:
        logger.error(f"Error saving conversation to file: {str(e)}", exc_info=True)

def load_conversation_from_file(roll_no=None):
    try:
        filename = f"interview_conversation_{roll_no}.txt" if roll_no else "interview_conversation.txt"
        logger.debug(f"Loading conversation from file: {filename}")
        
        if not os.path.exists(filename):
            logger.debug(f"File {filename} does not exist")
            return []
        
        with open(filename, "r") as f:
            lines = f.readlines()
        
        conversation = []
        for line in lines:
            if line.startswith("bot:") or line.startswith("user:"):
                speaker, text = line.split(":", 1)
                conversation.append({"speaker": speaker.strip(), "text": text.strip()})
            elif line.startswith("Question:"):
                question = line.split(":", 1)[1].strip()
                conversation.append({"question": question})
        
        logger.debug(f"Loaded {len(conversation)} conversation items from file")
        return conversation
    except Exception as e:
        logger.error(f"Error loading conversation from file: {str(e)}", exc_info=True)
        return []


INTRO_QUESTIONS = [
    "Tell me about yourself.",
    "Can you tell me a little about yourself?",
    "Please introduce yourself."
]

INTRO_FOLLOW_UPS = [
    "What are your long-range and short-range goals/objectives?",
    "What do you consider your 3 greatest strengths? Provide me an example of when you used your strengths.",
    "What is your greatest weakness?",
    "What qualifications do you have that make you think you will be successful?",
    "Why should we hire you?",
    "Why did you decide to seek a job with us?",
    "In what type of work environment do you perform best? How would you change your previous work environment to make it more productive?",
    "What type of individuals do you enjoy working with the most? What types of individuals do you like working with the least?",
    "Tell me about the best supervisor with whom you have worked. What was his or her management style?",
    "Tell me about the most difficult supervisor you have had. Why was it difficult?",
    "In what ways do you think you can make a contribution to our company?"
]

ENDING_QUESTIONS = [
    "Is there anything else you’d like to add or share that we haven’t discussed?",
    "Is there anything you wish I had asked you?",
    "Based on our conversation, how do you see yourself fitting into this role?",
    "What excites you most about the possibility of joining our team?",
    "Are there any concerns you have about this position or our company?"
]



def generate_questions_from_jd(jd_text, difficulty_level, roll_no=None):
    logger.debug(f"Generating questions from JD for difficulty: {difficulty_level}")
    
    # Check if jd_text is being passed correctly
    if not jd_text:
        logger.error("No JD text provided for question generation.")
        return []
    
    # Load previous questions to avoid repetition
    previous_questions = []
    filename = f"interview_conversation_{roll_no}.txt" if roll_no else "interview_conversation.txt"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("Question:"):
                    previous_questions.append(line.split(":", 1)[1].strip())
    
    logger.debug(f"Previous questions to avoid: {previous_questions[-5:] if previous_questions else 'None'}")
    
#     prompt = f"""
#     Generate an interview script based on the following job description. The interview should consist of:

# 1. One **introduction** question (about the candidate's background or motivation — NOT technical)
# 2. Three **technical** questions (related to the JD), each with one follow-up questions
# 3. One **behavioral** question (focused on soft skills, teamwork, problem-solving, etc.)

# Difficulty level: {difficulty_level}

# Instructions:
# - The **introduction question** should be simple and personal (e.g., background, interest in role — NOT technical).
# - The **technical questions 3* must reflect the difficulty level and relate to the job description.
# - Each technical question should have 1 follow-up questions.
# - The **behavioral question** must be non-technical and assess soft skills.
# - Avoid repeating these previous questions: {previous_questions[-5:] if previous_questions else "None"}

# Job Description:
# {jd_text}

# Format the output like this:

# Question 1: [introduction question]

# Question 2: [technical question]
#     Follow-up 1: [follow-up question]
   

# Question 3: [technical question]
#     Follow-up 1: [follow-up question]

# Question 4: [technical question]
#     Follow-up 1: [follow-up question]
   

# Question 5: [behavioral question]
#     """
    prompt = f"""
You are a senior technical interviewer.

🎯 Your task is to generate a structured interview script based on the following job description.

🧩 Required Output:
1. One **introduction question** from this list:
   - "Tell me about yourself."
   - "Can you tell me a little about yourself?"
   - "Please introduce yourself."

   🔁 Also include **one follow-up** selected from:
   - "What are your long-range and short-range goals/objectives?"
   - "What do you consider your 3 greatest strengths? Provide me an example of when you used your strengths."
   - "What is your greatest weakness?"
   - "What qualifications do you have that make you think you will be successful?"
   - "Why should we hire you?"
   - "Why did you decide to seek a job with us?"
   - "In what type of work environment do you perform best? How would you change your previous work environment to make it more productive?"
   - "What type of individuals do you enjoy working with the most? What types of individuals do you like working with the least?"
   - "Tell me about the best supervisor with whom you have worked. What was his or her management style?"
   - "Tell me about the most difficult supervisor you have had. Why did you find it difficult to work with him or her?"
   - "In what ways do you think you can make a contribution to our company?"

2. **Three technical questions** relevant to the job description and difficulty level: {difficulty_level}
   - Each technical question must be followed by a meaningful follow-up that is strictly based on the main question.
   - Technical content should align with the JD and difficulty level:
     - Beginner → basics and simple tasks
     - Medium → practical usage, implementation
     - Advanced → architecture, performance, scalability, edge cases

3. One **behavioral question** (e.g., teamwork, adaptability, communication).

4. One **ending question** from this list — but before asking, include a short polite closing statement like:

   - "Thanks for sharing your responses so far."
   - "We’ve covered most of what we planned to discuss."
   

Then immediately ask **one** of these ending questions:

   - "Is there anything else you’d like to add or share that we haven’t discussed?"
   - "Is there anything you wish I had asked you?"
   - "Based on our conversation, how do you see yourself fitting into this role?"
   - "What excites you most about the possibility of joining our team?"
   - "Are there any concerns you have about this position or our company?"


📋 Format your output exactly like this:

Question 1: <Intro question>
Follow-up 1: <Intro follow-up>

Question 2: <Technical Q1>
Follow-up 2: <Follow-up Q1>

Question 3: <Technical Q2>
Follow-up 3: <Follow-up Q2>

Question 4: <Technical Q3>
Follow-up 4: <Follow-up Q3>

Question 5: <Behavioral question>

Question 6: <Ending question>

💡 Avoid repeating these recent questions: {previous_questions[-5:] if previous_questions else "None"}

📝 Job Description:
{jd_text}
"""




    try:
        logger.debug("Sending prompt to OpenAI for question generation")
        response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000
)
        
        # Check if the response is valid
        if not response.choices or not response.choices[0].message.content.strip():
            logger.error("No valid choices found in OpenAI response.")
            return []

        script = response.choices[0].message.content
        logger.debug(f"Received response from OpenAI: {script[:100]}...")

        # Parse the questions from the response
        questions = []
        for line in script.split("\n"):
            if line.strip().startswith("Question") or line.strip().startswith("Follow-up"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    questions.append(parts[1].strip())

        
        logger.debug(f"Extracted {len(questions)} questions from response")
        return questions[:10]
    

    
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}", exc_info=True)

        # Fallback questions
        if difficulty_level == "beginner":
            logger.warning("Using fallback beginner questions")
            return [
                "Tell us about yourself and your background.",
                "What programming languages are you familiar with?",
                "Explain a basic programming concept you've learned recently.",
                "Have you worked on any small coding projects?",
                "Describe a time when you had to learn something new quickly."
            ]
        elif difficulty_level == "advanced":
            logger.warning("Using fallback advanced questions")
            return [
                "Walk us through your professional experience and key achievements.",
                "Explain a complex technical challenge you've solved recently.",
                "How would you design a scalable system for high traffic?",
                "Describe your approach to debugging complex issues.",
                "Tell us about a time you had to lead a technical team through a difficult project."
            ]
        else:  # medium
            logger.warning("Using fallback medium questions")
            return [
                "Tell us about your technical background and experience.",
                "Explain a technical concept you're comfortable with in detail.",
                "Describe a project where you implemented a technical solution.",
                "How do you approach learning new technologies?",
                "Describe a time you had to work in a team to solve a technical problem."
            ]





def generate_encouragement_prompt(conversation_history):
    try:
        logger.debug("Generating encouragement prompt")
        prompt = f"""
        The candidate has paused during their response. Generate a brief, encouraging prompt to:
        - Help them continue their thought
        - Be supportive and professional
        - Be concise (one short sentence)
        
        Current conversation context:
        {conversation_history[-2:] if len(conversation_history) > 2 else conversation_history}
        
        Return ONLY the prompt, nothing else.
        """
        
        logger.debug(f"Encouragement prompt context: {prompt}")
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        encouragement = response.choices[0].message.content.strip()
        logger.debug(f"Generated encouragement: {encouragement}")
        return encouragement
    except Exception as e:
        logger.error(f"Error generating encouragement prompt: {str(e)}", exc_info=True)
        return "Please continue with your thought."

def text_to_speech(text):
    try:
        logger.debug(f"Converting text to speech: {text[:50]}...")
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
        
        tts.save(temp_filename)
        logger.debug(f"TTS saved to temporary file: {temp_filename}")
        
        # audio = AudioSegment.from_mp3(temp_filename)
        wav_filename = temp_filename.replace('.mp3', '.wav')

        # audio.export(wav_filename, format="wav")
        logger.debug(f"Converted to WAV format: {wav_filename}")
        
        with open(wav_filename, 'rb') as f:
            audio_data = f.read()
        
        os.unlink(temp_filename)
        os.unlink(wav_filename)
        logger.debug("Temporary files cleaned up")
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error in text-to-speech: {str(e)}", exc_info=True)
        return None

def process_frame_for_gpt4v(frame):
    try:
        logger.debug("Processing frame for GPT-4 Vision")
        height, width = frame.shape[:2]
        if height > MAX_FRAME_SIZE or width > MAX_FRAME_SIZE:
            scale = MAX_FRAME_SIZE / max(height, width)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            logger.debug(f"Resized frame from {height}x{width} to {frame.shape[0]}x{frame.shape[1]}")
        
        _, buffer = cv2.imencode('.jpg', frame)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        logger.debug(f"Encoded frame to base64 (length: {len(base64_str)})")
        return base64_str
    except Exception as e:
        logger.error(f"Error processing frame for GPT-4V: {str(e)}")
        return ""

def analyze_visual_response(frame_base64, conversation_context):
    try:
        logger.debug("Analyzing visual response with GPT-4V")
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this interview candidate's appearance and environment. "
                                                f"Current conversation context: {conversation_context[-3:] if len(conversation_context) > 3 else conversation_context} "
                                                "Provide descriptive feedback on:" 
                                                "1. Professional appearance (description only)"
                                                "2. Body language and posture (description only)"
                                                "3. Environment appropriateness (description only)"
                                                "4. Any visual distractions (description only)"
                                                "Return ONLY a JSON object with these fields: "
                                                "professional_appearance, body_language, environment, distractions"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200
        )
        try:
            feedback = json.loads(response.choices[0].message.content)
            logger.debug(f"Received visual feedback: {feedback}")
            return feedback
        except json.JSONDecodeError:
            # Fallback if GPT doesn't return proper JSON
            feedback_text = response.choices[0].message.content
            logger.warning(f"GPT returned non-JSON response: {feedback_text}")
            return {
                "professional_appearance": "No specific feedback available",
                "body_language": "No specific feedback available",
                "environment": "No visual feedback available",
                "distractions": feedback_text
            }
    except Exception as e:
        logger.error(f"Error in visual analysis: {str(e)}", exc_info=True)
        return {
            "professional_appearance": "No visual feedback available",
            "body_language": "No visual feedback available",
            "environment": "No visual feedback available",
            "distractions": "No visual feedback available"
        }

def evaluate_response(answer, question, difficulty_level, visual_feedback=None):
    if len(answer.strip()) < 20:
        logger.warning("Answer too short for proper evaluation")
        return {
            "technical": 2.0,
            "communication": 2.0,
            "problem_solving": 2.0,
            "time_management": 2.0,
            "overall": 2.0
        }

    rating_prompt = f"""
    Analyze this interview response for a {difficulty_level} level candidate.
    Question: "{question}"
    Answer: "{answer}"

    Provide numeric ratings from 1-10 for:
    - Technical Knowledge (how accurate and deep was the technical content)
    - Communication (clarity, organization, articulation)
    - Problem Solving (logical approach, creativity in solutions)
    - Time Management (conciseness, staying on topic)
    - Overall (composite score)

    Return ONLY a JSON object with these numeric ratings, nothing else.
    Example: {{"technical": 7.5, "communication": 5.0, "problem_solving": 6.2, "time_management": 5.8, "overall": 6.1}}
    """

    try:
        logger.debug(f"Evaluating response for question: {question[:50]}...")
        logger.debug(f"Answer length: {len(answer)} characters")
        
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": rating_prompt}],
            temperature=0.4,
            max_tokens=200
        )
        ratings = json.loads(response.choices[0].message.content)
        logger.debug(f"Received ratings: {ratings}")
        
        # Ensure all ratings are floats
        return {k: float(v) for k, v in ratings.items()}
    except Exception as e:
        logger.error(f"Error evaluating response: {str(e)}", exc_info=True)
        return {
            "technical": 5.0,
            "communication": 5.0,
            "problem_solving": 5.0,
            "time_management": 5.0,
            "overall": 5.0
        }

def generate_interview_report(interview_data):
    try:
        logger.debug("Starting interview report generation")
        
        # Calculate duration
        duration = "N/A"
        if interview_data['start_time'] and interview_data['end_time']:
            total_secs = (interview_data['end_time'] - interview_data['start_time']).total_seconds()
            m, s = divmod(int(total_secs), 60)
            duration = f"{m}m {s}s"
            logger.debug(f"Interview duration: {duration}")

        # Calculate average rating (on a 0-10 scale)
        avg_rating = 0.0
        if interview_data['ratings']:
            avg_rating = sum(r['overall'] for r in interview_data['ratings']) / len(interview_data['ratings'])
            logger.debug(f"Average rating: {avg_rating:.1f}")
        
        # Convert to percentage (0-100 scale)
        avg_percentage = avg_rating * 10
        
        # Determine status based on percentage ranges
        if avg_percentage >= 75:
            status = "Very Good"
            status_class = "status-Very-Good"
        elif avg_percentage >= 60:
            status = "Good"
            status_class = "status-Good"
        elif avg_percentage >= 50:
            status = "Average"
            status_class = "status-Average"
        else:
            status = "Poor"
            status_class = "status-Poor"
        
        logger.debug(f"Performance status: {status}")

        # Prepare conversation history
        conversation_history = "\n".join(
            f"{item['speaker']}: {item['text']}" 
            for item in interview_data['conversation_history']
            if 'speaker' in item
        )
        logger.debug(f"Conversation history length: {len(conversation_history)} characters")

        # Generate the detailed report with structured format
        report_prompt = f"""
Generate a professional interview performance report focusing ONLY on strengths and areas for improvement. 
The report should be structured with clear sections and use only information from the interview.

Interview Difficulty Level: {interview_data['difficulty_level'].capitalize()}
Interview Duration: {duration}

Format the report with these EXACT sections:

### Key Strengths
<table class="report-table">
<tr><th>Area</th><th>Examples</th><th>Rating</th></tr>
[Create table rows for each strength with specific examples from interview]
</table>

### Areas for Improvement
<table class="report-table">
<tr><th>Area</th><th>Suggestions</th></tr>
[Create table rows for each improvement area with actionable suggestions]
</table>

### Visual Feedback Summary
<table class="report-table">
<tr><th>Area</th><th>Feedback</th></tr>
[Create table rows for visual feedback with descriptive feedback only]
</table>

Conversation Transcript:
{conversation_history}
"""
        logger.debug(f"Report prompt length: {len(report_prompt)} characters")

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": report_prompt}],
            temperature=0.5,
            max_tokens=1500
        )
        report_content = response.choices[0].message.content
        logger.debug(f"Generated report content length: {len(report_content)} characters")

        # Generate structured ratings
        rating_prompt = f"""
Based on this interview transcript, provide a JSON object with ratings (1-10) and analysis for:
1. Technical Knowledge (accuracy, depth)
2. Communication Skills (clarity, articulation)
3. Problem Solving (logic, creativity)
4. Time Management (conciseness)
5. Overall Performance (composite)

For each category, include:
- rating (1-10)
- strengths (bullet points)
- improvement_suggestions (bullet points)

Format:
{{
  "technical_knowledge": {{"rating": number, "strengths": [], "improvement_suggestions": []}},
  "communication_skills": {{"rating": number, "strengths": [], "improvement_suggestions": []}},
  "problem_solving": {{"rating": number, "strengths": [], "improvement_suggestions": []}},
  "time_management": {{"rating": number, "strengths": [], "improvement_suggestions": []}},
  "overall_performance": {{"rating": number}}
}}

Transcript:
{conversation_history}
"""
        logger.debug("Generating category ratings")
        rating_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": rating_prompt}],
            temperature=0.3
        )
        
        try:
            category_ratings = json.loads(rating_response.choices[0].message.content)
            logger.debug(f"Category ratings: {category_ratings}")
            # Ensure all ratings are floats
            for category in category_ratings:
                if 'rating' in category_ratings[category]:
                    category_ratings[category]['rating'] = float(category_ratings[category]['rating'])
        except:
            logger.warning("Failed to parse category ratings, using fallback")
            category_ratings = {
                "technical_knowledge": {"rating": float(avg_rating), "strengths": [], "improvement_suggestions": []},
                "communication_skills": {"rating": float(avg_rating), "strengths": [], "improvement_suggestions": []},
                "problem_solving": {"rating": float(avg_rating), "strengths": [], "improvement_suggestions": []},
                "time_management": {"rating": float(avg_rating), "strengths": [], "improvement_suggestions": []},
                "overall_performance": {"rating": float(avg_rating)}
            }

        # Process visual feedback (descriptive only)
        visual_feedback = {
            "professional_appearance": "No visual feedback available",
            "body_language": "No visual feedback available",
            "environment": "No visual feedback available",
            "distractions": "No visual feedback available",
            "summary": "No visual feedback collected"
        }
        
        if interview_data['visual_feedback_data']:
            try:
                logger.debug("Processing visual feedback data")
                from collections import Counter
                professional_appearance = []
                body_language = []
                environment = []
                distractions = []
                
                for feedback in interview_data['visual_feedback_data']:
                    if 'feedback' in feedback and isinstance(feedback['feedback'], dict):
                        visual_data = feedback['feedback']
                        professional_appearance.append(visual_data.get('professional_appearance', 'No feedback'))
                        body_language.append(visual_data.get('body_language', 'No feedback'))
                        environment.append(visual_data.get('environment', 'No feedback'))
                        distractions.append(visual_data.get('distractions', 'No feedback'))
                
                def most_common_feedback(feedback_list):
                    if not feedback_list:
                        return "No feedback available"
                    counts = Counter(feedback_list)
                    return counts.most_common(1)[0][0]
                
                visual_feedback = {
                    "professional_appearance": most_common_feedback(professional_appearance),
                    "body_language": most_common_feedback(body_language),
                    "environment": most_common_feedback(environment),
                    "distractions": most_common_feedback(distractions),
                    "summary": "Visual feedback collected throughout the interview"
                }
                logger.debug(f"Visual feedback summary: {visual_feedback}")
            except Exception as e:
                logger.error(f"Error processing visual feedback: {str(e)}")

        # Generate voice feedback
        voice_prompt = f"""
Create a concise 3-sentence spoken feedback summary based on this report:
{report_content}

The feedback should:
- Be professional and factual
- Mention one strength and one area for improvement
- Include visual feedback if available
- Be exactly 3 sentences
"""
        logger.debug("Generating voice feedback")
        voice_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": voice_prompt}],
            temperature=0.6
        )
        voice_feedback = voice_response.choices[0].message.content
        logger.debug(f"Voice feedback: {voice_feedback}")
        voice_audio = text_to_speech(voice_feedback) if voice_feedback else None

        # Create summary card HTML
        summary_card = f"""
<div class="interview-summary-card" style="
    display: flex;
    justify-content: space-between;
    background: linear-gradient(135deg,#6e8efb,#a777e3);
    padding: 1rem;
    border-radius: 8px;
    color: white;
    margin-bottom: 1rem;
    font-family: sans-serif;
">
<div>
    <div><small>Candidate Name</small><br><strong>{interview_data['student_info'].get('name', 'N/A')}</strong></div>
    <div style="margin-top:0.5rem;"><small>Roll No</small><br><strong>{interview_data['student_info'].get('roll_no', 'N/A')}</strong></div>
    <div style="margin-top:0.5rem;"><small>Batch No</small><br><strong>{interview_data['student_info'].get('batch_no', 'N/A')}</strong></div>
</div>
<div>
    <div><small>Center</small><br><strong>{interview_data['student_info'].get('center', 'N/A')}</strong></div>
    <div style="margin-top:0.5rem;"><small>Course</small><br><strong>{interview_data['student_info'].get('course', 'N/A')}</strong></div>
    <div style="margin-top:0.5rem;"><small>Evaluation Date</small><br><strong>{interview_data['student_info'].get('eval_date', 'N/A')}</strong></div>
</div>
<div style="align-self:center;">
    <span class="{status_class}" style="
        background: gold;
        color: black;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        font-weight: bold;
    ">{status}</span>
</div>
</div>
"""
        logger.debug("Created summary card HTML")

        # Combine summary card with report content
        full_report_html = summary_card + report_content

        return {
            "status": "success",
            "report_html": full_report_html,
            "category_ratings": category_ratings,
            "voice_feedback": voice_feedback,
            "voice_audio": voice_audio,
            "status_class": status_class,
            "visual_feedback": visual_feedback
        }

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }



# @app.route("/login", methods=["GET", "POST"])
# @app.route("/", methods=["GET", "POST"])
# def login():
#     logger.debug(f"Login route accessed with method: {request.method}")
#     if request.method == "POST":
#         username = request.form.get("username")
#         password = request.form.get("password")
#         logger.debug(f"Login attempt for username: {username}")
        
#         # Admin static login
#         if username == "admin" and password == "admin123":
#             logger.debug("Admin login successful")
#             session["user"] = username
#             session["role"] = "recruiter"
#             return redirect("/recruiter_home")
#         else:
#             try:
#                 conn = get_postgres_connection()
#                 if not conn:
#                     raise Exception("Could not connect to postgres")
                    
#                 cs = conn.cursor()
#                 logger.debug(f"Checking credentials for user: {username}")
#                 cs.execute("SELECT PASSWORD FROM REGISTER WHERE EMAIL_ID=%s;", (username,))
#                 row = cs.fetchone()
#                 cs.close()
#                 conn.close()
                
#                 if row and check_password_hash(row[0], password):
#                     logger.debug("User login successful")
#                     session["user"] = username
#                     session["role"] = "student"
#                     return redirect("/dashboard")
#                 else:
#                     logger.warning("Invalid credentials")
#                     return render_template("login.html", error="Invalid credentials")
#             except Exception as e:
#                 logger.error(f"Login error: {e}")
#                 return render_template("login.html", error="Error during login")
#     return render_template("login.html")



@app.route("/login", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def login():
    logger.debug(f"Login route accessed with method: {request.method}")
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        logger.debug(f"Login attempt for username: {username}")

        try:
            conn = get_postgres_connection()
            if not conn:
                raise Exception("Could not connect to postgres")
            
            cs = conn.cursor()

            # ✅ Check if user is admin
            cs.execute("SELECT PASSWORD FROM ADMIN WHERE USERNAME=%s;", (username,))
            admin_row = cs.fetchone()
            logger.debug(f"Fetched admin row: {admin_row}")
            
            if admin_row:
                # ✅ Check password hash for admin
                is_correct = check_password_hash(admin_row[0], password)
                logger.debug(f"Password check result for admin: {is_correct}")
                
                if is_correct:
                    logger.debug("Admin login successful")
                    session["user"] = username
                    session["role"] = "recruiter"
                    cs.close()
                    conn.close()
                    return redirect("/recruiter_home")

            # ✅ If not admin, check in REGISTER table for student
            cs.execute("SELECT PASSWORD FROM REGISTER WHERE EMAIL_ID=%s;", (username,))
            user_row = cs.fetchone()
            logger.debug(f"Fetched student row: {user_row}")

            if user_row:
                # ✅ Check password hash for student
                is_correct = check_password_hash(user_row[0], password)
                logger.debug(f"Password check result for student: {is_correct}")

                if is_correct:
                    logger.debug("Student login successful")
                    session["user"] = username
                    session["role"] = "student"
                    cs.close()
                    conn.close()
                    return redirect("/dashboard")

            cs.close()
            conn.close()
            logger.warning("Invalid credentials")
            return render_template("login.html", error="Invalid credentials")
        except Exception as e:
            logger.error(f"Login error: {e}")
            return render_template("login.html", error="Error during login")
    return render_template("login.html")




@app.route("/register", methods=["GET", "POST"])
def register():
    logger.debug(f"Register route accessed with method: {request.method}")
    if request.method == "POST":
        name = request.form.get("name")
        course_name = request.form.get("course_name")
        email_id = request.form.get("email_id")
        mobile_no = request.form.get("mobile_no")
        center = request.form.get("center")
        batch_no = request.form.get("batch_no")
        password = request.form.get("password")
        
        logger.debug(f"Registration attempt for email: {email_id}")
        
        hashed_password = generate_password_hash(password)
        student_id = str(uuid.uuid4())

        try:
            conn = get_postgres_connection()
            if not conn:
                raise Exception("Could not connect to postgres")
                
            cs = conn.cursor()
            logger.debug("Creating REGISTER table if not exists")
            cs.execute("""
    CREATE TABLE IF NOT EXISTS REGISTER (
        STUDENT_ID VARCHAR PRIMARY KEY,
        NAME VARCHAR,
        COURSE_NAME VARCHAR,
        EMAIL_ID VARCHAR,
        MOBILE_NO VARCHAR,
        CENTER VARCHAR,
        BATCH_NO VARCHAR,
        PASSWORD VARCHAR,
        CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

            """)

            logger.debug("Inserting new user registration")
            cs.execute("""
                INSERT INTO REGISTER (STUDENT_ID, NAME, COURSE_NAME, EMAIL_ID, MOBILE_NO, CENTER, BATCH_NO, PASSWORD)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """, (student_id, name, course_name, email_id, mobile_no, center, batch_no, hashed_password))
            conn.commit()
            cs.close()
            conn.close()
            logger.info(f"Registration successful for email: {email_id}")
            return render_template("login.html", message="Registration successful! Please login.")
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            return render_template("register.html", error="Registration failed. Please try again.")
    return render_template("register.html")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    logger.debug(f"Forgot password route accessed with method: {request.method}")
    if request.method == "POST":
        email = request.form.get("username")
        if not email:
            logger.warning("Empty email in forgot password request")
            return render_template("forgot_password.html", error="Please enter your email ID")
        
        # Generate a new random password
        new_password = ''.join(random.choices(string.ascii_letters + string.digits + "!@#$%^&*", k=10))
        hashed_password = generate_password_hash(new_password)
        logger.debug(f"Password reset for email: {email}")

        try:
            conn = get_postgres_connection()
            if not conn:
                raise Exception("Could not connect to postgres")
                
            cs = conn.cursor()

            # Check if user exists
            cs.execute("SELECT * FROM REGISTER WHERE EMAIL_ID=%s", (email,))
            user = cs.fetchone()
            if not user:
                logger.warning(f"No account found for email: {email}")
                return render_template("forgot_password.html", error="No account found with that email.")

            # Update the password in Snowflake
            logger.debug(f"Updating password for email: {email}")
            cs.execute("UPDATE REGISTER SET PASSWORD=%s WHERE EMAIL_ID=%s", (hashed_password, email))
            conn.commit()

            # Send email via Outlook SMTP
            msg = MIMEText(f"""Hello,

Your new password is: {new_password}

Please login here: {LOGIN_URL}
For security, please change your password after login.

Thanks,
Interview Bot Team
""")
            msg['Subject'] = 'Password Reset - Interview Bot'
            msg['From'] = OUTLOOK_EMAIL
            msg['To'] = email

            logger.debug(f"Sending password reset email to: {email}")
            with smtplib.SMTP('smtp.office365.com', 587) as server:
                server.starttls()
                server.login(OUTLOOK_EMAIL, OUTLOOK_PASSWORD)
                server.send_message(msg)

            cs.close()
            conn.close()

            logger.info(f"Password reset email sent to: {email}")
            return render_template("forgot_password.html", message="A new password has been sent to your email.")
        except Exception as e:
            logger.error(f"Error resetting password: {e}")
            return render_template("forgot_password.html", error="Error resetting password.")

    return render_template("forgot_password.html")

@app.route("/recruiter_home")
def recruiter_home():
    logger.debug("Recruiter home route accessed")
    if "user" not in session or session.get("role") != "recruiter":
        logger.warning("Unauthorized access to recruiter home")
        return redirect(url_for("login"))

    try:
        conn = get_postgres_connection()
        if not conn:
            raise Exception("Could not connect to postgres")
            
        cs = conn.cursor()

        # Fetch interview ratings
        logger.debug("Fetching interview ratings from postgres")
        cs.execute("""
            SELECT roll_no, technical_rating, communication_rating, problem_solving_rating,
                   time_management_rating, total_rating, interview_ts
            FROM interview_rating
            ORDER BY interview_ts DESC
            LIMIT 100
        """)
        ratings_rows = cs.fetchall()
        ratings_cols = [desc[0].lower() for desc in cs.description]
        logger.debug(f"Fetched {len(ratings_rows)} interview ratings")

        # Prepare interview ratings data JSON (only required columns)
        interview_ratings_json = json.dumps([
            {
                "roll_no": row[0],
                "technical_rating": row[1],
                "communication_rating": row[2],
                "problem_solving_rating": row[3],
                "time_management_rating": row[4],
                "total_rating": row[5],
                "interview_ts": row[6].strftime('%Y-%m-%d') if row[6] else None
            }
            for row in ratings_rows
        ])

        # Fetch student info
        logger.debug("Fetching student info from Snowflake")
        cs.execute("""
            SELECT student_name, roll_no, batch_no, center, course, evaluation_date, difficulty_level, interview_ts
            FROM student_info
            ORDER BY interview_ts DESC
            LIMIT 100
        """)
        students_rows = cs.fetchall()
        students_cols = [desc[0].lower() for desc in cs.description]
        logger.debug(f"Fetched {len(students_rows)} student records")

        # Fetch visual feedback data
        logger.debug("Fetching visual feedback from Snowflake")
        cs.execute("""
            SELECT roll_no, professional_appearance, body_language, environment, 
                   distractions, interview_ts
            FROM visual_feedback
            ORDER BY interview_ts DESC
            LIMIT 100
        """)
        visual_feedback_rows = cs.fetchall()
        visual_feedback_cols = [desc[0].lower() for desc in cs.description]
        logger.debug(f"Fetched {len(visual_feedback_rows)} visual feedback records")

        # Prepare student info data JSON (only required columns)
        student_info_json = json.dumps([
            {
                "student_name": row[0],
                "center": row[3],
                "course": row[4],
                "evaluation_date": row[5].strftime('%Y-%m-%d') if isinstance(row[5], (datetime)) else row[5],
                "difficulty_level": row[6]
            }
            for row in students_rows
        ])

        cs.close()
        conn.close()

        return render_template(
            "recruiter_home.html",
            username=session.get("user"),
            interview_ratings=ratings_rows,
            interview_ratings_cols=ratings_cols,
            student_info=students_rows,
            student_info_cols=students_cols,
            visual_feedback=visual_feedback_rows,
            visual_feedback_cols=visual_feedback_cols,
            interview_ratings_json=interview_ratings_json,
            student_info_json=student_info_json
        )
    except Exception as e:
        logger.error(f"Error in recruiter_home: {str(e)}", exc_info=True)
        return f"Error loading data: {e}"

@app.route("/dashboard")
def dashboard():
    logger.debug("Dashboard route accessed")
    if "user" not in session:
        logger.warning("Unauthorized access to dashboard")
        return redirect(url_for("login"))

    try:
        conn = get_postgres_connection()
        if not conn:
            raise Exception("Could not connect to Snowflake")
            
        cs = conn.cursor()

        # Fetch interview_rating data ordered by interview timestamp
        logger.debug("Fetching interview ratings for dashboard")
        cs.execute("""
            SELECT roll_no, technical_rating, communication_rating, problem_solving_rating,
                   time_management_rating, total_rating, interview_ts
            FROM interview_rating
            WHERE total_rating IS NOT NULL
            ORDER BY interview_ts
            LIMIT 200
        """)
        rows = cs.fetchall()
        cols = [desc[0].lower() for desc in cs.description]
        logger.debug(f"Fetched {len(rows)} interview ratings")

        # Convert to DataFrame
        df_ratings = pd.DataFrame(rows, columns=cols)

        # Calculate averages per interview index
        df_ratings['interview_number'] = range(1, len(df_ratings) + 1)

        # Skill wise averages across all interviews for bar chart
        skill_avg = {
            "Technical Knowledge": df_ratings["technical_rating"].mean(),
            "Communication": df_ratings["communication_rating"].mean(),
            "Problem Solving": df_ratings["problem_solving_rating"].mean(),
            "Time Management": df_ratings["time_management_rating"].mean()
        }
        logger.debug(f"Calculated skill averages: {skill_avg}")

        # Prepare data for line graph: interview_number vs total_rating avg (or total_rating)
        line_data = {
            "interview_numbers": df_ratings["interview_number"].tolist(),
            "avg_ratings": df_ratings["total_rating"].tolist()
        }

        # KPI calculations
        average_rating = round(df_ratings["total_rating"].mean(), 1) if not df_ratings.empty else 0
        completed_interviews = len(df_ratings)
        logger.debug(f"Dashboard KPIs - Avg rating: {average_rating}, Completed interviews: {completed_interviews}")

        cs.close()
        conn.close()

        # Convert to JSON for passing to JS
        skill_avg_json = json.dumps(skill_avg)
        line_data_json = json.dumps(line_data)

        return render_template(
            "dashboard.html",
            user_name=session.get("user"),
            skill_avg=skill_avg_json,
            line_data=line_data_json,
            average_rating=average_rating,
            completed_interviews=completed_interviews
        )

    except Exception as e:
        logger.error(f"Error in dashboard: {str(e)}", exc_info=True)
        return f"Error: {e}"

@app.route("/interview")
def interview_bot():
    logger.debug("Interview bot route accessed")
    if "user" not in session:
        logger.warning("Unauthorized access to interview bot")
        return redirect(url_for("login"))
    
    # Initialize with a minimal session structure
    session['interview_data'] = {
        'questions': [],
        'current_question': 0,
        'interview_started': False,
        'student_info': {
            'name': '',
            'roll_no': '',
            'batch_no': '',
            'center': '',
            'course': '',
            'eval_date': ''
        }
    }
    logger.debug("Initialized new interview session data")
    return render_template("interview_bot.html")

@app.route("/logout")
def logout():
    logger.debug(f"Logout requested by user: {session.get('user')}")
    session.clear()
    return redirect(url_for("login"))

@app.route('/start_interview', methods=['POST'])
def start_interview():
    logger.debug("Start interview endpoint called")
    if "user" not in session:
        logger.warning("Unauthenticated start interview attempt")
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
    data = request.get_json()
    session['interview_data'] = init_interview_data()
    interview_data = session['interview_data']
    
    interview_data['difficulty_level'] = data.get('difficulty_level', 'medium')
    interview_data['student_info'] = {
        'name': data.get('name', ''),
        'roll_no': data.get('roll_no', ''),
        'batch_no': data.get('batch_no', ''),
        'center': data.get('center', ''),
        'course': data.get('course', ''),
        'eval_date': data.get('eval_date', '')
    }
    interview_data['jd_text'] = data.get('jd_text', '')
    interview_data['start_time'] = datetime.now(timezone.utc)
    interview_data['last_activity_time'] = datetime.now(timezone.utc)
    
    logger.debug(f"Starting interview with data: {interview_data['student_info']}")
    logger.debug(f"Difficulty level: {interview_data['difficulty_level']}")
    logger.debug(f"JD text length: {len(interview_data['jd_text'])} characters")
    
    try:
        questions = generate_questions_from_jd(
            interview_data['jd_text'],
            interview_data['difficulty_level'],
            interview_data['student_info'].get('roll_no')
        )
        
        if not questions:
            logger.error("No questions generated, using fallback questions.")
            questions = ["Tell us about yourself.", "What programming languages do you know?", "Explain a basic project you've worked on."]
        
        interview_data['questions'] = questions
        interview_data['interview_started'] = True  # Mark interview as started
        session['interview_data'] = interview_data
        
        logger.info(f"Interview started with {len(questions)} questions")
        return jsonify({
            "status": "started",
            "total_questions": len(interview_data['questions']),
            "welcome_message": f"Welcome to the interview. Let's begin with the first question."
        })
    except Exception as e:
        logger.error(f"Error in start_interview: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/upload_jd', methods=['POST'])
def upload_jd():
    logger.debug("JD upload endpoint called")
    if "user" not in session:
        logger.warning("Unauthenticated JD upload attempt")
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
    if 'jd_file' not in request.files:
        logger.warning("No file in JD upload request")
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
        
    file = request.files['jd_file']
    if file.filename == '':
        logger.warning("Empty filename in JD upload")
        return jsonify({"status": "error", "message": "No file selected"}), 400
        
    logger.debug(f"Processing JD file: {file.filename}")
    jd_text = extract_text_from_file(file)
    if not jd_text:
        logger.warning("Could not extract text from JD file")
        return jsonify({"status": "error", "message": "Could not extract text from file"}), 400
        
    logger.info(f"Successfully extracted JD text (length: {len(jd_text)} characters)")
    return jsonify({
        "status": "success",
        "jd_text": jd_text
    })

@app.route('/get_question', methods=['GET'])
def get_question():
    logger.debug("Get question endpoint called")
    if "user" not in session:
        logger.warning("Unauthenticated get question attempt")
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
    interview_data = session.get('interview_data', None)
    
    if not interview_data or not interview_data.get('interview_started', False):  # Ensure interview has started
        logger.warning("Attempt to get question before interview started")
        return jsonify({"status": "not_started"})
    
    if interview_data['current_question'] >= len(interview_data['questions']):
        logger.info("All questions have been asked")
        return jsonify({"status": "completed"})
    
    current_q = interview_data['questions'][interview_data['current_question']]
    interview_data['current_question'] += 1
    
    interview_data['conversation_history'].append({"speaker": "bot", "text": current_q})
    interview_data['current_answer'] = ""
    interview_data['waiting_for_answer'] = True  # Set flag to indicate we're waiting for answer
    save_conversation_to_file([{"speaker": "bot", "text": current_q}], interview_data['student_info'].get('roll_no'))
    interview_data['last_activity_time'] = datetime.now(timezone.utc)
    session['interview_data'] = interview_data
    
    logger.debug(f"Question {interview_data['current_question']}: {current_q[:50]}...")
    
    audio_data = text_to_speech(current_q)
    
    return jsonify({
        "status": "success",
        "question": current_q,
        "audio": audio_data,
        "question_number": interview_data['current_question'],
        "total_questions": len(interview_data['questions'])
    })

@app.route('/process_answer', methods=['POST'])
def process_answer():
    logger.debug("Process answer endpoint called")
    if "user" not in session:
        logger.warning("Unauthenticated process answer attempt")
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
    interview_data = session.get('interview_data', init_interview_data())
    
    if interview_data.get('is_processing_answer', False):
        logger.warning("Answer processing already in progress")
        return jsonify({"status": "processing"})
    
    interview_data['is_processing_answer'] = True
    session['interview_data'] = interview_data
    
    if not interview_data['interview_started']:
        logger.warning("Attempt to process answer before interview started")
        interview_data['is_processing_answer'] = False
        session['interview_data'] = interview_data
        return jsonify({"status": "error", "message": "Interview not started"}), 400
    
    data = request.get_json()
    answer = data.get('answer', '').strip()
    frame_data = data.get('frame', None)
    audio_data = data.get('audio', None)
    is_final = data.get('is_final', False)
    speaking_time = data.get('speaking_time', 0)
    
    logger.debug(f"Processing answer (is_final: {is_final}, speaking_time: {speaking_time}s)")
    logger.debug(f"Answer text length: {len(answer)} characters")
    
    interview_data['interview_time_used'] += speaking_time
    
    if interview_data['interview_time_used'] >= INTERVIEW_DURATION:
        logger.info("Interview duration limit reached")
        interview_data['end_time'] = datetime.now(timezone.utc)
        interview_data['is_processing_answer'] = False
        session['interview_data'] = interview_data
        return jsonify({
            "status": "interview_complete",
            "message": "Interview duration limit reached"
        })
    
    if not is_final:
        logger.debug("Accumulating partial answer")
        interview_data['current_answer'] = answer
        session['interview_data'] = interview_data
        interview_data['is_processing_answer'] = False
        session['interview_data'] = interview_data
        return jsonify({
            "status": "answer_accumulated",
            "remaining_time": max(0, INTERVIEW_DURATION - interview_data['interview_time_used'])
        })
    
    if not answer and interview_data['current_answer']:
        answer = interview_data['current_answer']
    
    if not answer:
        logger.warning("Empty answer received")
        interview_data['is_processing_answer'] = False
        session['interview_data'] = interview_data
        return jsonify({"status": "error", "message": "Empty answer"}), 400
    
    if audio_data:
        try:
            logger.debug("Processing audio data with VAD")
            # Use WebRTC VAD to detect speech
            has_speech, speech_ratio = process_audio_from_base64(audio_data)
            interview_data['speech_detected'] = has_speech
            interview_data['last_speech_time'] = datetime.now(timezone.utc) if has_speech else None
            logger.debug(f"Speech detection - has_speech: {has_speech}, ratio: {speech_ratio:.2f}")
        except Exception as e:
            logger.error(f"Error processing audio with VAD: {str(e)}", exc_info=True)
    
    current_question = interview_data['conversation_history'][-1]['text']
    
    interview_data['answers'].append(answer)
    interview_data['conversation_history'].append({"speaker": "user", "text": answer})
    interview_data['current_answer'] = ""
    interview_data['waiting_for_answer'] = False  # Clear flag as answer is received
    save_conversation_to_file([{"speaker": "user", "text": answer}], interview_data['student_info'].get('roll_no'))
    interview_data['last_activity_time'] = datetime.now(timezone.utc)
    
    visual_feedback = None
    current_time = datetime.now().timestamp()
    if frame_data and (current_time - interview_data['last_frame_time']) > FRAME_CAPTURE_INTERVAL:
        try:
            logger.debug("Processing frame data")
            frame_bytes = base64.b64decode(frame_data.split(',')[1])
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                frame_base64 = process_frame_for_gpt4v(frame)
                visual_feedback = analyze_visual_response(
                    frame_base64,
                    interview_data['conversation_history'][-3:]
                )
                if visual_feedback:
                    interview_data['visual_feedback'].append(visual_feedback)
                    interview_data['visual_feedback_data'].append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "feedback": visual_feedback
                    })
                    interview_data['last_frame_time'] = current_time
                    logger.debug(f"Visual feedback: {visual_feedback}")
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}", exc_info=True)
    
    logger.debug("Evaluating response")
    rating = evaluate_response(
        answer, 
        current_question, 
        interview_data['difficulty_level'],
        visual_feedback
    )
    interview_data['ratings'].append(rating)
    logger.debug(f"Response rating: {rating}")
    
    interview_data['is_processing_answer'] = False
    session['interview_data'] = interview_data
    
    return jsonify({
        "status": "answer_processed",
        "current_question": interview_data['current_question'],
        "total_questions": len(interview_data['questions']),
        "interview_complete": interview_data['current_question'] >= len(interview_data['questions']),
        "remaining_time": INTERVIEW_DURATION
    })

@app.route('/check_speech', methods=['POST'])
def check_speech():
    logger.debug("Check speech endpoint called")
    if "user" not in session:
        logger.warning("Unauthenticated check speech attempt")
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
    interview_data = session.get('interview_data', init_interview_data())
    
    if not interview_data['interview_started']:
        logger.warning("Attempt to check speech before interview started")
        return jsonify({"status": "not_started"})
    
    data = request.get_json()
    audio_data = data.get('audio', None)
    
    if not audio_data:
        logger.warning("No audio data in check speech request")
        return jsonify({"status": "error", "message": "No audio data"}), 400
    
    try:
        logger.debug("Checking speech in audio data")
        has_speech, speech_ratio = process_audio_from_base64(audio_data)
        interview_data['speech_detected'] = has_speech
        interview_data['last_speech_time'] = datetime.now(timezone.utc) if has_speech else None
        session['interview_data'] = interview_data
        
        speech_ended = False
        silence_duration = 0
        if interview_data['last_speech_time']:
            silence_duration = (datetime.now(timezone.utc) - interview_data['last_speech_time']).total_seconds()
            speech_ended = silence_duration > PAUSE_THRESHOLD
        
        logger.debug(f"Speech detection - has_speech: {has_speech}, ratio: {speech_ratio:.2f}, silence_duration: {silence_duration:.1f}s, speech_ended: {speech_ended}")
        
        return jsonify({
            "status": "success",
            "speech_detected": has_speech,
            "speech_ratio": speech_ratio,
            "speech_ended": speech_ended,
            "silence_duration": silence_duration if has_speech else 0
        })
    except Exception as e:
        logger.error(f"Error checking speech: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/check_pause', methods=['GET'])
def check_pause():
    logger.debug("Check pause endpoint called")
    if "user" not in session:
        logger.warning("Unauthenticated check pause attempt")
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
    interview_data = session.get('interview_data', init_interview_data())
    
    if not interview_data['interview_started']:
        logger.warning("Attempt to check pause before interview started")
        return jsonify({"status": "not_started"})
    
    # Only check for pause if we're actually waiting for an answer
    if not interview_data.get('waiting_for_answer', False):
        logger.debug("Not waiting for answer, skip pause check")
        return jsonify({"status": "active"})
    
    current_time = datetime.now(timezone.utc)
    last_activity = interview_data['last_activity_time']
    seconds_since_activity = (current_time - last_activity).total_seconds() if last_activity else 0
    
    logger.debug(f"Time since last activity: {seconds_since_activity:.1f}s (threshold: {PAUSE_THRESHOLD}s)")
    
    if seconds_since_activity > PAUSE_THRESHOLD:
        logger.info("Pause detected, generating encouragement")
        encouragement = generate_encouragement_prompt(interview_data['conversation_history'])
        audio_data = text_to_speech(encouragement)
        interview_data['last_activity_time'] = current_time
        session['interview_data'] = interview_data
        
        return jsonify({
            "status": "pause_detected",
            "prompt": encouragement,
            "audio": audio_data
        })
    
    return jsonify({"status": "active"})

@app.route('/generate_report', methods=['GET'])
def generate_report():
    logger.debug("Generate report endpoint called")
    if "user" not in session:
        logger.warning("Unauthenticated generate report attempt")
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
    interview_data = session.get('interview_data', init_interview_data())
    
    if not interview_data['interview_started']:
        logger.warning("Attempt to generate report before interview started")
        return jsonify({"status": "error", "message": "Interview not started"}), 400
    
    # If report was already generated, return the cached version
    if interview_data.get('report_generated', False):
        logger.debug("Report already generated, returning cached version")
        # Load conversation history from file to ensure consistency
        roll_no = interview_data['student_info'].get('roll_no')
        if roll_no:
            interview_data['conversation_history'] = load_conversation_from_file(roll_no)
        
        report = generate_interview_report(interview_data)
        return jsonify({
            "status": "success",
            "report": report['report_html'],
            "ratings": report['category_ratings'],
            "voice_feedback": report['voice_feedback'],
            "voice_audio": report['voice_audio'],
            "status_class": report['status_class'],
            "visual_feedback": report.get('visual_feedback', {})
        })
    
    if not interview_data['end_time']:
        interview_data['end_time'] = datetime.now(timezone.utc)
        session['interview_data'] = interview_data
    
    report = generate_interview_report(interview_data)
    
    if report['status'] == 'error':
        logger.error(f"Error generating report: {report['message']}")
        return jsonify(report), 500

    # Mark report as generated to prevent duplicate entries
    interview_data['report_generated'] = True
    session['interview_data'] = interview_data

    try:
        conn = get_postgres_connection()
        if not conn:
            raise Exception("Could not connect to Snowflake")
            
        cs = conn.cursor()
        
        # Check if this student already has an entry
        roll_no = interview_data['student_info'].get('roll_no') or session.get('user')
        interview_ts = interview_data['end_time']
        
        # Create tables if they don't exist
        logger.debug("Creating tables if they don't exist")
        cs.execute("""
            CREATE TABLE IF NOT EXISTS REGISTER (
                STUDENT_ID VARCHAR PRIMARY KEY,
                NAME VARCHAR,
                COURSE_NAME VARCHAR,
                EMAIL_ID VARCHAR,
                MOBILE_NO VARCHAR,
                CENTER VARCHAR,
                BATCH_NO VARCHAR,
                PASSWORD VARCHAR,
                CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS interview_rating(
                roll_no TEXT,
                technical_rating FLOAT,
                communication_rating FLOAT,
                problem_solving_rating FLOAT,
                time_management_rating FLOAT,
                total_rating FLOAT,
                interview_ts TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS student_info (
                student_name TEXT,
                roll_no TEXT,
                batch_no TEXT,
                center TEXT,
                course TEXT,
                evaluation_date TEXT,
                difficulty_level TEXT,
                interview_ts TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS visual_feedback (
                roll_no TEXT,
                professional_appearance TEXT,
                body_language TEXT,
                environment TEXT,
                distractions TEXT,
                interview_ts TIMESTAMP
            );
        """)
        
        # Insert interview ratings
        logger.debug("Inserting interview ratings into Snowflake")
        cs.execute("""
            INSERT INTO interview_rating
              (roll_no, technical_rating, communication_rating,
               problem_solving_rating, time_management_rating,
               total_rating, interview_ts)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """, (
            roll_no,
            report['category_ratings']['technical_knowledge']['rating'],
            report['category_ratings']['communication_skills']['rating'],
            report['category_ratings']['problem_solving']['rating'],
            report['category_ratings']['time_management']['rating'],
            report['category_ratings']['overall_performance']['rating'],
            interview_ts
         ))
        
        # Insert student information
        logger.debug("Inserting student info into Snowflake")
        cs.execute("""
            INSERT INTO student_info
              (student_name, roll_no, batch_no, center, course,
               evaluation_date, difficulty_level, interview_ts)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            interview_data['student_info'].get('name', ''),
            roll_no,
            interview_data['student_info'].get('batch_no', ''),
            interview_data['student_info'].get('center', ''),
            interview_data['student_info'].get('course', ''),
            interview_data['student_info'].get('eval_date', ''),
            interview_data['difficulty_level'],
            interview_ts
         ))

        # Insert visual feedback data if available
        if interview_data['visual_feedback_data']:
            try:
                logger.debug("Inserting visual feedback into Snowflake")
                # Get the most common feedback from all visual feedback entries
                professional_appearance = []
                body_language = []
                environment = []
                distractions = []
                
                for feedback in interview_data['visual_feedback_data']:
                    if 'feedback' in feedback and isinstance(feedback['feedback'], dict):
                        visual_data = feedback['feedback']
                        professional_appearance.append(visual_data.get('professional_appearance', 'No feedback'))
                        body_language.append(visual_data.get('body_language', 'No feedback'))
                        environment.append(visual_data.get('environment', 'No feedback'))
                        distractions.append(visual_data.get('distractions', 'No feedback'))
                
                # Get the most frequent feedback for each category
                def most_common_feedback(feedback_list):
                    if not feedback_list:
                        return "No feedback available"
                    counts = Counter(feedback_list)
                    return counts.most_common(1)[0][0]
                
                visual_feedback_data = {
                    "professional_appearance": most_common_feedback(professional_appearance),
                    "body_language": most_common_feedback(body_language),
                    "environment": most_common_feedback(environment),
                    "distractions": most_common_feedback(distractions)
                }
                
                cs.execute("""
                    INSERT INTO visual_feedback
                      (roll_no, professional_appearance, body_language,
                       environment, distractions, interview_ts)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (
                    roll_no,
                    visual_feedback_data['professional_appearance'],
                    visual_feedback_data['body_language'],
                    visual_feedback_data['environment'],
                    visual_feedback_data['distractions'],
                    interview_ts
                ))
                logger.info("Successfully saved visual feedback to Snowflake")
            except Exception as e:
                logger.error(f"Error saving visual feedback: {str(e)}", exc_info=True)

        conn.commit()
        logger.info("Successfully saved all interview data to Snowflake")
    except Exception as e:
        logger.error(f"Snowflake insert failed: {e}")
    finally:
        if 'cs' in locals(): cs.close()
        if 'conn' in locals(): conn.close()
 
    return jsonify({
        "status": "success",
        "report": report['report_html'],
        "ratings": report['category_ratings'],
        "voice_feedback": report['voice_feedback'],
        "voice_audio": report['voice_audio'],
        "status_class": report['status_class'],
        "visual_feedback": report.get('visual_feedback', {})
    })

@app.route('/reset_interview', methods=['POST'])
def reset_interview():
    logger.debug("Reset interview endpoint called")
    if "user" not in session:
        logger.warning("Unauthenticated reset interview attempt")
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
    session.clear()
    session['interview_data'] = init_interview_data()
    logger.info("Interview session reset")
    return jsonify({"status": "success", "message": "Interview reset successfully"})

@app.route('/export_interview_ratings')
def export_interview_ratings():
    logger.debug("Export interview ratings endpoint called")
    if "user" not in session or session.get("role") != "recruiter":
        logger.warning("Unauthorized export attempt")
        return redirect(url_for("login"))
    try:
        conn = get_postgres_connection()
        if not conn:
            raise Exception("Could not connect to Snowflake")
            
        cs = conn.cursor()
        interview_data = session.get('interview_data', init_interview_data())

        if not interview_data['interview_started']:
            logger.warning("Attempt to check pause before interview started")
            return jsonify({"status": "not_started"})
        
        # Only check for pause if we're actually waiting for an answer
        if not interview_data.get('waiting_for_answer', False):
            logger.debug("Not waiting for answer, skip pause check")
            return jsonify({"status": "active"})
        
        current_time = datetime.now(timezone.utc)
        last_activity = interview_data['last_activity_time']
        seconds_since_activity = (current_time - last_activity).total_seconds() if last_activity else 0
        
        logger.debug(f"Time since last activity: {seconds_since_activity:.1f}s (threshold: {PAUSE_THRESHOLD}s)")
        
        if seconds_since_activity > PAUSE_THRESHOLD:
            logger.info("Pause detected, generating encouragement")
            encouragement = generate_encouragement_prompt(interview_data['conversation_history'])
            audio_data = text_to_speech(encouragement)
            interview_data['last_activity_time'] = current_time
            session['interview_data'] = interview_data
            
            return jsonify({
                "status": "pause_detected",
                "prompt": encouragement,
                "audio": audio_data
            })
        
        return jsonify({"status": "active"})
    except Exception as e:
        logger.error(f"Error in export_interview_ratings: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/generate_report', methods=['GET'])
# def generate_report():
#     logger.debug("Generate report endpoint called")
#     if "user" not in session:
#         logger.warning("Unauthenticated generate report attempt")
#         return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
#     interview_data = session.get('interview_data', init_interview_data())
    
#     if not interview_data['interview_started']:
#         logger.warning("Attempt to generate report before interview started")
#         return jsonify({"status": "error", "message": "Interview not started"}), 400
    
#     # If report was already generated, return the cached version
#     if interview_data.get('report_generated', False):
#         logger.debug("Report already generated, returning cached version")
#         # Load conversation history from file to ensure consistency
#         roll_no = interview_data['student_info'].get('roll_no')
#         if roll_no:
#             interview_data['conversation_history'] = load_conversation_from_file(roll_no)
        
#         report = generate_interview_report(interview_data)
#         return jsonify({
#             "status": "success",
#             "report": report['report_html'],
#             "ratings": report['category_ratings'],
#             "voice_feedback": report['voice_feedback'],
#             "voice_audio": report['voice_audio'],
#             "status_class": report['status_class'],
#             "visual_feedback": report.get('visual_feedback', {})
#         })
    
#     if not interview_data['end_time']:
#         interview_data['end_time'] = datetime.now(timezone.utc)
#         session['interview_data'] = interview_data
    
#     report = generate_interview_report(interview_data)
    
#     if report['status'] == 'error':
#         logger.error(f"Error generating report: {report['message']}")
#         return jsonify(report), 500

#     # Mark report as generated to prevent duplicate entries
#     interview_data['report_generated'] = True
#     session['interview_data'] = interview_data

#     try:
#         conn = get_postgres_connection()
#         if not conn:
#             raise Exception("Could not connect to Snowflake")
            
#         cs = conn.cursor()
        
#         # Check if this student already has an entry
#         roll_no = interview_data['student_info'].get('roll_no') or session.get('user')
#         interview_ts = interview_data['end_time']
        
#         # Create tables if they don't exist
#         logger.debug("Creating tables if they don't exist")
#         cs.execute("""
#             CREATE TABLE IF NOT EXISTS interview_rating(
#               roll_no TEXT,
#               technical_rating FLOAT,
#               communication_rating FLOAT,
#               problem_solving_rating FLOAT,
#               time_management_rating FLOAT,
#               total_rating FLOAT,
#               interview_ts TIMESTAMP
#             );
#         """)
        
#         cs.execute("""
#             CREATE TABLE IF NOT EXISTS student_info (
#               student_name TEXT,
#               roll_no TEXT,
#               batch_no TEXT,
#               center TEXT,
#               course TEXT,
#               evaluation_date TEXT,
#               difficulty_level TEXT,
#               interview_ts TIMESTAMP
#             );
#         """)
        
#         cs.execute("""
#             CREATE TABLE IF NOT EXISTS visual_feedback (
#               roll_no TEXT,
#               professional_appearance TEXT,
#               body_language TEXT,
#               environment TEXT,
#               distractions TEXT,
#               interview_ts TIMESTAMP
#             );
#         """)

#         # Insert interview ratings
#         logger.debug("Inserting interview ratings into Snowflake")
#         cs.execute("""
#             INSERT INTO interview_rating
#               (roll_no, technical_rating, communication_rating,
#                problem_solving_rating, time_management_rating,
#                total_rating, interview_ts)
#             VALUES (%s, %s, %s, %s, %s, %s, %s);
#         """, (
#             roll_no,
#             report['category_ratings']['technical_knowledge']['rating'],
#             report['category_ratings']['communication_skills']['rating'],
#             report['category_ratings']['problem_solving']['rating'],
#             report['category_ratings']['time_management']['rating'],
#             report['category_ratings']['overall_performance']['rating'],
#             interview_ts
#          ))
        
#         # Insert student information
#         logger.debug("Inserting student info into Snowflake")
#         cs.execute("""
#             INSERT INTO student_info
#               (student_name, roll_no, batch_no, center, course,
#                evaluation_date, difficulty_level, interview_ts)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
#         """, (
#             interview_data['student_info'].get('name', ''),
#             roll_no,
#             interview_data['student_info'].get('batch_no', ''),
#             interview_data['student_info'].get('center', ''),
#             interview_data['student_info'].get('course', ''),
#             interview_data['student_info'].get('eval_date', ''),
#             interview_data['difficulty_level'],
#             interview_ts
#          ))

#         # Insert visual feedback data if available
#         if interview_data['visual_feedback_data']:
#             try:
#                 logger.debug("Inserting visual feedback into Snowflake")
#                 # Get the most common feedback from all visual feedback entries
#                 professional_appearance = []
#                 body_language = []
#                 environment = []
#                 distractions = []
                
#                 for feedback in interview_data['visual_feedback_data']:
#                     if 'feedback' in feedback and isinstance(feedback['feedback'], dict):
#                         visual_data = feedback['feedback']
#                         professional_appearance.append(visual_data.get('professional_appearance', 'No feedback'))
#                         body_language.append(visual_data.get('body_language', 'No feedback'))
#                         environment.append(visual_data.get('environment', 'No feedback'))
#                         distractions.append(visual_data.get('distractions', 'No feedback'))
                
#                 # Get the most frequent feedback for each category
#                 def most_common_feedback(feedback_list):
#                     if not feedback_list:
#                         return "No feedback available"
#                     counts = Counter(feedback_list)
#                     return counts.most_common(1)[0][0]
                
#                 visual_feedback_data = {
#                     "professional_appearance": most_common_feedback(professional_appearance),
#                     "body_language": most_common_feedback(body_language),
#                     "environment": most_common_feedback(environment),
#                     "distractions": most_common_feedback(distractions)
#                 }
                
#                 cs.execute("""
#                     INSERT INTO visual_feedback
#                       (roll_no, professional_appearance, body_language,
#                        environment, distractions, interview_ts)
#                     VALUES (%s, %s, %s, %s, %s, %s);
#                 """, (
#                     roll_no,
#                     visual_feedback_data['professional_appearance'],
#                     visual_feedback_data['body_language'],
#                     visual_feedback_data['environment'],
#                     visual_feedback_data['distractions'],
#                     interview_ts
#                 ))
#                 logger.info("Successfully saved visual feedback to Snowflake")
#             except Exception as e:
#                 logger.error(f"Error saving visual feedback: {str(e)}", exc_info=True)

#         conn.commit()
#         logger.info("Successfully saved all interview data to Snowflake")
#     except Exception as e:
#         logger.error(f"Snowflake insert failed: {e}")
#     finally:
#         if 'cs' in locals(): cs.close()
#         if 'conn' in locals(): conn.close()
 
#     return jsonify({
#         "status": "success",
#         "report": report['report_html'],
#         "ratings": report['category_ratings'],
#         "voice_feedback": report['voice_feedback'],
#         "voice_audio": report['voice_audio'],
#         "status_class": report['status_class'],
#         "visual_feedback": report.get('visual_feedback', {})
#     })



# @app.route('/reset_interview', methods=['POST'])
# def reset_interview():
#     logger.debug("Reset interview endpoint called")
#     if "user" not in session:
#         logger.warning("Unauthenticated reset interview attempt")
#         return jsonify({"status": "error", "message": "Not authenticated"}), 401
        
#     session.clear()
#     session['interview_data'] = init_interview_data()
#     logger.info("Interview session reset")
#     return jsonify({"status": "success", "message": "Interview reset successfully"})

# @app.route('/export_interview_ratings')
# def export_interview_ratings():
#     logger.debug("Export interview ratings endpoint called")
#     if "user" not in session or session.get("role") != "recruiter":
#         logger.warning("Unauthorized export attempt")
#         return redirect(url_for("login"))
#     try:
#         conn = get_postgres_connection()
#         if not conn:
#             raise Exception("Could not connect to Snowflake")
            
#         cs = conn.cursor()
#         logger.debug("Fetching interview ratings for export")
#         cs.execute("""
#             SELECT roll_no, technical_rating, communication_rating, problem_solving_rating,
#                    time_management_rating, total_rating, interview_ts
#             FROM interview_rating
#             ORDER BY interview_ts DESC
#             LIMIT 200
#         """)
#         rows = cs.fetchall()
#         cols = [desc[0] for desc in cs.description]
#         cs.close()
#         conn.close()

#         df = pd.DataFrame(rows, columns=cols)
#         logger.debug(f"Prepared {len(df)} records for export")

#         output = io.BytesIO()
#         with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#             df.to_excel(writer, index=False, sheet_name='Interview Ratings')
#         output.seek(0)

#         logger.info("Successfully exported interview ratings")
#         return send_file(output,
#                  download_name="interview_ratings.xlsx",
#                  as_attachment=True,
#                  mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

#     except Exception as e:
#         logger.error(f"Error exporting interview ratings: {str(e)}", exc_info=True)
#         return f"Error exporting interview ratings: {e}", 500



@app.route('/export_student_info')
def export_student_info():
    logger.debug("Export student info endpoint called")
    if "user" not in session or session.get("role") != "recruiter":
        logger.warning("Unauthorized export attempt")
        return redirect(url_for("login"))
    try:
        conn = get_postgres_connection()
        if not conn:
            raise Exception("Could not connect to Snowflake")
            
        cs = conn.cursor()
        logger.debug("Fetching student info for export")
        cs.execute("""
            SELECT student_name, roll_no, batch_no, center, course, evaluation_date, difficulty_level, interview_ts
            FROM student_info
            ORDER BY interview_ts DESC
            LIMIT 200
        """)
        rows = cs.fetchall()
        cols = [desc[0] for desc in cs.description]
        cs.close()
        conn.close()

        df = pd.DataFrame(rows, columns=cols)
        logger.debug(f"Prepared {len(df)} records for export")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Student Info')
        output.seek(0)

        logger.info("Successfully exported student info")
        return send_file(output,
                 download_name="student_info.xlsx",
                 as_attachment=True,
                 mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        logger.error(f"Error exporting student info: {str(e)}", exc_info=True)
        return f"Error exporting student info: {e}", 500

@app.route('/export_visual_feedback')
def export_visual_feedback():
    logger.debug("Export visual feedback endpoint called")
    if "user" not in session or session.get("role") != "recruiter":
        logger.warning("Unauthorized export attempt")
        return redirect(url_for("login"))
    try:
        conn = get_postgres_connection()
        if not conn:
            raise Exception("Could not connect to Snowflake")
            
        cs = conn.cursor()
        logger.debug("Fetching visual feedback for export")
        cs.execute("""
            SELECT roll_no, professional_appearance, body_language, environment, 
                   distractions, interview_ts
            FROM visual_feedback
            ORDER BY interview_ts DESC
            LIMIT 200
        """)
        rows = cs.fetchall()
        cols = [desc[0] for desc in cs.description]
        cs.close()
        conn.close()

        df = pd.DataFrame(rows, columns=cols)
        logger.debug(f"Prepared {len(df)} records for export")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Visual Feedback')
        output.seek(0)

        logger.info("Successfully exported visual feedback")
        return send_file(output,
                 download_name="visual_feedback.xlsx",
                 as_attachment=True,
                 mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        logger.error(f"Error exporting visual feedback: {str(e)}", exc_info=True)
        return f"Error exporting visual feedback: {e}", 500

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, np.integer)):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    app.json_encoder = JSONEncoder
    app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'), debug=True)