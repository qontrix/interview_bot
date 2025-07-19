
# from flask import Flask, request, jsonify
# import openai
# import os
# import logging
# import re

# app = Flask(__name__)
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

# # Set your API key
# openai.api_key = os.getenv("OPENAI_API_KEY") or "your-openai-api-key-here"

# def generate_questions_from_jd(jd_text, difficulty_level, roll_no=None):
#     logger.debug("Generating structured interview questions")

#     if not jd_text:
#         return []

#     previous_questions = []
#     filename = f"interview_conversation_{roll_no}.txt" if roll_no else "interview_conversation.txt"
#     if os.path.exists(filename):
#         with open(filename, "r") as f:
#             for line in f:
#                 if line.startswith("Question:"):
#                     previous_questions.append(line.split(":", 1)[1].strip())

#     prompt = f"""
# You are a senior technical interviewer tasked with designing a well-structured and role-specific interview script 
# based on the following Job Description.

# Please generate exactly 8 questions as per the format and structure below:

# 1. **Introduction Question** (Understand the candidate’s background relevant to the role)
# 2. **Three Technical Questions**, each with **one meaningful Follow-up question**:
#    - Ensure relevance to the JD
#    - Vary difficulty based on this level: {difficulty_level}
#    - Cover both conceptual knowledge and real-world application
# 3. **One Behavioral Question** (Focus on collaboration, ownership, communication, or adaptability)

# 🧠 **Guidelines:**
# - For `{difficulty_level}` level, adjust technical depth:
#   - Beginner: Fundamental concepts, tools usage, basic problem-solving
#   - Medium: Intermediate problem-solving, implementation choices, system design thinking
#   - Advanced: Architectural thinking, scalability, performance optimization, edge cases
# - Avoid repeating these recent questions: {previous_questions[-5:] if previous_questions else "None"}

# 📋 **Output Format:**
# Question 1: [Introduction]
# Question 2: [Technical Q1]
# Follow-up 1: [Follow-up Q1]
# Question 3: [Technical Q2]
# Follow-up 2: [Follow-up Q2]
# Question 4: [Technical Q3]
# Follow-up 3: [Follow-up Q3]
# Question 5: [Behavioral]

# 🎯 **Important**:
# - Questions must feel like real interview scenarios
# - Refer to domain-relevant tools, libraries, or practices where appropriate
# - Keep each question concise but clear
# - Do NOT include answers, just the questions

# 📝 Job Description:
# {jd_text}
# """



#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.7,
#             max_tokens=1200
#         )

#         raw_output = response.choices[0].message.content.strip()
#         raw_output = re.sub(r"```(json)?", "", raw_output).strip()

#         # Parse into structured format
#         structured = []
#         temp_question = {}
#         lines = raw_output.splitlines()

#         for line in lines:
#             line = line.strip()
#             if re.match(r"^Question \d:", line):
#                 if temp_question:
#                     structured.append(temp_question)
#                 temp_question = {
#                     "question": line.split(":", 1)[1].strip(),
#                     "type": ""
#                 }
#                 if "intro" in line.lower():
#                     temp_question["type"] = "introduction"
#                 elif "behavioral" in line.lower():
#                     temp_question["type"] = "behavioral"
#                 else:
#                     temp_question["type"] = "technical"

#             elif re.match(r"^Follow-up Question \d:", line):
#                 if temp_question and temp_question.get("type") == "technical":
#                     temp_question["follow_up"] = line.split(":", 1)[1].strip()

#         if temp_question:
#             structured.append(temp_question)

#         return structured

#     except Exception as e:
#         logger.error(f"OpenAI error: {e}", exc_info=True)
#         return []

# @app.route("/generate-questions", methods=["POST"])
# def generate_questions_api():
#     data = request.get_json()
#     jd_text = data.get("jd_text")
#     difficulty = data.get("difficulty_level", "medium")
#     roll_no = data.get("roll_no", None)

#     questions = generate_questions_from_jd(jd_text, difficulty, roll_no)
#     return jsonify({"questions": questions})

# if __name__ == "__main__":
#     app.run(debug=True)






from flask import Flask, request, jsonify
import openai
import os
import logging
import re

app = Flask(__name__)

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY") or "your-openai-api-key-here"

def generate_questions_from_jd(jd_text, difficulty_level, roll_no=None):
    logger.debug("Generating structured interview questions")

    if not jd_text:
        logger.error("No JD text provided.")
        return []

    previous_questions = []
    filename = f"interview_conversation_{roll_no}.txt" if roll_no else "interview_conversation.txt"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("Question:"):
                    previous_questions.append(line.split(":", 1)[1].strip())

    prompt = f"""
You are a senior technical interviewer designing structured interview questions from a Job Description.

🎯 Requirements:
- 1 Introduction Question (about candidate's background relevant to backend roles)
- 3 Technical Questions based on this JD: `{jd_text}`
  - Each technical question MUST have 1 meaningful follow-up
  - Questions must reflect difficulty level: {difficulty_level}
- 1 Behavioral Question (NOT technical — focused on soft skills/team/collaboration)

⛔ Avoid repeating previous questions: {previous_questions[-5:] if previous_questions else "None"}

📋 Format strictly like this:
Question 1: [Introduction Question]
Question 2: [Technical Q1]
Follow-up 1: [Follow-up to Q1]
Question 3: [Technical Q2]
Follow-up 2: [Follow-up to Q2]
Question 4: [Technical Q3]
Follow-up 3: [Follow-up to Q3]
Question 5: [Behavioral Question]
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )

        raw_output = response.choices[0].message.content.strip()
        logger.debug("\nFull GPT Response:\n" + raw_output)

        # Parse questions
            # Parse questions into structured format
                # Parse questions into structured format
        structured = []
        current = {}

        for line in raw_output.splitlines():
            line = line.strip()
            if not line:
                continue

            # Match question and follow-up lines
            if re.match(r"^Question\s*\d+:", line, re.IGNORECASE):
                if current:
                    structured.append(current)

                question_text = re.sub(r"^Question\s*\d+:\s*", "", line).strip()
                current = {
                    "question": question_text,
                    "type": "technical"
                }

                if len(structured) == 0:
                    current["type"] = "introduction"
                elif len(structured) >= 4:
                    current["type"] = "behavioral"

            elif re.match(r"^Follow[- ]?up\s*\d+:", line, re.IGNORECASE):
                follow_up_text = re.sub(r"^Follow[- ]?up\s*\d+:\s*", "", line).strip()
                if current:
                    current["follow_up"] = follow_up_text

        if current:
            structured.append(current)

        # ✅ Pretty print in CMD
        print("\n📋 Generated Interview Questions:")
        for q in structured:
            print(f"\n🟢 Type: {q['type'].capitalize()}")
            print(f"👉 Q: {q['question']}")
            if 'follow_up' in q:
                print(f"   🔁 Follow-up: {q['follow_up']}")


        return structured


    except Exception as e:
        logger.error(f"Error from OpenAI: {str(e)}", exc_info=True)
        return []

# API route
@app.route("/generate-questions", methods=["POST"])
def generate_questions_api():
    data = request.get_json()
    jd_text = data.get("jd_text")
    difficulty = data.get("difficulty_level", "medium")
    roll_no = data.get("roll_no", None)

    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    questions = generate_questions_from_jd(jd_text, difficulty, roll_no)
    return jsonify({"questions": questions}), 200

if __name__ == "__main__":
    app.run(debug=True)
