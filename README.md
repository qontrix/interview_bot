# AI Interview System

An AI-powered interview system that generates questions based on uploaded job descriptions and evaluates candidate responses.

## Features

- Upload job descriptions in PDF, Word, or Text format
- AI-generated interview questions based on the job description
- Question pattern: 1 introduction, 3 technical, and 1 behavioral question
- Real-time answer evaluation
- Modern and responsive UI

## Setup

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'  # Unix/MacOS
   set OPENAI_API_KEY=your-api-key-here       # Windows
   ```

6. Run the backend server:
   ```bash
   python app.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Upload a job description file (PDF, Word, or Text format)
3. Click "Start Interview" to begin
4. Answer each question as it appears
5. Submit your answer to proceed to the next question
6. View the interview history and evaluations in real-time

## Technologies Used

- Frontend: React.js
- Backend: Flask (Python)
- AI: OpenAI GPT-3.5
- File Processing: PyPDF2, python-docx

## Note

Make sure to keep your OpenAI API key secure and never commit it to version control. 