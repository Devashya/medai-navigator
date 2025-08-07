from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
import shutil
import os
import torch

app = FastAPI()

# Create uploads directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")


# 1. Upload PDF report endpoint
import pdfplumber
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    # Save the uploaded PDF file
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text from PDF
    extracted_text = ""
    with pdfplumber.open(file_location) as pdf:
        for page in pdf.pages:
            extracted_text += page.extract_text() or ""

    # Shorten the text if it's too long (BART model limit ~1024 tokens)
    max_input_length = 1024  # approx characters, not tokens
    trimmed_text = extracted_text[:max_input_length]

    # Generate summary
    summary_list = summarizer(trimmed_text, max_length=130, min_length=30, do_sample=False)
    summary = summary_list[0]['summary_text']

    return {
        "filename": file.filename,
        "summary": summary
    }



# Data model for /ask-question input
class QuestionInput(BaseModel):
    question: str
    context: str


# 3. Ask question endpoint (dummy)
@app.post("/ask-question")
async def ask_question(input: QuestionInput):
    # Dummy response for now
    answer = "This is a placeholder answer. Model integration coming soon."
    return {"answer": answer}

class TextInput(BaseModel):
    text: str

# 4. Generate text-to-speech endpoint (stub)
@app.post("/generate-tts")
async def generate_tts(input: TextInput):
    # Placeholder response
    return {"message": "TTS functionality coming soon."}
