from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List 
import logging
import httpx
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # <-- includes timestamp
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = FastAPI()

# Enable CORS so frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to ["http://localhost:3000"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

def cosine_similarity(v1, v2):
    from numpy import dot
    from numpy.linalg import norm
    return dot(v1, v2) / (norm(v1) * norm(v2))


class MatchRequest(BaseModel):
    resume: str
    jd: str

class MatchResponse(BaseModel):
    score: float
    missingKeywords: List[str]
    suggestions: List[str]

    
@app.post("/match", response_model=MatchResponse)
async def match_resume(request: MatchRequest):
    # Log the incoming request
    logging.info(f"Received request: {request.dict()}")

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": {
            "source_sentence": request.resume,
            "sentences": [request.jd]
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(HF_API_URL, headers=headers, json=payload)

        # Check for Hugging Face API errors
        if response.status_code != 200:
            logging.error(f"Hugging Face API Error: {response.status_code}, {response.text}")
            return {"score": 0, "missingKeywords": [], "suggestions": ["Error generating similarity score."]}

        try:
            similarity = response.json()[0]
        except Exception as e:
            logging.error(f"Error decoding similarity response: {e}")
            return {"score": 0, "missingKeywords": [], "suggestions": ["Similarity parsing failed."]}

        score = round(similarity * 100, 2)

    result = {
        "score": score,
        "missingKeywords": ["Docker", "Kubernetes", "CI/CD"],  # Can enhance later
        "suggestions": [
            "Include more cloud-related experience.",
            "Highlight agile methodologies.",
            "Mention containerization tools."
        ]
    }

    # Log the response before returning
    logging.info(f"Returning response: {result}")

    return result