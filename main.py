from fastapi import FastAPI, Request, HTTPException  
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

raw_origins = os.getenv("ALLOWED_ORIGINS", "")
origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

# Enable CORS so frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_URL = os.getenv("HF_API_URL")

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
async def match_resume(req: Request, body: MatchRequest):
    # Log the incoming request
    logging.info(f"Request headers: {dict(req.headers)}")
    logging.info(f"Request client: {req.client}")
    logging.info(f"Request method: {req.method}, URL: {req.url}")
    logging.info(f"Request body: {body.dict()}")

    origin = req.headers.get("origin")
    
    if origin not in origins:
        raise HTTPException(status_code=403, detail="Forbidden origin")

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": {
            "source_sentence": body.resume,
            "sentences": [body.jd]
        }
    }

    logging.info(f"Sending request to Hugging Face API:")
    logging.info(f"URL: {HF_API_URL}")
    masked_headers = {k: (v if k.lower() != "authorization" else "Bearer ***") for k, v in headers.items()}
    logging.info(f"Headers: {masked_headers}")
    logging.info(f"Payload: {payload}")

    async with httpx.AsyncClient() as client:

        try:
            response = await client.post(HF_API_URL, headers=headers, json=payload)
        except Exception as e:
            logging.error(f"HTTP request to Hugging Face failed: {e}")
            raise HTTPException(status_code=502, detail="Failed to contact Hugging Face API")

        logging.info(f"Hugging Face API response status: {response.status_code}")
        logging.info(f"Hugging Face API response body: {response.text}")

        if response.status_code != 200:
            logging.error(f"Hugging Face API Error: {response.status_code}, {response.text}")
            return {
                "score": 0,
                "missingKeywords": [],
                "suggestions": ["Error generating similarity score."]
            }

        try:
            similarity = response.json()[0]
        except Exception as e:
            logging.error(f"Error decoding similarity response: {e}")
            return {
                "score": 0,
                "missingKeywords": [],
                "suggestions": ["Similarity parsing failed."]
            }

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

    logging.info(f"Returning response: {result}")

    return result