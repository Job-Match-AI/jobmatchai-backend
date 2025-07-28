# JobMatchAI - Backend

🚀 An intelligent job-resume matcher using Hugging Face sentence similarity models.

## 📌 Features

- Accepts resume and job description as input
- Computes similarity using Hugging Face Transformers (`all-MiniLM-L6-v2`)
- FastAPI backend
- Environment variables secured via `.env`
- Easy to deploy and extend

---

## ⚙️ Tech Stack

- Python 3.10+
- FastAPI
- Uvicorn
- Hugging Face Inference API
- python-dotenv

---

## 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/Job-Match-AI/jobmatchai-backend.git
cd jobmatchai-backend

# Create .env file with your Hugging Face API key
echo "HF_API_KEY=your_huggingface_api_key" > .env

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn main:app --reload
