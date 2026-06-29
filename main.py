from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import get_rag_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json, re
from news import fetch_articles, generate_current_affairs

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://tpsc-frontend.vercel.app"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

class MCQRequest(BaseModel):
    topic: str
    num_questions: int = 5

@app.post("/generate-mcq")
async def generate_mcq(body: MCQRequest):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

    prompt = f"""Generate {body.num_questions} multiple choice questions about "{body.topic}" 
for the TPSC (Tripura Public Service Commission) exam.

Return ONLY a JSON array, no explanation, no markdown, just raw JSON like this:
[
  {{
    "question": "Question text here?",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "A) option1",
    "explanation": "Brief explanation why this is correct"
  }}
]"""

    response = llm.invoke(prompt)
    text = response.content.strip()
    text = re.sub(r"```json|```", "", text).strip()
    questions = json.loads(text)
    return {"questions": questions}

rag_chain = get_rag_chain()

@app.post("/ask")
async def ask_question(body: Question):
    answer = rag_chain.invoke(body.question)
    return {"answer": answer}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/current-affairs")
async def current_affairs():
    articles = fetch_articles(max_articles=5)
    if not articles:
        return {"questions": [], "message": "No articles found"}
    questions = generate_current_affairs(articles)
    return {"questions": questions}