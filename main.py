from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import get_rag_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json, re
from news import fetch_articles, generate_current_affairs
from datetime import date
import httpx
from bs4 import BeautifulSoup
from fastapi import UploadFile, File
import tempfile
from fastapi import Form, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
import tempfile

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

class FlashcardRequest(BaseModel):
    topic: str
    num_cards: int = 10

@app.post("/generate-flashcards")
async def generate_flashcards(body: FlashcardRequest):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
    
    # First retrieve relevant context from your database
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    from langchain_chroma import Chroma
    import os
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(body.topic)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""Based on the following study material, create {body.num_cards} flashcards about "{body.topic}" for TPSC exam preparation.

Study material:
{context}

Return ONLY a JSON array, no explanation, no markdown:
[
  {{
    "front": "Term or concept here",
    "back": "Clear explanation or answer here"
  }}
]"""

    response = llm.invoke(prompt)
    text = response.content.strip()
    text = re.sub(r"```json|```", "", text).strip()
    cards = json.loads(text)
    return {{"cards": cards}}

@app.get("/daily-trivia")
async def daily_trivia():
    today = date.today()
    today_str = today.strftime("%d-%m-%Y")
    
    # Drishti IAS daily current affairs URL
    url = f"https://www.drishtiias.com/current-affairs-news-analysis-editorials/news-analysis/{today_str}"
    
    content = ""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            res = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(res.text, "html.parser")
            
            # Extract only article content, skip nav/footer
            article = soup.find("div", class_="article-detail") or \
                      soup.find("div", class_="content-area") or \
                      soup.find("main") or \
                      soup.find("article")
            
            if article:
                content = article.get_text(separator=" ", strip=True)[:5000]
            else:
                # Fall back to paragraphs only
                paragraphs = soup.find_all("p")
                content = " ".join([p.get_text() for p in paragraphs[:30]])[:5000]
    except Exception as e:
        print(f"Scraping failed: {e}")
        content = ""

    # If scraping failed or content too short, use Groq with general knowledge
    if len(content) < 200:
        content = f"Today is {today.strftime('%B %d, %Y')}. Generate questions about recent Indian current affairs including economy, polity, environment, science and international relations relevant for TPSC exam."

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

    prompt = f"""Generate exactly 5 current affairs trivia questions for TPSC exam preparation based on recent Indian news.

Context: {content}

Rules:
- Questions must be about real, verifiable facts
- Do NOT ask about website downloads or navigation
- Focus on: government schemes, international relations, economy, environment, science
- Make questions genuinely useful for a state PCS exam

Return ONLY a JSON array:
[
  {{
    "question": "Question here?",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "A) option1",
    "explanation": "Brief explanation"
  }}
]"""

    response = llm.invoke(prompt)
    text = re.sub(r"```json|```", "", response.content.strip()).strip()
    questions = json.loads(text)
    return {"date": today.isoformat(), "questions": questions}

@app.post("/chat-with-doc")
async def chat_with_doc(question: str = Form(...), file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Load and chunk the PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Create temporary in-memory vector store
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"temp_{tmp_path[-6:]}",
    )
    
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    
    prompt = f"""Answer the question based on the uploaded document.

Context from document:
{context}

Question: {question}

If the answer is not in the document, say so clearly."""

    response = llm.invoke(prompt)
    
    # Cleanup
    os.unlink(tmp_path)
    db.delete_collection()
    
    return {"answer": response.content}