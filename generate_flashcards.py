from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from supabase import create_client
import os, json, re

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

PRESET_TOPICS = [
    "Indian Polity",
    "Indian History",
    "Indian Economy",
    "Indian Geography",
    "Science & Technology",
    "Tripura History",
    "Tripura Geography",
    "Environment & Ecology",
]

def get_context(topic):
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(topic)
    return "\n\n".join([d.page_content for d in docs])

def generate_cards(topic):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
    context = get_context(topic)

    prompt = f"""Based on the following study material, create 15 flashcards about "{topic}" for TPSC exam preparation.

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
    text = re.sub(r"```json|```", "", response.content.strip()).strip()
    return json.loads(text)

def main():
    for topic in PRESET_TOPICS:
        print(f"Generating: {topic}...")
        try:
            cards = generate_cards(topic)
            rows = [{"topic": topic, "front": c["front"], "back": c["back"]} for c in cards]
            supabase.table("flashcards").insert(rows).execute()
            print(f"  Saved {len(rows)} cards for {topic}")
        except Exception as e:
            print(f"  Failed {topic}: {e}")

if __name__ == "__main__":
    main()