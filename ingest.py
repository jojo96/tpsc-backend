from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
import os

CHROMA_PATH = "chroma_db"

def ingest_pdfs(root_folder: str):
    docs = []
    
    # Walk through all subfolders
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".pdf"):
                full_path = os.path.join(dirpath, file)
                print(f"Loading: {full_path}")
                try:
                    loader = PyPDFLoader(full_path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Skipping {file}: {e}")

    print(f"\nLoaded {len(docs)} pages total")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Done. Stored in ChromaDB at {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_pdfs("./pdfs")