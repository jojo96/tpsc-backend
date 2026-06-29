from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = "chroma_db"

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for TPSC (Tripura Public Service Commission) exam preparation.
Use the following context to answer the question. If you don't know, say so.

Context: {context}

Question: {question}
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def ask(question: str) -> str:
    chain = get_rag_chain()
    return chain.invoke(question)

if __name__ == "__main__":
    print(ask("What is the Sixth Schedule of the Indian Constitution?"))