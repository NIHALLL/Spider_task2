# ============================================================
# Standard Library
# ============================================================
import os
import re
# 
from collections import defaultdict
import xml.etree.ElementTree as ET

# ============================================================
# Data Processing
# ============================================================
import pandas as pd

# ============================================================
# LangChain
# ============================================================
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ============================================================
# Ollama
# ============================================================
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# ============================================================
# Vector Database
# ============================================================
from langchain_chroma import Chroma

# ============================================================
# Configuration
# ============================================================

KNOWLEDGE_SOURCES = {
    "medquad": {
        "path": "./data/medquad",
        "type": "xml",
        "source": "MedQuAD",
    },
    "who": {
        "path": "./data/who",
        "type": "pdf",
        "source": "WHO",
    },
    "cdc": {
        "path": "./data/cdc",
        "type": "pdf",
        "source": "CDC",
    },
    "nice": {
        "path": "./data/nice",
        "type": "pdf",
        "source": "NICE",
    },
}
CHROMA_DIR = "./chroma_db"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

TOP_K = 5

CONFIDENCE_HIGH = 0.7
CONFIDENCE_LOW = 0.4
# ============================================================
# Safety Layer
# ============================================================

EMERGENCY_PATTERNS = [
    "heart attack",
    "stroke",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "overdose",
    "unconscious",
    "severe chest pain",
]

UNSAFE_PATTERNS = [
    "maximum lethal dose",
    "how to overdose",
    "kill myself",
    "harm myself",
    "illegal prescription",
]
def safety_check(query: str):
    query = query.lower()

    for pattern in EMERGENCY_PATTERNS:
        if pattern in query:
            return (
                False,
                "This appears to describe a medical emergency. "
                "Please contact your local emergency services immediately."
            )

    for pattern in UNSAFE_PATTERNS:
        if pattern in query:
            return (
                False,
                "I'm unable to assist with requests that could cause harm."
            )

    return (True, None)
def load_pdf_source(pdf_dir: str, source_name: str):
    documents = []
    global_idx = 0

    for dirpath, _, filenames in os.walk(pdf_dir):
        for filename in filenames:
            if not filename.endswith(".pdf"):
                continue

            filepath = os.path.join(dirpath, filename)

            try:
                loader = PyPDFLoader(filepath)
                pages = loader.load()
            except Exception as e:
                print(f"Skipping {filepath}: failed to load ({e})")
                continue

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            chunks = text_splitter.split_documents(pages)

            for chunk in chunks:
                chunk.metadata["source"] = source_name
                chunk.metadata["file"] = filename
                chunk.metadata["chunk_id"] = global_idx
                documents.append(chunk)
                global_idx += 1

    if global_idx == 0:
        raise FileNotFoundError(f"No usable PDF pages found under {pdf_dir}")

    return documents
def load_medquad_xml():
    folder = KNOWLEDGE_SOURCES["medquad"]["path"]
    documents = []
    global_idx = 0

    for dirpath, _, filenames in os.walk(folder):
        for filename in filenames:
            if not filename.endswith(".xml"):
                continue

            filepath = os.path.join(dirpath, filename)

            try:
                tree = ET.parse(filepath)
            except ET.ParseError as e:
                print(f"Skipping {filepath}: malformed XML ({e})")
                continue

            root = tree.getroot()
            focus = root.findtext("Focus", default="").strip()

            for qa_pair in root.findall(".//QAPair"):
                question_el = qa_pair.find("Question")
                answer_el = qa_pair.find("Answer")

                if question_el is None or answer_el is None:
                    continue

                question = (question_el.text or "").strip()
                answer = (answer_el.text or "").strip()

                if not question or not answer:
                    continue

                qtype = question_el.get("qtype", "unknown")
                text = f"""Question:{question}Answer:{answer}"""

                doc = Document(
                    page_content=text,
                    metadata={
                        "source": KNOWLEDGE_SOURCES["medquad"]["source"],
                        "file": filename,
                        "focus": focus,
                        "qtype": qtype,
                        "chunk_id": global_idx,
                    }
                )
                documents.append(doc)
                global_idx += 1

    if global_idx == 0:
        raise FileNotFoundError(f"No usable XML Q&A pairs found under {folder}")

    return documents
def build_vectorstore(documents):

    embeddings = OllamaEmbeddings( model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(documents=documents,embedding=embeddings,persist_directory=CHROMA_DIR)

    return vectorstore
def retrieve_documents(vectorstore, query):

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=TOP_K
    )

    return results

def group_sources(results):
# makes a default dictionary for who,medquad etc....

    grouped_sources = defaultdict(list)

    for doc, score in results:
        source = doc.metadata["source"]
        grouped_sources[source].append(doc)

    return grouped_sources

def build_prompt(query, results):

    context = ""

    for doc, score in results:

        context += (
            f"Source: {doc.metadata['source']}\n"
            f"{doc.page_content}\n\n"
        )

    prompt = f"""
                    You are a trustworthy and very professional medical assistant.

                    Use ONLY the information provided in the context.

                    Rules:
                    - Do not use outside knowledge.
                    - Do not make assumptions.
                    - Do not diagnose diseases.
                    - Do not recommend medication dosages.
                    - If the context is insufficient, clearly say so.
                    - Cite the source whenever possible.

                    Context:
                    {context}

                    Question:
                    {query}

                    Answer:"""
    return prompt
def generate_response(prompt):
    llm=OllamaLLM(model=LLM_MODEL)
    response=llm.invoke(prompt)

    return response

def estimate_confidence(results):

    if not results:
        return 0.0, "LOW"

    best_score = max(score for _, score in results)

    sources = {
        doc.metadata["source"]
        for doc, _ in results
    }
# what if there are infinite sources?
    source_bonus = 0.05 * (len(sources) - 1)

    confidence = min(best_score + source_bonus, 1.0)

    if confidence >= CONFIDENCE_HIGH:
        label = "HIGH"

    elif confidence >= CONFIDENCE_LOW:
        label = "MEDIUM"

    else:
        label = "LOW"

    return confidence, label

def query_system(vectorstore,query):
    safe , message = safety_check(query)
    if not safe:
        return{
            "response":message,
            "confidence":"n/a",
            "sources":{}
            }
    results = retrieve_documents(vectorstore, query)
    confidence_score, confidence_label = estimate_confidence(results)
    grouped_sources=group_sources(results)
    prompt = build_prompt(query, results)
    response = generate_response(prompt)
    return {
        "response": response,
        "confidence": confidence_label,
        "score": confidence_score,
        "sources": grouped_sources
    }
def main():

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(CHROMA_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
    else:
        documents = []

        documents.extend(load_medquad_xml())
        documents.extend(load_pdf_source(KNOWLEDGE_SOURCES["who"]["path"], KNOWLEDGE_SOURCES["who"]["source"]))
        documents.extend(load_pdf_source(KNOWLEDGE_SOURCES["cdc"]["path"], KNOWLEDGE_SOURCES["cdc"]["source"]))
        documents.extend(load_pdf_source(KNOWLEDGE_SOURCES["nice"]["path"], KNOWLEDGE_SOURCES["nice"]["source"]))

        vectorstore = build_vectorstore(documents)

    while True:
        query = input("\nEnter your medical question (or type 'exit'): ")

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        result = query_system(vectorstore, query)

        print("\nAnswer:\n")
        print(result["response"])
        print("\nConfidence:", result["confidence"])
if __name__ == "__main__":
    main()