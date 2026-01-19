# Importing required librarries

import os
import nltk
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct



from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

from dotenv import load_dotenv
load_dotenv()
nltk.download("punkt")

#Global Models and Qdrant

semantic_embedder = HuggingFaceEmbeddings(
model_name = 'sentence-transformers/all-MiniLM-L6-v2'

)

bge = SentenceTransformer('BAAI/bge-base-en')
bge.max_seq_length=512

reranker= CrossEncoder("BAAI/bge-reranker-base")

qdrant=QdrantClient(path="qdrant_db")
collection_name='rag_collection'

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#Chunking

def chunk_document(text):
    semantic_chunker=SemanticChunker(
    embeddings=semantic_embedder,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=95

    )
    semantic_chunks=semantic_chunker.split_text(text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", "!", "?"]

    )

    final_chunks=[]
    for chunk in semantic_chunks:
        final_chunks.extend(recursive_splitter.split_text(chunk))

    return final_chunks


# Embeddings

def embed_text(text_list):
    return bge.encode(text_list, normalize_embeddings=True).tolist()

# Ingesting

def ingest_documents(texts):
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE,

        ),
    )
    points=[]
    idx=0

    for text in texts:
        chunks=chunk_document(text)

        for chunk in chunks:
            vec = embed_text([chunk])[0]
            points.append(PointStruct(id=idx, vector=vec, payload={'text': chunk}))
            idx+=1

    qdrant.upsert(collection_name=collection_name, points=points)
    return idx    


#Retrival + reranking

def vector_search(query, top_k=10):
    vec = embed_text([query])[0]

    results = qdrant.query_points(
        collection_name=collection_name,
        query=vec,
        limit=top_k
    )
    
    docs = [{"text": p.payload.get("text", ""), "score_before": p.score}
            for p in results.points]
    
    return docs

def rerank_results(query, docs, top_k=5):

    pairs=[(query, d["text"]) for d in docs]
    scores = reranker.predict(pairs)

    for d,s in zip(docs, scores):
        d["score_after"]=float(s)

    ranked = sorted (docs, key=lambda x: x["score_after"], reverse=True)
    return ranked[:top_k]

def build_prompt(question, passages):
    context = "\n".join([f"- {p['text']}" for p in passages])

    return f"""
You are a helpful assistant. Use ONLY the context given below.
If you do not find the answer in context, say: "I don't know".

### CONTEXT
{context}

### QUESTION
{question}

### ANSWER
"""

def generate_answer(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

#Full pipeline

def answer_questions(question):
    retrieved=vector_search(question)
    reranked = rerank_results(question, retrieved)
    prompt = build_prompt(question, reranked)
    answer = generate_answer(prompt)
    return answer, reranked