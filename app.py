from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client import models
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = QdrantClient(url="http://localhost:6333")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

EMBEDDING_MODEL = "BAAI/bge-small-en"
SPARSE_COLLECTION_NAME = 'course_faq_sparse'
HYBRID_COLLECTION_NAME = 'course_faq_hybrid'
SEMANTIC_COLLECTION_NAME = 'course_faq'

class QueryRequest(BaseModel):
    question: str
    search_type: str

def semantic_search(query: str, limit: int = 3):
    results = client.query_points(
        collection_name=SEMANTIC_COLLECTION_NAME,
        query=models.Document(
            text=query,
            model=EMBEDDING_MODEL
        ),
        limit=limit,
        with_payload=True
    )
    return results.points

def sparse_search(query: str, limit: int = 3):
    results = client.query_points(
        collection_name=SPARSE_COLLECTION_NAME,
        query=models.Document(
            text=query,
            model="Qdrant/bm25",
        ),
        using="bm25",
        limit=limit,
        with_payload=True,
    )
    return results.points

def hybrid_search(query: str, limit: int = 3):
    results = client.query_points(
        collection_name=HYBRID_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model=EMBEDDING_MODEL,
                ),
                using="bge_small",
                limit=(5 * limit),
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25",
                ),
                using="bm25",
                limit=(5 * limit),
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
    )
    return results.points

def format_context(points):
    content = ""
    for point in points:
        content += f"Course: {point.payload['course']}\n"
        content += f"Section: {point.payload['section']}\n"
        content += f"Text: {point.payload['text']}\n"
        content += "-" * 20 + "\n"
    return content

def generate_answer(question: str, context: str):
    prompt_template = f"""
        You are a helpful and concise Teaching Assistant. 
        Your task is to answer the student's question based ONLY on the provided Context.
        
        GUIDELINES:
        1. If the answer is contained within the Context, provide a clear and helpful response.
        2. If the answer is NOT in the Context or is unrelated, strictly respond with: "I'm sorry, but I don't have that information based on the course materials."
        3. Do not use outside knowledge or make up facts.

        CONTEXT:
        {context}

        QUESTION: 
        {question}

        ANSWER:
        """
    prompt = prompt_template.format(question, context)
    
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )
    
    return completion.choices[0].message.content

@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.post("/search")
async def search(request: QueryRequest):
    question = request.question
    search_type = request.search_type
    
    if search_type == "semantic":
        points = semantic_search(question)
    elif search_type == "sparse":
        points = sparse_search(question)
    elif search_type == "hybrid":
        points = hybrid_search(question)
    else:
        return {"error": "Invalid search type"}
    
    context = format_context(points)
    answer = generate_answer(question, context)
    
    return {"answer": answer}
