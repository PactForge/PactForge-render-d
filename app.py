from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
import google.generativeai as genai
import time
from docx import Document
import os

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

client = genai.Client(api_key=GOOGLE_API_KEY)

# ✅ Define config as a plain dict
model_config = {
    "temperature": 0.75,
    "top_p": 0.9
}

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

DATA_PATH = "./Clauses"
collections = {}
clientdb = chromadb.Client()
agreement_types = ["rent", "nda", "employment", "franchise", "contractor"]

try:
    for name in agreement_types:
        col = clientdb.get_or_create_collection(name=f"{name}_agreements")
        clauses = []
        path = f"{DATA_PATH}/{name}.docx"
        doc = Document(path)
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                clauses.append(text)
        embeds, ids, docs = [], [], []
        for i, clause in enumerate(clauses):
            embed = client.embed_content(
                model="models/embedding-001",
                content=clause,
                task_type="retrieval_document"
            )
            time.sleep(0.4)
            embeds.append(embed['embedding'])
            ids.append(f"{name}-{i}")
            docs.append(clause)
        col.add(embeddings=embeds, ids=ids, documents=docs)
        collections[name] = col
except Exception as e:
    print(f"Error initializing database: {e}")

class AgreementInput(BaseModel):
    agreement_type: str
    important_info: str
    extra_info: str

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "PactForge API is running"}

@app.post("/generate")
async def generate_agreement(data: AgreementInput):
    try:
        user_input = data.important_info + "\n" + data.extra_info
        embed = client.embed_content(
            model="models/embedding-001",
            content=user_input,
            task_type="retrieval_query"
        )
        query_embedding = embed['embedding']
        db = collections.get(data.agreement_type)
        if not db:
            raise HTTPException(status_code=400, detail="Invalid agreement type")
        
        results = db.query(query_embeddings=[query_embedding], n_results=5)
        relevant_docs = results['documents'][0]

        with open("./sampleagreements/sample.txt", "r") as f:
            sample_agreements = f.read()

        prompt = f"""
            You are a helpful AI assistant for law agreement generation.

            The agreement type is: {data.agreement_type}
            Important information: {data.important_info}
            Additional input: {data.extra_info}

            Relevant clauses:
            {relevant_docs}

            Format the agreement similar to:
            {sample_agreements}

            Make it clear, complete, and legally sound. Output only the agreement text.
        """

        # ✅ Use GenerationConfig to wrap model_config
        generation_config = genai.types.GenerationConfig(**model_config)

        response = client.generate_content(
            model="gemini-pro",
            contents=prompt,
            generation_config=generation_config
        )
        return {"agreement": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing (optional)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
