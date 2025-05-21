import os
import uuid
import time
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Set Gemini as the embedding model
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/embedding-001")

# Load your JSON file as a document
json_path = os.path.join(os.path.dirname(__file__), "pccoe_knowledge.json")
documents = SimpleDirectoryReader(input_files=[json_path]).load_data()

# Build a vector index over the JSON document
index = VectorStoreIndex.from_documents(documents)

# Create a query engine using Gemini (GoogleGenerativeAI)
llm = GoogleGenAI(model="models/gemini-1.5-flash", temperature=0)
query_engine = index.as_query_engine(llm=llm)

app = FastAPI()

# CORS for Flutter/web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_keys = os.getenv("API_KEYS", "").split(",")

API_KEYS = set(api_keys)


session_histories = {}
SESSION_TIMEOUT = 60 * 30  # 30 minutes

class ChatRequest(BaseModel):
    session_id: str = None
    query: str

@app.post("/chat")
async def chat(body: ChatRequest, x_api_key: str = Header(None)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Clean up expired sessions
    now = time.time()
    expired = [sid for sid, s in session_histories.items() if now - s["last_active"] > SESSION_TIMEOUT]
    for sid in expired:
        del session_histories[sid]

    # If no session_id, create a new session
    session_id = body.session_id
    if not session_id or session_id not in session_histories:
        session_id = str(uuid.uuid4())
        session_histories[session_id] = {"history": [], "last_active": now}

    # Get history for this session
    session = session_histories[session_id]
    history = session["history"]
    session["last_active"] = now

    # Add user query to history
    history.append({"role": "user", "content": body.query})
    # Compose prompt from history with concise instruction
    system_instruction = (
        "You are a helpful assistant for PCCoE. "
        "Answer concisely, clearly, and in no more than 4-5 sentences."
    )
    prompt = system_instruction + "\n"
    for turn in history:
        prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
    prompt += "Assistant:"
    # Call your RAG/LLM here
    answer = str(query_engine.query(prompt))
    # Add assistant answer to history
    history.append({"role": "assistant", "content": answer})

    return {"answer": answer}