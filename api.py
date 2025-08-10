from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from utils.database_utils import VectorDatabase
from utils.chat_utils import BOT_FRAMEWORK

app = FastAPI(
    title="Generalized AI Assistant",
    version="0.1",
    description="Generalized framework for building AI Assistants"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    database = VectorDatabase(configuration_filepath="./config.yml") #.database
    if database is None:
        print("Warning: Vector database is None. Please ensure the collection exists.")

    chatbot = BOT_FRAMEWORK(configuration_filepath='./config.yml',
                            vector_database=database)
except Exception as e:
    print(f"Error initializing database or chatbot: {e}")
    database = None
    chatbot = None


class QueryRequest(BaseModel):
    query: str
    language: str = None


class ChatResponse(BaseModel):
    bot_reply: str


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Assistant API"}


@app.post("/api/chat_response", response_model=ChatResponse)
async def chat_response(request: QueryRequest):
    try:
        if chatbot is None:
            raise HTTPException(
                status_code=503, detail="Chatbot service unavailable")

        if not request.query:
            raise HTTPException(status_code=400, detail="No query provided")

        query = request.query.lower().strip()
        ai_response = chatbot.get_llm_response(query=query)
        print(f"Assistant: {ai_response}")

        return ChatResponse(bot_reply=ai_response)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error Generating Response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
