import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controller.chat_controller import router as chat_router
from service.worker import ChatWorker
from agent.agent import RAGAgent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config.database import connect_to_mongo, close_mongo_connection
from config.logging import get_logger

logger = get_logger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

worker = None

@app.on_event("startup")
async def startup_event():
    global worker
    print(">>> STARTUP EVENT TRIGGERED <<<", flush=True)
    
    try:
        # Initialize database connection (sync)
        print(">>> Connecting to MongoDB...", flush=True)
        connect_to_mongo()
        print(">>> MongoDB connected.", flush=True)
        
        # Read agent configuration from environment variables
        import os
        
        llm_type = os.getenv("LLM_TYPE", "gemini")
        llm_config = {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": os.getenv("GEMINI_MODEL"),
        }
        
        # Agent settings
        enable_planner = os.getenv("ENABLE_PLANNER", "true").lower() == "true"
        use_conversation_history = os.getenv("USE_CONVERSATION_HISTORY", "true").lower() == "true"
        max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))
        
        # Vector database settings
        vector_db_backend = os.getenv("VECTOR_DB_BACKEND", "in-memory")
        
        print(f">>> Config: llm_type={llm_type}, enable_planner={enable_planner}", flush=True)
        
        # Initialize Agent with environment configuration
        print(">>> Initializing Agent...", flush=True)
        agent = RAGAgent(
            llm_type=llm_type,
            llm_config=llm_config,
            enable_planner=enable_planner,
            use_conversation_history=use_conversation_history
        )
        print(">>> Agent Initialized.", flush=True)
        
        loop = asyncio.get_running_loop()
        print(">>> Starting Worker...", flush=True)
        worker = ChatWorker(agent, main_loop=loop)
        worker.start()
        print(">>> Worker Started.", flush=True)
        
    except Exception as e:
        print(f">>> STARTUP ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown_event():
    if worker:
        worker.stop()
    
    # Close database connection (sync)
    close_mongo_connection()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
