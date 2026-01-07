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
        vector_db_backend = os.getenv("VECTOR_DB_BACKEND", "chroma")  # Changed default to chroma
        chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./storage/chroma")
        use_google_embedding = os.getenv("USE_GOOGLE_EMBEDDING", "false").lower() == "true"
        
        print(f">>> Config: llm_type={llm_type}, enable_planner={enable_planner}, vector_db_backend={vector_db_backend}", flush=True)
        
        # Initialize Vector Database with ChromaDB backend
        print(f">>> Initializing VectorDatabase with backend: {vector_db_backend}...", flush=True)
        vector_db = None
        
        if vector_db_backend == "chroma":
            from agent.memory import VectorDatabase
            
            # Initialize Google Gemini embedding function for ChromaDB
            embedding_function = None
            try:
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                
                if gemini_api_key:
                    print(">>> Using Google Gemini Embedding for ChromaDB...", flush=True)
                    from agent.llm.gemini import get_google_embedding_function
                    embedding_function = get_google_embedding_function(
                        api_key=gemini_api_key,
                        model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
                    )
                    print(">>> Google Gemini Embedding initialized successfully", flush=True)
                else:
                    print(">>> ERROR: GEMINI_API_KEY not found in .env file", flush=True)
                    print(">>> Please add GEMINI_API_KEY to your .env file", flush=True)
                    raise ValueError("GEMINI_API_KEY is required for ChromaDB")
            except Exception as e:
                print(f">>> ERROR: Could not initialize Google Gemini embedding: {e}", flush=True)
                print(">>> Please configure GEMINI_API_KEY in .env file", flush=True)
                raise
            
            # Create VectorDatabase with ChromaDB backend
            vector_db = VectorDatabase(
                backend_type="chroma",
                persist_directory=chroma_persist_dir,
                embedding_function=embedding_function
            )
            print(f">>> VectorDatabase initialized with ChromaDB (persist_dir: {chroma_persist_dir})", flush=True)
        else:
            from agent.memory import VectorDatabase
            vector_db = VectorDatabase(backend_type=vector_db_backend)
            print(f">>> VectorDatabase initialized with {vector_db_backend} backend", flush=True)
        
        # Initialize Agent with environment configuration
        print(">>> Initializing Agent...", flush=True)
        agent = RAGAgent(
            vector_db=vector_db,  # Pass the configured vector_db
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
