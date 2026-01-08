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
agent = None

@app.on_event("startup")
async def startup_event():
    global worker, agent
    
    try:
        # Initialize database connection
        connect_to_mongo()
        
        # Read agent configuration from environment variables
        import os
        
        llm_type = os.getenv("LLM_TYPE", "gemini").lower()
        
        # Get LLM config based on type
        llm_config = {}
        if llm_type == "gemini":
            llm_config = {
                "api_key": os.getenv("GEMINI_API_KEY"),
                "model": os.getenv("GEMINI_MODEL"),
            }
        elif llm_type == "openai":
            llm_config = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("OPENAI_MODEL"),
                "base_url": os.getenv("OPENAI_BASE_URL"),
            }
        elif llm_type == "vnpt":
            llm_config = {
                "api_key": os.getenv("VNPT_AUTHORIZATION"),
                "token_id": os.getenv("VNPT_TOKEN_ID"),
                "token_key": os.getenv("VNPT_TOKEN_KEY"),
                "model": os.getenv("VNPT_MODEL"),
            }
        
        # Agent settings
        enable_planner = os.getenv("ENABLE_PLANNER", "true").lower() == "true"
        use_conversation_history = os.getenv("USE_CONVERSATION_HISTORY", "true").lower() == "true"
        max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))
        
        # Vector database settings
        vector_db_backend = os.getenv("VECTOR_DB_BACKEND", "chroma")
        chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./storage/chroma")
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()  # "gemini" or "openai"
        
        # Log configuration
        model_name = llm_config.get("model", "default")
        print(f">>> Config: LLM={llm_type.upper()}({model_name}), Embedding={embedding_provider.upper()}, VectorDB={vector_db_backend}", flush=True)
        vector_db = None
        
        if vector_db_backend == "chroma":
            from agent.memory import VectorDatabase
            
            # Initialize embedding function for ChromaDB (independent from LLM type)
            embedding_function = None
            
            try:
                if embedding_provider == "openai":
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    if not openai_api_key:
                        raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
                    from agent.llm.openai import get_openai_embedding_function
                    embedding_function = get_openai_embedding_function(
                        api_key=openai_api_key,
                        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                        base_url=os.getenv("OPENAI_BASE_URL")
                    )
                else:  # Default to Gemini
                    gemini_api_key = os.getenv("GEMINI_API_KEY")
                    if not gemini_api_key:
                        raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
                    from agent.llm.gemini import get_google_embedding_function
                    embedding_function = get_google_embedding_function(
                        api_key=gemini_api_key,
                        model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
                    )
            except Exception as e:
                print(f">>> ERROR: Could not initialize {embedding_provider} embedding: {e}", flush=True)
                raise
            
            # Create VectorDatabase with ChromaDB backend
            vector_db = VectorDatabase(
                backend_type="chroma",
                persist_directory=chroma_persist_dir,
                embedding_function=embedding_function
            )
        else:
            from agent.memory import VectorDatabase
            vector_db = VectorDatabase(backend_type=vector_db_backend)
        
        # Initialize Agent
        agent = RAGAgent(
            vector_db=vector_db,
            llm_type=llm_type,
            llm_config=llm_config,
            enable_planner=enable_planner,
            use_conversation_history=use_conversation_history
        )
        
        # Start worker
        loop = asyncio.get_running_loop()
        worker = ChatWorker(agent, main_loop=loop)
        worker.start()
        print(">>> Server started successfully", flush=True)
        
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
