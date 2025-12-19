import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controller.chat_controller import router as chat_router
from service.worker import ChatWorker
from agent.agent import RAGAgent

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
    # Initialize Agent
    # We assume environment variables are set for LLM configuration if needed
    # or relying on default behavior of RAGAgent
    agent = RAGAgent(llm_type="gemini") # Explicitly trying gemini as per code hint
    
    loop = asyncio.get_running_loop()
    worker = ChatWorker(agent, main_loop=loop)
    worker.start() # Now synchronous spawning threads

@app.on_event("shutdown")
async def shutdown_event():
    if worker:
        worker.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
