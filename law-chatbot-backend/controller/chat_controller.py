from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse
from model.chat import ChatQueueModel, ChatHistoryModel
from config.database import get_database
from service.stream_manager import manager
from datetime import datetime
import asyncio
import json

router = APIRouter()

@router.post("/chat")
async def chat(chat_request: ChatQueueModel):
    db = get_database()
    
    # Ensure time is set if not provided
    if not chat_request.time:
        chat_request.time = datetime.now()
        
    chat_request.is_processed = False
    
    # PyMongo sync operation
    db.chat_queue.insert_one(chat_request.dict())
    
    return {"status": "queued", "session_id": chat_request.session_id}

@router.get("/take_all_chat")
async def take_all_chat(session_id: str = Query(...)):
    db = get_database()
    # PyMongo sync operation
    cursor = db.history_chat.find({"session_id": session_id}).sort("time", 1)
    history = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"]) # Convert ObjectId to string
        history.append(doc)
    return history

@router.get("/stream")
async def stream(session_id: str = Query(...)):
    async def event_generator():
        queue = await manager.connect(session_id)
        try:
            while True:
                # Wait for message
                data = await queue.get()
                # Yield SSE format
                yield f"data: {json.dumps({'content': data})}\n\n"
        except asyncio.CancelledError:
            manager.disconnect(session_id)
            print(f"Client {session_id} disconnected")

    return StreamingResponse(event_generator(), media_type="text/event-stream")
