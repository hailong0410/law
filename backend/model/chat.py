from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ChatQueueModel(BaseModel):
    session_id: str
    chat_content: str
    time: datetime = Field(default_factory=datetime.now)
    is_processed: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123",
                "chat_content": "Tell me about the law",
                "time": "2023-10-01T12:00:00",
                "is_processed": False
            }
        }

class ChatHistoryModel(BaseModel):
    session_id: str
    chat_content: str
    time: datetime = Field(default_factory=datetime.now)
    role: str # 'user' or 'assistant'

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123",
                "chat_content": "Hello",
                "time": "2023-10-01T12:00:00",
                "role": "user"
            }
        }
