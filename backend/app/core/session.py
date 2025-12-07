import redis.asyncio as redis
from config import settings

from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id: str)-> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id=session_id,url=settings.REDIS_URL)