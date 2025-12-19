import os
import motor.motor_asyncio

MONGO_DETAILS = os.getenv("MONGO_DETAILS", "mongodb://localhost:27017")

def get_database():
    # Helper to get database with new client to avoid event loop issues in threads
    # For production, you might want a singleton per loop or connection pooling properly
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
    return client.law_chatbot
