import asyncio
from typing import Dict

class ConnectionManager:
    def __init__(self):
        # Dictionary to hold queues for each session_id
        self.active_connections: Dict[str, asyncio.Queue] = {}

    async def connect(self, session_id: str):
        if session_id not in self.active_connections:
            self.active_connections[session_id] = asyncio.Queue()
        # You might want to handle multiple connections for the same session differently,
        # but for now, we assume one queue per session is sufficient or they share it (though Queue isn't broadcast).
        # To support multiple tabs for same session, we might need a list of queues.
        # For simplicity/mvp: one queue per session_id. 
        # If a new connect comes, we repurpose the existing queue or create new one?
        # Let's ensure it exists.
        return self.active_connections[session_id]

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].put(message)

manager = ConnectionManager()
