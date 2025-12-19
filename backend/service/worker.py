import asyncio
import logging
from config.database import get_database
from service.stream_manager import manager
from agent.agent import RAGAgent

logger = logging.getLogger(__name__)

import threading

class ChatWorker:
    def __init__(self, agent: RAGAgent, main_loop: asyncio.AbstractEventLoop):
        self.agent = agent
        self.main_loop = main_loop
        self.running = False
        self.threads = []

    def start(self, worker_count: int = 5):
        self.running = True
        logger.info(f"Starting {worker_count} worker threads...")
        for i in range(worker_count):
            t = threading.Thread(target=self._run_worker_thread, args=(i,))
            t.daemon = True # Daemon threads exit when main program exits
            t.start()
            self.threads.append(t)

    def _run_worker_thread(self, worker_id: int):
        logger.info(f"Thread {worker_id} started")
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._process_queue_loop(worker_id))
        except Exception as e:
            logger.error(f"Thread {worker_id} crashed: {e}")
        finally:
            loop.close()

    async def _process_queue_loop(self, worker_id: int):
        logger.info(f"Worker {worker_id} entering loop")
        
        # We need to get a database instance that is safe for this loop.
        # get_database() now creates a new client which is safe.
        db = get_database()
        collection = db.chat_queue

        while self.running:
            try:
                # atomically find and mark as processed
                chat_item = await collection.find_one_and_update(
                    {"is_processed": False},
                    {"$set": {"is_processed": True}},
                    sort=[("time", 1)]
                )

                if chat_item:
                    session_id = chat_item["session_id"]
                    content = chat_item["chat_content"]
                    
                    logger.info(f"Worker {worker_id} processing chat for session {session_id}")

                    # Call agent
                    try:
                        response = await self.agent.chat(session_id, content)
                        
                        # Send result to stream manager
                        # manager.send_message is async/awaitable.
                        # Wait! existing stream_manager.manager holds asyncio.Queue bound to MAIN LOOP?
                        # If we push to it from a different loop/thread, asyncio.Queue isn't thread-safe?
                        # asyncio.Queue is NOT thread-safe.
                        
                        # We need to use `run_coroutine_threadsafe` to push to the queue on the MAIN loop
                        # if the manager was created on the main loop.
                        # But manager.active_connections is a dict of asyncio.queue.
                        
                        # This is tricky. SSE endpoint is running in Main Thread (FastAPI).
                        # Worker is in Thread X.
                        # Thread X cannot await queue.put() of a queue belonging to Main Loop.
                        
                        # We must invoke send_message on the main loop.
                        # Usually we capture the main loop somewhere?
                        # Or use a thread-safe implementation.
                        
                        await self._safe_send_message(session_id, response)
                        
                    except Exception as e:
                        logger.error(f"Error processing chat {chat_item['_id']}: {e}")
                        pass

                else:
                    # No items, sleep briefly
                    await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(5)

    def stop(self):
        self.running = False
