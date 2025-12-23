import asyncio
import logging
import threading
import time
from typing import List
from pathlib import Path

from config.database import get_database
from service.stream_manager import manager
from agent.agent import RAGAgent
from config.logging import get_logger

logger = get_logger(__name__)


class ChatWorker:
    def __init__(
        self,
        agent: RAGAgent,
        main_loop: asyncio.AbstractEventLoop,
        worker_count: int = 5,
        poll_interval: float = 1.0,
    ):
        self.agent = agent
        self.main_loop = main_loop
        self.worker_count = worker_count
        self.poll_interval = poll_interval

        self._stop_event = threading.Event()
        self.threads: List[threading.Thread] = []
        
        # Load system prompt from file
        self.system_prompt = self._load_system_prompt()

    # =========================
    # HELPER METHODS
    # =========================
    def _load_system_prompt(self) -> str:
        """Load system prompt from file"""
        system_prompt_file = Path(__file__).parent / "system_prompt.txt"
        try:
            if system_prompt_file.exists():
                with open(system_prompt_file, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
                logger.info(f"Loaded system prompt from {system_prompt_file}")
                return prompt
        except Exception as e:
            logger.warning(f"Failed to load system prompt: {e}")
        return None

    # =========================
    # PUBLIC API
    # =========================
    def start(self):
        print(f">>> Worker.start() called with count={self.worker_count}", flush=True)
        logger.info(f"Starting {self.worker_count} chat worker threads")

        for worker_id in range(self.worker_count):
            t = threading.Thread(
                target=self._thread_entry,
                args=(worker_id,),
                daemon=True,
                name=f"chat-worker-{worker_id}",
            )
            t.start()
            self.threads.append(t)

    def stop(self):
        logger.info("Stopping chat workers...")
        self._stop_event.set()

        for t in self.threads:
            t.join(timeout=5)

    # =========================
    # THREAD ENTRY
    # =========================
    def _thread_entry(self, worker_id: int):
        """
        Each thread owns its own asyncio event loop
        """
        print(f">>> Worker-{worker_id} Thread Started", flush=True)
        logger.info(f"[Worker-{worker_id}] Thread started")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._worker_loop(worker_id))
        except Exception as e:
            logger.exception(f"[Worker-{worker_id}] Crashed: {e}")
        finally:
            loop.close()
            logger.info(f"[Worker-{worker_id}] Loop closed")

    # =========================
    # WORKER LOOP (SYNC with async agent calls)
    # =========================
    async def _worker_loop(self, worker_id: int):
        db = get_database()
        collection = db.chat_queue

        logger.info(f"[Worker-{worker_id}] Entering consume loop")

        while not self._stop_event.is_set():
            try:
                # PyMongo sync operation - thread-safe
                chat_item = collection.find_one_and_update(
                    {"is_processed": False},
                    {
                        "$set": {
                            "is_processed": True,
                            "worker_id": worker_id,
                        }
                    },
                    sort=[("time", 1)],
                )

                if not chat_item:
                    await asyncio.sleep(self.poll_interval)
                    continue

                session_id = chat_item["session_id"]
                content = chat_item["chat_content"]

                logger.info(
                    f"[Worker-{worker_id}] Processing session={session_id}"
                )

                try:
                    logger.info(f"[Worker-{worker_id}] START processing session={session_id}")
                    logger.info(f"[Worker-{worker_id}] Content: {content}")
                    logger.info(f"[Worker-{worker_id}] System prompt: {self.system_prompt}")
                    
                    start_time = time.time()
                    
                    response = await self.agent.chat(
                        session_id, 
                        content, 
                        system_prompt=self.system_prompt
                    )
                    
                    process_time = time.time() - start_time
                    logger.info(f"[Worker-{worker_id}] DONE processing session={session_id} in {process_time:.2f}s")
                    
                    self._send_to_main_loop(session_id, response)
                except Exception as e:
                    logger.exception(
                        f"[Worker-{worker_id}] Agent error: {e}"
                    )

            except Exception as e:
                logger.exception(
                    f"[Worker-{worker_id}] Worker loop error: {e}"
                )
                await asyncio.sleep(3)

    # =========================
    # CROSS-THREAD COMMUNICATION
    # =========================
    def _send_to_main_loop(self, session_id: str, message: str):
        """
        Send message safely to main asyncio loop (SSE/WebSocket)
        """
        future = asyncio.run_coroutine_threadsafe(
            manager.send_message(session_id, message),
            self.main_loop,
        )

        # Optional: wait for completion / log errors
        try:
            future.result(timeout=10)
        except Exception as e:
            logger.error(f"Send message failed: {e}")
