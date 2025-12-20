import asyncio
import logging
import threading
import time
from typing import List

from config.database import get_database
from service.stream_manager import manager
from agent.agent import RAGAgent

logger = logging.getLogger(__name__)


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

    # =========================
    # PUBLIC API
    # =========================
    def start(self):
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
                    response = await self.agent.chat(session_id, content)
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
