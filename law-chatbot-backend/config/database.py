"""
MongoDB database configuration and connection management.

This module provides database connection setup, configuration management,
and utility functions for MongoDB operations in the Law Chatbot API.
It handles both async and sync MongoDB connections with proper error handling.

Author: Law Chatbot Team
Version: 1.0.0
"""

import os
import traceback
from typing import Optional
from threading import Lock

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

from .logging import get_logger

# Initialize logger
logger = get_logger(__name__)


class DatabaseConfig:
    """
    MongoDB database configuration and connection management.

    This class handles MongoDB connection configuration including host,
    port, credentials, and database name. It supports both URI-based
    and individual parameter configuration.
    """

    def __init__(self):
        """
        Initialize database configuration from environment variables.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        # MongoDB connection settings
        self.mongo_host = os.getenv("DB_HOST", "localhost")
        self.mongo_port = int(os.getenv("DB_PORT", "27017"))
        self.mongo_username = os.getenv("DB_USERNAME", "")
        self.mongo_password = os.getenv("DB_PASSWORD", "")
        self.mongo_database = os.getenv("DB_NAME", "law_chatbot")
        self.connection_string = os.getenv("MONGO_URI")
        
        # Build connection string
        if not self.connection_string:
            if self.mongo_username and self.mongo_password:
                self.connection_string = (
                    f"mongodb://{self.mongo_username}:{self.mongo_password}"
                    f"@{self.mongo_host}:{self.mongo_port}/{self.mongo_database}"
                    "?authSource=admin"
                )
            else:
                self.connection_string = (
                    f"mongodb://{self.mongo_host}:{self.mongo_port}"
                    f"/{self.mongo_database}"
                )
        else:
            # Extract database name from connection string, removing query parameters
            db_part = self.connection_string.split("/")[-1]
            self.mongo_database = db_part.split("?")[0]

    def get_client(self) -> AsyncIOMotorClient:
        """
        Get async MongoDB client.

        Returns:
            AsyncIOMotorClient: Async MongoDB client instance

        Raises:
            ConnectionError: If client creation fails
        """
        return AsyncIOMotorClient(self.connection_string)

    def get_sync_client(self) -> MongoClient:
        """
        Get sync MongoDB client.

        Returns:
            MongoClient: Sync MongoDB client instance

        Raises:
            ConnectionError: If client creation fails
        """
        return MongoClient(self.connection_string)


# Global database instance
db_config = DatabaseConfig()


class DatabaseManager:
    """
    Manages MongoDB database connections.

    This class handles the lifecycle of MongoDB connections including
    connection establishment, health checks, and cleanup operations.
    Uses singleton pattern for thread-safe sync client.
    """

    def __init__(self):
        """
        Initialize database manager.

        Sets up connection state variables for tracking client and database instances.
        """
        self.async_client: Optional[AsyncIOMotorClient] = None
        self.async_database = None
        
        # Singleton sync client with thread-safe initialization
        self._sync_client: Optional[MongoClient] = None
        self._sync_lock = Lock()

    async def connect(self):
        """
        Create async database connection.

        Establishes connection to MongoDB, performs health check, and creates indexes.

        Raises:
            ConnectionError: If MongoDB connection fails
            TimeoutError: If connection times out
            ValueError: If connection configuration is invalid
        """
        try:
            logger.info("Attempting to connect to MongoDB...")
            logger.info(
                "Connection string: %s",
                db_config.connection_string.replace(db_config.mongo_password, "***")
                if db_config.mongo_password
                else db_config.connection_string,
            )

            self.async_client = db_config.get_client()
            self.async_database = self.async_client[db_config.mongo_database]

            # Test connection
            await self.async_client.admin.command("ping")
            logger.info("✅ Connected to MongoDB: %s", db_config.mongo_database)

            # Create indexes
            await self.create_indexes()

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error("❌ Failed to connect to MongoDB: %s", e)
            logger.error("Traceback: %s", traceback.format_exc())
            raise e

    async def close(self):
        """
        Close async database connection.

        Properly closes the MongoDB client connection and cleans up resources.
        """
        if self.async_client:
            self.async_client.close()
            logger.info("🔌 Disconnected from async MongoDB")

    async def create_indexes(self):
        """
        Create database indexes.

        Creates necessary indexes for optimal database performance.

        Raises:
            ValueError: If index creation fails
            TypeError: If database operations are invalid
            AttributeError: If database structure is invalid
        """
        try:
            # Chat queue collection indexes
            await self.async_database.chat_queue.create_index("session_id")
            await self.async_database.chat_queue.create_index("is_processed")
            await self.async_database.chat_queue.create_index("time")

            # Chat history collection indexes
            await self.async_database.history_chat.create_index("session_id")
            await self.async_database.history_chat.create_index("time")

            logger.info("📊 Database indexes created successfully")

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("⚠️ Failed to create indexes: %s", e)

    def get_database(self):
        """
        Get sync database instance (thread-safe singleton).

        Returns:
            Database: MongoDB database instance

        Raises:
            ConnectionError: If sync client creation fails
        """
        if self._sync_client is None:
            with self._sync_lock:
                # Double-check locking pattern
                if self._sync_client is None:
                    logger.info("Creating singleton sync MongoDB client...")
                    self._sync_client = db_config.get_sync_client()
                    
                    # Test connection
                    try:
                        self._sync_client.admin.command("ping")
                        logger.info("✅ Sync MongoDB client connected")
                    except Exception as e:
                        logger.error("❌ Sync client connection failed: %s", e)
                        raise ConnectionError(f"Failed to connect sync client: {e}")

        return self._sync_client[db_config.mongo_database]

    def get_async_database(self):
        """
        Get async database instance.

        Returns:
            Database: Async MongoDB database instance or None if not connected
        """
        if self.async_database is None:
            logger.error("Async database is None! Connection may have failed.")
        return self.async_database


# Global database manager instance
db_manager = DatabaseManager()


async def connect_to_mongo():
    """
    Connect to MongoDB (async).

    Establishes connection to MongoDB using the global database manager.

    Raises:
        ConnectionError: If MongoDB connection fails
        TimeoutError: If connection times out
        ValueError: If connection configuration is invalid
    """
    await db_manager.connect()


async def close_mongo_connection():
    """
    Close MongoDB connection (async).

    Properly closes the MongoDB connection and cleans up resources.
    """
    await db_manager.close()


def get_database():
    """
    Get sync database instance (thread-safe).

    Returns:
        Database: MongoDB database instance

    Raises:
        ConnectionError: If database connection fails
    """
    return db_manager.get_database()


def get_async_database():
    """
    Get async database instance.

    Returns:
        Database: Async MongoDB database instance or None if not connected
    """
    return db_manager.get_async_database()

