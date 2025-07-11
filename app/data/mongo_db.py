"""
Manages the connection to the MongoDB database.
"""

import os
import sys
import pymongo
from typing import Optional, Dict, List, Union, Any
from datetime import datetime, timedelta, UTC
import pandas as pd
import logging
import time # Import time module for sleep
from configparser import ConfigParser
from pymongo.errors import ConnectionFailure, OperationFailure

# Add the parent directory to the path to import app modules
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, app_dir)

# Also add the project root to the path
project_root = os.path.dirname(app_dir)
sys.path.insert(0, project_root)

from app.config.constants import DEFAULT_CONFIG_PATH, MONGO_DB_NAME, MONGO_COLLECTION_NAME

try:
    from app.util.logger import get_logger
    logger = get_logger(__name__)
    print(f"Successfully imported logger from {app_dir}")
except ImportError as e:
    print(f"Warning: Could not import app logger: {e}")
    print(f"Current sys.path: {sys.path}")
    # Fallback logging if app logger is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# Constants for messages
_MSG_NOT_CONNECTED = "Not connected to MongoDB"


class MongoDBClient:
    """Handles the connection and authentication to the MongoDB database."""

    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        """
        Initializes the MongoDB connection using credentials from the config file.

        Args:
            config_path (str): Path to the configuration file. Defaults to DEFAULT_CONFIG_PATH.
        """
        self.config = ConfigParser(interpolation=None)
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.config.read(config_path)
        
        if 'MONGO' not in self.config or 'ConnectionString' not in self.config['MONGO']:
            logger.error("The [MONGO] section or 'ConnectionString' key is missing in your config file.")
            logger.error(f"Please check your configuration file at: {config_path}")
            raise KeyError("MongoDB 'ConnectionString' not found in config file.")
            
        self.connection_string = self.config['MONGO']['ConnectionString']
        
        # Connection attempt parameters
        self.max_retries = 3
        self.retry_delay = 5 # seconds
        
        self.client: Optional[pymongo.MongoClient] = None
        self._connected = False
        self._initialized = False
        self.connect()

    def connect(self):
        """Establishes connection to the MongoDB database with retry logic."""
        if self._connected:
            logger.info("Already connected to MongoDB.")
            return True

        attempts = 0
        while attempts < self.max_retries:
            attempts += 1
            logger.info(f"Initializing MongoDB connection (Attempt {attempts}/{self.max_retries})...")
            
            try:
                # Add serverSelectionTimeoutMS to fail faster if server is not available
                self.client = pymongo.MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
                # The ping command is cheap and does not require auth, but forces connection.
                self.client.admin.command('ping')
                logger.info("MongoDB connection successful.")
                self._connected = True
                return True # Successfully connected
            except OperationFailure as e:
                if e.code == 8000: # AtlasError: bad auth
                    logger.error(f"MongoDB Authentication Failed (Attempt {attempts}): {e.details.get('errmsg', 'No details')}")
                    logger.error("Please check the following in your config.ini:")
                    logger.error("1. Correct username and password in your ConnectionString.")
                    logger.error("2. Ensure your current IP address is whitelisted in MongoDB Atlas.")
                    logger.error("3. If your password has special characters, ensure they are URL-encoded.")
                else:
                    logger.error(f"A database operation failed on connection (Attempt {attempts}): {e}")
            except ConnectionFailure as e:
                logger.error(f"MongoDB connection failed on attempt {attempts}: {e}")
                logger.error("Please check your network settings and the server address in your ConnectionString.")
            except Exception as e:
                logger.error(f"An unexpected error occurred during MongoDB connection attempt {attempts}: {e}")

            # If we created a client instance but failed to connect, close it.
            if self.client:
                self.client.close()

            # Wait before the next retry, unless it's the last attempt
            if attempts < self.max_retries:
                logger.info(f"Waiting {self.retry_delay} seconds before next connection attempt...")
                time.sleep(self.retry_delay)

        # If loop finishes without returning True, connection failed
        logger.error(f"Failed to connect to MongoDB after {self.max_retries} attempts.")
        self._connected = False
        return False

    def list_collection_names(self, db_name: str) -> Optional[List[str]]:
        """
        Lists the names of all collections in a given database.

        Args:
            db_name (str): The name of the database to query.

        Returns:
            Optional[List[str]]: A list of collection names, or None if not connected.
        """
        if not self._connected or not self.client:
            logger.error(_MSG_NOT_CONNECTED)
            return None
        
        try:
            db = self.client[db_name]
            collections = db.list_collection_names()
            logger.info(f"Successfully retrieved collections from database '{db_name}'.")
            return collections
        except Exception as e:
            logger.error(f"An error occurred while listing collections from '{db_name}': {e}")
            return None

    def shutdown(self):
        """Closes the MongoDB connection."""
        if self._connected and self.client:
            logger.info("Closing MongoDB connection...")
            self.client.close()
            self._connected = False
            logger.info("MongoDB connection closed.")
        else:
            logger.info("MongoDB connection already closed or not established.")

    def is_connected(self) -> bool:
        """
        Checks if the client is still connected to the MongoDB server by pinging it.
        
        Returns:
            bool: True if connected, False otherwise.
        """
        if not self._connected or not self.client:
            return False
        
        try:
            # The 'ping' command is cheap and does not require auth.
            self.client.admin.command('ping')
            return True
        except ConnectionFailure:
            logger.warning("MongoDB connection lost (ping failed).")
            self._connected = False
            return False

    def get_connection_status(self) -> bool:
        """Returns the internal connection status flag."""
        return self._connected
    
    def add_one(self, db_name: str, collection_name: str, document: Dict[str, Any]) -> Optional[Any]:
        """
        Inserts a single document into a specified collection.

        Args:
            db_name (str): The name of the database.
            collection_name (str): The name of the collection.
            document (Dict[str, Any]): The document to insert.

        Returns:
            Optional[Any]: The ID of the inserted document, or None on failure.
        """
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return None
        
        try:
            db = self.client[db_name]
            collection = db[collection_name]
            result = collection.insert_one(document)
            logger.info(f"Successfully inserted document with ID {result.inserted_id} into '{db_name}.{collection_name}'.")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to insert document into '{db_name}.{collection_name}': {e}")
            return None

    def add_many(self, db_name: str, collection_name: str, documents: List[Dict[str, Any]]) -> Optional[List[Any]]:
        """
        Inserts multiple documents into a specified collection.

        Args:
            db_name (str): The name of the database.
            collection_name (str): The name of the collection.
            documents (List[Dict[str, Any]]): A list of documents to insert.

        Returns:
            Optional[List[Any]]: A list of IDs for the inserted documents, or None on failure.
        """
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return None
        
        if not documents:
            logger.warning("Document list is empty, nothing to insert.")
            return []
            
        try:
            db = self.client[db_name]
            collection = db[collection_name]
            result = collection.insert_many(documents)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} documents into '{db_name}.{collection_name}'.")
            return result.inserted_ids
        except Exception as e:
            logger.error(f"Failed to insert documents into '{db_name}.{collection_name}': {e}")
            return None

    def query_distinct(self, db_name: str, collection_name: str, key: str, filter: Dict[str, Any] = None) -> Optional[List[Any]]:
        """
        Finds the distinct values for a specified key in a collection.

        Args:
            db_name (str): The name of the database.
            collection_name (str): The name of the collection.
            key (str): The key for which to return distinct values.
            filter (Dict[str, Any], optional): A query that matches documents. Defaults to None.

        Returns:
            Optional[List[Any]]: A list of distinct values, or None on failure.
        """
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return None
        
        try:
            db = self.client[db_name]
            collection = db[collection_name]
            distinct_values = collection.distinct(key, filter or {})
            logger.info(f"Successfully retrieved {len(distinct_values)} distinct values for key '{key}' from '{db_name}.{collection_name}'.")
            return distinct_values
        except Exception as e:
            logger.error(f"Failed to query distinct values from '{db_name}.{collection_name}': {e}")
            return None

    def query_one(self, db_name: str, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Finds and returns a single document matching the query.

        Args:
            db_name (str): The name of the database.
            collection_name (str): The name of the collection.
            query (Dict[str, Any]): The query to match a document.

        Returns:
            Optional[Dict[str, Any]]: A single document, or None if no match is found or on failure.
        """
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return None
            
        try:
            db = self.client[db_name]
            collection = db[collection_name]
            document = collection.find_one(query)
            if document:
                logger.info(f"Successfully found document in '{db_name}.{collection_name}'.")
            else:
                logger.info(f"No document found matching query in '{db_name}.{collection_name}'.")
            return document
        except Exception as e:
            logger.error(f"Failed to query one document from '{db_name}.{collection_name}': {e}")
            return None

    def query_all(self, db_name: str, collection_name: str, query: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Finds and returns all documents matching the query.

        Args:
            db_name (str): The name of the database.
            collection_name (str): The name of the collection.
            query (Dict[str, Any], optional): The query to match documents. Defaults to None, which matches all.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of documents, or None on failure.
        """
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return None
            
        try:
            db = self.client[db_name]
            collection = db[collection_name]
            cursor = collection.find(query or {})
            documents = list(cursor)
            logger.info(f"Successfully found {len(documents)} documents in '{db_name}.{collection_name}'.")
            return documents
        except Exception as e:
            logger.error(f"Failed to query all documents from '{db_name}.{collection_name}': {e}")
            return None

    def delete_one(self, db_name: str, collection_name: str, query: Dict[str, Any]) -> Optional[int]:
        """
        Deletes a single document matching the query.

        Args:
            db_name (str): The name of the database.
            collection_name (str): The name of the collection.
            query (Dict[str, Any]): The query to match the document to delete.

        Returns:
            Optional[int]: The number of documents deleted (0 or 1), or None on failure.
        """
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return None
            
        try:
            db = self.client[db_name]
            collection = db[collection_name]
            result = collection.delete_one(query)
            logger.info(f"Successfully deleted {result.deleted_count} document(s) from '{db_name}.{collection_name}'.")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to delete one document from '{db_name}.{collection_name}': {e}")
            return None

    def delete_many(self, db_name: str, collection_name: str, query: Dict[str, Any]) -> Optional[int]:
        """
        Deletes all documents matching the query.

        Args:
            db_name (str): The name of the database.
            collection_name (str): The name of the collection.
            query (Dict[str, Any]): The query to match the documents to delete.

        Returns:
            Optional[int]: The number of documents deleted, or None on failure.
        """
        if not self.is_connected():
            logger.error(_MSG_NOT_CONNECTED)
            return None
            
        try:
            db = self.client[db_name]
            collection = db[collection_name]
            result = collection.delete_many(query)
            logger.info(f"Successfully deleted {result.deleted_count} document(s) from '{db_name}.{collection_name}'.")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to delete documents from '{db_name}.{collection_name}': {e}")
            return None


if __name__ == "__main__":
    mongo_db = MongoDBClient()
    print("MongoDB Client Object:", mongo_db.client)
    if mongo_db.get_connection_status():
        print("Connection Status: Connected")
        
        # --- Test the list_collection_names function ---
        db_to_query = MONGO_DB_NAME 
        print(f"Attempting to list collections from database: '{db_to_query}'")
        collections = mongo_db.list_collection_names(db_to_query)
        if collections is not None:
            print("Available collections:", collections)
        else:
            print("Could not retrieve collections.")

        # --- Test add_one and add_many ---
        db_for_insert = MONGO_DB_NAME
        collection_for_insert = "test_samples"
        
        print("\n--- Testing add_one and add_many ---")
        
        # Test add_one
        print(f"Attempting to insert one document into '{db_for_insert}.{collection_for_insert}'...")
        doc_one = {"name": "Test Movie", "year": 2025, "timestamp": datetime.now(UTC)}
        inserted_id = mongo_db.add_one(db_for_insert, collection_for_insert, doc_one)
        if inserted_id:
            print(f"Document inserted with ID: {inserted_id}")

        # Test add_many
        print(f"\nAttempting to insert multiple documents into '{db_for_insert}.{collection_for_insert}'...")
        docs_many = [
            {"name": "Test Movie 2", "year": 2026, "timestamp": datetime.now(UTC)},
            {"name": "Test Movie 3", "year": 2027, "timestamp": datetime.now(UTC)}
        ]
        inserted_ids = mongo_db.add_many(db_for_insert, collection_for_insert, docs_many)
        if inserted_ids:
            print(f"Documents inserted with IDs: {inserted_ids}")
        
        # --- Test Query Functions ---
        print("\n--- Testing Query Functions ---")
        
        # Test query_one
        print("\nQuerying for one document (year: 2025)...")
        one_result = mongo_db.query_one(db_for_insert, collection_for_insert, {"year": 2025})
        if one_result:
            print(f"Found one document: {one_result}")

        # Test query_all
        print("\nQuerying for all documents in the collection...")
        all_results = mongo_db.query_all(db_for_insert, collection_for_insert)
        if all_results:
            print(f"Found {len(all_results)} documents in total.")

        # Test query_distinct
        print("\nQuerying for distinct years...")
        distinct_years = mongo_db.query_distinct(db_for_insert, collection_for_insert, "year")
        if distinct_years:
            print(f"Distinct years found: {sorted(distinct_years)}")

        # --- Test Delete Functions ---
        print("\n--- Testing Delete Functions ---")
        
        # Test delete_one
        print("\nDeleting one document (year: 2025)...")
        deleted_count_one = mongo_db.delete_one(db_for_insert, collection_for_insert, {"year": 2025})
        if deleted_count_one is not None:
            print(f"Deleted {deleted_count_one} document.")
            # Verify it's gone
            remaining_docs = mongo_db.query_all(db_for_insert, collection_for_insert)
            if remaining_docs is not None:
                print(f"Documents remaining: {len(remaining_docs)}")
        
        # Test delete_many
        print("\nDeleting remaining documents (year >= 2026)...")
        deleted_count_many = mongo_db.delete_many(db_for_insert, collection_for_insert, {"year": {"$gte": 2026}})
        if deleted_count_many is not None:
            print(f"Deleted {deleted_count_many} documents.")
            # Verify they're gone
            final_docs = mongo_db.query_all(db_for_insert, collection_for_insert)
            if final_docs is not None:
                print(f"Documents remaining: {len(final_docs)}")

        # Clean up the test collection
        if mongo_db.is_connected():
            print(f"\nCleaning up test collection '{collection_for_insert}'...")
            try:
                mongo_db.client[db_for_insert].drop_collection(collection_for_insert)
                print("Test collection dropped successfully.")
            except Exception as e:
                print(f"Could not drop test collection: {e}")
        
        # --- Test connection status functions ---
        print(f"\nIs connected? (ping test): {mongo_db.is_connected()}")
        print(f"Get connection status (flag): {mongo_db.get_connection_status()}")

        # --- Test shutdown ---
        mongo_db.shutdown()
        print(f"Is connected after shutdown? (ping test): {mongo_db.is_connected()}")
        print(f"Get connection status after shutdown (flag): {mongo_db.get_connection_status()}")

    else:
        print("Connection Status: Failed")