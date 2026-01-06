import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Singleton-style database connection
_client = None

def get_db():
    global _client
    if _client is None:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError("MONGODB_URI not found in .env file")
        
        # Connect to Atlas
        _client = MongoClient(uri, maxPoolSize=50, minPoolSize=10)
        print("--- ✅ Connected to MongoDB Atlas (Nepali Portal) ---")
    
    return _client.nepali_news_db