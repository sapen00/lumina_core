import os
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient

# This finds the .env file even if it's one folder up
load_dotenv(find_dotenv())

# 1. Get variables and check for typos
# Make sure the string inside getenv matches your .env file exactly!
uri = os.getenv("MONGODB_URI") 
db_name = os.getenv("DB_NAME", "nepali_news_db")

if not uri:
    print("❌ ERROR: MONGO_URI is None. Check your .env file key name!")
    exit()

print(f"📡 Attempting to connect to: {uri[:20]}...")

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]
    
    # Trigger a quick command to check if the connection actually works
    client.admin.command('ping')
    print(f"✅ Connected to MongoDB Atlas: {db_name}")

    # 2. Reset all articles
    art_result = db.articles.update_many(
        {}, 
        {"$set": {"cluster_id": None}}
    )

    # 3. Clear clusters
    clus_result = db.clusters.delete_many({})

    print(f"--- 🧹 Database Cleaned ---")
    print(f"Articles Reset: {art_result.modified_count}")
    print(f"Clusters Deleted: {clus_result.deleted_count}")

except Exception as e:
    print(f"❌ Connection Failed: {e}")