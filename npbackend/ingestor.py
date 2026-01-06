import os
import datetime
import requests
from datasets import load_dataset
from ai_engine import NepaliAIEngine

class NepaliIngestor:
    def __init__(self, db):
        self.db = db
        self.ai = NepaliAIEngine()
        self.api_key = os.getenv("NEWSDATA_API_KEY") 
        self.hf_token = os.getenv("HF_TOKEN")

    # ... (sync_huggingface_dataset remains the same)

    def fetch_and_process(self, query=None, country="np", category=None):
        """
        Fetches live news from NewsData.io. 
        Fixed to correctly pass the 'category' parameter to the API.
        """
        if not self.api_key:
            print("❌ No API Key for NewsData.io found.")
            return

        print(f"--- 🌍 Fetching Live News (Query: {query}, Category: {category}) ---")
        
        base_url = "https://newsdata.io/api/1/latest"
        params = {
            "apikey": self.api_key,
            "country": country,
            "language": "ne",
        }
        
        # KEY FIX: Pass category and query to the params if they exist
        if query:
            params["q"] = query
        if category:
            params["category"] = category

        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if data.get("status") != "success":
                # Detailed error logging helps debug API limit issues
                print(f"❌ API Error: {data.get('results', {}).get('message', 'Unknown error')}")
                return

            articles_to_insert = []
            for art in data.get("results", []):
                # Check duplication by Link
                if self.db.articles.find_one({"url": art.get("link")}):
                    continue
                
                # Combine title and description for a richer embedding
                full_text = f"{art.get('title')} {art.get('description') or ''}"
                
                doc = {
                    "title": art.get("title"),
                    "content": art.get("description") or art.get("content") or "",
                    "url": art.get("link"),
                    "image_url": art.get("image_url"),
                    "source": art.get("source_id"),
                    # Store the category we fetched, or fall back to API's category
                    "category": category if category else art.get("category", ["विविध"])[0],
                    "embedding": self.ai.get_embedding(full_text),
                    "created_at": datetime.datetime.utcnow(),
                    "cluster_id": None
                }
                articles_to_insert.append(doc)

            if articles_to_insert:
                self.db.articles.insert_many(articles_to_insert)
                print(f"✅ Added {len(articles_to_insert)} new live articles.")
            else:
                print("⏭️ No new unique articles found.")

        except Exception as e:
            print(f"❌ NewsData.io Fetch Error: {e}")

    def fetch_trending_nepal_news(self):
        """
        Loops through key categories to populate the home feed.
        """
        trending_categories = ["top", "politics", "world", "sports", "technology"]
        
        for category in trending_categories:
            print(f"🔥 Triggering refresh for: {category}")
            self.fetch_and_process(country="np", category=category)