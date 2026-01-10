import numpy as np
import datetime
from datetime import timedelta
from contextlib import asynccontextmanager
from sklearn.cluster import AgglomerativeClustering
from apscheduler.schedulers.background import BackgroundScheduler

from fastapi import FastAPI, BackgroundTasks, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from database import get_db
from ai_engine import NepaliAIEngine
from ingestor import NepaliIngestor

# --- SHARED FUNCTIONS ---

def perform_clustering_logic():
    db = get_db()
    ai = NepaliAIEngine()
    print("--- 🧩 Starting On-Demand Clustering Job ---")
    
    # Use UTC now for consistency
    recent_time = datetime.datetime.utcnow() - timedelta(days=2)
    articles = list(db.articles.find({
        "cluster_id": None,
        "created_at": {"$gte": recent_time}
    }))
    
    if len(articles) < 3:
        print("⏭️ Not enough new articles to cluster.")
        return

    vectors = np.array([a['embedding'] for a in articles])
    
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.25,
        metric='cosine',
        linkage='average'
    )
    labels = model.fit_predict(vectors)
    
    grouped = {}
    for idx, label in enumerate(labels):
        if label not in grouped: grouped[label] = []
        grouped[label].append(articles[idx])
        
    clusters_created = 0
    for label, group in grouped.items():
        if len(group) < 2: continue
        
        group_texts = [a['content'] for a in group if a['content']]
        summary = ai.summarize_cluster(group_texts)
        
        main_img = next((a['image_url'] for a in group if a.get('image_url')), None)
        main_cat = group[0].get('category', 'General')
        
        cluster_doc = {
            "summary": summary,
            "title": group[0]['title'],
            "category": main_cat,
            "article_count": len(group),
            "image_url": main_img,
            "articles": [{"id": str(a['_id']), "title": a['title'], "source": a['source']} for a in group],
            "created_at": datetime.datetime.utcnow()
        }
        
        res = db.clusters.insert_one(cluster_doc)
        new_cluster_id = str(res.inserted_id)
        
        ids = [a['_id'] for a in group]
        db.articles.update_many({"_id": {"$in": ids}}, {"$set": {"cluster_id": new_cluster_id}})
        clusters_created += 1

    print(f"--- ✅ Formed {clusters_created} new Nepali news clusters. ---")

async def on_demand_fetch_and_cluster(query_str: str):
    db = get_db()
    ingestor = NepaliIngestor(db)
    print(f"🚀 On-Demand Fetching for: {query_str}")
    ingestor.fetch_and_process(query=query_str)
    perform_clustering_logic()

# --- SCHEDULER SETUP ---

scheduler = BackgroundScheduler()

def trending_refresh_job():
    print("🌅 Running scheduled trending news refresh...")
    db = get_db()
    ingestor = NepaliIngestor(db)
    ingestor.fetch_trending_nepal_news()
    perform_clustering_logic()

@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    # Wait 5 seconds for the AI Engine to warm up before starting the sync
    await asyncio.sleep(5) 
    
    scheduler.add_job(trending_refresh_job, 'date') 
    scheduler.add_job(trending_refresh_job, 'interval', hours=1)
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(title="Lumina Nepali Portal", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

@app.get("/search")
async def nepali_hybrid_search(
    background_tasks: BackgroundTasks, 
    q: str = Query(..., min_length=2)
):
    db = get_db()
    ai = NepaliAIEngine()
    query_vector = ai.get_embedding(q)
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index", 
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 50, 
                "limit": 10
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$match": {"score": {"$gte": 0.60}}}, 
        {"$project": {"embedding": 0}}
    ]
    
    results = list(db.articles.aggregate(pipeline))

    if len(results) < 3:
        background_tasks.add_task(on_demand_fetch_and_cluster, query_str=q)
        if not results:
            return {
                "status": "ingesting",
                "message": "तपाईंको खोजी अनुसार कुनै समाचार भेटिएन। हामी ताजा समाचार खोज्दैछौं...",
                "results": []
            }

    formatted = []
    for res in results:
        res["_id"] = str(res["_id"])
        formatted.append(res)
    return {"status": "success", "results": formatted}

@app.get("/feed")
def get_nepali_feed(category: str = None):
    db = get_db()
    query = {}
    if category:
        query = {"category": {"$regex": f"^{category}$", "$options": "i"}}
    
    clusters = list(db.clusters.find(query).sort("created_at", -1).limit(20))
    for c in clusters:
        c["_id"] = str(c["_id"])
    return clusters

@app.get("/admin/stats")
async def get_ingestion_stats():
    db = get_db()
    twenty_four_hours_ago = datetime.datetime.utcnow() - timedelta(hours=24)
    
    pipeline = [
        {"$match": {"created_at": {"$gte": twenty_four_hours_ago}}},
        {"$sortByCount": "$category"}
    ]
    category_stats = list(db.articles.aggregate(pipeline))
    unique_clusters = db.articles.distinct("cluster_id", {"cluster_id": {"$ne": None}})
    total_recent = db.articles.count_documents({"created_at": {"$gte": twenty_four_hours_ago}})

    return {
        "total_articles_24h": total_recent,
        "clusters_active": len(unique_clusters),
        "breakdown": category_stats
    }

@app.post("/admin/refresh-trending")
def manual_trending_refresh(background_tasks: BackgroundTasks):
    background_tasks.add_task(trending_refresh_job)
    return {"message": "Trending refresh started in background."}

@app.post("/trigger-clustering")
def manual_clustering():
    perform_clustering_logic()
    return {"message": "Clustering process finished."}

@app.delete("/admin/reset-database")
async def reset_database(confirm: bool = Query(False)):
    """
    DANGER: This will delete all articles and clusters from the database.
    Must pass ?confirm=true to execute.
    """
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="You must confirm deletion by setting the 'confirm' query parameter to true."
        )

    db = get_db()
    
    # Delete all documents from the two main collections
    articles_res = db.articles.delete_many({})
    clusters_res = db.clusters.delete_many({})
    
    return {
        "status": "success",
        "message": "Database cleared successfully.",
        "deleted_counts": {
            "articles": articles_res.deleted_count,
            "clusters": clusters_res.deleted_count
        }
    }