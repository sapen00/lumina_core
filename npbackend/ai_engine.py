import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class NepaliAIEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NepaliAIEngine, cls).__new__(cls)
            print("--- 🇳🇵 Initializing Nepali AI Engine ---")
            
            # 1. SEARCH/EMBEDDING MODEL (Using multilingual MiniLM - essentially a distilled mBERT)
            # This converts Nepali text into numbers for vector search.
            cls._instance.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # 2. CLASSIFICATION (Zero-Shot)
            cls._instance.classifier = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli",
                device=-1 # CPU for now, change to 0 for GPU
            )
            
            # 3. SUMMARIZATION (Upgraded from simple mBERT to mBART or mT5)
            # mBERT cannot generate text effectively. We use a model fine-tuned for Nepali summarization.
            # 'GenzNepal/mt5-summarize-nepali' is a good candidate for this specific task.
            print("--- 🧠 Loading Summarization Model (mT5/mBART) ---")
            try:
                cls._instance.summ_tokenizer = AutoTokenizer.from_pretrained("GenzNepal/mt5-summarize-nepali")
                cls._instance.summ_model = AutoModelForSeq2SeqLM.from_pretrained("GenzNepal/mt5-summarize-nepali")
            except Exception as e:
                print(f"⚠️ Could not load specific Nepali model, falling back to mBART-50: {e}")
                cls._instance.summ_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
                cls._instance.summ_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50")

            # Standard Nepali Categories
            cls._instance.categories = ["राजनीति", "खेलकुद", "मनोरञ्जन", "प्रविधि", "अर्थतन्त्र", "शिक्षा", "समाज"]

        return cls._instance

    def get_embedding(self, text):
        return self.embedder.encode(text).tolist()

    def classify_nepali(self, text):
        """Categorizes Nepali text into Devanagari labels"""
        try:
            # We classify using English labels internally for better accuracy with BART, 
            # then map to Nepali, or pass Nepali directly if the model supports it well.
            # Using direct Nepali here:
            res = self.classifier(text[:500], self.categories)
            return res['labels'][0]
        except:
            return "विविध" # Miscellaneous

    def summarize_cluster(self, texts):
        """
        Takes a list of article contents (a cluster) and generates a single Nepali summary.
        """
        if not texts: return ""
        
        # Combine first 2-3 sentences of top 3 articles to form input context
        combined_text = " ".join([t[:500] for t in texts[:3]])
        
        try:
            inputs = self.summ_tokenizer(
                "summarize: " + combined_text, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            )
            
            summary_ids = self.summ_model.generate(
                inputs["input_ids"], 
                max_length=150, 
                min_length=40, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
            
            summary = self.summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"Summarization failed: {e}")
            return texts[0][:200] + "..." # Fallback to truncation