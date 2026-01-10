import torch
import threading
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5Tokenizer, MT5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

class NepaliAIEngine:
    _instance = None
    _lock = threading.Lock()  # Prevents race conditions during initialization

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-check to ensure another thread didn't create it 
                # while we were waiting for the lock
                if cls._instance is None:
                    cls._instance = super(NepaliAIEngine, cls).__new__(cls)
                    cls._instance._initialize_engine()
        return cls._instance

    def _initialize_engine(self):
        """Internal method to load models once."""
        print("--- 🇳🇵 Initializing Lumina Nepali AI Engine ---")
        
        # 1. Hardware Detection
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- 🚀 Running on: {self.device_name.upper()} ---")
        
        # 2. Search/Embedding Model
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # 3. Classification (Zero-Shot)
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli",
            device=self.device_id
        )
        
        # 4. Custom Summarization Model
        self.my_model_id = "sapen-00/nepali_news_summ"
        print(f"--- 🧠 Loading CUSTOM Nepali Model: {self.my_model_id} ---")
        
        try:
            self.summ_tokenizer = T5Tokenizer.from_pretrained(self.my_model_id)
            self.summ_model = MT5ForConditionalGeneration.from_pretrained(self.my_model_id)
            
            if self.device_name == "cuda":
                self.summ_model = self.summ_model.to("cuda")
            print("✅ Custom model loaded successfully!")
                
        except Exception as e:
            print(f"⚠️ Could not load your model, falling back to mBART: {e}")
            self.summ_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
            self.summ_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50")
            if self.device_name == "cuda":
                self.summ_model = self.summ_model.to("cuda")

        # Standard Nepali Categories
        self.categories = ["राजनीति", "खेलकुद", "मनोरञ्जन", "प्रविधि", "अर्थतन्त्र", "शिक्षा", "समाज"]

    def get_embedding(self, text):
        return self.embedder.encode(text).tolist()

    def classify_nepali(self, text):
        try:
            res = self.classifier(text[:500], self.categories)
            return res['labels'][0]
        except:
            return "विविध"

    def summarize_cluster(self, texts):
        if not texts: 
            return ""
        
        # Check if model is actually loaded to prevent attribute errors
        if not hasattr(self, 'summ_model') or self.summ_model is None:
            return texts[0][:150] + "..."

        combined_text = " ".join([t[:500] for t in texts[:2]])
        
        try:
            input_text = "summarize: " + combined_text
            
            inputs = self.summ_tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to(self.device_name if self.device_name == "cuda" else "cpu")
            
            summary_ids = self.summ_model.generate(
                inputs["input_ids"], 
                max_length=80, 
                min_length=15, 
                length_penalty=1.5, 
                num_beams=4, 
                early_stopping=True
            )
            
            summary = self.summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            print(f"Summarization failed: {e}")
            return texts[0][:150] + "..."