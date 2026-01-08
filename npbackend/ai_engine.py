import torch
from sentence_transformers import SentenceTransformer
# We use explicit MT5 classes to ensure your model loads correctly
from transformers import pipeline, T5Tokenizer, MT5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

class NepaliAIEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NepaliAIEngine, cls).__new__(cls)
            print("--- 🇳🇵 Initializing Lumina Nepali AI Engine ---")
            
            # Use GPU if available, else CPU
            device_id = 0 if torch.cuda.is_available() else -1
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"--- 🚀 Running on: {device_name.upper()} ---")
            
            # 1. SEARCH/EMBEDDING MODEL
            cls._instance.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # 2. CLASSIFICATION (Zero-Shot)
            cls._instance.classifier = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli",
                device=device_id
            )
            
            # 3. YOUR CUSTOM SUMMARIZATION MODEL
            cls._instance.my_model_id = "sapen-00/nepali_news_summ"
            print(f"--- 🧠 Loading CUSTOM Nepali Model: {cls._instance.my_model_id} ---")
            
            try:
                # Using T5Tokenizer and MT5ForConditionalGeneration explicitly fixes the 'Unrecognized' error
                cls._instance.summ_tokenizer = T5Tokenizer.from_pretrained(cls._instance.my_model_id)
                cls._instance.summ_model = MT5ForConditionalGeneration.from_pretrained(cls._instance.my_model_id)
                
                if device_name == "cuda":
                    cls._instance.summ_model = cls._instance.summ_model.to("cuda")
                print("✅ Custom model loaded successfully!")
                    
            except Exception as e:
                print(f"⚠️ Could not load your model, falling back to mBART: {e}")
                # Fallback also needs to be robust
                cls._instance.summ_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
                cls._instance.summ_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50")

            # Standard Nepali Categories
            cls._instance.categories = ["राजनीति", "खेलकुद", "मनोरञ्जन", "प्रविधि", "अर्थतन्त्र", "शिक्षा", "समाज"]

        return cls._instance

    def get_embedding(self, text):
        return self.embedder.encode(text).tolist()

    def classify_nepali(self, text):
        try:
            res = self.classifier(text[:500], self.categories)
            return res['labels'][0]
        except:
            return "विविध"

    def summarize_cluster(self, texts):
        if not texts: return ""
        
        # We only use the first 2 articles to keep the summary concise and fast
        combined_text = " ".join([t[:500] for t in texts[:2]])
        
        try:
            # Prefix 'summarize: ' is usually required for T5-based models
            input_text = "summarize: " + combined_text
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = self.summ_tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to(device)
            
            # Generate shorter output since you trained for headlines/summaries
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