from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import os
import joblib
from core.time_index import causal_slice


class TopicModeler:
    def __init__(self, n_topics=10):
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        
    def fit(self, texts):
        print(f"Fitting LDA on {len(texts)} articles...")
        tfidf = self.vectorizer.fit_transform(texts)
        self.lda.fit(tfidf)
        
    def get_topics(self, texts):
        tfidf = self.vectorizer.transform(texts)
        topic_dist = self.lda.transform(tfidf)
        return topic_dist.argmax(axis=1), topic_dist

    def save_model(self, path):
        joblib.dump({"vectorizer": self.vectorizer, "lda": self.lda}, path)
        
    def load_model(self, path):
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.lda = data["lda"]

def process_topics(csv_path, output_path, n_topics=10, train_cutoff_date=None, model_path="data/processed/lda_model.joblib"):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Ensure date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    modeler = TopicModeler(n_topics=n_topics)
    
    # Train Logic
    if train_cutoff_date and 'date' in df.columns:
        print(f"Training Topic Model on data <= {train_cutoff_date}...")
        train_df = causal_slice(df.set_index('date'), train_cutoff_date).reset_index()
        train_headlines = train_df['headline'].astype(str).tolist()
        modeler.fit(train_headlines)
        modeler.save_model(model_path)
    elif os.path.exists(model_path):
        print(f"Loading existing Topic Model from {model_path}...")
        modeler.load_model(model_path)
    else:
        print("Warning: Fitting on ALL data (Potential Leakage if backtesting). Provide train_cutoff_date for strict causal safety.")
        headlines = df['headline'].astype(str).tolist()
        modeler.fit(headlines)
        modeler.save_model(model_path)
    
    # Transform (Apply to all)
    all_headlines = df['headline'].astype(str).tolist()
    topic_ids, probs = modeler.get_topics(all_headlines)
    
    df['topic_id'] = topic_ids
    df['topic_confidence'] = probs.max(axis=1)
    
    df.to_csv(output_path, index=False)
    print(f"Topic modeling completed. Saved to {output_path}")
    return df

if __name__ == "__main__":
    raw_news = "data/raw/news_raw.csv"
    output = "data/processed/news_with_topics.csv"
    process_topics(raw_news, output)
