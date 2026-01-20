import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingestion.news_ingestor import NewsIngestionManager, MockNewsProvider
from news_engine.sentiment_analyzer import process_sentiment
from news_engine.topic_modeler import process_topics
from features.event_alignment import align_news_to_inflections
from news_engine.relevance_model import RelevanceModel, build_relevance_dataset
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, COMMODITIES
from utils.logger import setup_logger

logger = setup_logger("news_pipeline", log_file="news_pipeline.log")

def main():
    logger.info("Starting News Intelligence Pipeline...")
    
    # 1. Ingest News
    news_raw = os.path.join(RAW_DATA_DIR, "news_raw.csv")
    manager = NewsIngestionManager(MockNewsProvider())
    logger.info("Ingesting historical news...")
    manager.ingest("2010-01-01", "2024-12-31", news_raw)
    
    # 2. Process Sentiment & Topics
    logger.info("Processing sentiment and topics...")
    news_intel = os.path.join(PROCESSED_DATA_DIR, "news_with_intel.csv")
    df = process_sentiment(news_raw, news_intel)
    df = process_topics(news_intel, news_intel) # Overwrite with topics
    
    # 3. Align to Inflection Points
    logger.info("Aligning news to market inflection points...")
    inf_path = os.path.join(PROCESSED_DATA_DIR, "inflection_points.csv")
    study_list_path = os.path.join(PROCESSED_DATA_DIR, "news_study_list.csv")
    study_list = align_news_to_inflections(news_intel, inf_path)
    study_list.to_csv(study_list_path, index=False)
    
    # 4. Train/Run Relevance Model
    logger.info("Training and running news relevance model...")
    X, y = build_relevance_dataset(df, study_list)
    rel_model = RelevanceModel()
    rel_model.fit(X, y)
    df['relevance_prob'] = rel_model.predict_relevance(X)
    
    # 5. Aggregate News to Daily Features
    logger.info("Aggregating news to daily features...")
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    df['date'] = df['timestamp_utc'].dt.date
    
    daily_news = df.groupby('date').agg({
        'compound': ['mean', 'count'],
        'relevance_prob': 'mean'
    })
    
    # Flatten columns
    daily_news.columns = ['news_sentiment_mean', 'news_volume', 'news_relevance_mean']
    daily_news.index = pd.to_datetime(daily_news.index)
    
    # 6. Update Feature Store
    logger.info("Updating Feature Store V2...")
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store.csv")
    try:
        feature_store = pd.read_csv(store_path, index_col=0, parse_dates=True)
        final_store = feature_store.join(daily_news, how='left').fillna(0)
        
        final_output = os.path.join(PROCESSED_DATA_DIR, "feature_store_v2.csv")
        final_store.to_csv(final_output)
        logger.info(f"Feature Store V2 built with {len(final_store.columns)} columns.")
        logger.info(f"Saved to {final_output}")
    except Exception as e:
        logger.error(f"Failed to update feature store: {e}")

    logger.info("News Pipeline Complete.")

if __name__ == "__main__":
    main()
