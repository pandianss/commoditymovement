import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR

def align_news_to_inflections(news_path, inflection_path, window_days=1):
    """
    Finds news articles that fall within a window of inflection points.
    """
    if not (os.path.exists(news_path) and os.path.exists(inflection_path)):
        print("Required files not found.")
        return pd.DataFrame()
        
    news_df = pd.read_csv(news_path)
    inf_df = pd.read_csv(inflection_path, index_col=0, parse_dates=True)
    
    # Ensure news timestamps are datetime
    news_df['timestamp_utc'] = pd.to_datetime(news_df['timestamp_utc'])
    news_df['date'] = news_df['timestamp_utc'].dt.date
    news_df['date'] = pd.to_datetime(news_df['date'])
    
    aligned_events = []
    
    for idx, row in inf_df.iterrows():
        # Define window
        start_win = idx - pd.Timedelta(days=window_days)
        end_win = idx + pd.Timedelta(days=window_days)
        
        # Filter news for this window and commodity (if specified in news)
        mask = (news_df['date'] >= start_win) & (news_df['date'] <= end_win)
        
        # If news has a 'commodity' field, match it
        comm_name = row['commodity']
        comm_mask = (news_df['commodity'] == comm_name) | (news_df['commodity'].isna())
        
        window_news = news_df[mask & comm_mask]
        
        for _, news_item in window_news.iterrows():
            aligned_events.append({
                "inflection_date": idx,
                "commodity": comm_name,
                "move_type": row['move_type'],
                "magnitude": row['magnitude'],
                "headline": news_item['headline'],
                "source": news_item['source'],
                "news_date": news_item['date']
            })
            
    return pd.DataFrame(aligned_events)

def main():
    news_path = "data/raw/news_raw.csv"
    inf_path = os.path.join(PROCESSED_DATA_DIR, "inflection_points.csv")
    
    # For demo purposes, if news_raw doesn't exist, we run the ingestor first
    if not os.path.exists(news_path):
        from data_ingestion.news_ingestor import NewsIngestionManager, MockNewsProvider
        manager = NewsIngestionManager(MockNewsProvider())
        manager.ingest("2010-01-01", "2024-12-31", news_path)
        
    study_list = align_news_to_inflections(news_path, inf_path)
    
    if not study_list.empty:
        output_path = os.path.join(PROCESSED_DATA_DIR, "news_study_list.csv")
        study_list.to_csv(output_path, index=False)
        print(f"Built Study List with {len(study_list)} aligned news items.")
        print(f"Top entries:\n{study_list.head()}")
    else:
        print("No news aligned to inflection points.")

if __name__ == "__main__":
    main()
