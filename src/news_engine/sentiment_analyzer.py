from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import os

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def get_sentiment(self, text):
        """
        Returns a dictionary with pos, neg, neu, and compound scores.
        """
        if not isinstance(text, str):
            return {"compound": 0, "pos": 0, "neg": 0, "neu": 1}
        return self.analyzer.polarity_scores(text)

def process_sentiment(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    analyzer = SentimentAnalyzer()
    
    print(f"Processing sentiment for {len(df)} items...")
    
    # Calculate scores
    scores = df['headline'].apply(analyzer.get_sentiment)
    
    # Expand scores into columns
    scores_df = pd.DataFrame(scores.tolist())
    df = pd.concat([df, scores_df], axis=1)
    
    df.to_csv(output_path, index=False)
    print(f"Sentiment analysis completed. Saved to {output_path}")
    return df

if __name__ == "__main__":
    # Test on study list
    raw_news = "data/raw/news_raw.csv"
    output = "data/processed/news_with_sentiment.csv"
    process_sentiment(raw_news, output)
