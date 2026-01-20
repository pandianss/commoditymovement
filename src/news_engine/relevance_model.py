import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RelevanceModel:
    """
    Predicts P(news moves commodity | context).
    Uses sentiment, topic, and historical market context.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def prepare_data(self, news_with_intel, inflection_points):
        """
        Labels news items as 'impactful' if they occur near an inflection point.
        """
        # Cross join news and inflections is too big, let's use the study list logic
        # For simplicity, we assume 'impactful' if in study list
        pass # To be implemented in the integration script

    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict_relevance(self, X):
        return self.model.predict_proba(X)[:, 1]

def build_relevance_dataset(news_df, study_list_df):
    """
    Creates a training set where 'y' is 1 if the news is in the study list 
    (aligned to a move) and 0 otherwise.
    """
    # Features: sentiment (compound, pos, neg), topic_id, length of headline
    news_df['is_impactful'] = 0
    # Match on headline + source (proxy for unique ID)
    impact_keys = set(study_list_df['headline'] + study_list_df['source'])
    news_df.loc[(news_df['headline'] + news_df['source']).isin(impact_keys), 'is_impactful'] = 1
    
    # Feature engineering for the model
    X = news_df[['compound', 'pos', 'neg', 'topic_id']].copy()
    X['headline_len'] = news_df['headline'].apply(len)
    y = news_df['is_impactful']
    
    return X, y

if __name__ == "__main__":
    # This requires both sentiment and topics to be run first
    pass
