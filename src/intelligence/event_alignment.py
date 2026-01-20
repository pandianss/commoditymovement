import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class EventAligner:
    """
    Aligns news events to market windows to create an 'Event Study' dataset.
    Supporting 'Market Intelligence Core' Requirement A.
    """
    
    def __init__(self, market_data: pd.DataFrame, news_data: pd.DataFrame):
        """
        :param market_data: DataFrame with index=Datetime, columns=[commodity, close, etc.]
        :param news_data: DataFrame with columns=[timestamp_utc, commodity, headline, etc.]
        """
        self.market_data = market_data.sort_index()
        self.news_data = news_data
        
        # Ensure timestamps
        if not isinstance(self.news_data['timestamp_utc'].dtype, pd.DatetimeTZDtype):
             self.news_data['timestamp_utc'] = pd.to_datetime(self.news_data['timestamp_utc'])
    
    def align_to_windows(self, lookback_days=1, lookforward_days=5):
        """
        Aligns each news item with market data surrounding it.
        
        Computes:
        - Pre-event trend (lookback)
        - Post-event response (lookforward)
        """
        aligned_events = []
        
        # Group by commodity to speed up (avoid unrelated matching)
        for commodity, group in self.news_data.groupby('commodity'):
            if commodity not in self.market_data['commodity'].values:
                logger.warning(f"Commodity {commodity} in news but not in market data.")
                continue
                
            comm_market = self.market_data[self.market_data['commodity'] == commodity].sort_index()
            
            for _, news_item in group.iterrows():
                event_time = news_item['timestamp_utc']
                event_date = event_time.normalize() # Midnight of event day
                
                # Define Indices
                start_date = event_date - timedelta(days=lookback_days)
                end_date = event_date + timedelta(days=lookforward_days)
                
                # Slice Market Data
                window = comm_market.loc[start_date:end_date]
                
                if window.empty:
                    continue
                    
                # Calculate simple metrics (Placeholders for 'ImpactAnalyzer')
                # Price at event (or closest prev close)
                try:
                    event_idx = window.index.get_indexer([event_date], method='nearest')[0]
                    price_at_event = window.iloc[event_idx]['close']
                    
                    # Forward returns
                    fwd_returns = {}
                    for i in range(1, lookforward_days + 1):
                        target_date = event_date + timedelta(days=i)
                        # Find closest actual trading data
                        target_indices = window.index.get_indexer([target_date], method='nearest')
                        if len(target_indices) > 0:
                            p_t = window.iloc[target_indices[0]]['close']
                            ret = (p_t - price_at_event) / price_at_event
                            fwd_returns[f'fwd_ret_{i}d'] = ret
                        else:
                            fwd_returns[f'fwd_ret_{i}d'] = None
                            
                    event_record = news_item.to_dict()
                    event_record.update({
                        'event_price': price_at_event,
                        **fwd_returns
                    })
                    aligned_events.append(event_record)
                    
                except Exception as e:
                    logger.debug(f"Failed to align event at {event_date}: {e}")
                    continue

        return pd.DataFrame(aligned_events)
