import schedule
import time
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from data_ingestion.bse_fetcher import BSEHistoricalFetcher
from intelligence.spike_detector import SpikeDetector
from intelligence.news_correlator import NewsCorrelator

logger = setup_logger("continuous_cycle", log_file="continuous_cycle.log")

def run_analysis_cycle():
    """
    Executes one full cycle of the intelligence pipeline.
    1. Fetch Latest Data
    2. Detect Anomalies
    3. Correlate News
    4. Generate/Update Report (Placeholder)
    """
    logger.info(f"Starting Analysis Cycle at {datetime.now()}")
    
    # 1. Fetch Latest Data
    try:
        fetcher = BSEHistoricalFetcher()
        new_data = fetcher.fetch_latest()
        if new_data.empty:
            logger.info("No new data found for today yet.")
        else:
            logger.info(f"Ingested {len(new_data)} new records.")
            
            # 2. Detect Anomalies on New Data
            detector = SpikeDetector()
            spikes = detector.detect_spikes(new_data)
            
            if not spikes.empty:
                logger.info(f"DETECTED {len(spikes)} REAL-TIME ANOMALIES!")
                
                # 3. Correlate with News
                correlator = NewsCorrelator()
                context_spikes = correlator.correlate_spikes(spikes)
                
                # Append to persistent record
                history_path = "data/processed/spikes_with_context.csv"
                header = not os.path.exists(history_path)
                context_spikes.to_csv(history_path, mode='a', header=header, index=False)
                logger.info(f"Updated Risk Map with new anomalies: {history_path}")
                
            else:
                logger.info("No significant anomalies detected in new data.")
                
    except Exception as e:
        logger.error(f"Error in analysis cycle: {e}")

    logger.info("Cycle Complete.\n")

def start_continuous_loop(interval_minutes=60):
    """
    Starts the continuous loop.
    """
    logger.info(f"Starting Continuous Intelligence Engine (Interval: {interval_minutes} mins)")
    
    # Run once immediately
    run_analysis_cycle()
    
    # Schedule
    schedule.every(interval_minutes).minutes.do(run_analysis_cycle)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    # For demonstration/testing, we might run just once or start the loop
    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        try:
             # Default to 1 hour for "Continuous" (since BSE updates EOD, but we poll for intraday readiness)
            start_continuous_loop(interval_minutes=60)
        except KeyboardInterrupt:
            logger.info("Stopping Continuous Engine.")
    else:
        run_analysis_cycle()
