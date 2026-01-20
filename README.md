# Commodity Movement Prediction System

A sophisticated pipeline for predicting commodity price movements using macroeconomic drivers, market technicals, and news intelligence.

## Features
- **Data Ingestion**: Multi-source data ingestion (Yahoo Finance, Alpha Vantage).
- **News Engine**: VADER-based sentiment analysis, topic modeling, and historical event alignment.
- **Predictive Modeling**: Temporal Convolutional Network (TCN) with quantile heads for probabilistic forecasting.
- **Strategy Layer**: Persistence-based trend following strategies.
- **Live Intelligence**: Real-time monitoring loop with intraday shock detection and news polling.

## Architecture
The project is organized as follows:
- `src/data_ingestion`: Modules for fetching market and news data.
- `src/features`: Feature engineering, including event alignment and feature store management.
- `src/news_engine`: NLP pipeline for sentiment and topic analysis.
- `src/models`: TCN and Transformer model architectures.
- `src/strategies`: Trading signal generation logic.
- `src/utils`: Shared utilities including a centralized logger.

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up environment variables:
   Copy `.env.example` to `.env` and add your `ALPHA_VANTAGE_API_KEY`.
3. Run the daily update:
   ```bash
   python src/run_daily_update.py
   ```
4. Run the live intelligence monitor:
   ```bash
   python src/run_live_intelligence.py 30
   ```
   *(30 is the poll interval in minutes)*

## Stability & Logging
The system uses a centralized logging utility located in `src/utils/logger.py`. Logs are stored in the `/logs` directory at the project root. The live intelligence monitor (`run_live_intelligence.py`) includes refined error handling to ensure continuous operation during intermittent API or network failures.
