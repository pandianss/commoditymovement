import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    date: pd.Timestamp
    asset: str
    quantity: float
    price: float
    cost: float
    direction: int # 1 or -1

@dataclass
class PortfolioState:
    date: pd.Timestamp
    cash: float
    positions: Dict[str, float] # asset -> quantity
    equity: float

class BacktestEngine:
    """
    Enhanced Backtesting Engine with capital accounting and transaction costs.
    Supporting Requirement F.
    """
    
    def __init__(self, initial_capital=100_000.0, commission_bps=10, slippage_bps=5):
        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10000.0
        self.slippage_rate = slippage_bps / 10000.0
        
        self.cash = initial_capital
        self.positions = {} # asset -> quantity
        self.history: List[PortfolioState] = []
        self.trades: List[Trade] = []
        
    def run_backtest(self, price_data: pd.DataFrame, allocations: pd.DataFrame):
        """
        Executes backtest based on daily allocations.
        :param price_data: DataFrame with index=Date, columns=[Asset_Close...]
        :param allocations: DataFrame with columns=[date, asset, weight]
        """
        # Align dates
        # Ensure allocations has date as datetime
        allocations['date'] = pd.to_datetime(allocations['date'])
        
        # Iterate daily
        dates = price_data.index.unique().sort_values()
        
        for date in dates:
            # 1. Update Portfolio Value (Mark-to-Market)
            current_prices = price_data.loc[date]
            port_value = self.cash
            for asset, qty in self.positions.items():
                if asset in current_prices:
                    port_value += qty * current_prices[asset]
            
            # Record State (Pre-trade)
            self.history.append(PortfolioState(date, self.cash, self.positions.copy(), port_value))
            
            # 2. Rebalance (if allocations exist for this day)
            daily_allocs = allocations[allocations['date'] == date]
            if not daily_allocs.empty:
                self._rebalance(date, daily_allocs, current_prices, port_value)
                
        return pd.DataFrame([vars(s) for s in self.history])

    def _rebalance(self, date, allocations, current_prices, total_equity):
        """
        Adjusts positions to match target weights.
        """
        for _, row in allocations.iterrows():
            asset = row['asset']
            target_weight = row['weight']
            
            if asset not in current_prices:
                continue
                
            price = current_prices[asset]
            target_value = total_equity * target_weight
            current_qty = self.positions.get(asset, 0)
            current_value = current_qty * price
            
            diff_value = target_value - current_value
            
            # Calculate quantity to trade
            # Note: Approximating execution at Close price
            
            # Apply Slippage
            # Buy: Price * (1 + slip), Sell: Price * (1 - slip)
            exec_price = price * (1 + self.slippage_rate) if diff_value > 0 else price * (1 - self.slippage_rate)
            
            qty_to_trade = diff_value / exec_price
            
            # Calculate Commission
            trade_cost = abs(diff_value) * self.commission_rate
            
            # Update Cash
            self.cash -= (diff_value + trade_cost)
            
            # Update Position
            self.positions[asset] = current_qty + qty_to_trade
            
            # Log Trade
            if abs(qty_to_trade) > 0.0001:
                self.trades.append(Trade(
                    date, asset, qty_to_trade, exec_price, trade_cost, 
                    1 if qty_to_trade > 0 else -1
                ))

