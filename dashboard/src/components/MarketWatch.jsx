import React from 'react';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';

const MarketWatch = ({ marketData, predictions, activeTab, setActiveTab }) => {
    const commodities = Object.keys(marketData);

    return (
        <div className="glass h-full flex flex-col overflow-hidden">
            <div className="p-3 border-b border-white/5 flex justify-between items-center bg-white/5">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Market Watch</span>
                <Activity size={14} className="text-blue-500" />
            </div>

            <div className="overflow-y-auto flex-grow">
                {commodities.map((ticker) => {
                    // Mocking previous close for change calc if not real
                    const price = marketData[ticker]?.Close?.toFixed(2) || "0.00";
                    const pred = predictions[ticker] ? predictions[ticker][predictions[ticker].length - 1] : null;
                    const signal = pred ? (pred.p50 > 0.001 ? "BUY" : (pred.p50 < -0.001 ? "SELL" : "NEUTRAL")) : "WAIT";
                    const signalColor = signal === "BUY" ? "text-green-400" : (signal === "SELL" ? "text-red-400" : "text-slate-500");

                    return (
                        <div
                            key={ticker}
                            onClick={() => setActiveTab(ticker)}
                            className={`p-3 border-b border-white/5 cursor-pointer transition-all hover:bg-white/5 ${activeTab === ticker ? 'bg-blue-500/10 border-l-2 border-l-blue-500' : ''}`}
                        >
                            <div className="flex justify-between items-center mb-1">
                                <span className="font-bold text-sm text-slate-200">{ticker.replace('=F', '')}</span>
                                <span className="font-mono text-sm text-white">₹{price}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className={`text-[10px] font-bold ${signalColor} px-1.5 py-0.5 rounded bg-white/5`}>
                                    {signal} {pred && Math.abs(pred.p50 * 100).toFixed(1)}%
                                </span>
                                {/* Mock Change */}
                                <span className="text-[10px] text-slate-500 font-mono">+0.45%</span>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default MarketWatch;
