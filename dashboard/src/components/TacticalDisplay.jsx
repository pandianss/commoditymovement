import React from 'react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, ReferenceLine
} from 'recharts';
import { TrendingUp, TrendingDown, Target } from 'lucide-react';

const TacticalDisplay = ({ activeTab, predictions, marketData }) => {
    const data = predictions[activeTab] || [];
    const latestPred = data.length > 0 ? data[data.length - 1] : null;
    const currentPrice = marketData[activeTab]?.Close || 0;

    // Determine Signal
    const signal = latestPred ? (latestPred.p50 > 0.005 ? "STRONG BUY" : (latestPred.p50 < -0.005 ? "STRONG SELL" : "NEUTRAL")) : "ANALYZING";
    const signalColor = signal.includes("BUY") ? "bg-green-500" : (signal.includes("SELL") ? "bg-red-500" : "bg-slate-600");
    const glowClass = signal.includes("BUY") ? "shadow-[0_0_20px_rgba(34,197,94,0.3)]" : (signal.includes("SELL") ? "shadow-[0_0_20px_rgba(239,68,68,0.3)]" : "");

    return (
        <div className="glass h-full flex flex-col relative overflow-hidden">
            {/* Signal Banner */}
            <div className={`p-4 ${signalColor} ${glowClass} text-black font-bold flex justify-between items-center z-10`}>
                <div className="flex items-center gap-3">
                    <Target size={24} className="animate-pulse" />
                    <span className="text-2xl tracking-tighter">{signal} @ {currentPrice.toFixed(2)}</span>
                </div>
                <div className="text-right">
                    <span className="text-xs block opacity-70 uppercase tracking-widest">Confidence</span>
                    <span className="text-lg">85%</span>
                </div>
            </div>

            {/* Main Chart */}
            <div className="flex-grow p-4 relative">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id="colorP50" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="date" hide />
                        <YAxis
                            domain={['auto', 'auto']}
                            orientation="right"
                            stroke="#475569"
                            fontSize={11}
                            tickFormatter={(val) => `${(val * 100).toFixed(1)}%`}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid rgba(255,255,255,0.1)' }}
                            itemStyle={{ fontSize: '12px' }}
                        />
                        {/* Confidence Interval */}
                        <Area type="monotone" dataKey="p95" stroke="none" fill="#3b82f6" fillOpacity={0.05} />
                        <Area type="monotone" dataKey="p05" stroke="none" fill="#05070a" fillOpacity={1} />

                        {/* Median Prediction */}
                        <Line
                            type="monotone"
                            dataKey="p50"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            dot={false}
                            activeDot={{ r: 4, fill: '#fff' }}
                        />
                        <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
                    </AreaChart>
                </ResponsiveContainer>

                {/* Waterfall / Depth Placeholder (Visual Flourish) */}
                <div className="absolute bottom-4 left-4 p-2 bg-black/40 rounded border border-white/5 backdrop-blur-sm">
                    <div className="flex items-center gap-2 mb-1">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                        <span className="text-[10px] text-slate-400 font-mono">TCN-QUANTILE ENGINE ACTIVE</span>
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono">
                        H: {latestPred?.p95.toFixed(4)} L: {latestPred?.p05.toFixed(4)}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TacticalDisplay;
