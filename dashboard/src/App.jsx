import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, ReferenceLine
} from 'recharts';
import {
  TrendingUp, TrendingDown, AlertTriangle, Newspaper, Activity, BarChart3, Clock, Zap
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = "http://localhost:8000/api";

const App = () => {
  const [marketData, setMarketData] = useState({});
  const [predictions, setPredictions] = useState({ GOLD: [] });
  const [news, setNews] = useState([]);
  const [shocks, setShocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('GOLD');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [marketRes, predRes, newsRes, shockRes] = await Promise.all([
          axios.get(`${API_BASE}/market-data`),
          axios.get(`${API_BASE}/predictions`),
          axios.get(`${API_BASE}/news`),
          axios.get(`${API_BASE}/shocks`)
        ]);
        setMarketData(marketRes.data);
        setPredictions(predRes.data);
        setNews(newsRes.data);
        setShocks(shockRes.data);
        setLoading(false);
      } catch (err) {
        console.error("Error fetching data:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#05070a]">
        <motion.div
          animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="text-blue-500 font-bold"
        >
          LOGGING INTO INTELLIGENCE CONSOLE...
        </motion.div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-[1600px] mx-auto min-h-screen">
      {/* Header */}
      <header className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold premium-gradient m-0">INTELLIGENCE CONSOLE</h1>
          <p className="text-slate-500 text-sm mt-1 uppercase tracking-widest font-medium flex items-center gap-2">
            <Activity size={14} className="text-green-500" /> Continuous Monitoring Active
          </p>
        </div>
        <div className="flex gap-4">
          <div className="glass px-4 py-2 flex items-center gap-3">
            <Clock size={16} className="text-slate-400" />
            <span className="text-slate-300 font-mono text-sm">{new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-12 gap-6">
        {/* Shocks & Summary Bar */}
        <div className="col-span-12 glass p-4 flex gap-8 items-center overflow-x-auto whitespace-nowrap">
          <div className="flex items-center gap-2 px-4 border-r border-white/5">
            <Zap size={18} className="text-yellow-500" />
            <span className="text-xs font-bold text-slate-400 uppercase">Live Pulse:</span>
          </div>
          {shocks.map((shock, i) => (
            <motion.div
              key={i}
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 px-3 py-1 rounded-full"
            >
              <AlertTriangle size={14} className="text-red-500 shadow-sm" />
              <span className="text-sm font-bold text-red-400">{shock.ticker}</span>
              <span className="text-xs text-red-500/80">{(shock.magnitude * 100).toFixed(2)}% move detected</span>
            </motion.div>
          ))}
        </div>

        {/* Main Chart Area */}
        <div className="col-span-12 lg:col-span-8 space-y-6">
          <section className="glass p-6 min-h-[500px] flex flex-col">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-semibold m-0 flex items-center gap-2">
                <BarChart3 size={20} className="text-blue-500" />
                Predictive Quantile Forecasts
              </h2>
              <div className="flex bg-white/5 p-1 rounded-lg">
                {Object.keys(marketData).map(ticker => {
                  const names = {
                    'GC=F': 'GOLD',
                    'SI=F': 'SILVER',
                    'CL=F': 'CRUDE OIL',
                    'HG=F': 'COPPER',
                    'NG=F': 'NAT GAS'
                  };
                  return (
                    <button
                      key={ticker}
                      onClick={() => setActiveTab(ticker)}
                      className={`px-3 py-1 text-xs font-bold rounded-md transition-all ${activeTab === ticker ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}
                      title={ticker}
                    >
                      {names[ticker] || ticker.replace('=F', '')}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="flex-grow">
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={(predictions[activeTab] || []).map(p => ({
                  ...p,
                  p05: Math.max(-0.2, Math.min(0.2, p.p05)),
                  p50: Math.max(-0.2, Math.min(0.2, p.p50)),
                  p95: Math.max(-0.2, Math.min(0.2, p.p95)),
                }))}>
                  <defs>
                    <linearGradient id="colorPred" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.1} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="date" hide />
                  <YAxis
                    domain={[-0.20, 0.20]}
                    allowDataOverflow={true}
                    stroke="#475569"
                    fontSize={10}
                    tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    itemStyle={{ fontSize: '12px' }}
                    formatter={(val) => `${(val * 100).toFixed(2)}%`}
                    labelFormatter={(label) => new Date(label).toLocaleDateString()}
                  />
                  {/* Prediction Range */}
                  <Area type="monotone" dataKey="p95" stroke="none" fill="#3b82f6" fillOpacity={0.05} />
                  <Area type="monotone" dataKey="p05" stroke="none" fill="#05070a" fillOpacity={1} />

                  {/* Median Prediction */}
                  <Line type="monotone" dataKey="p50" stroke="#3b82f6" strokeWidth={3} dot={false} animationDuration={2000} />
                  <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
                </AreaChart>
              </ResponsiveContainer>
              <div className="mt-4 p-4 bg-blue-500/5 rounded-xl border border-blue-500/10 flex justify-between items-center">
                <p className="text-sm text-blue-400/80 m-0 italic">
                  * TCN Quantile Engine: The shaded region represents a 90% confidence interval for next-day price movement.
                </p>
                <div className="flex gap-4 text-xs text-slate-400">
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500"></span> Median</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500/20"></span> 90% CI</span>
                </div>
              </div>
            </div>
          </section>

          {/* Intelligence Grid */}
          <div className="grid grid-cols-3 gap-6">
            <div className="glass p-4">
              <span className="card-title block">Sentiment Index</span>
              <div className="flex items-end gap-2">
                <span className="text-3xl font-bold text-white">0.68</span>
                <span className="text-sm text-green-400 flex items-center mb-1"><TrendingUp size={12} /> +12%</span>
              </div>
            </div>
            <div className="glass p-4">
              <span className="card-title block">System Uptime</span>
              <div className="flex items-end gap-2">
                <span className="text-3xl font-bold text-white">99.9%</span>
                <span className="text-sm text-slate-400 mb-1 uppercase tracking-widest text-[10px]">Stable</span>
              </div>
            </div>
            <div className="glass p-4">
              <span className="card-title block">Retrain Signal</span>
              <div className="flex items-end gap-2">
                <span className="text-3xl font-bold text-white uppercase italic text-yellow-500">Idle</span>
                <span className="text-sm text-slate-400 mb-1 uppercase tracking-widest text-[10px]">No Drift</span>
              </div>
            </div>
          </div>
        </div>

        {/* Side Panel: News Intelligence */}
        <div className="col-span-12 lg:col-span-4 flex flex-col h-full">
          <section className="glass p-6 overflow-hidden flex flex-col h-[740px]">
            <div className="flex items-center gap-2 mb-6">
              <Newspaper size={20} className="text-blue-500" />
              <h2 className="text-xl font-semibold m-0">News Intelligence</h2>
            </div>
            <div className="flex-grow overflow-y-auto pr-2">
              <AnimatePresence>
                {news.map((item, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="mb-4 p-4 rounded-xl bg-white/5 border border-white/5 hover:border-white/10 transition-all cursor-pointer group"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span className="text-[10px] font-bold text-blue-500/80 uppercase tracking-widest">{item.source}</span>
                      <div className={`px-2 py-0.5 rounded text-[9px] font-bold ${item.compound > 0 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                        {item.compound > 0 ? "BULLISH" : "BEARISH"} ({item.compound.toFixed(2)})
                      </div>
                    </div>
                    <h3 className="text-sm font-medium text-slate-200 group-hover:text-blue-400 transition-colors line-clamp-2">
                      {item.headline}
                    </h3>
                    <div className="mt-3 flex items-center justify-between">
                      <span className="text-[10px] text-slate-500 font-mono">
                        {new Date(item.timestamp_utc).toLocaleDateString()}
                      </span>
                      <span className="text-[10px] bg-white/5 px-2 py-0.5 rounded text-slate-400">
                        Relevance: {(item.relevance_prob * 100).toFixed(0)}%
                      </span>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default App;
