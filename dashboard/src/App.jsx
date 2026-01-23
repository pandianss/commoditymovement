import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Clock, Lock, Unlock, Monitor
} from 'lucide-react';
import { TickerProvider, useTicker } from './TickerProvider';
import MarketWatch from './components/MarketWatch';
import TacticalDisplay from './components/TacticalDisplay';
import LogStream from './components/LogStream';

const API_BASE = "http://localhost:8000/api";

const DashboardContent = () => {
  const [marketData, setMarketData] = useState({});
  const [predictions, setPredictions] = useState({ GOLD: [] });
  const [logs, setLogs] = useState([]);
  const [systemStatus, setSystemStatus] = useState({
    heartbeat: "LOADING...",
    last_cycle: "...",
    risk_mandate: "...",
    version: "2.0-Alpha"
  });
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('GOLD');
  const { isConnected } = useTicker();

  // Helper to add logs
  const addLog = (msg, type = "info") => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => [...prev.slice(-49), { time, message: msg, type }]);
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [marketRes, predRes, newsRes, shockRes, statusRes] = await Promise.all([
          axios.get(`${API_BASE}/market-data`),
          axios.get(`${API_BASE}/predictions`),
          axios.get(`${API_BASE}/news`),
          axios.get(`${API_BASE}/shocks`),
          axios.get(`${API_BASE}/system-status`)
        ]);

        setMarketData(marketRes.data);
        setPredictions(predRes.data);
        setSystemStatus(statusRes.data);

        if (shockRes.data.length > 0) {
          const latestShock = shockRes.data[0];
          // Simple logic to add shocks to logs if they are new (mocked for now)
          // addLog(`SHOCK: ${latestShock.ticker} ${latestShock.magnitude}`, "alert");
        }

        setLoading(false);
      } catch (err) {
        console.error("Error fetching data:", err);
        addLog("Connection failed", "alert");
      }
    };

    fetchData();
    addLog("System initialized", "success");
    addLog("Connecting to TCN Matrix...", "info");

    const interval = setInterval(() => {
      fetchData();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleLogin = async () => {
    try {
      const res = await axios.get(`${API_BASE}/auth/kite/login`);
      window.location.href = res.data.login_url;
    } catch (e) {
      console.error("Login failed", e);
    }
  };

  if (loading) {
    return <div className="bg-[#05070a] h-screen text-blue-500 flex items-center justify-center font-mono uppercase tracking-widest animate-pulse">Initializing Pro Terminal...</div>;
  }

  return (
    <div className="bg-[#05070a] h-screen w-screen flex flex-col overflow-hidden font-sans text-slate-300">
      {/* 1. Header */}
      <header className="h-12 border-b border-white/10 flex justify-between items-center px-4 bg-[#0d1117] flex-shrink-0">
        <div className="flex items-center gap-3">
          <Monitor size={18} className="text-blue-500" />
          <h1 className="text-sm font-bold tracking-widest text-white uppercase">Intelligence Console <span className="text-blue-500">PRO</span></h1>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={handleLogin}
            className={`text-xs font-bold flex items-center gap-2 px-3 py-1 rounded bg-white/5 hover:bg-white/10 transition-all ${isConnected ? 'text-green-400' : 'text-slate-400'}`}
          >
            {isConnected ? <Unlock size={12} /> : <Lock size={12} />}
            {isConnected ? "KITE ACTIVE" : "CONNECT KITE"}
          </button>
          <div className="text-xs font-mono text-slate-500 flex items-center gap-2">
            <Clock size={12} />
            {new Date().toLocaleTimeString()}
          </div>
        </div>
      </header>

      {/* 2. Main Grid Layout */}
      <div className="flex-grow grid grid-cols-12 gap-0 overflow-hidden">

        {/* Left: Market Watch (2 cols) */}
        <div className="col-span-12 md:col-span-3 lg:col-span-2 border-r border-white/10 bg-[#06080b] h-full overflow-hidden">
          <MarketWatch
            marketData={marketData}
            predictions={predictions}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
          />
        </div>

        {/* Center: Tactical Display (7 cols) */}
        <div className="col-span-12 md:col-span-6 lg:col-span-7 bg-gradient-to-b from-[#05070a] to-[#0a0f16] relative h-full overflow-hidden">
          <TacticalDisplay
            activeTab={activeTab}
            predictions={predictions}
            marketData={marketData}
          />
        </div>

        {/* Right: Log Stream (3 cols) */}
        <div className="col-span-12 md:col-span-3 border-l border-white/10 bg-[#06080b] h-full overflow-hidden">
          <LogStream logs={logs} systemStatus={systemStatus} />
        </div>

      </div>
    </div>
  );
};

const App = () => {
  return (
    <TickerProvider>
      <DashboardContent />
    </TickerProvider>
  );
};

export default App;
