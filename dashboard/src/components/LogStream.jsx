import React, { useEffect, useRef } from 'react';
import { Terminal, Shield, Cpu } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const LogStream = ({ logs, systemStatus }) => {
    const scrollRef = useRef(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="glass h-full flex flex-col">
            <div className="p-3 border-b border-white/5 flex justify-between items-center bg-white/5">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                    <Terminal size={12} /> System Nervous System
                </span>
                <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${systemStatus.heartbeat === "OPERATIONAL" ? "bg-green-500 shadow-[0_0_5px_rgba(34,197,94,0.5)]" : "bg-red-500"}`}></span>
                </div>
            </div>

            {/* Metrics Row */}
            <div className="grid grid-cols-2 gap-2 p-2 border-b border-white/5">
                <div className="bg-white/5 p-2 rounded text-center">
                    <span className="block text-[10px] text-slate-500 uppercase">Risk (VaR)</span>
                    <span className="text-sm font-bold text-red-400">12.4%</span>
                </div>
                <div className="bg-white/5 p-2 rounded text-center">
                    <span className="block text-[10px] text-slate-500 uppercase">Latency</span>
                    <span className="text-sm font-bold text-green-400">24ms</span>
                </div>
            </div>

            {/* Log Feed */}
            <div ref={scrollRef} className="flex-grow overflow-y-auto p-2 space-y-1 font-mono text-[10px]">
                {logs.length === 0 && <div className="text-slate-600 italic p-2">Waiting for system events...</div>}
                <AnimatePresence>
                    {logs.map((log, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="border-l-2 border-white/10 pl-2 py-1 text-slate-300 hover:bg-white/5 transition-colors"
                        >
                            <span className="text-slate-500 mr-2">[{log.time}]</span>
                            <span className={log.type === "alert" ? "text-red-400 font-bold" : (log.type === "success" ? "text-green-400" : "")}>
                                {log.message}
                            </span>
                        </motion.div>
                    ))}
                </AnimatePresence>
            </div>

            <div className="p-2 border-t border-white/5 text-[9px] text-slate-600 text-center uppercase tracking-widest">
                Ver: {systemStatus.version} | Mode: {systemStatus.risk_mandate}
            </div>
        </div>
    );
};

export default LogStream;
