import React, { createContext, useContext, useEffect, useState, useRef } from 'react';
import { KiteTicker } from 'kiteconnect';
import axios from 'axios';

const TickerContext = createContext();

const API_BASE = "http://localhost:8000/api";

export const TickerProvider = ({ children }) => {
    const [ticks, setTicks] = useState({});
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);
    const tickerRef = useRef(null);

    useEffect(() => {
        const initTicker = async () => {
            try {
                // Fetch active token from our backend
                const res = await axios.get(`${API_BASE}/auth/kite/token`);
                const { access_token, api_key } = res.data;

                if (!access_token || !api_key) {
                    console.warn("Kite Token missing. Ticker disabled.");
                    return;
                }

                console.log("Initializing Kite Ticker...");
                const ticker = new KiteTicker({
                    api_key: api_key,
                    access_token: access_token,
                });

                ticker.autoReconnect(true, 10, 5);

                ticker.on("ticks", (newTicks) => {
                    // console.log("Ticks", newTicks);
                    const tickMap = {};
                    newTicks.forEach(t => {
                        tickMap[t.instrument_token] = t;
                    });
                    setTicks(prev => ({ ...prev, ...tickMap }));
                });

                ticker.on("connect", () => {
                    console.log("Kite Ticker Connected");
                    setIsConnected(true);
                    // Subscribe to Gold/Silver tokens (example tokens, replace with real ones from instrument list)
                    // Gold 1g (MCX): 256265 ? Need to look up real tokens
                    // For now, subscribing to indices/demo tokens
                    // ticker.subscribe([738561]); 
                    // ticker.setMode(ticker.modeFull, [738561]);
                });

                ticker.on("disconnect", () => setIsConnected(false));
                ticker.on("error", (e) => {
                    console.error("Ticker Error", e);
                    setError(e.message);
                });

                ticker.connect();
                tickerRef.current = ticker;

            } catch (err) {
                console.error("Failed to init ticker:", err);
                setError("Auth missing");
            }
        };

        initTicker();

        return () => {
            if (tickerRef.current) {
                tickerRef.current.disconnect();
            }
        };
    }, []);

    const subscribe = (tokens) => {
        if (tickerRef.current && isConnected) {
            tickerRef.current.subscribe(tokens);
            tickerRef.current.setMode(tickerRef.current.modeFull, tokens);
        }
    };

    return (
        <TickerContext.Provider value={{ ticks, isConnected, error, subscribe }}>
            {children}
        </TickerContext.Provider>
    );
};

export const useTicker = () => useContext(TickerContext);
