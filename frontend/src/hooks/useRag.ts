import { useState, useCallback, useEffect } from 'react';
import type { RagResponse, RagStats } from '../types/api';

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8001';
const HISTORY_KEY = 'tibert_rag_history';

function loadHistory(): Array<{ q: string; a: string }> {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveHistory(h: Array<{ q: string; a: string }>) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(h)); } catch {}
}

async function callRag<T>(path: string, body?: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: body ? 'POST' : 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export function useRag() {
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [chunks, setChunks] = useState<RagResponse['retrieved_chunks']>([]);
  const [retrieveTime, setRetrieveTime] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<Array<{ q: string; a: string }>>([]);

  // Load persisted history on mount
  useEffect(() => { setHistory(loadHistory()); }, []);

  const ask = useCallback(async (question: string, language = '藏文', topK = 5) => {
    setLoading(true);
    setError(null);
    setAnswer(null);
    setChunks([]);
    setRetrieveTime(null);
    try {
      const res = await callRag<RagResponse>('/rag', { question, language, top_k: topK });
      setAnswer(res.answer);
      setChunks(res.retrieved_chunks);
      setRetrieveTime(res.retrieve_time_s);
      setHistory(prev => {
        const next = [{ q: question, a: res.answer }, ...prev].slice(0, 50);
        saveHistory(next);
        return next;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const getStats = useCallback(async (): Promise<RagStats> => {
    return callRag<RagStats>('/rag/stats');
  }, []);

  return { ask, getStats, loading, answer, chunks, retrieveTime, error, history };
}
