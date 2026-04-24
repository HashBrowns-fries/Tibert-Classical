import { useState, useCallback } from 'react';
import type { GemmaLookupResponse, RagResponse, RagChunk } from '../types/api';

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8001';

async function callApi<T>(path: string, body?: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: body ? 'POST' : 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export function useLookup() {
  const [loading, setLoading] = useState(false);
  const [lookupResult, setLookupResult] = useState<GemmaLookupResponse | null>(null);
  const [ragAnswer, setRagAnswer] = useState<string | null>(null);
  const [ragChunks, setRagChunks] = useState<RagChunk[]>([]);
  const [retrieveTime, setRetrieveTime] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const lookup = useCallback(async (text: string, dictNames?: string[]) => {
    setLoading(true);
    setError(null);
    setLookupResult(null);
    setRagAnswer(null);
    setRagChunks([]);
    setRetrieveTime(null);

    try {
      // ① gemma 分词 + 词典查询
      const res = await callApi<GemmaLookupResponse>('/gemma/lookup', {
        text,
        dict_names: dictNames ?? null,
      });
      setLookupResult(res);

      // ② RAG 问答（后台）
      try {
        const ragRes = await callApi<RagResponse>('/rag', {
          question: text,
          language: '藏文',
          top_k: 5,
        });
        setRagAnswer(ragRes.answer);
        setRagChunks(ragRes.retrieved_chunks);
        setRetrieveTime(ragRes.retrieve_time_s);
      } catch (ragErr) {
        // RAG 失败不影响词典展示，仅记录到控制台
        console.warn('[useLookup] RAG unavailable:', ragErr);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const clear = useCallback(() => {
    setLookupResult(null);
    setRagAnswer(null);
    setRagChunks([]);
    setRetrieveTime(null);
    setError(null);
  }, []);

  return { lookup, clear, loading, lookupResult, ragAnswer, ragChunks, retrieveTime, error };
}
