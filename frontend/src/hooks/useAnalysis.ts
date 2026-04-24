import { useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type { AnalyzeResponse, PosResponse, CorpusStats, LookupResponse, CorpusSentencesResponse } from '../types/api';
import { useAnalysisStore } from '../stores/analysisStore';

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8001';

// Cached Tauri detection — set once per page lifetime
let _isTauri: boolean | null = null;
async function isTauri(): Promise<boolean> {
  if (_isTauri !== null) return _isTauri;
  try {
    // @ts-ignore
    _isTauri = typeof window !== 'undefined' && !!window.__TAURI__;
  } catch {
    _isTauri = false;
  }
  return _isTauri;
}

async function callTauri<T>(cmd: string, args: Record<string, unknown>): Promise<T> {
  return invoke<T>(cmd, args);
}

async function callHttp<T>(path: string, body?: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: body ? 'POST' : 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json() as Promise<T>;
}

export function useAnalysis() {
  const { setCurrent, setLoading, setError, addToHistory } = useAnalysisStore();

  // ── Fast POS tagging (legacy, used for quick preview) ─────────────────────
  const posTag = useCallback(async (text: string): Promise<PosResponse> => {
    try {
      if (await isTauri()) {
        return callTauri<PosResponse>('pos_tag', { text });
      } else {
        return callHttp<PosResponse>('/pos', { text });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      throw err;
    }
  }, [setError]);

  // ── Full analysis: lotsawa + TiBERT hybrid pipeline ─────────────────────────
  const analyze = useCallback(
    async (text: string, useLlm = true) => {
      setError(null);
      setLoading(true);
      setCurrent(null);
      try {
        const result: AnalyzeResponse = await isTauri()
          ? await callTauri<AnalyzeResponse>('analyze', { text, useLlm })
          : await callHttp<AnalyzeResponse>('/analyze', { text, useLlm });
        setCurrent(result);
        addToHistory(text, result);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [setCurrent, setLoading, setError, addToHistory]
  );

  const getCorpusStats = useCallback(async (): Promise<CorpusStats> => {
    try {
      return await isTauri()
        ? await callTauri<CorpusStats>('get_corpus_stats', {})
        : await callHttp<CorpusStats>('/corpus/stats');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      throw err;
    }
  }, [setError]);

  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      if (await isTauri()) {
        return callTauri<boolean>('check_health', {});
      } else {
        const res = await fetch(`${API}/health`);
        return res.ok;
      }
    } catch { return false; }
  }, []);

  const lookup = useCallback(
    async (word: string, dictName?: string, includeVerbs = true): Promise<LookupResponse> => {
      return await isTauri()
        ? callTauri<LookupResponse>('lookup', {
            word,
            dict_names: dictName ? [dictName] : null,
            include_verbs: includeVerbs,
          })
        : callHttp<LookupResponse>('/lookup', {
            word,
            dict_name: dictName ?? null,
            include_verbs: includeVerbs,
          });
    },
    []
  );

  const getCorpusSentences = useCallback(
    async (params: {
      collection?: string;
      page?: number;
      page_size?: number;
      search?: string;
    }): Promise<CorpusSentencesResponse> => {
      if (await isTauri()) {
        return callTauri<CorpusSentencesResponse>('get_corpus_sentences', params);
      }
      const qs = new URLSearchParams();
      if (params.collection) qs.set('collection', params.collection);
      if (params.page) qs.set('page', String(params.page));
      if (params.page_size) qs.set('page_size', String(params.page_size));
      if (params.search) qs.set('search', params.search);
      const qsStr = qs.toString() ? `?${qs.toString()}` : '';
      return callHttp<CorpusSentencesResponse>(`/corpus/sentences${qsStr}`);
    },
    []
  );

  return { analyze, posTag, getCorpusStats, checkHealth, lookup, getCorpusSentences };
}
