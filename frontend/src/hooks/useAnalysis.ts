import { useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type { AnalyzeResponse, PosResponse, CorpusStats, LookupResponse } from '../types/api';
import { useAnalysisStore } from '../stores/analysisStore';

const getApiBase = () => {
  const hostname = window.location.hostname;
  const port = 8000;
  return `http://${hostname}:${port}`;
};

async function isTauri(): Promise<boolean> {
  try {
    // @ts-ignore
    return typeof window !== 'undefined' && !!window.__TAURI__;
  } catch {
    return false;
  }
}

async function callTauri<T>(cmd: string, args: Record<string, unknown>): Promise<T> {
  return invoke<T>(cmd, args);
}

async function callHttp<T>(path: string, body?: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${getApiBase()}${path}`, {
    method: body ? 'POST' : 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json() as Promise<T>;
}

export function useAnalysis() {
  const { setCurrent, setLoading, setError, addToHistory } = useAnalysisStore();

  // ── Phase 1: POS tagging — fast ───────────────────────────────────────────
  const posTag = useCallback(async (text: string): Promise<PosResponse> => {
    try {
      const tauri = await isTauri();
      if (tauri) {
        return callTauri<PosResponse>('pos_tag', { text });
      } else {
        return callHttp<PosResponse>('/pos', { text });
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      throw err;
    }
  }, [setError]);

  // ── Phase 2: Full analysis — POS + LLM, async ─────────────────────────────
  const analyze = useCallback(
    async (text: string, useLlm = true) => {
      setError(null);

      // ① POS — fast, show immediately
      setLoading(true);
      let posResult: PosResponse;
      try {
        posResult = await posTag(text);
      } catch (err) {
        setLoading(false);
        throw err;
      }

      // Show POS results right away (no waiting for LLM)
      const immediateResult: AnalyzeResponse = {
        ...posResult,
        llm_explanation: undefined,
        structure: undefined,
      };
      setCurrent(immediateResult);

      // ② LLM — background, update when ready
      if (useLlm) {
        try {
          const tauri = await isTauri();
          const fullResult: AnalyzeResponse = tauri
            ? await callTauri<AnalyzeResponse>('analyze', { text, useLlm: true })
            : await callHttp<AnalyzeResponse>('/analyze', { text, useLlm: true });
          setCurrent(fullResult);
          addToHistory(text, fullResult);
        } catch {
          // LLM failed silently — POS is already displayed
        }
      } else {
        addToHistory(text, immediateResult);
      }
      setLoading(false);
    },
    [setCurrent, setLoading, setError, addToHistory, posTag]
  );

  const getCorpusStats = useCallback(async (): Promise<CorpusStats> => {
    try {
      const tauri = await isTauri();
      if (tauri) {
        return callTauri<CorpusStats>('get_corpus_stats', {});
      } else {
        return callHttp<CorpusStats>('/corpus/stats');
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      throw err;
    }
  }, [setError]);

  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const tauri = await isTauri();
      if (tauri) {
        return callTauri<boolean>('check_health', {});
      } else {
        const res = await fetch(`${getApiBase()}/health`);
        return res.ok;
      }
    } catch {
      return false;
    }
  }, []);

  const lookup = useCallback(
    async (word: string, dictName?: string, includeVerbs = true): Promise<LookupResponse> => {
      const tauri = await isTauri();
      if (tauri) {
        return callTauri<LookupResponse>('lookup', {
          word,
          dict_names: dictName ? [dictName] : null,
          include_verbs: includeVerbs,
        });
      } else {
        return callHttp<LookupResponse>('/lookup', {
          word,
          dict_name: dictName ?? null,
          include_verbs: includeVerbs,
        });
      }
    },
    []
  );

  return { analyze, posTag, getCorpusStats, checkHealth, lookup };
}
