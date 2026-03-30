import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { AnalyzeResponse } from '../types/api';

export type Theme = 'light' | 'dark' | 'system';

interface HistoryEntry {
  id: string;
  text: string;
  result: AnalyzeResponse;
  timestamp: number;
}

interface AnalysisStore {
  // 当前分析结果
  current: AnalyzeResponse | null;
  isLoading: boolean;
  error: string | null;

  // 分析历史
  history: HistoryEntry[];

  // 主题
  theme: Theme;

  // 操作
  setCurrent: (result: AnalyzeResponse | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  addToHistory: (text: string, result: AnalyzeResponse) => void;
  clearHistory: () => void;
  setTheme: (theme: Theme) => void;
}

export const useAnalysisStore = create<AnalysisStore>()(
  persist(
    (set) => ({
      current: null,
      isLoading: false,
      error: null,
      history: [],
      theme: 'system',

      setCurrent: (result) => set({ current: result, error: null }),
      setLoading: (isLoading) => set({ isLoading }),
      setError: (error) => set({ error }),

      addToHistory: (text, result) =>
        set((state) => ({
          history: [
            {
              id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
              text,
              result,
              timestamp: Date.now(),
            },
            ...state.history,
          ].slice(0, 50), // 最多保留 50 条
        })),

      clearHistory: () => set({ history: [] }),

      setTheme: (theme) => set({ theme }),
    }),
    {
      name: 'tibert-analysis-store',
      partialize: (state) => ({
        history: state.history.slice(0, 50),
        theme: state.theme,
      }),
    }
  )
);
