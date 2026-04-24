import { useCallback, useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { useAnalysis } from '../hooks/useAnalysis';
import { useAnalysisStore } from '../stores/analysisStore';
import { TokenDisplay } from '../components/TokenDisplay';
import { GrammarPanel } from '../components/GrammarPanel';
import type { AnalyzeResponse, LookupEntry } from '../types/api';

// Module-level dict color helper (used by DictPanel too)
const _DICT_COLORS: Record<string, string> = {
  RangjungYeshe: '#c084fc',
  MonlamTibEng: '#60a5fa',
  MonlamTibetan: '#60a5fa',
  DagYig: '#34d399',
  BodDag: '#fbbf24',
};
function _dictColor(dictName: string): string {
  for (const [k, v] of Object.entries(_DICT_COLORS)) {
    if (dictName.includes(k)) return v;
  }
  return '#d4a853';
}

export function AnalyzerPage() {
  const location = useLocation();
  const { current, isLoading, error, history } = useAnalysisStore();
  const { analyze } = useAnalysis();
  const [inputText, setInputText] = useState('');
  const [useLlm, setUseLlm] = useState(true);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [dictEntries, setDictEntries] = useState<Record<string, LookupEntry[]>>({});

  // Example sentences for first-time users
const EXAMPLE_SENTENCES = [
  { text: 'བདག་གི་ཕ་མ་ལ་སོང་', label: '属格 · 为格示例' },
  { text: 'བོད་གི་ཡུལ་ལྷོ་ལ་སོང་', label: '经典例句' },
];

// 从其他页面跳转时预填
  useEffect(() => {
    const state = location.state as { text?: string } | null;
    if (state?.text) {
      setInputText(state.text);
      analyze(state.text, true).catch(() => {});
      window.history.replaceState({}, '');
    }
  }, [location.state]); // eslint-disable-line

  const handleAnalyze = useCallback(
    async (text: string, useLlmVal = useLlm) => {
      if (!text.trim()) return;
      setDictEntries({}); // clear old dict entries
      await analyze(text.trim(), useLlmVal);
    },
    [analyze, useLlm]
  );

  // When a new result arrives, merge dict entries from tokens (backend dict lookup)
  // with any additional entries from frontend /lookup calls (supplementary cache)
  useEffect(() => {
    if (!current) return;
    const tokens = current.tokens.filter(t => t.token !== '་' && t.token !== '།' && t.token !== '༔');
    const newEntries: Record<string, LookupEntry[]> = {};
    for (const t of tokens) {
      if (!newEntries[t.token]) {
        // Prefer backend dict_entries, fallback to frontend state cache
        newEntries[t.token] = (t.dict_entries && t.dict_entries.length > 0)
          ? t.dict_entries
          : (dictEntries[t.token] ?? []);
      }
    }
    setDictEntries(prev => ({ ...prev, ...newEntries }));
    // Scroll to results on desktop (mobile users are already at top)
    if (window.innerWidth > 900) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [current]);

  return (
    <div
      className="bg-grad-hero"
      style={{ minHeight: 'calc(100vh - 60px)' }}
    >
      {/* Page header */}
      <div
        style={{
          padding: '2.5rem 3rem 1.5rem',
          borderBottom: '1px solid rgba(255,255,255,0.04)',
        }}
      >
        {/* Tibetan decorative header */}
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '1.5rem', marginBottom: '0.5rem' }}>
          <h1
            className="font-display animate-fade-up"
            style={{
              fontSize: '2.25rem',
              fontWeight: 700,
              color: '#e8e0d0',
              letterSpacing: '-0.01em',
              lineHeight: 1.1,
            }}
          >
            古典藏文分析器
          </h1>
          <div
            className="tibetan-xl animate-fade-up delay-100"
            style={{ color: 'rgba(201,74,74,0.6)', fontFamily: 'var(--font-tibetan)' }}
          >
            བོད་རྩལ་ཚིག་བགྱི་བརྟག་ཚིག
          </div>
        </div>
        <p
          className="animate-fade-up delay-200"
          style={{ fontSize: '0.85rem', color: '#8b8070', fontFamily: 'var(--font-sans)' }}
        >
          lotsawa + TiBERT 分词标注 · MiniMax 语法解释
        </p>
      </div>

{/* Main content: split panel */}
      <div
        className="analyzer-grid"
        style={{
          width: '100%',
          padding: '1.5rem 2rem',
        }}
      >
        {/* ── LEFT: Input ── */}
        <div
          className="animate-fade-up"
          style={{ minWidth: 0 }}
        >
          {/* Tibetan text input */}
          <div style={{ marginBottom: '1.25rem' }}>
            <label
              style={{
                display: 'block',
                fontSize: '0.75rem',
                fontWeight: 600,
                color: '#8b8070',
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                marginBottom: '0.5rem',
                fontFamily: 'var(--font-sans)',
              }}
            >
              藏文文本
            </label>
            <textarea
              className="input-tibetan"
              rows={7}
              placeholder="输入或粘贴古典藏文…"
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  handleAnalyze(inputText);
                }
              }}
            />
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginTop: '0.375rem',
                fontSize: '0.7rem',
                color: '#4a4540',
                fontFamily: 'var(--font-sans)',
              }}
            >
              <span>{inputText.length} 字符</span>
              <span>Ctrl+Enter 分析</span>
            </div>

            {/* Example sentences — only shown when input is empty */}
            {!inputText && (
              <div style={{ marginTop: '0.75rem', display: 'flex', flexWrap: 'wrap', gap: '0.375rem', alignItems: 'center' }}>
                <span style={{ fontSize: '0.65rem', color: '#4a4540', fontFamily: 'var(--font-sans)', marginRight: '0.25rem' }}>示例：</span>
                {EXAMPLE_SENTENCES.map((ex) => (
                  <button
                    key={ex.text}
                    onClick={() => {
                      setInputText(ex.text);
                      handleAnalyze(ex.text);
                    }}
                    style={{
                      background: 'rgba(201,74,74,0.08)',
                      border: '1px solid rgba(201,74,74,0.2)',
                      borderRadius: '99px',
                      padding: '0.2rem 0.625rem',
                      cursor: 'pointer',
                      color: 'rgba(201,74,74,0.7)',
                      fontSize: '0.68rem',
                      fontFamily: 'var(--font-sans)',
                      transition: 'all 0.15s ease',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                    }}
                    onMouseEnter={e => {
                      (e.currentTarget as HTMLElement).style.background = 'rgba(201,74,74,0.15)';
                      (e.currentTarget as HTMLElement).style.color = '#c94a4a';
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLElement).style.background = 'rgba(201,74,74,0.08)';
                      (e.currentTarget as HTMLElement).style.color = 'rgba(201,74,74,0.7)';
                    }}
                  >
                    <span className="tibetan" style={{ fontSize: '0.8rem' }}>{ex.text}</span>
                    <span style={{ fontSize: '0.6rem', opacity: 0.7 }}>({ex.label})</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {/* LLM toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', flexShrink: 0 }}>
              <div
                style={{
                  fontSize: '0.8rem',
                  color: '#8b8070',
                  fontFamily: 'var(--font-sans)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                }}
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                </svg>
                LLM 解释
              </div>
              <button
                role="switch"
                aria-checked={useLlm}
                onClick={() => setUseLlm(v => !v)}
                style={{
                  width: '36px',
                  height: '20px',
                  borderRadius: '99px',
                  background: useLlm ? '#c94a4a' : 'rgba(255,255,255,0.1)',
                  border: 'none',
                  cursor: 'pointer',
                  position: 'relative',
                  transition: 'background 0.2s ease',
                  flexShrink: 0,
                }}
              >
                <span
                  style={{
                    display: 'block',
                    width: '16px',
                    height: '16px',
                    borderRadius: '50%',
                    background: 'white',
                    position: 'absolute',
                    top: '2px',
                    left: useLlm ? '18px' : '2px',
                    transition: 'left 0.2s ease',
                    boxShadow: '0 1px 4px rgba(0,0,0,0.3)',
                  }}
                />
              </button>
            </label>

            {/* Analyze */}
            <button
              className="btn-primary"
              onClick={() => handleAnalyze(inputText)}
              disabled={isLoading || !inputText.trim()}
              style={{ flex: 1 }}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin-slow w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                  </svg>
                  分析中…
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
                  </svg>
                  分析
                </>
              )}
            </button>

            {/* Clear */}
            <button
              className="btn-ghost"
              onClick={() => setInputText('')}
              disabled={!inputText}
              style={{ padding: '0.625rem 0.875rem' }}
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Error */}
          {error && (
            <div
              className="animate-fade-up"
              style={{
                marginTop: '1rem',
                padding: '0.75rem 1rem',
                borderRadius: '0.75rem',
                background: 'rgba(220,38,38,0.1)',
                border: '1px solid rgba(220,38,38,0.2)',
                color: '#f87171',
                fontSize: '0.85rem',
                fontFamily: 'var(--font-sans)',
              }}
            >
              ⚠️ {error}
            </div>
          )}

          {/* History */}
          {history.length > 0 && (
            <div style={{ marginTop: '1.5rem' }}>
              <button
                className="btn-ghost"
                onClick={() => setHistoryOpen(v => !v)}
                style={{
                  width: '100%',
                  justifyContent: 'space-between',
                  fontSize: '0.8rem',
                  color: '#8b8070',
                }}
              >
                <span>历史记录 ({history.length})</span>
                <svg
                  className="w-4 h-4 transition-transform"
                  style={{ transform: historyOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}
                  viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {historyOpen && (
                <div
                  className="animate-fade-up"
                  style={{
                    marginTop: '0.5rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.375rem',
                    maxHeight: '200px',
                    overflowY: 'auto',
                  }}
                >
                  {history.slice(0, 10).map(entry => (
                    <button
                      key={entry.id}
                      onClick={() => {
                        setInputText(entry.text);
                        handleAnalyze(entry.text, !!entry.result.llm_explanation);
                      }}
                      className="btn-ghost"
                      style={{
                        justifyContent: 'flex-start',
                        textAlign: 'left',
                        fontSize: '0.8rem',
                        padding: '0.4rem 0.75rem',
                        color: '#8b8070',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      <span
                        style={{
                          fontSize: '0.65rem',
                          color: '#4a4540',
                          marginRight: '0.5rem',
                          flexShrink: 0,
                        }}
                      >
                        {new Date(entry.timestamp).toLocaleTimeString('zh-CN', {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </span>
                      <span className="tibetan-sm" style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {entry.text}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── RIGHT: Results ── */}
        <div
          className="animate-fade-up delay-200"
          style={{ minWidth: 0 }}
        >
          {current ? (
            <ResultView result={current} loading={isLoading} dictEntries={dictEntries} />
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                  <rect x="12" y="8" width="40" height="48" rx="4" stroke="currentColor" strokeWidth="1.5" />
                  <path d="M20 20h24M20 28h18M20 36h20M20 44h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                  <circle cx="50" cy="50" r="10" fill="var(--color-crimson)" opacity="0.15" stroke="var(--color-crimson)" strokeWidth="1.5" />
                  <path d="M47 50l6-6M53 50l-6-6" stroke="var(--color-crimson)" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </div>
              <p style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                等待分析
              </p>
              <p style={{ fontSize: '0.8rem', color: '#4a4540', fontFamily: 'var(--font-sans)' }}>
                左侧输入藏文后点击「分析」
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ResultView({ result, loading, dictEntries }: {
  result: AnalyzeResponse;
  loading: boolean;
  dictEntries: Record<string, LookupEntry[]>;
}) {
  const caseParticles = result.tokens.filter(t => t.is_case_particle);
  const stats = result.stats;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Stats row */}
      <div
        className="card"
        style={{ padding: '1rem 1.25rem' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
          {[
            { label: '名词', value: stats.nouns, color: '#a78bfa', bg: 'rgba(124,58,237,0.12)' },
            { label: '动词', value: stats.verbs, color: '#2dd4bf', bg: 'rgba(13,148,136,0.12)' },
            { label: '格助词', value: stats.case_particles, color: '#d97706', bg: 'rgba(217,119,6,0.12)' },
            { label: '音节', value: stats.syllable_count, color: '#e8e0d0', bg: 'rgba(255,255,255,0.06)' },
          ].map(({ label, value, color, bg }) => (
            <div
              key={label}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.4rem 0.875rem',
                borderRadius: '99px',
                background: bg,
                border: `1px solid ${color}22`,
              }}
              className="animate-count-up"
            >
              <span
                className="stat-number"
                style={{ fontSize: '1.25rem', color }}
              >
                {value}
              </span>
              <span style={{ fontSize: '0.7rem', color, fontFamily: 'var(--font-sans)', fontWeight: 500 }}>
                {label}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Original text */}
      <div
        className="card"
        style={{
          padding: '1rem 1.25rem',
          background: 'rgba(201,74,74,0.04)',
          borderColor: 'rgba(201,74,74,0.12)',
        }}
      >
        <div
          style={{
            fontSize: '0.7rem',
            fontWeight: 600,
            color: 'rgba(201,74,74,0.6)',
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            marginBottom: '0.375rem',
            fontFamily: 'var(--font-sans)',
          }}
        >
          原文
        </div>
        <div className="tibetan-lg" style={{ color: '#e8e0d0' }}>
          {result.original}
        </div>
      </div>

      {/* Token display */}
      <div className="panel-section">
        <div className="panel-header">
          <span
            style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              color: '#8b8070',
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              fontFamily: 'var(--font-sans)',
            }}
          >
            词性标注
          </span>
          <span style={{ fontSize: '0.7rem', color: '#4a4540', fontFamily: 'var(--font-sans)' }}>
            鼠标悬停查看详情
          </span>
        </div>
        <div className="panel-body" style={{ padding: '1.25rem' }}>
          <TokenDisplay tokens={result.tokens} dictEntries={dictEntries} />
        </div>
      </div>

      {/* 词典释义 */}
      <DictPanel tokens={result.tokens} dictEntries={dictEntries} />

      {/* Case particles */}
      {caseParticles.length > 0 ? (
        <div className="panel-section">
          <div className="panel-header">
            <span
              style={{
                fontSize: '0.75rem',
                fontWeight: 600,
                color: '#d97706',
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                fontFamily: 'var(--font-sans)',
              }}
            >
              ★ 格助词
            </span>
          </div>
          <div style={{ padding: '1rem 1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {caseParticles.map((t, i) => (
              <div
                key={i}
                className="animate-fade-up"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  padding: '0.625rem 0.875rem',
                  borderRadius: '0.625rem',
                  background: 'rgba(217,119,6,0.08)',
                  border: '1px solid rgba(217,119,6,0.15)',
                  animationDelay: `${i * 60}ms`,
                }}
              >
                <span
                  className="tibetan"
                  style={{
                    fontWeight: 700,
                    color: '#d97706',
                    fontSize: '1.1rem',
                    minWidth: '2.5rem',
                    textAlign: 'center',
                  }}
                >
                  {t.token}
                </span>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.7rem',
                    color: '#d97706',
                    background: 'rgba(217,119,6,0.12)',
                    padding: '0.15rem 0.5rem',
                    borderRadius: '99px',
                  }}
                >
                  {t.case_name}
                </span>
                <span
                  style={{
                    fontSize: '0.8rem',
                    color: '#8b8070',
                    fontFamily: 'var(--font-sans)',
                  }}
                >
                  {t.case_desc}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        stats.case_particles === 0 && stats.syllable_count > 0 && (
          <div
            className="panel-section"
            style={{ opacity: 0.5 }}
          >
            <div className="panel-header">
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#d97706', letterSpacing: '0.06em', textTransform: 'uppercase', fontFamily: 'var(--font-sans)' }}>
                ★ 格助词
              </span>
            </div>
            <div style={{ padding: '0.875rem 1.25rem', fontSize: '0.8rem', color: '#4a4540', fontFamily: 'var(--font-sans)' }}>
              本句未检测到格助词（属格 / 作格 / 为格 / 离格等）
            </div>
          </div>
        )
      )}

      {/* LLM explanation */}
      <GrammarPanel
        explanation={result.llm_explanation ?? undefined}
        structure={result.structure ?? undefined}
        original={result.original}
        loading={loading && !result.llm_explanation}
      />

      {/* Export */}
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        <button
          className="btn-ghost"
          style={{ fontSize: '0.8rem' }}
          onClick={() => {
            try { navigator.clipboard?.writeText(JSON.stringify(result, null, 2)); } catch(e) {}
          }}
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184" />
          </svg>
          复制 JSON
        </button>
        <button
          className="btn-ghost"
          style={{ fontSize: '0.8rem' }}
          onClick={() => {
            const text = result.tokens
              .map(t => `${t.token}\t${t.pos}\t${t.pos_zh}`)
              .join('\n');
            try { navigator.clipboard?.writeText(text); } catch(e) {}
          }}
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184" />
          </svg>
          复制标注
        </button>
      </div>
    </div>
  );
}

// ── DictPanel: inline dictionary lookup per token ──────────────────────────────

interface DictPanelProps {
  tokens: AnalyzeResponse['tokens'];
  dictEntries: Record<string, LookupEntry[]>;
}

function DictPanel({ tokens, dictEntries }: DictPanelProps) {
  const [open, setOpen] = useState(false);

  // Deduplicate tokens (keep first occurrence)
  const uniqueTokens = tokens.filter(
    (t, i, arr) => t.token !== '་' && t.token !== '།' && t.token !== '༔' && arr.findIndex(x => x.token === t.token) === i
  );

  const tokenKey = tokens.map(t => t.token).join('');
  useEffect(() => {
    setOpen(false);
  }, [tokenKey]);

  // Count how many tokens have dictionary entries (tracked via tokenKey dep)

  return (
    <div
      style={{
        borderRadius: '12px',
        border: '1px solid rgba(255,255,255,0.06)',
        overflow: 'hidden',
        background: 'rgba(255,255,255,0.025)',
      }}
    >
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          width: '100%',
          padding: '0.75rem 1rem',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          color: '#8b8070',
          fontFamily: 'var(--font-sans)',
          transition: 'color 0.2s',
        }}
        onMouseEnter={e => (e.currentTarget.style.color = '#e8e0d0')}
        onMouseLeave={e => (e.currentTarget.style.color = '#8b8070')}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <svg style={{ width: '14px', height: '14px', color: '#d4a853' }} viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
          </svg>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
            词典释义
          </span>
          <span style={{ fontSize: '0.65rem', color: 'rgba(139,128,112,0.5)' }}>
            {uniqueTokens.length} 词
          </span>
        </div>
        <svg
          style={{
            width: '14px', height: '14px',
            transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.25s ease',
          }}
          viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div
          style={{
            borderTop: '1px solid rgba(255,255,255,0.05)',
            padding: '0.75rem 1rem 1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.5rem',
            maxHeight: '480px',
            overflowY: 'auto',
          }}
        >
          {uniqueTokens.map((t, i) => {
            const entries = dictEntries[t.token] ?? [];
            const hasDef = entries.length > 0;

            return (
              <div
                key={t.token}
                className="animate-fade-up"
                style={{ animationDelay: `${i * 30}ms` }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                  <span className="tibetan" style={{ fontWeight: 700, fontSize: '1rem', color: t.is_case_particle ? '#d97706' : '#e8e0d0' }}>
                    {t.token}
                  </span>
                  <span style={{ fontSize: '0.65rem', color: 'rgba(139,128,112,0.5)', fontFamily: 'var(--font-mono)' }}>
                    {t.pos_zh}
                  </span>
                </div>

                {hasDef ? entries.slice(0, 3).map((e, j) => (
                  <div
                    key={j}
                    style={{
                      display: 'flex',
                      gap: '0.5rem',
                      padding: '0.2rem 0.5rem',
                      borderLeft: `2px solid ${_dictColor(e.dict_name)}44`,
                      marginBottom: '0.15rem',
                    }}
                  >
                    <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: _dictColor(e.dict_name), opacity: 0.8, flexShrink: 0, paddingTop: '0.05rem', minWidth: '5rem' }}>
                      {e.dict_name}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: 'rgba(232,224,208,0.75)', fontFamily: 'var(--font-sans)', lineHeight: 1.5 }}>
                      {e.definition.length > 120 ? e.definition.slice(0, 120) + '…' : e.definition}
                    </span>
                  </div>
                )) : (
                  <div style={{ fontSize: '0.72rem', color: '#4a4540', fontStyle: 'italic', paddingLeft: '0.5rem', fontFamily: 'var(--font-sans)' }}>
                    词典未收录
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

