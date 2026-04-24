import { useState, useCallback, useEffect } from 'react';
import { useRag } from '../hooks/useRag';
import type { RagStats } from '../types/api';

const LANGUAGES = [
  { value: '藏文', label: '藏文' },
  { value: '中文', label: '中文' },
  { value: '英文', label: '英文' },
];

export function RagPage() {
  const { ask, getStats, loading, answer, chunks, retrieveTime, error, history } = useRag();
  const [question, setQuestion] = useState('');
  const [language, setLanguage] = useState('藏文');
  const [topK, setTopK] = useState(5);
  const [stats, setStats] = useState<RagStats | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  useEffect(() => {
    getStats().then(setStats).catch(() => {});
  }, [getStats]);

  const handleAsk = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault();
      if (!question.trim()) return;
      await ask(question.trim(), language, topK);
    },
    [question, language, topK, ask]
  );

  return (
    <div className="bg-grad-hero" style={{ minHeight: 'calc(100vh - 60px)' }}>
      {/* Header */}
      <div style={{ padding: '2.5rem 3rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
        <h1
          className="font-display animate-fade-up"
          style={{ fontSize: '2.25rem', fontWeight: 700, color: '#e8e0d0', letterSpacing: '-0.01em', marginBottom: '0.5rem' }}
        >
          RAG 佛典问答
        </h1>
        <p style={{ color: '#8b8070', fontSize: '0.9rem' }}>
          基于藏文佛典语料库，RAG 检索 + MiniMax 生成
        </p>
      </div>

      <div className="rag-grid" style={{ height: 'calc(100vh - 180px)' }}>
        {/* Left: Input + Answer */}
        <div style={{ padding: '2rem 2.5rem', overflowY: 'auto', borderRight: '1px solid rgba(255,255,255,0.04)' }}>
          {/* Stats bar */}
          {stats && (
            <div
              style={{
                display: 'flex', gap: '1.5rem', padding: '0.75rem 1rem',
                borderRadius: '10px', background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)', marginBottom: '1.5rem',
                fontSize: '0.75rem', color: '#8b8070', fontFamily: 'var(--font-sans)',
              }}
            >
              <span>📚 {stats.total_chunks.toLocaleString()} chunks</span>
              <span>🔍 {stats.embedding_model?.split('/').pop() ?? stats.embedding_model}</span>
              <span>🧠 {stats.llm_model?.split('/').pop() ?? stats.llm_model}</span>
            </div>
          )}

          {/* Input form */}
          <form onSubmit={handleAsk} style={{ marginBottom: '1.5rem' }}>
            <textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder="输入问题，例如：བོད་གི་ཡུལ་ལྷོ་ལ་སོང་ 是什么意思？"
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (!loading && question.trim()) handleAsk();
                }
              }}
              style={{
                width: '100%', minHeight: '90px', padding: '0.875rem 1rem',
                borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)',
                background: 'rgba(255,255,255,0.04)', color: '#e8e0d0',
                fontSize: '1rem', fontFamily: 'var(--font-sans)', resize: 'vertical',
                outline: 'none', marginBottom: '0.75rem',
                boxSizing: 'border-box',
              }}
              onFocus={e => (e.target.style.borderColor = 'rgba(201,74,74,0.4)')}
              onBlur={e => (e.target.style.borderColor = 'rgba(255,255,255,0.08)')}
            />

            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
              <select
                value={language}
                onChange={e => setLanguage(e.target.value)}
                style={{
                  padding: '0.5rem 0.75rem', borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,0.08)',
                  background: 'rgba(255,255,255,0.05)', color: '#e8e0d0',
                  fontSize: '0.85rem', cursor: 'pointer',
                }}
              >
                {LANGUAGES.map(l => (
                  <option key={l.value} value={l.value}>{l.label}</option>
                ))}
              </select>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#8b8070', fontSize: '0.8rem' }}>
                <span>top_k:</span>
                <input
                  type="range" min={1} max={20} value={topK}
                  onChange={e => setTopK(Number(e.target.value))}
                  style={{ accentColor: '#c94a4a' }}
                />
                <span style={{ minWidth: '1.5rem' }}>{topK}</span>
              </div>

              <button
                type="submit"
                disabled={loading || !question.trim()}
                style={{
                  marginLeft: 'auto', padding: '0.6rem 1.5rem',
                  borderRadius: '10px', border: 'none', cursor: loading ? 'not-allowed' : 'pointer',
                  background: loading
                    ? 'rgba(201,74,74,0.4)'
                    : 'linear-gradient(135deg, #c94a4a 0%, #7a1e1e 100%)',
                  color: 'white', fontWeight: 600, fontSize: '0.9rem',
                  boxShadow: loading ? 'none' : '0 4px 14px rgba(201,74,74,0.35)',
                  transition: 'all 0.2s',
                  opacity: loading ? 0.7 : 1,
                }}
              >
                {loading ? '⏳ 生成中...' : '🔍 提问'}
              </button>
            </div>
          </form>

          {/* Example questions — shown when no question entered yet */}
          {!question && (
            <div style={{ marginBottom: '1.5rem' }}>
              <div style={{ fontSize: '0.65rem', color: '#4a4540', fontFamily: 'var(--font-sans)', marginBottom: '0.5rem', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                示例问题
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.375rem' }}>
                {[
                  { q: 'བོད་གི་ཡུལ་ལྷོ་ལ་སོང་这句的语法结构是什么？', lang: '藏文' },
                  { q: '什么是属格助词？', lang: '中文' },
                  { q: 'Explain the case particles in Classical Tibetan.', lang: '英文' },
                ].map((ex) => (
                  <button
                    key={ex.q}
                    onClick={() => { setQuestion(ex.q); setLanguage(ex.lang); }}
                    style={{
                      background: 'rgba(201,74,74,0.06)',
                      border: '1px solid rgba(201,74,74,0.15)',
                      borderRadius: '99px',
                      padding: '0.25rem 0.75rem',
                      cursor: 'pointer',
                      color: 'rgba(201,74,74,0.65)',
                      fontSize: '0.72rem',
                      fontFamily: 'var(--font-sans)',
                      transition: 'all 0.15s',
                    }}
                    onMouseEnter={e => {
                      (e.currentTarget as HTMLElement).style.background = 'rgba(201,74,74,0.12)';
                      (e.currentTarget as HTMLElement).style.color = '#c94a4a';
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLElement).style.background = 'rgba(201,74,74,0.06)';
                      (e.currentTarget as HTMLElement).style.color = 'rgba(201,74,74,0.65)';
                    }}
                  >
                    {ex.q.length > 30 ? ex.q.slice(0, 30) + '…' : ex.q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div style={{
              padding: '0.875rem 1rem', borderRadius: '10px', marginBottom: '1rem',
              background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)',
              color: '#fca5a5', fontSize: '0.85rem', fontFamily: 'var(--font-sans)',
            }}>
              ⚠️ {error}
            </div>
          )}

          {/* Answer */}
          {answer && (
            <div>
              {retrieveTime !== null && (
                <div style={{ fontSize: '0.75rem', color: '#8b8070', marginBottom: '0.75rem' }}>
                  ⏱ 检索耗时 {retrieveTime.toFixed(2)}s · {chunks.length} 个相关片段
                </div>
              )}
              <div
                style={{
                  padding: '1.25rem 1.5rem', borderRadius: '14px',
                  background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.08)',
                  color: '#e8e0d0', fontSize: '0.95rem', lineHeight: 1.7,
                  fontFamily: 'var(--font-sans)', whiteSpace: 'pre-wrap',
                }}
              >
                {answer}
              </div>
            </div>
          )}

          {/* Retrieved chunks */}
          {chunks.length > 0 && (
            <div style={{ marginTop: '1.5rem' }}>
              <div style={{ fontSize: '0.8rem', color: '#8b8070', marginBottom: '0.75rem', fontWeight: 600 }}>
                📄 检索到的相关片段（{chunks.length}）
              </div>
              {chunks.map((c, i) => (
                <div
                  key={i}
                  style={{
                    padding: '0.875rem 1rem', borderRadius: '10px', marginBottom: '0.5rem',
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid rgba(255,255,255,0.05)',
                    fontSize: '0.8rem', fontFamily: 'var(--font-sans)',
                  }}
                >
                  <div style={{ color: '#8b8070', marginBottom: '0.25rem', fontSize: '0.7rem' }}>
                    [{i + 1}] {c.source} · distance={c.distance.toFixed(3)}
                  </div>
                  <div style={{ color: '#c4b8a8', lineHeight: 1.6 }} className="tibetan">
                    {c.text.length > 300 ? c.text.slice(0, 300) + '…' : c.text}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: History */}
        <div style={{ padding: '1.5rem', overflowY: 'auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <span style={{ fontSize: '0.8rem', color: '#8b8070', fontWeight: 600 }}>📝 历史记录</span>
            {history.length > 0 && (
              <button
                onClick={() => setShowHistory(!showHistory)}
                style={{ fontSize: '0.7rem', color: '#c94a4a', background: 'none', border: 'none', cursor: 'pointer' }}
              >
                {showHistory ? '收起' : '展开'}
              </button>
            )}
          </div>
          {history.length === 0 ? (
            <div style={{ color: '#8b8070', fontSize: '0.8rem', textAlign: 'center', padding: '2rem 0' }}>
              暂无历史记录
            </div>
          ) : showHistory ? (
            history.slice().reverse().map((h, i) => (
              <div
                key={i}
                style={{
                  padding: '0.875rem', borderRadius: '10px', marginBottom: '0.5rem',
                  background: 'rgba(255,255,255,0.03)',
                  border: '1px solid rgba(255,255,255,0.05)',
                  cursor: 'pointer',
                }}
                onClick={() => setQuestion(h.q)}
                title="点击填入问题"
              >
                <div style={{ color: '#e8e0d0', fontSize: '0.8rem', marginBottom: '0.25rem', fontWeight: 500 }}>
                  Q: {h.q.slice(0, 60)}{h.q.length > 60 ? '…' : ''}
                </div>
                <div style={{ color: '#8b8070', fontSize: '0.75rem' }}>
                  A: {h.a.slice(0, 80)}{h.a.length > 80 ? '…' : ''}
                </div>
              </div>
            ))
          ) : (
            <div style={{ color: '#8b8070', fontSize: '0.8rem' }}>
              {history.length} 条记录
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
