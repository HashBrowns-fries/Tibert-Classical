import { useState } from 'react';
import { useLookup } from '../hooks/useLookup';
import type { LookupEntry } from '../types/api';

const DICT_COLORS: Record<string, { accent: string; bg: string }> = {
  RangjungYeshe:  { accent: '#c94a4a', bg: 'rgba(201,74,74,0.08)' },
  DagYig:         { accent: '#d97706', bg: 'rgba(217,119,6,0.08)' },
  Dungkar:        { accent: '#7c3aed', bg: 'rgba(124,58,237,0.08)' },
  MonlamTibetan: { accent: '#0d9488', bg: 'rgba(13,148,136,0.08)' },
  MonlamTibEng:  { accent: '#2563eb', bg: 'rgba(37,99,235,0.08)' },
  tibChinmo:      { accent: '#16a34a', bg: 'rgba(22,163,74,0.08)' },
  HanTb:          { accent: '#9333ea', bg: 'rgba(147,51,234,0.08)' },
  'dz-en':        { accent: '#0284c7', bg: 'rgba(2,132,199,0.08)' },
  dzongkha:       { accent: '#d97706', bg: 'rgba(217,119,6,0.08)' },
  default:        { accent: '#8b8070', bg: 'rgba(139,128,112,0.08)' },
};

function getDictColor(name: string) {
  return DICT_COLORS[name] ?? DICT_COLORS['default'];
}

export function LookupPage() {
  const { lookup, clear, loading, lookupResult, ragAnswer, ragChunks, retrieveTime, error } = useLookup();
  const [text, setText] = useState('');

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!text.trim()) return;
    lookup(text.trim());
  };

  return (
    <div className="bg-grad-hero" style={{ minHeight: 'calc(100vh - 60px)' }}>
      {/* Header */}
      <div style={{ padding: '2rem 2.5rem 1.25rem', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
        <h1 className="font-display animate-fade-up"
          style={{ fontSize: '2rem', fontWeight: 700, color: '#e8e0d0', marginBottom: '0.25rem' }}>
          藏文词典
          <span style={{ marginLeft: '0.75rem', fontSize: '1.25rem', opacity: 0.5, fontFamily: 'var(--font-tibetan)' }}>
            ཚིག་ཐ་མ་གཅད་གཅོད
          </span>
        </h1>
        <p style={{ fontSize: '0.8rem', color: '#8b8070' }}>
          gemma-2-mitra-it 分词 · SQLite 词典（541,618 条）· RAG 佛典释义
        </p>
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} style={{ padding: '1.25rem 2.5rem', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <input
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="输入藏文，例如：བོད་གི་ཡུལ་ལྷོ་ལ་སོང་"
            className="input-tibetan"
            style={{ flex: 1, fontSize: '1.1rem', padding: '0.75rem 1rem' }}
          />
          <button type="submit" disabled={loading || !text.trim()} className="btn-primary">
            {loading ? '⏳ 分词中…' : '🔍 查询'}
          </button>
          {lookupResult && (
            <button type="button" onClick={clear} className="btn-primary"
              style={{ background: 'rgba(255,255,255,0.06)', color: '#8b8070', boxShadow: 'none' }}>
              清空
            </button>
          )}
        </div>
      </form>

      {/* Error */}
      {error && (
        <div style={{ margin: '1rem 2.5rem', padding: '0.75rem 1rem', borderRadius: '10px',
          background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', color: '#fca5a5', fontSize: '0.85rem' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Syllable chips */}
      {lookupResult && lookupResult.syllables.length > 0 && (
        <div style={{ padding: '1rem 2.5rem', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.7rem', color: '#8b8070', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', flexShrink: 0 }}>
              音节
            </span>
            {lookupResult.syllables.map((s, i) => (
              <span
                key={i}
                className="tibetan"
                style={{
                  padding: '0.25rem 0.625rem',
                  borderRadius: '8px',
                  background: 'rgba(201,74,74,0.1)',
                  border: '1px solid rgba(201,74,74,0.25)',
                  color: '#e8d0c0',
                  fontSize: '1rem',
                  fontFamily: 'var(--font-tibetan)',
                }}
              >
                {s}
              </span>
            ))}
            <span style={{ fontSize: '0.7rem', color: '#4a4540', marginLeft: '0.5rem' }}>
              {lookupResult.total} 条词典释义
            </span>
          </div>
        </div>
      )}

      {/* Main content: dict + RAG */}
      {lookupResult && (
        <div style={{ display: 'grid', gridTemplateColumns: lookupResult.total > 0 ? '1fr 380px' : '1fr', height: 'calc(100vh - 280px)' }}>
          {/* Left: dictionary entries */}
          {lookupResult.total > 0 && (
            <DictEntries entries={lookupResult.entries} />
          )}

          {/* Right: RAG answer */}
          {(ragAnswer || ragChunks.length > 0) && (
            <div style={{ padding: '1.5rem', borderLeft: '1px solid rgba(255,255,255,0.04)', overflowY: 'auto' }}>
              {retrieveTime !== null && (
                <div style={{ fontSize: '0.7rem', color: '#8b8070', marginBottom: '0.75rem' }}>
                  ⏱ {retrieveTime.toFixed(1)}s · {ragChunks.length} 相关片段
                </div>
              )}
              {ragAnswer && (
                <div style={{ marginBottom: '1.25rem' }}>
                  <div style={{ fontSize: '0.7rem', color: '#8b8070', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.5rem' }}>
                    gemma RAG 释义
                  </div>
                  <div style={{
                    padding: '1rem 1.25rem', borderRadius: '12px',
                    background: 'rgba(201,74,74,0.06)',
                    border: '1px solid rgba(201,74,74,0.15)',
                    color: '#c8c0b0', fontSize: '0.875rem', lineHeight: 1.7,
                    fontFamily: 'var(--font-sans)', whiteSpace: 'pre-wrap',
                  }}>
                    {ragAnswer}
                  </div>
                </div>
              )}
              {ragChunks.length > 0 && (
                <div>
                  <div style={{ fontSize: '0.7rem', color: '#8b8070', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.5rem' }}>
                    相关佛典片段
                  </div>
                  {ragChunks.map((c, i) => (
                    <div key={i} style={{
                      padding: '0.75rem 1rem', borderRadius: '10px', marginBottom: '0.5rem',
                      background: 'rgba(255,255,255,0.03)',
                      border: '1px solid rgba(255,255,255,0.05)',
                      fontSize: '0.75rem',
                    }}>
                      <div style={{ color: '#8b8070', marginBottom: '0.25rem' }}>
                        [{i+1}] {c.source} · dist={c.distance.toFixed(2)}
                      </div>
                      <div style={{ color: '#c4b8a8', lineHeight: 1.6 }} className="tibetan">
                        {c.text.length > 200 ? c.text.slice(0, 200) + '…' : c.text}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!lookupResult && !loading && !error && (
        <div style={{ textAlign: 'center', padding: '4rem 2rem', color: '#4a4540' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.3 }}>
            <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
            </svg>
          </div>
          <p style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', color: '#8b8070' }}>
            输入藏文，查询词典释义
          </p>
          <p style={{ fontSize: '0.8rem', color: '#4a4540', marginTop: '0.5rem' }}>
            例如：<span className="tibetan" style={{ color: '#c94a4a' }}>བོད</span> · 7 部词典 · 541,618 条
          </p>
        </div>
      )}
    </div>
  );
}

function DictEntries({ entries }: { entries: LookupEntry[] }) {
  // Group entries by word
  const byWord: Record<string, typeof entries> = {};
  for (const e of entries) {
    const key = e.word ?? '(无音节)';
    if (!byWord[key]) byWord[key] = [];
    byWord[key].push(e);
  }
  const words = Object.keys(byWord);

  return (
    <div style={{ padding: '1.25rem 1.5rem', overflowY: 'auto' }}>
      {words.map(word => (
        <div key={word} style={{ marginBottom: '1.25rem' }}>
          {/* Syllable header */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            <span className="tibetan" style={{ fontSize: '1.5rem', fontWeight: 700, color: '#e8d0c0', fontFamily: 'var(--font-tibetan)' }}>
              {word}
            </span>
            <span style={{ fontSize: '0.65rem', color: '#4a4540', padding: '0.15rem 0.4rem', borderRadius: '4px', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)' }}>
              {byWord[word].length} 条释义
            </span>
          </div>

          {/* Entry cards */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', paddingLeft: '0.5rem' }}>
            {byWord[word].map((entry, i) => {
              const color = getDictColor(entry.dict_name);
              return (
                <div key={i} style={{
                  padding: '0.625rem 0.875rem', borderRadius: '8px',
                  background: color.bg,
                  borderLeft: `2px solid ${color.accent}`,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                    <span style={{ fontSize: '0.7rem', fontWeight: 600, color: color.accent, fontFamily: 'var(--font-sans)' }}>
                      {entry.dict_name}
                    </span>
                  </div>
                  <span style={{ fontSize: '0.8rem', color: '#b0a898', lineHeight: 1.6, fontFamily: 'var(--font-sans)' }}>
                    {entry.definition.replace(/<[^>]+>/g, '').slice(0, 300)}
                    {entry.definition.length > 300 ? '…' : ''}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
