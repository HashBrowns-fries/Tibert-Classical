import { useState, useCallback } from 'react';
import { useAnalysis } from '../hooks/useAnalysis';
import type { LookupResponse, LookupEntry, VerbEntry } from '../types/api';

const AVAILABLE_DICTS = [
  { key: '', label: '全部词典' },
  { key: 'RangjungYeshe', label: 'Rangjung Yeshe' },
  { key: 'DagYig', label: 'Dag Yig' },
  { key: 'Dungkar', label: 'Dungkar' },
  { key: 'MonlamTibetan', label: 'Monlam 藏藏' },
  { key: 'MonlamTibEng', label: 'Monlam 藏英' },
  { key: 'tibChinmo', label: '藏汉摩庄' },
  { key: 'HanTb', label: '汉藏' },
  { key: 'dz-en', label: '宗卡英' },
  { key: 'dzongkha', label: '宗卡语' },
];

const DICT_COLORS: Record<string, { accent: string; bg: string }> = {
  RangjungYeshe: { accent: '#c94a4a', bg: 'rgba(201,74,74,0.08)' },
  DagYig:        { accent: '#d97706', bg: 'rgba(217,119,6,0.08)' },
  Dungkar:       { accent: '#7c3aed', bg: 'rgba(124,58,237,0.08)' },
  MonlamTibetan:{ accent: '#0d9488', bg: 'rgba(13,148,136,0.08)' },
  MonlamTibEng: { accent: '#2563eb', bg: 'rgba(37,99,235,0.08)' },
  tibChinmo:     { accent: '#16a34a', bg: 'rgba(22,163,74,0.08)' },
  HanTb:         { accent: '#9333ea', bg: 'rgba(147,51,234,0.08)' },
  'dz-en':       { accent: '#0284c7', bg: 'rgba(2,132,199,0.08)' },
  dzongkha:      { accent: '#d97706', bg: 'rgba(217,119,6,0.08)' },
  default:       { accent: '#8b8070', bg: 'rgba(139,128,112,0.08)' },
};

function getDictColor(dictName: string) {
  return DICT_COLORS[dictName] ?? DICT_COLORS['default'];
}

export function LookupPage() {
  const { lookup } = useAnalysis();
  const [query, setQuery] = useState('');
  const [selectedDict, setSelectedDict] = useState('');
  const [includeVerbs, setIncludeVerbs] = useState(true);
  const [result, setResult] = useState<LookupResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault();
      if (!query.trim()) return;
      setLoading(true);
      setError(null);
      setResult(null);
      try {
        const res = await lookup(
          query.trim(),
          selectedDict || undefined,
          includeVerbs
        );
        setResult(res);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    },
    [query, selectedDict, includeVerbs, lookup]
  );

  return (
    <div className="bg-grad-hero" style={{ minHeight: 'calc(100vh - 60px)' }}>
      {/* Header */}
      <div
        style={{
          padding: '2.5rem 3rem 1.5rem',
          borderBottom: '1px solid rgba(255,255,255,0.04)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '1.5rem', marginBottom: '0.5rem' }}>
          <h1
            className="font-display animate-fade-up"
            style={{ fontSize: '2.25rem', fontWeight: 700, color: '#e8e0d0', letterSpacing: '-0.01em', lineHeight: 1.1 }}
          >
            藏文词典
          </h1>
          <div
            className="tibetan-xl animate-fade-up delay-100"
            style={{ color: 'rgba(212,168,83,0.6)', fontFamily: 'var(--font-tibetan)' }}
          >
            ཚིག་ཐ་མ་གཅད་གཅོད
          </div>
        </div>
        <p
          className="animate-fade-up delay-200"
          style={{ fontSize: '0.85rem', color: '#8b8070', fontFamily: 'var(--font-sans)' }}
        >
          12 部藏文词典 · 2,489 动词词干形态 · StarDict 即时查询
        </p>
      </div>

      {/* Search */}
      <form
        onSubmit={handleSearch}
        className="animate-fade-up delay-200"
        style={{ padding: '1.5rem 2rem 0', maxWidth: '1400px', margin: '0 auto' }}
      >
        {/* Query input */}
        <div style={{ position: 'relative', marginBottom: '1rem' }}>
          <div
            style={{
              position: 'absolute',
              left: '1rem',
              top: '50%',
              transform: 'translateY(-50%)',
              color: '#4a4540',
              pointerEvents: 'none',
            }}
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
            </svg>
          </div>
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="输入藏文词语或动词词干（Wylie 罗马字，如 kum, ker）"
            className="input-tibetan"
            style={{
              paddingLeft: '2.75rem',
              paddingRight: '9rem',
              fontSize: '1.2rem',
              paddingTop: '0.875rem',
              paddingBottom: '0.875rem',
            }}
          />
          <button
            type="submit"
            className="btn-primary"
            disabled={loading || !query.trim()}
            style={{
              position: 'absolute',
              right: '0.5rem',
              top: '50%',
              transform: 'translateY(-50%)',
            }}
          >
            {loading ? (
              <>
                <svg className="animate-spin-slow w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" strokeLinecap="round" />
                </svg>
                查询中…
              </>
            ) : (
              <>
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
                </svg>
                查询
              </>
            )}
          </button>
        </div>

        {/* Dict selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
          <span
            style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              color: '#8b8070',
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              fontFamily: 'var(--font-sans)',
              flexShrink: 0,
            }}
          >
            词典
          </span>
          {AVAILABLE_DICTS.map(({ key, label }) => (
            <button
              key={key}
              type="button"
              onClick={() => setSelectedDict(key)}
              className={`btn-chip ${selectedDict === key ? 'active' : ''}`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Verb lexicon toggle */}
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
          <div
            style={{
              width: '36px',
              height: '20px',
              borderRadius: '99px',
              background: includeVerbs ? '#c94a4a' : 'rgba(255,255,255,0.1)',
              border: 'none',
              cursor: 'pointer',
              position: 'relative',
              transition: 'background 0.2s ease',
            }}
            onClick={() => setIncludeVerbs(v => !v)}
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
                left: includeVerbs ? '18px' : '2px',
                transition: 'left 0.2s ease',
                boxShadow: '0 1px 4px rgba(0,0,0,0.3)',
              }}
            />
          </div>
          <span style={{ fontSize: '0.8rem', color: '#8b8070', fontFamily: 'var(--font-sans)' }}>
            同时查询动词词干词典（Wylie 罗马字）
          </span>
        </label>
      </form>

      {/* Error */}
      {error && (
        <div
          style={{
            margin: '1rem 2rem 0',
            padding: '0.75rem 1rem',
            borderRadius: '0.75rem',
            background: 'rgba(220,38,38,0.1)',
            border: '1px solid rgba(220,38,38,0.2)',
            color: '#f87171',
            fontSize: '0.85rem',
            fontFamily: 'var(--font-sans)',
            maxWidth: '1400px',
            marginLeft: 'auto',
            marginRight: 'auto',
          }}
        >
          ⚠️ {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <LookupResultView result={result} />
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <div className="empty-state" style={{ maxWidth: '1400px', margin: '2rem auto', padding: '3rem 2rem' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.25 }}>
            <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
              <rect x="10" y="6" width="44" height="52" rx="4" stroke="currentColor" strokeWidth="1.5" />
              <path d="M10 14h44" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M20 28h8M20 36h16M20 44h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity="0.5" />
              <circle cx="50" cy="52" r="10" fill="var(--color-gold)" opacity="0.1" stroke="var(--color-gold)" strokeWidth="1.5" />
              <path d="M47 52h6M50 49v6" stroke="var(--color-gold)" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </div>
          <p style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', fontWeight: 600, color: '#8b8070', marginBottom: '0.25rem' }}>
            输入词语开始查询
          </p>
          <p style={{ fontSize: '0.8rem', color: '#4a4540', fontFamily: 'var(--font-sans)' }}>
            例如：<span className="tibetan" style={{ color: '#d97706' }}>བོད</span>、 kum、 ker、 dkar
          </p>
        </div>
      )}
    </div>
  );
}

function LookupResultView({ result }: { result: LookupResponse }) {
  const { entries, verb_entries } = result;

  if (entries.length === 0 && (!verb_entries || verb_entries.length === 0)) {
    return (
      <div
        className="animate-fade-up"
        style={{
          maxWidth: '1400px',
          margin: '2rem auto',
          padding: '0 2rem',
        }}
      >
        <div
          style={{
            padding: '2rem',
            borderRadius: '1rem',
            background: 'rgba(139,128,112,0.06)',
            border: '1px solid rgba(139,128,112,0.1)',
            textAlign: 'center',
            color: '#8b8070',
            fontFamily: 'var(--font-sans)',
          }}
        >
          未找到「{result.word}」的释义
        </div>
      </div>
    );
  }

  return (
    <div
      className="animate-fade-up"
      style={{ maxWidth: '1400px', margin: '1.5rem auto', padding: '0 2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}
    >
      {/* Dictionary entries */}
      {entries.length > 0 && (
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              marginBottom: '1rem',
            }}
          >
            <svg style={{ width: '14px', height: '14px', color: '#c94a4a' }} viewBox="0 0 24 24" fill="currentColor">
              <path d="M4 4h16v16H4V4z" opacity="0.3" />
              <path d="M2 2h20v20H2V2zm2 2v16h16V4H4z" />
            </svg>
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
              词典释义 ({entries.length})
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))', gap: '0.75rem' }}>
            {entries.map((entry: LookupEntry) => {
              const color = getDictColor(entry.dict_name);
              return (
                <div
                  key={entry.dict_name}
                  className="card animate-fade-up"
                  style={{
                    overflow: 'hidden',
                    transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
                  }}
                >
                  {/* Header bar */}
                  <div
                    style={{
                      padding: '0.625rem 1rem',
                      borderBottom: `1px solid ${color.accent}20`,
                      background: color.bg,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                    }}
                  >
                    <div
                      style={{
                        width: '6px',
                        height: '6px',
                        borderRadius: '50%',
                        background: color.accent,
                        flexShrink: 0,
                      }}
                    />
                    <span
                      style={{
                        fontSize: '0.8rem',
                        fontWeight: 600,
                        color: color.accent,
                        fontFamily: 'var(--font-sans)',
                      }}
                    >
                      {entry.dict_name}
                    </span>
                  </div>
                  {/* Definition */}
                  <div style={{ padding: '0.875rem 1rem' }}>
                    <DefinitionText text={entry.definition} accent={color.accent} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Verb stem table */}
      {verb_entries && verb_entries.length > 0 && (
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              marginBottom: '1rem',
            }}
          >
            <svg style={{ width: '14px', height: '14px', color: '#d97706' }} viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
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
              动词词干形态 ({verb_entries.length})
            </span>
          </div>

          <div className="card" style={{ overflow: 'hidden' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr
                  style={{
                    background: 'rgba(217,119,6,0.06)',
                    borderBottom: '1px solid rgba(217,119,6,0.15)',
                  }}
                >
                  {['词干', '现在时', '过去时', '将来时', '命令式', '释义'].map(h => (
                    <th
                      key={h}
                      style={{
                        padding: '0.625rem 1rem',
                        fontSize: '0.7rem',
                        fontWeight: 600,
                        color: '#d97706',
                        letterSpacing: '0.04em',
                        textTransform: 'uppercase',
                        textAlign: 'left',
                        fontFamily: 'var(--font-sans)',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {verb_entries.map((ve: VerbEntry, i: number) => (
                  <tr
                    key={i}
                    style={{
                      borderTop: i > 0 ? '1px solid rgba(255,255,255,0.04)' : undefined,
                    }}
                  >
                    {[
                      ve.headword,
                      stripLabel(ve.present),
                      stripLabel(ve.past),
                      stripLabel(ve.future),
                      stripLabel(ve.imperative),
                      stripLabel(ve.meaning),
                    ].map((cell, j) => (
                      <td
                        key={j}
                        style={{
                          padding: '0.625rem 1rem',
                          fontSize: j === 0 ? '0.95rem' : '0.8rem',
                          color: j === 0 ? '#d97706' : '#8b8070',
                          fontFamily: j === 0 ? 'var(--font-tibetan)' : 'var(--font-sans)',
                          fontWeight: j === 0 ? 700 : 400,
                          maxWidth: j === 0 ? '80px' : '200px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {cell || '—'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function stripLabel(s?: string) {
  if (!s) return undefined;
  return s.replace(/^(Present|Past|Future|Imperative|Meaning):\s*/i, '').trim();
}

function DefinitionText({ text, accent }: { text: string; accent: string }) {
  // Strip HTML tags, limit length
  const stripped = text
    .replace(/<[^>]+>/g, '')
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&')
    .trim();
  const limited = stripped.length > 400 ? stripped.slice(0, 400) + '…' : stripped;

  return (
    <span
      style={{
        fontSize: '0.875rem',
        color: '#c8c0b0',
        lineHeight: '1.7',
        fontFamily: 'var(--font-sans)',
      }}
    >
      {limited}
    </span>
  );
}
