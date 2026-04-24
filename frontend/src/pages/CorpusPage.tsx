import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysis } from '../hooks/useAnalysis';
import type { CorpusSentencesResponse, CorpusSentence } from '../types/api';

type Tab = 'stats' | 'reader';

const COLLECTION_COLORS: Record<string, string> = {
  DharmaDownload:    '#60a5fa',
  DrikungChetsang:   '#4ade80',
  GuruLamaWorks:     '#c94a4a',
  KarmaDelek:        '#d4a853',
  PalriParkhang:     '#a78bfa',
  Shechen:           '#2dd4bf',
  eKangyur:          '#f87171',
  DrikungKagyu:      '#fbbf24',
  RiBoGdemand:       '#818cf8',
  Dzongkhag:         '#34d399',
  Unknown:           '#8b8070',
};

function collectionColor(name: string): string {
  return COLLECTION_COLORS[name] ?? COLLECTION_COLORS.Unknown;
}

// ── Sentence card ──────────────────────────────────────────────────────────────
function SentenceCard({
  sentence,
  onAnalyze,
}: {
  sentence: CorpusSentence;
  onAnalyze: (text: string) => void;
}) {
  return (
    <div
      className="animate-fade-up"
      style={{
        borderRadius: '12px',
        padding: '0.875rem 1rem',
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.06)',
        transition: 'border-color 0.2s',
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(201,74,74,0.2)';
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(255,255,255,0.06)';
      }}
    >
      {/* Header: collection badge + analyze button */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
        <span
          style={{
            fontSize: '0.65rem',
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            color: collectionColor(sentence.collection),
            background: `${collectionColor(sentence.collection)}18`,
            padding: '0.15rem 0.5rem',
            borderRadius: '99px',
            letterSpacing: '0.03em',
          }}
        >
          {sentence.collection}
        </span>
        <button
          onClick={() => onAnalyze(sentence.text)}
          style={{
            fontSize: '0.7rem',
            fontFamily: 'var(--font-sans)',
            color: '#c94a4a',
            background: 'none',
            border: '1px solid rgba(201,74,74,0.2)',
            borderRadius: '99px',
            padding: '0.15rem 0.625rem',
            cursor: 'pointer',
            transition: 'all 0.15s',
            letterSpacing: '0.03em',
          }}
          onMouseEnter={e => {
            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(201,74,74,0.08)';
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLButtonElement).style.background = 'none';
          }}
        >
          分析 →
        </button>
      </div>

      {/* Tibetan text */}
      <div
        className="tibetan"
        style={{
          fontSize: '0.95rem',
          lineHeight: 1.9,
          color: '#e8e0d0',
          marginBottom: '0.375rem',
          overflow: 'hidden',
          display: '-webkit-box',
          WebkitLineClamp: 4,
          WebkitBoxOrient: 'vertical',
          wordBreak: 'break-word' as const,
        }}
      >
        {sentence.text}
      </div>

      {/* Syllable count */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ fontSize: '0.65rem', color: 'rgba(139,128,112,0.4)', fontFamily: 'var(--font-sans)' }}>
          {sentence.syllables.length} 音节
        </span>
        <span style={{ fontSize: '0.6rem', color: 'rgba(139,128,112,0.25)', fontFamily: 'var(--font-mono)' }}>
          {sentence.id}
        </span>
      </div>
    </div>
  );
}

// ── Corpus Reader tab ──────────────────────────────────────────────────────────
function CorpusReader({ onAnalyze }: { onAnalyze: (text: string) => void }) {
  const { getCorpusSentences } = useAnalysis();

  const [data, setData] = useState<CorpusSentencesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [selectedCollection, setSelectedCollection] = useState<string>('');
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 20;

  const load = useCallback(
    async (coll: string, q: string, p: number) => {
      setLoading(true);
      setError(null);
      try {
        const result = await getCorpusSentences({
          collection: coll || undefined,
          search: q || undefined,
          page: p,
          page_size: PAGE_SIZE,
        });
        setData(result);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [getCorpusSentences]
  );

  // Reload on filter/page change
  useEffect(() => {
    load(selectedCollection, search, page);
  }, [selectedCollection, search, page, load]);

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <div>
      {/* Controls */}
      <div
        className="animate-fade-up"
        style={{
          display: 'flex',
          gap: '0.75rem',
          marginBottom: '1.25rem',
          flexWrap: 'wrap',
          alignItems: 'center',
        }}
      >
        {/* Collection filter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.7rem', color: '#8b8070', fontFamily: 'var(--font-sans)', flexShrink: 0 }}>
            收藏集
          </span>
          <select
            value={selectedCollection}
            onChange={e => { setSelectedCollection(e.target.value); setPage(1); }}
            style={{
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: '8px',
              color: '#e8e0d0',
              fontSize: '0.78rem',
              fontFamily: 'var(--font-sans)',
              padding: '0.375rem 0.75rem',
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            <option value="">全部</option>
            {data?.collections.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        {/* Search */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: 1, minWidth: '180px' }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <input
              type="text"
              placeholder="全文搜索…"
              value={searchInput}
              onChange={e => setSearchInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') {
                  setSearch(searchInput.trim());
                  setPage(1);
                }
              }}
              style={{
                width: '100%',
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '8px',
                color: '#e8e0d0',
                fontSize: '0.78rem',
                fontFamily: 'var(--font-sans)',
                padding: '0.375rem 2rem 0.375rem 0.75rem',
                outline: 'none',
              }}
            />
            {searchInput && (
              <button
                onClick={() => { setSearchInput(''); setSearch(''); setPage(1); }}
                style={{
                  position: 'absolute', right: '0.5rem', top: '50%', transform: 'translateY(-50%)',
                  background: 'none', border: 'none', cursor: 'pointer', color: '#8b8070', fontSize: '0.75rem',
                }}
              >
                ×
              </button>
            )}
          </div>
          <button
            onClick={() => { setSearch(searchInput.trim()); setPage(1); }}
            style={{
              background: 'rgba(201,74,74,0.12)',
              border: '1px solid rgba(201,74,74,0.25)',
              borderRadius: '8px',
              color: '#c94a4a',
              fontSize: '0.78rem',
              fontFamily: 'var(--font-sans)',
              padding: '0.375rem 0.75rem',
              cursor: 'pointer',
            }}
          >
            搜索
          </button>
        </div>

        {/* Count */}
        <div style={{ fontSize: '0.72rem', color: '#8b8070', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>
          {data ? `${data.total.toLocaleString()} 条` : '…'}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{ color: '#f87171', fontSize: '0.8rem', fontFamily: 'var(--font-sans)', marginBottom: '1rem' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              style={{
                borderRadius: '12px',
                height: '100px',
                background: 'rgba(255,255,255,0.03)',
                animation: 'pulse 1.5s ease-in-out infinite',
                animationDelay: `${i * 0.1}s`,
              }}
            />
          ))}
        </div>
      )}

      {/* Sentence list */}
      {!loading && data && (
        <>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem', marginBottom: '1.25rem' }}>
            {data.sentences.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '3rem', color: 'rgba(139,128,112,0.35)', fontSize: '0.85rem', fontFamily: 'var(--font-sans)' }}>
                未找到匹配句子
              </div>
            ) : (
              data.sentences.map((s) => (
                <SentenceCard key={s.id} sentence={s} onAnalyze={onAnalyze} />
              ))
            )}
          </div>

          {/* Pagination */}
          {!loading && data.total > PAGE_SIZE && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page <= 1}
                style={{
                  width: '32px', height: '32px', borderRadius: '8px',
                  background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
                  color: page <= 1 ? '#4a4540' : '#e8e0d0',
                  cursor: page <= 1 ? 'not-allowed' : 'pointer',
                  fontSize: '0.8rem', display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}
              >
                ‹
              </button>

              {/* Page numbers */}
              {(() => {
                const pages: (number | '…')[] = [];
                const cur = page;
                const total = totalPages;
                if (total <= 7) {
                  for (let i = 1; i <= total; i++) pages.push(i);
                } else {
                  pages.push(1);
                  if (cur > 3) pages.push('…');
                  for (let i = Math.max(2, cur - 1); i <= Math.min(total - 1, cur + 1); i++) pages.push(i);
                  if (cur < total - 2) pages.push('…');
                  pages.push(total);
                }
                return pages.map((p, i) =>
                  p === '…' ? (
                    <span key={`ellipsis-${i}`} style={{ color: '#4a4540', padding: '0 0.25rem' }}>…</span>
                  ) : (
                    <button
                      key={p}
                      onClick={() => setPage(p)}
                      style={{
                        minWidth: '32px', height: '32px', borderRadius: '8px',
                        background: p === cur ? 'rgba(201,74,74,0.15)' : 'rgba(255,255,255,0.04)',
                        border: `1px solid ${p === cur ? 'rgba(201,74,74,0.35)' : 'rgba(255,255,255,0.08)'}`,
                        color: p === cur ? '#c94a4a' : '#e8e0d0',
                        cursor: 'pointer', fontSize: '0.78rem', fontFamily: 'var(--font-mono)',
                        fontWeight: p === cur ? 600 : 400,
                      }}
                    >
                      {p}
                    </button>
                  )
                );
              })()}

              <button
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                style={{
                  width: '32px', height: '32px', borderRadius: '8px',
                  background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
                  color: page >= totalPages ? '#4a4540' : '#e8e0d0',
                  cursor: page >= totalPages ? 'not-allowed' : 'pointer',
                  fontSize: '0.8rem', display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}
              >
                ›
              </button>

              <span style={{ fontSize: '0.7rem', color: '#8b8070', fontFamily: 'var(--font-mono)', marginLeft: '0.25rem' }}>
                {page}/{totalPages}
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Stats tab ─────────────────────────────────────────────────────────────────
function StatsTab() {
  const navigate = useNavigate();
  const { getCorpusStats } = useAnalysis();
  const [stats, setStats] = useState<{
    total_sentences: number;
    total_collections: number;
    collections: { name: string; count: number }[];
    pos_dataset_stats: Record<string, { sentences: number; max_length: number }>;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getCorpusStats()
      .then(setStats)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [getCorpusStats]);

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '3rem', color: 'rgba(139,128,112,0.5)', fontFamily: 'var(--font-sans)', gap: '0.5rem' }}>
        <svg style={{ width: '16px', height: '16px', animation: 'spin 1.5s linear infinite', color: '#c94a4a' }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" strokeLinecap="round" />
        </svg>
        加载中…
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ borderRadius: '14px', padding: '1rem 1.25rem', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)', fontFamily: 'var(--font-sans)', fontSize: '0.85rem', color: '#f87171' }}>
        ⚠️ 无法加载语料库统计：{error}
      </div>
    );
  }

  const collections = stats?.collections ?? [];
  const maxCount = collections[0]?.count ?? 1;

  return (
    <div>
      {/* Stats dashboard */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8 animate-fade-up">
        {[
          {
            label: '句子总数', value: stats?.total_sentences?.toLocaleString() ?? '—',
            accent: '#60a5fa', bg: 'rgba(37,99,235,0.1)', border: 'rgba(37,99,235,0.2)',
            icon: (
              <svg style={{ width: '18px', height: '18px', color: '#60a5fa', opacity: 0.7 }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
            ),
          },
          {
            label: '收藏集', value: stats?.total_collections ?? '—',
            accent: '#a78bfa', bg: 'rgba(124,58,237,0.1)', border: 'rgba(124,58,237,0.2)',
            icon: (
              <svg style={{ width: '18px', height: '18px', color: '#a78bfa', opacity: 0.7 }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
              </svg>
            ),
          },
          {
            label: '训练集', value: stats?.pos_dataset_stats?.train?.sentences?.toLocaleString() ?? '—',
            accent: '#4ade80', bg: 'rgba(22,163,74,0.1)', border: 'rgba(22,163,74,0.2)',
            icon: (
              <svg style={{ width: '18px', height: '18px', color: '#4ade80', opacity: 0.7 }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
              </svg>
            ),
          },
          {
            label: '测试集', value: stats?.pos_dataset_stats?.test?.sentences?.toLocaleString() ?? '—',
            accent: '#c94a4a', bg: 'rgba(201,74,74,0.1)', border: 'rgba(201,74,74,0.2)',
            icon: (
              <svg style={{ width: '18px', height: '18px', color: '#c94a4a', opacity: 0.7 }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            ),
          },
        ].map(({ label, value, accent, bg, border, icon }) => (
          <div key={label} style={{ borderRadius: '16px', padding: '1.125rem 1rem', background: bg, border: `1px solid ${border}`, boxShadow: '0 4px 16px rgba(0,0,0,0.2)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.625rem' }}>
              <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(0,0,0,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {icon}
              </div>
            </div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: 'clamp(1.4rem, 3vw, 1.75rem)', fontWeight: 700, color: accent, lineHeight: 1, marginBottom: '0.3rem' }}>
              {value}
            </div>
            <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'rgba(139,128,112,0.6)', fontWeight: 500 }}>
              {label}
            </div>
          </div>
        ))}
      </div>

      {/* Collections */}
      <div className="mb-8 animate-fade-up" style={{ animationDelay: '60ms' }}>
        <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' as const, color: 'rgba(139,128,112,0.5)', marginBottom: '0.875rem' }}>
          收藏集
        </div>
        {collections.length === 0 ? (
          <div style={{ borderRadius: '14px', border: '1px dashed rgba(255,255,255,0.08)', padding: '2.5rem', textAlign: 'center', color: 'rgba(139,128,112,0.35)', fontSize: '0.8rem', fontFamily: 'var(--font-sans)' }}>
            暂无收藏集数据
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {collections.map((c) => (
              <div key={c.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.875rem 1rem', borderRadius: '14px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', gap: '1rem' }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', fontWeight: 600, color: '#e8e0d0', marginBottom: '0.2rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {c.name}
                  </div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'rgba(139,128,112,0.5)' }}>
                    {c.count.toLocaleString()} 句子
                  </div>
                </div>
                <div style={{ width: '80px', height: '3px', background: 'rgba(255,255,255,0.06)', borderRadius: '2px', overflow: 'hidden', flexShrink: 0 }}>
                  <div style={{ height: '100%', width: `${Math.min(100, (c.count / maxCount) * 100)}%`, background: `linear-gradient(to right, ${collectionColor(c.name)}, ${collectionColor(c.name)}88)`, borderRadius: '2px' }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* POS Dataset */}
      {stats?.pos_dataset_stats && (
        <div className="mb-8 animate-fade-up" style={{ animationDelay: '120ms' }}>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' as const, color: 'rgba(139,128,112,0.5)', marginBottom: '0.875rem' }}>
            标注数据集
          </div>
          <div style={{ borderRadius: '14px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.06)' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: 'rgba(0,0,0,0.2)' }}>
                  {(['数据集', '句子数', '最大长度'] as const).map((h, i) => (
                    <th key={h} style={{ padding: '0.625rem 1rem', textAlign: 'left', fontFamily: 'var(--font-sans)', fontSize: '0.7rem', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' as const, color: 'rgba(139,128,112,0.5)', borderBottom: i < 2 ? '1px solid rgba(255,255,255,0.04)' : 'none' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(['train', 'dev', 'test'] as const).map((split, i) => {
                  const s = stats.pos_dataset_stats?.[split];
                  return (
                    <tr key={split} style={{ borderBottom: i < 2 ? '1px solid rgba(255,255,255,0.04)' : 'none' }}>
                      {[
                        { text: ['训练集', '验证集', '测试集'][i], color: '#e8e0d0' },
                        { text: s?.sentences?.toLocaleString() ?? '—', color: 'rgba(139,128,112,0.7)' },
                        { text: s?.max_length ?? '—', color: 'rgba(139,128,112,0.7)' },
                      ].map(({ text, color }, j) => (
                        <td key={j} style={{ padding: '0.75rem 1rem', fontFamily: j === 0 ? 'var(--font-sans)' : 'var(--font-mono)', fontSize: j === 0 ? '0.8rem' : '0.78rem', fontWeight: j === 0 ? 500 : 400, color }}>
                          {text}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Info banner */}
      <div className="animate-fade-up" style={{ borderRadius: '14px', padding: '1rem 1.25rem', background: 'rgba(212,168,83,0.06)', border: '1px solid rgba(212,168,83,0.18)', animationDelay: '160ms' }}>
        <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', color: 'rgba(139,128,112,0.7)', lineHeight: 1.6 }}>
          💡 SegPOS 语料库共约 18.6M 句原始标注，采样 330k 句用于训练 POS 分类器
        </div>
        <button
          onClick={() => navigate('/')}
          style={{ marginTop: '0.5rem', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: '0.78rem', color: '#c94a4a', padding: 0, opacity: 0.8, transition: 'opacity 0.2s' }}
          onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.opacity = '1'; }}
          onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.opacity = '0.8'; }}
        >
          → 使用分析器分析自己的句子
        </button>
      </div>
    </div>
  );
}

// ── Main CorpusPage ───────────────────────────────────────────────────────────
export function CorpusPage() {
  const navigate = useNavigate();
  const [tab, setTab] = useState<Tab>('stats');

  const handleAnalyze = (text: string) => {
    navigate('/', { state: { text } });
  };

  return (
    <div className="max-w-6xl mx-auto px-6 pt-8 pb-12">
      {/* Page header */}
      <div className="mb-6 animate-fade-up">
        <div style={{ fontFamily: 'var(--font-display)', fontSize: 'clamp(1.6rem, 4vw, 2.2rem)', fontWeight: 700, color: '#e8e0d0', lineHeight: 1.1 }}>
          语料库
        </div>
        <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', color: 'rgba(139,128,112,0.7)', marginTop: '0.4rem', letterSpacing: '0.04em' }}>
          SegPOS 标注语料库 · 浏览与统计
        </div>
        <div style={{ marginTop: '0.75rem', height: '1px', background: 'linear-gradient(to right, rgba(212,168,83,0.4), transparent)', maxWidth: '280px' }} />
      </div>

      {/* Tabs */}
      <div
        className="animate-fade-up"
        style={{ display: 'flex', gap: '0.25rem', marginBottom: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.06)' }}
      >
        {([
          { key: 'stats',   label: '统计', icon: '📊' },
          { key: 'reader',   label: '语料阅读器', icon: '📖' },
        ] as const).map(({ key, label, icon }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            style={{
              padding: '0.5rem 1rem',
              background: 'none',
              border: 'none',
              borderBottom: `2px solid ${tab === key ? '#c94a4a' : 'transparent'}`,
              color: tab === key ? '#e8e0d0' : 'rgba(139,128,112,0.5)',
              fontFamily: 'var(--font-sans)',
              fontSize: '0.8rem',
              fontWeight: tab === key ? 600 : 400,
              cursor: 'pointer',
              transition: 'all 0.15s',
              display: 'flex',
              alignItems: 'center',
              gap: '0.375rem',
              marginBottom: '-1px',
            }}
          >
            <span>{icon}</span>
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === 'stats' ? <StatsTab /> : <CorpusReader onAnalyze={handleAnalyze} />}
    </div>
  );
}
