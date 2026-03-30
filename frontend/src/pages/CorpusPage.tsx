import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysis } from '../hooks/useAnalysis';

export function CorpusPage() {
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
  }, []);

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-6 py-12">
        <div
          className="flex items-center justify-center py-20 animate-pulse"
          style={{ fontFamily: 'var(--font-sans)', color: 'rgba(139,128,112,0.5)' }}
        >
          <svg
            style={{ width: '16px', height: '16px', marginRight: '0.5rem', color: '#c94a4a', animation: 'spin 1.5s linear infinite' }}
            viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
          >
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" strokeLinecap="round" />
          </svg>
          加载语料库统计…
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div
        className="max-w-6xl mx-auto px-6 py-8"
      >
        <div
          style={{
            borderRadius: '14px',
            padding: '1rem 1.25rem',
            background: 'rgba(239,68,68,0.08)',
            border: '1px solid rgba(239,68,68,0.25)',
          }}
        >
          <div
            style={{
              fontFamily: 'var(--font-sans)',
              fontSize: '0.85rem',
              color: '#f87171',
              marginBottom: '0.5rem',
            }}
          >
            ⚠️ 无法加载语料库：{error}
          </div>
          <div
            style={{
              fontFamily: 'var(--font-sans)',
              fontSize: '0.78rem',
              color: 'rgba(139,128,112,0.6)',
            }}
          >
            请确保 FastAPI 服务器已启动：
            <code
              style={{
                marginLeft: '0.5rem',
                background: 'rgba(255,255,255,0.06)',
                padding: '0.1rem 0.4rem',
                borderRadius: '4px',
                fontSize: '0.75rem',
              }}
            >
              tibert serve
            </code>
          </div>
        </div>
      </div>
    );
  }

  const collections = stats?.collections ?? [];
  const maxCount = collections[0]?.count ?? 1;

  return (
    <div className="max-w-6xl mx-auto px-6 pt-8 pb-12">
      {/* Page header */}
      <div className="mb-8 animate-fade-up" style={{ animationDelay: '0ms' }}>
        <div
          style={{
            fontFamily: 'var(--font-display)',
            fontSize: 'clamp(1.6rem, 4vw, 2.2rem)',
            fontWeight: 700,
            color: '#e8e0d0',
            lineHeight: 1.1,
          }}
        >
          语料库
        </div>
        <div
          style={{
            fontFamily: 'var(--font-sans)',
            fontSize: '0.8rem',
            color: 'rgba(139,128,112,0.7)',
            marginTop: '0.4rem',
            letterSpacing: '0.04em',
          }}
        >
          SegPOS 标注语料库统计信息
        </div>
        <div
          style={{
            marginTop: '0.75rem',
            height: '1px',
            background: 'linear-gradient(to right, rgba(212,168,83,0.4), transparent)',
            maxWidth: '280px',
          }}
        />
      </div>

      {/* Stats dashboard */}
      <div
        className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8 animate-fade-up"
        style={{ animationDelay: '60ms' }}
      >
        {[
          {
            label: '句子总数',
            value: stats?.total_sentences?.toLocaleString() ?? '—',
            accent: '#60a5fa',
            bg: 'rgba(37,99,235,0.1)',
            border: 'rgba(37,99,235,0.2)',
            icon: (
              <svg style={{ width: '18px', height: '18px', color: '#60a5fa', opacity: 0.7 }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
            ),
          },
          {
            label: '收藏集',
            value: stats?.total_collections ?? '—',
            accent: '#a78bfa',
            bg: 'rgba(124,58,237,0.1)',
            border: 'rgba(124,58,237,0.2)',
            icon: (
              <svg style={{ width: '18px', height: '18px', color: '#a78bfa', opacity: 0.7 }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
              </svg>
            ),
          },
          {
            label: '训练集',
            value: stats?.pos_dataset_stats?.train?.sentences?.toLocaleString() ?? '—',
            accent: '#4ade80',
            bg: 'rgba(22,163,74,0.1)',
            border: 'rgba(22,163,74,0.2)',
            icon: (
              <svg style={{ width: '18px', height: '18px', color: '#4ade80', opacity: 0.7 }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
              </svg>
            ),
          },
          {
            label: '测试集',
            value: stats?.pos_dataset_stats?.test?.sentences?.toLocaleString() ?? '—',
            accent: '#c94a4a',
            bg: 'rgba(201,74,74,0.1)',
            border: 'rgba(201,74,74,0.2)',
            icon: (
              <svg style={{ width: '18px', height: '18px', color: '#c94a4a', opacity: 0.7 }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            ),
          },
        ].map(({ label, value, accent, bg, border, icon }) => (
          <div
            key={label}
            style={{
              borderRadius: '16px',
              padding: '1.125rem 1rem',
              background: bg,
              border: `1px solid ${border}`,
              boxShadow: '0 4px 16px rgba(0,0,0,0.2)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '0.625rem' }}>
              <div
                style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '10px',
                  background: 'rgba(0,0,0,0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {icon}
              </div>
            </div>
            <div
              style={{
                fontFamily: 'var(--font-display)',
                fontSize: 'clamp(1.4rem, 3vw, 1.75rem)',
                fontWeight: 700,
                color: accent,
                lineHeight: 1,
                marginBottom: '0.3rem',
              }}
            >
              {value}
            </div>
            <div
              style={{
                fontFamily: 'var(--font-sans)',
                fontSize: '0.72rem',
                color: 'rgba(139,128,112,0.6)',
                fontWeight: 500,
              }}
            >
              {label}
            </div>
          </div>
        ))}
      </div>

      {/* Collections */}
      <div className="mb-8 animate-fade-up" style={{ animationDelay: '120ms' }}>
        <div
          style={{
            fontFamily: 'var(--font-sans)',
            fontSize: '0.75rem',
            fontWeight: 700,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            color: 'rgba(139,128,112,0.5)',
            marginBottom: '0.875rem',
          }}
        >
          收藏集
        </div>
        {collections.length === 0 ? (
          <div
            style={{
              borderRadius: '14px',
              border: '1px dashed rgba(255,255,255,0.08)',
              padding: '2.5rem',
              textAlign: 'center',
              color: 'rgba(139,128,112,0.35)',
              fontSize: '0.8rem',
              fontFamily: 'var(--font-sans)',
            }}
          >
            暂无收藏集数据
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {collections.map((c) => (
              <div
                key={c.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.875rem 1rem',
                  borderRadius: '14px',
                  background: 'rgba(255,255,255,0.03)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  gap: '1rem',
                }}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontFamily: 'var(--font-sans)',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      color: '#e8e0d0',
                      marginBottom: '0.2rem',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {c.name}
                  </div>
                  <div
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.68rem',
                      color: 'rgba(139,128,112,0.5)',
                    }}
                  >
                    {c.count.toLocaleString()} 句子
                  </div>
                </div>
                {/* Progress bar */}
                <div
                  style={{
                    width: '80px',
                    height: '3px',
                    background: 'rgba(255,255,255,0.06)',
                    borderRadius: '2px',
                    overflow: 'hidden',
                    flexShrink: 0,
                  }}
                >
                  <div
                    style={{
                      height: '100%',
                      width: `${Math.min(100, (c.count / maxCount) * 100)}%`,
                      background: 'linear-gradient(to right, #c94a4a, #d4a853)',
                      borderRadius: '2px',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* POS Dataset */}
      {stats?.pos_dataset_stats && (
        <div className="mb-8 animate-fade-up" style={{ animationDelay: '160ms' }}>
          <div
            style={{
              fontFamily: 'var(--font-sans)',
              fontSize: '0.75rem',
              fontWeight: 700,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              color: 'rgba(139,128,112,0.5)',
              marginBottom: '0.875rem',
            }}
          >
            标注数据集
          </div>
          <div
            style={{
              borderRadius: '14px',
              overflow: 'hidden',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: 'rgba(0,0,0,0.2)' }}>
                  {['数据集', '句子数', '最大长度'].map((h, i) => (
                    <th
                      key={h}
                      style={{
                        padding: '0.625rem 1rem',
                        textAlign: 'left',
                        fontFamily: 'var(--font-sans)',
                        fontSize: '0.7rem',
                        fontWeight: 600,
                        letterSpacing: '0.06em',
                        textTransform: 'uppercase',
                        color: 'rgba(139,128,112,0.5)',
                        borderBottom: i < 2 ? '1px solid rgba(255,255,255,0.04)' : 'none',
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(['train', 'dev', 'test'] as const).map((split, i) => {
                  const s = stats.pos_dataset_stats?.[split];
                  return (
                    <tr
                      key={split}
                      style={{
                        borderBottom: i < 2 ? '1px solid rgba(255,255,255,0.04)' : 'none',
                      }}
                    >
                      {[
                        { text: ['训练集', '验证集', '测试集'][i], color: '#e8e0d0' },
                        { text: s?.sentences?.toLocaleString() ?? '—', color: 'rgba(139,128,112,0.7)' },
                        { text: s?.max_length ?? '—', color: 'rgba(139,128,112,0.7)' },
                      ].map(({ text, color }, j) => (
                        <td
                          key={j}
                          style={{
                            padding: '0.75rem 1rem',
                            fontFamily: j === 0 ? 'var(--font-sans)' : 'var(--font-mono)',
                            fontSize: j === 0 ? '0.8rem' : '0.78rem',
                            fontWeight: j === 0 ? 500 : 400,
                            color,
                          }}
                        >
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
      <div
        className="animate-fade-up"
        style={{
          borderRadius: '14px',
          padding: '1rem 1.25rem',
          background: 'rgba(212,168,83,0.06)',
          border: '1px solid rgba(212,168,83,0.18)',
          animationDelay: '200ms',
        }}
      >
        <div
          style={{
            fontFamily: 'var(--font-sans)',
            fontSize: '0.8rem',
            color: 'rgba(139,128,112,0.7)',
            lineHeight: 1.6,
          }}
        >
          💡 SegPOS 语料库共约 18.6M 句原始标注，采样 330k 句用于训练 POS 分类器
          （Test WF1 = 90.2%）
        </div>
        <button
          onClick={() => navigate('/')}
          style={{
            marginTop: '0.5rem',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontFamily: 'var(--font-sans)',
            fontSize: '0.78rem',
            color: '#c94a4a',
            padding: 0,
            opacity: 0.8,
            transition: 'opacity 0.2s',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; }}
          onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.8'; }}
        >
          → 使用分析器分析自己的句子
        </button>
      </div>
    </div>
  );
}
