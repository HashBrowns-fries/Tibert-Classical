import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAnalysisStore } from '../stores/analysisStore';

function getApiBase() {
  return `http://${window.location.hostname}:8001`;
}

const NAV_ITEMS = [
  {
    to: '/',
    label: '分析器',
    desc: 'POS 标注 & LLM',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 01-1.59.659H9.06a2.25 2.25 0 01-1.591-.659L5 14.5" />
      </svg>
    ),
  },
  {
    to: '/rag',
    label: 'RAG',
    desc: '佛典问答',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M12 21a9.004 9.004 0 008.716 6.748M12 21a9.004 9.004 0 01-8.716-6.748M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0 0112 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 013 12c0-1.605.42-3.113 1.157-4.418" />
      </svg>
    ),
  },
  {
    to: '/dict',
    label: '词典',
    desc: '多词典查询',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
      </svg>
    ),
  },
  {
    to: '/learner',
    label: '学习',
    desc: '格助词 & 卡片',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M4.26 10.147a60.436 60.436 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.57 50.57 0 00-2.658-.813A59.905 59.905 0 0112 3.493a59.905 59.905 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.697 50.697 0 0112 13.489a50.702 50.702 0 017.74-3.342M6.75 15a.75.75 0 100-1.5.75.75 0 000 1.5zm0 0v-3.675A55.378 55.378 0 0112 8.443m-7.007 11.55A5.981 5.981 0 006.75 15.75v-1.5" />
      </svg>
    ),
  },
  {
    to: '/corpus',
    label: '语料库',
    desc: 'SegPOS 统计',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round"
          d="M20.25 8.511c.884.284 1.5 1.128 1.5 2.097v4.286c0 1.136-.847 2.1-1.98 2.193-.34.027-.68.052-1.02.072v3.091l-3-3c-1.354 0-2.694-.055-4.02-.163a2.115 2.115 0 01-.825-.242m9.345-8.334a2.126 2.126 0 00-.476-.095 48.64 48.64 0 00-8.048 0c-1.131.094-1.976 1.057-1.976 2.192v4.286c0 .837.46 1.58 1.155 1.951m9.345-8.334V6.637c0-1.621-1.152-3.026-2.76-3.235A48.455 48.455 0 0011.25 3c-2.115 0-4.198.137-6.24.402-1.608.209-2.76 1.614-2.76 3.235v6.226c0 1.621 1.152 3.026 2.76 3.235.577.075 1.157.137 1.74.194v3.096a48.62 48.62 0 00-1.022-.123m9.345 1.899a48.62 48.62 0 00-1.022-.123 48.62 48.62 0 00-1.022.123m9.345-1.899v3.096c.577-.057 1.163-.119 1.74-.194 1.608-.209 2.76-1.614 2.76-3.235V9.414c0-1.621-1.152-3.026-2.76-3.235a48.455 48.455 0 00-1.75-.195" />
      </svg>
    ),
  },
];

export function Header() {
  const location = useLocation();
  const { theme, setTheme } = useAnalysisStore();
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  useEffect(() => {
    fetch(`${getApiBase()}/health`)
      .then(r => setApiOk(r.ok))
      .catch(() => setApiOk(false));
  }, []);

  const themes: Array<'light' | 'dark' | 'system'> = ['light', 'dark', 'system'];
  const themeLabels = { light: '☀️', dark: '🌙', system: '🖥️' };

  return (
    <header
      className="sticky top-0 z-50 border-b border-glass"
      style={{
        background: 'rgba(8,9,15,0.85)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
      }}
    >
      {/* Crimson glow top border */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '1px',
          background: 'linear-gradient(to right, transparent, rgba(201,74,74,0.5), transparent)',
        }}
      />

      <div
        className="max-w-7xl mx-auto px-6"
        style={{ display: 'flex', alignItems: 'center', height: '60px', gap: '2rem' }}
      >
        {/* Logo */}
        <Link
          to="/"
          style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', textDecoration: 'none', flexShrink: 0 }}
        >
          <div
            style={{
              width: '34px',
              height: '34px',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, #c94a4a 0%, #7a1e1e 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 14px rgba(201,74,74,0.35)',
              flexShrink: 0,
            }}
          >
            <span
              className="tibetan"
              style={{
                color: 'white',
                fontWeight: 700,
                fontSize: '1rem',
                lineHeight: 1,
              }}
            >
              ཐ
            </span>
          </div>
          <div>
            <div
              style={{
                fontFamily: 'var(--font-display)',
                fontWeight: 700,
                fontSize: '1.05rem',
                color: '#e8e0d0',
                lineHeight: 1.1,
                letterSpacing: '0.01em',
              }}
            >
              Classical Tibetan
            </div>
            <div
              style={{
                fontFamily: 'var(--font-sans)',
                fontSize: '0.65rem',
                color: 'rgba(201,74,74,0.7)',
                fontWeight: 600,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
              }}
            >
              NLP
            </div>
          </div>
        </Link>

        {/* Nav */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', flex: 1 }}>
          {NAV_ITEMS.map((item) => {
            const isActive = location.pathname === item.to ||
              (item.to !== '/' && location.pathname.startsWith(item.to));
            return (
              <Link
                key={item.to}
                to={item.to}
                className={`nav-link ${isActive ? 'active' : ''}`}
              >
                {item.icon}
                <span>{item.label}</span>
                <span
                  style={{
                    fontSize: '0.7rem',
                    color: 'inherit',
                    opacity: 0.5,
                    fontFamily: 'var(--font-sans)',
                    fontWeight: 400,
                    marginLeft: '0.125rem',
                  }}
                >
                  {item.desc}
                </span>
              </Link>
            );
          })}
        </nav>

        {/* Right: API status + theme */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexShrink: 0 }}>
          {/* API status */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.375rem',
              padding: '0.3rem 0.625rem',
              borderRadius: '99px',
              border: '1px solid rgba(255,255,255,0.07)',
              background: 'rgba(255,255,255,0.03)',
              fontSize: '0.7rem',
              fontFamily: 'var(--font-sans)',
              fontWeight: 500,
              color: '#8b8070',
            }}
          >
            <div
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: apiOk === null
                  ? '#4a4540'
                  : apiOk
                    ? '#22c55e'
                    : '#ef4444',
                boxShadow: apiOk
                  ? '0 0 6px rgba(34,197,94,0.6)'
                  : 'none',
                animation: apiOk ? 'pulse-ring 2s ease-in-out infinite' : 'none',
              }}
            />
            {apiOk === null ? '连接中…' : apiOk ? 'API 在线' : '离线'}
          </div>

          {/* Theme cycle */}
          <button
            onClick={() => {
              const idx = themes.indexOf(theme);
              setTheme(themes[(idx + 1) % themes.length]);
            }}
            style={{
              width: '34px',
              height: '34px',
              borderRadius: '8px',
              border: '1px solid rgba(255,255,255,0.08)',
              background: 'rgba(255,255,255,0.04)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '1rem',
              transition: 'all 0.2s ease',
              color: '#e8e0d0',
            }}
            title={`主题: ${theme === 'dark' ? '深色' : theme === 'light' ? '浅色' : '跟随系统'}`}
          >
            {themeLabels[theme]}
          </button>
        </div>
      </div>
    </header>
  );
}
