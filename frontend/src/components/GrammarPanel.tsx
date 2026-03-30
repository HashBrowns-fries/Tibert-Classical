import ReactMarkdown from 'react-markdown';

interface GrammarPanelProps {
  explanation?: string;
  structure?: string;
  original: string;
  loading?: boolean;
}

export function GrammarPanel({ explanation, structure, original, loading }: GrammarPanelProps) {
  if (loading) {
    return (
      <div className="llm-panel" style={{ minHeight: '120px', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <svg
          className="animate-spin-slow"
          style={{ width: '20px', height: '20px', color: '#c94a4a', flexShrink: 0 }}
          viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
        >
          <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" strokeLinecap="round" />
        </svg>
        <div>
          <div style={{ fontSize: '0.85rem', color: '#8b8070', fontFamily: 'var(--font-sans)' }}>
            LLM 正在分析中…
          </div>
          <div style={{ fontSize: '0.75rem', color: '#4a4540', marginTop: '0.25rem', fontFamily: 'var(--font-sans)' }}>
            基于 DashScope Qwen 模型生成语法解释
          </div>
        </div>
      </div>
    );
  }

  if (!explanation) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* LLM header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0 0.25rem',
        }}
      >
        <svg style={{ width: '16px', height: '16px', color: '#c94a4a' }} viewBox="0 0 24 24" fill="currentColor">
          <path d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
        </svg>
        <span
          style={{
            fontSize: '0.75rem',
            fontWeight: 600,
            color: '#c94a4a',
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            fontFamily: 'var(--font-sans)',
          }}
        >
          LLM 语法分析
        </span>
        <span
          style={{
            fontSize: '0.65rem',
            color: '#4a4540',
            fontFamily: 'var(--font-sans)',
          }}
        >
          DashScope Qwen
        </span>
      </div>

      {/* Explanation — rendered as Markdown */}
      <div className="llm-panel animate-fade-up">
        <ReactMarkdown
          components={{
            h1: ({ children }) => (
              <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '1.2rem', fontWeight: 700, color: '#e8e0d0', marginBottom: '0.5rem', marginTop: 0 }}>
                {children}
              </h1>
            ),
            h2: ({ children }) => (
              <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', fontWeight: 700, color: '#d4a853', marginBottom: '0.4rem', marginTop: '0.75rem', borderBottom: '1px solid rgba(212,168,83,0.2)', paddingBottom: '0.25rem' }}>
                {children}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 style={{ fontFamily: 'var(--font-sans)', fontSize: '0.85rem', fontWeight: 600, color: '#c94a4a', marginBottom: '0.3rem', marginTop: '0.6rem' }}>
                {children}
              </h3>
            ),
            p: ({ children }) => (
              <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.875rem', color: '#e8e0d0', lineHeight: 1.8, marginBottom: '0.6rem', marginTop: 0 }}>
                {children}
              </p>
            ),
            ul: ({ children }) => (
              <ul style={{ fontFamily: 'var(--font-sans)', fontSize: '0.875rem', color: '#e8e0d0', lineHeight: 1.8, paddingLeft: '1.25rem', marginBottom: '0.5rem' }}>
                {children}
              </ul>
            ),
            ol: ({ children }) => (
              <ol style={{ fontFamily: 'var(--font-sans)', fontSize: '0.875rem', color: '#e8e0d0', lineHeight: 1.8, paddingLeft: '1.25rem', marginBottom: '0.5rem' }}>
                {children}
              </ol>
            ),
            li: ({ children }) => (
              <li style={{ marginBottom: '0.2rem' }}>
                <span style={{ color: '#d97706', marginRight: '0.3rem' }}>·</span>
                {children}
              </li>
            ),
            strong: ({ children }) => (
              <strong style={{ color: '#d4a853', fontWeight: 600 }}>
                {children}
              </strong>
            ),
            em: ({ children }) => (
              <em style={{ color: '#c94a4a', fontStyle: 'italic' }}>
                {children}
              </em>
            ),
            code: ({ children }) => (
              <code style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.78rem',
                background: 'rgba(255,255,255,0.06)',
                padding: '0.1rem 0.35rem',
                borderRadius: '4px',
                color: '#a78bfa',
              }}>
                {children}
              </code>
            ),
            blockquote: ({ children }) => (
              <blockquote style={{
                borderLeft: '3px solid rgba(201,74,74,0.4)',
                paddingLeft: '0.75rem',
                marginLeft: 0,
                color: 'rgba(232,224,208,0.7)',
                fontStyle: 'italic',
                marginBottom: '0.5rem',
              }}>
                {children}
              </blockquote>
            ),
            table: ({ children }) => (
              <div style={{ overflowX: 'auto', marginBottom: '0.75rem' }}>
                <table style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontFamily: 'var(--font-sans)',
                  fontSize: '0.8rem',
                }}>
                  {children}
                </table>
              </div>
            ),
            thead: ({ children }) => (
              <thead style={{ background: 'rgba(0,0,0,0.2)' }}>
                {children}
              </thead>
            ),
            tbody: ({ children }) => <tbody>{children}</tbody>,
            tr: ({ children }) => (
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                {children}
              </tr>
            ),
            th: ({ children }) => (
              <th style={{
                padding: '0.4rem 0.75rem',
                textAlign: 'left',
                fontWeight: 600,
                color: '#d4a853',
                fontSize: '0.72rem',
                letterSpacing: '0.04em',
                whiteSpace: 'nowrap',
              }}>
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td style={{
                padding: '0.4rem 0.75rem',
                color: 'rgba(232,224,208,0.8)',
                lineHeight: 1.5,
              }}>
                {children}
              </td>
            ),
            hr: () => <hr style={{ border: 'none', borderTop: '1px solid rgba(255,255,255,0.06)', margin: '0.75rem 0' }} />,
            a: ({ href, children }) => (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: '#c94a4a', textDecoration: 'underline', textDecorationColor: 'rgba(201,74,74,0.4)' }}
              >
                {children}
              </a>
            ),
          }}
        >
          {explanation}
        </ReactMarkdown>
      </div>

      {/* Structure */}
      {structure && (
        <div className="card" style={{ padding: '0.875rem 1rem' }}>
          <div
            style={{
              fontSize: '0.7rem',
              fontWeight: 600,
              color: '#8b8070',
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              marginBottom: '0.5rem',
              fontFamily: 'var(--font-sans)',
            }}
          >
            句法结构
          </div>
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.8rem',
              color: '#a78bfa',
              lineHeight: '1.8',
              letterSpacing: '0.02em',
            }}
          >
            {structure}
          </div>
        </div>
      )}
    </div>
  );
}
