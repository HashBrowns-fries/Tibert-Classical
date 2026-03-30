import { useState, useRef, useEffect } from 'react';
import type { TokenResponse } from '../types/api';

interface TooltipData {
  token: string;
  pos: string;
  pos_zh: string;
  case_name?: string;
  case_desc?: string;
}

function getPosChipStyle(pos: string): {
  bg: string;
  text: string;
  border: string;
  glow?: string;
} {
  if (pos === 'punc') return { bg: 'rgba(107,114,128,0.08)', text: '#6b7280', border: 'rgba(107,114,128,0.2)' };
  if (pos.startsWith('case')) return { bg: 'rgba(217,119,6,0.15)', text: '#d97706', border: 'rgba(217,119,6,0.3)', glow: '0 0 8px rgba(217,119,6,0.15)' };
  if (pos.startsWith('n.')) return { bg: 'rgba(124,58,237,0.12)', text: '#a78bfa', border: 'rgba(124,58,237,0.25)' };
  if (pos.startsWith('v.')) return { bg: 'rgba(13,148,136,0.12)', text: '#2dd4bf', border: 'rgba(13,148,136,0.25)' };
  if (pos === 'adj') return { bg: 'rgba(22,163,74,0.1)', text: '#4ade80', border: 'rgba(22,163,74,0.2)' };
  if (pos.startsWith('adv')) return { bg: 'rgba(37,99,235,0.1)', text: '#60a5fa', border: 'rgba(37,99,235,0.2)' };
  if (pos.startsWith('neg')) return { bg: 'rgba(220,38,38,0.1)', text: '#f87171', border: 'rgba(220,38,38,0.2)' };
  return { bg: 'rgba(139,128,112,0.08)', text: '#8b8070', border: 'rgba(139,128,112,0.15)' };
}

export function TokenDisplay({ tokens }: { tokens: TokenResponse[] }) {
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Close tooltip on scroll
  useEffect(() => {
    const handler = () => setTooltip(null);
    window.addEventListener('scroll', handler, true);
    return () => window.removeEventListener('scroll', handler, true);
  }, []);

  function showTooltip(e: React.MouseEvent<HTMLSpanElement>, token: TokenResponse) {
    const rect = containerRef.current?.getBoundingClientRect();
    const x = e.clientX;
    const y = e.clientY;
    // Keep tooltip within viewport
    setTooltipPos({ x: Math.min(x + 12, window.innerWidth - 300), y: y - 8 });
    setTooltip({
      token: token.token,
      pos: token.pos,
      pos_zh: token.pos_zh,
      case_name: token.case_name ?? undefined,
      case_desc: token.case_desc ?? undefined,
    });
  }

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      <div
        className="tibetan"
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.25rem',
          lineHeight: '2.4',
          fontSize: '1.15rem',
        }}
      >
        {tokens.map((t, i) => {
          const isSep = t.token === '་' || t.token === '།';
          const isCase = t.is_case_particle;
          const style = getPosChipStyle(t.pos);

          if (isSep) {
            return (
              <span
                key={i}
                style={{
                  color: 'rgba(139,128,112,0.3)',
                  fontSize: '0.8rem',
                  lineHeight: '2.4',
                  userSelect: 'none',
                  marginLeft: '-0.1rem',
                }}
              >
                {t.token === '་' ? '·' : '⸱'}
              </span>
            );
          }

          return (
            <span
              key={i}
              className="token-chip"
              style={{
                background: style.bg,
                border: `1px solid ${style.border}`,
                boxShadow: style.glow ?? 'none',
                position: 'relative',
              }}
              onMouseEnter={e => showTooltip(e, t)}
              onMouseLeave={() => setTooltip(null)}
            >
              <span
                style={{
                  color: isCase ? '#d97706' : '#e8e0d0',
                  fontWeight: isCase ? 700 : 600,
                  fontFamily: 'var(--font-tibetan)',
                  lineHeight: '1.3',
                  fontSize: '1.1rem',
                }}
              >
                {t.token}
              </span>
              <span
                className="token-chip-pos"
                style={{ color: style.text }}
              >
                {t.pos_zh}
              </span>
            </span>
          );
        })}
      </div>

      {/* Glassmorphism tooltip */}
      {tooltip && (
        <div
          className="token-tooltip"
          style={{
            left: tooltipPos.x,
            top: tooltipPos.y,
          }}
        >
          <div
            className="tibetan"
            style={{
              fontWeight: 700,
              fontSize: '1.2rem',
              color: '#e8e0d0',
              marginBottom: '0.375rem',
              lineHeight: '1.3',
            }}
          >
            {tooltip.token}
          </div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              marginBottom: tooltip.case_name ? '0.25rem' : 0,
            }}
          >
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.65rem',
                color: '#8b8070',
                background: 'rgba(255,255,255,0.06)',
                padding: '0.15rem 0.4rem',
                borderRadius: '4px',
              }}
            >
              {tooltip.pos}
            </span>
            <span
              style={{
                fontFamily: 'var(--font-sans)',
                fontSize: '0.8rem',
                fontWeight: 500,
                color: '#e8e0d0',
              }}
            >
              {tooltip.pos_zh}
            </span>
          </div>
          {tooltip.case_name && (
            <div
              style={{
                fontFamily: 'var(--font-sans)',
                fontSize: '0.75rem',
                color: '#d97706',
                marginBottom: '0.2rem',
              }}
            >
              ★ {tooltip.case_name}
            </div>
          )}
          {tooltip.case_desc && (
            <div
              style={{
                fontFamily: 'var(--font-sans)',
                fontSize: '0.75rem',
                color: '#8b8070',
                lineHeight: '1.4',
              }}
            >
              {tooltip.case_desc}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
