import { useState, useRef, useEffect } from 'react';
import type { TokenResponse, LookupEntry } from '../types/api';

interface TooltipData {
  token: string;
  pos: string;
  pos_zh: string;
  case_name?: string;
  case_desc?: string;
  dictEntries: LookupEntry[];
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
  if (pos.startsWith('v.') || pos === 'VERB') return { bg: 'rgba(13,148,136,0.12)', text: '#2dd4bf', border: 'rgba(13,148,136,0.25)' };
  if (pos === 'adj' || pos === 'ADJ') return { bg: 'rgba(22,163,74,0.1)', text: '#4ade80', border: 'rgba(22,163,74,0.2)' };
  if (pos.startsWith('adv') || pos === 'ADV') return { bg: 'rgba(37,99,235,0.1)', text: '#60a5fa', border: 'rgba(37,99,235,0.2)' };
  if (pos.startsWith('neg')) return { bg: 'rgba(220,38,38,0.1)', text: '#f87171', border: 'rgba(220,38,38,0.2)' };
  return { bg: 'rgba(139,128,112,0.08)', text: '#8b8070', border: 'rgba(139,128,112,0.15)' };
}

const DICT_COLORS: Record<string, string> = {
  RangjungYeshe: '#c084fc',
  MonlamTibEng: '#60a5fa',
  DagYig: '#34d399',
  BodDag: '#fbbf24',
};

function dictColor(dictName: string): string {
  for (const [k, v] of Object.entries(DICT_COLORS)) {
    if (dictName.includes(k)) return v;
  }
  return '#d4a853';
}

export function TokenDisplay({
  tokens,
  dictEntries = {},
}: {
  tokens: TokenResponse[];
  dictEntries?: Record<string, LookupEntry[]>;
}) {
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = () => setTooltip(null);
    window.addEventListener('scroll', handler, true);
    return () => window.removeEventListener('scroll', handler, true);
  }, []);

  function showTooltip(e: React.MouseEvent<HTMLSpanElement>, token: TokenResponse) {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const x = Math.min(rect.left + window.scrollX, window.innerWidth - 320);
    const y = rect.bottom + window.scrollY + 4;
    setTooltipPos({ x, y });
    setTooltip({
      token: token.token,
      pos: token.pos,
      pos_zh: token.pos_zh,
      case_name: token.case_name ?? undefined,
      case_desc: token.case_desc ?? undefined,
      dictEntries: dictEntries[token.token] ?? [],
    });
  }

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      {/* Tibetan syllable ribbon */}
      <div
        className="tibetan"
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.2rem',
          lineHeight: '2.4',
          fontSize: '1.15rem',
        }}
      >
        {tokens.map((t, i) => {
          const isSep = t.token === '་' || t.token === '།';
          const isCase = t.is_case_particle;
          const style = getPosChipStyle(t.pos);
          const hasDict = (dictEntries[t.token]?.length ?? 0) > 0;

          if (isSep) {
            return (
              <span
                key={i}
                style={{
                  color: 'rgba(139,128,112,0.35)',
                  fontSize: '0.75rem',
                  lineHeight: '2.4',
                  userSelect: 'none',
                  marginLeft: '-0.1rem',
                  fontFamily: 'var(--font-sans)',
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
                cursor: 'pointer',
                position: 'relative',
              }}
              onMouseEnter={e => showTooltip(e, t)}
              onMouseLeave={() => setTooltip(null)}
            >
              {/* Dict dot indicator */}
              {hasDict && !isCase && (
                <span
                  style={{
                    position: 'absolute',
                    top: '-3px',
                    right: '-3px',
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    background: '#22c55e',
                    boxShadow: '0 0 4px rgba(34,197,94,0.6)',
                  }}
                />
              )}
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

      {/* Rich tooltip */}
      {tooltip && (
        <div
          style={{
            position: 'fixed',
            left: tooltipPos.x,
            top: tooltipPos.y,
            width: '300px',
            background: 'rgba(12,12,18,0.95)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '12px',
            padding: '0.875rem 1rem',
            zIndex: 9999,
            boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
            animation: 'fade-in 0.12s ease-out',
          }}
        >
          {/* Tibetan headword */}
          <div className="tibetan" style={{ fontWeight: 700, fontSize: '1.25rem', color: '#e8e0d0', marginBottom: '0.5rem', lineHeight: '1.3' }}>
            {tooltip.token}
          </div>

          {/* POS + case */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap', marginBottom: '0.35rem' }}>
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.62rem',
                color: '#8b8070',
                background: 'rgba(255,255,255,0.06)',
                padding: '0.1rem 0.35rem',
                borderRadius: '4px',
              }}
            >
              {tooltip.pos}
            </span>
            <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', fontWeight: 500, color: '#e8e0d0' }}>
              {tooltip.pos_zh}
            </span>
            {tooltip.case_name && (
              <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', color: '#d97706', fontWeight: 600 }}>
                ★ {tooltip.case_name}
              </span>
            )}
          </div>

          {/* Case description */}
          {tooltip.case_desc && (
            <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', color: '#8b8070', lineHeight: 1.5, marginBottom: '0.3rem' }}>
              {tooltip.case_desc}
            </div>
          )}

          {/* Dictionary entries */}
          {tooltip.dictEntries.length > 0 && (
            <div style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.62rem', fontWeight: 600, color: '#4a4540', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '0.35rem' }}>
                词典释义
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', maxHeight: '180px', overflowY: 'auto' }}>
                {tooltip.dictEntries.slice(0, 5).map((e, j) => (
                  <div key={j} style={{ display: 'flex', gap: '0.5rem' }}>
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.6rem',
                        color: dictColor(e.dict_name),
                        background: `${dictColor(e.dict_name)}18`,
                        padding: '0.1rem 0.35rem',
                        borderRadius: '4px',
                        border: `1px solid ${dictColor(e.dict_name)}33`,
                        flexShrink: 0,
                        minWidth: '4rem',
                        textAlign: 'center',
                      }}
                    >
                      {e.dict_name.replace('RangjungYeshe','RY').replace('MonlamTibEng','MT')}
                    </span>
                    <span
                      style={{
                        fontFamily: 'var(--font-sans)',
                        fontSize: '0.73rem',
                        color: 'rgba(232,224,208,0.75)',
                        lineHeight: 1.5,
                      }}
                    >
                      {e.definition.length > 100 ? e.definition.slice(0, 100) + '…' : e.definition}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {tooltip.dictEntries.length === 0 && (
            <div style={{ marginTop: '0.4rem', paddingTop: '0.4rem', borderTop: '1px solid rgba(255,255,255,0.06)', fontFamily: 'var(--font-sans)', fontSize: '0.7rem', color: '#4a4540', fontStyle: 'italic' }}>
              词典中未收录此词
            </div>
          )}
        </div>
      )}
    </div>
  );
}
