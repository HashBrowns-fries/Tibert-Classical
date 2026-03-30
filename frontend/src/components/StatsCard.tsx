interface StatsCardProps {
  stats: {
    nouns: number;
    verbs: number;
    case_particles: number;
    syllable_count: number;
  };
}

const STAT_ITEMS = [
  {
    key: 'nouns' as const,
    label: '名词',
    accent: '#a78bfa',
    bg: 'rgba(124,58,237,0.1)',
    border: 'rgba(124,58,237,0.2)',
  },
  {
    key: 'verbs' as const,
    label: '动词',
    accent: '#2dd4bf',
    bg: 'rgba(13,148,136,0.1)',
    border: 'rgba(13,148,136,0.2)',
  },
  {
    key: 'case_particles' as const,
    label: '格助词 ★',
    accent: '#d97706',
    bg: 'rgba(217,119,6,0.12)',
    border: 'rgba(217,119,6,0.25)',
    glow: '0 0 10px rgba(217,119,6,0.1)',
  },
  {
    key: 'syllable_count' as const,
    label: '音节',
    accent: 'rgba(139,128,112,0.7)',
    bg: 'rgba(255,255,255,0.03)',
    border: 'rgba(255,255,255,0.07)',
  },
];

export function StatsCard({ stats }: StatsCardProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {STAT_ITEMS.map(({ key, label, accent, bg, border, glow }) => (
        <div
          key={key}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            borderRadius: '99px',
            padding: '0.3rem 0.75rem 0.3rem 0.5rem',
            background: bg,
            border: `1px solid ${border}`,
            boxShadow: glow ?? 'none',
          }}
        >
          {/* Number */}
          <span
            style={{
              fontFamily: 'var(--font-display)',
              fontSize: '1rem',
              fontWeight: 700,
              color: accent,
              lineHeight: 1,
              minWidth: '1.4em',
              textAlign: 'center',
            }}
          >
            {stats[key]}
          </span>
          {/* Label */}
          <span
            style={{
              fontFamily: 'var(--font-sans)',
              fontSize: '0.72rem',
              fontWeight: 500,
              color: accent,
              opacity: 0.85,
              whiteSpace: 'nowrap',
            }}
          >
            {label}
          </span>
        </div>
      ))}
    </div>
  );
}
