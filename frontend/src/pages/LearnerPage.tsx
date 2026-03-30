import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CASE_PARTICLES } from '../components/CaseParticleGuide';

const POS_REFS = [
  { tag: 'n.count', tag_zh: '普通名词', desc: '指一般事物的名称', accent: '#a78bfa' },
  { tag: 'n.prop', tag_zh: '专有名词', desc: '人名、地名、书名等专有名称', accent: '#a78bfa' },
  { tag: 'n.rel', tag_zh: '关系名词', desc: '表示相对关系的名词', accent: '#a78bfa' },
  { tag: 'n.mass', tag_zh: '物质名词', desc: '不可数物质', accent: '#a78bfa' },
  { tag: 'v.past', tag_zh: '过去时', desc: '表示过去发生的动作', accent: '#2dd4bf' },
  { tag: 'v.pres', tag_zh: '现在时', desc: '表示现在进行的动作', accent: '#2dd4bf' },
  { tag: 'v.fut', tag_zh: '将来时', desc: '表示将来发生的动作', accent: '#2dd4bf' },
  { tag: 'v.invar', tag_zh: '不变词', desc: '不随时态变化的动词', accent: '#2dd4bf' },
  { tag: 'v.aux', tag_zh: '助动词', desc: '辅助主要动词', accent: '#2dd4bf' },
  { tag: 'v.cop', tag_zh: '系词', desc: '"是"、"为"', accent: '#2dd4bf' },
  { tag: 'adj', tag_zh: '形容词', desc: '修饰名词', accent: '#4ade80' },
  { tag: 'cv.*', tag_zh: '副动词', desc: '连接动词', accent: '#60a5fa' },
  { tag: 'cl.*', tag_zh: '小品词', desc: '语气词', accent: '#fb923c' },
  { tag: 'd.*', tag_zh: '限定词', desc: '指示词', accent: '#f472b6' },
  { tag: 'num.*', tag_zh: '数词', desc: '数量词', accent: '#fbbf24' },
  { tag: 'adv.*', tag_zh: '副词', desc: '修饰动词/形容词', accent: '#60a5fa' },
  { tag: 'punc', tag_zh: '标点', desc: '音节分隔符 ་ 或句末 །', accent: '#6b7280' },
  { tag: 'neg', tag_zh: '否定词', desc: '"不"、"没有"', accent: '#f87171' },
  { tag: 'skt', tag_zh: '梵语音译', desc: '来自梵语的音译词', accent: '#d4a853' },
];

type Tab = 'particles' | 'pos' | 'flashcard';

export function LearnerPage() {
  const [tab, setTab] = useState<Tab>('particles');

  return (
    <div className="max-w-6xl mx-auto px-6 pt-8 pb-12">
      {/* Page header */}
      <div
        className="mb-8 animate-fade-up"
        style={{ animationDelay: '0ms' }}
      >
        <div
          style={{
            fontFamily: 'var(--font-display)',
            fontSize: 'clamp(1.6rem, 4vw, 2.2rem)',
            fontWeight: 700,
            color: '#e8e0d0',
            lineHeight: 1.1,
            letterSpacing: '-0.01em',
          }}
        >
          学习助手
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
          格助词速查 · POS 标签参考 · 翻转卡片
        </div>
        {/* Gold rule */}
        <div
          style={{
            marginTop: '0.75rem',
            height: '1px',
            background: 'linear-gradient(to right, rgba(212,168,83,0.4), transparent)',
            maxWidth: '280px',
          }}
        />
      </div>

      {/* Tab bar */}
      <div
        className="flex gap-0 mb-8 animate-fade-up"
        style={{ animationDelay: '60ms' }}
      >
        {(
          [
            { key: 'particles', label: '格助词表', sub: '8+2 经典格' },
            { key: 'pos', label: 'POS 标签', sub: '36 类' },
            { key: 'flashcard', label: '翻转卡片', sub: '记忆练习' },
          ] as const
        ).map(({ key, label, sub }) => {
          const isActive = tab === key;
          return (
            <button
              key={key}
              onClick={() => setTab(key)}
              style={{
                position: 'relative',
                padding: '0.625rem 1.25rem 0.625rem 1.25rem',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-start',
                gap: '0.1rem',
                borderBottom: isActive
                  ? '2px solid #c94a4a'
                  : '2px solid transparent',
                transition: 'all 0.2s ease',
                color: isActive ? '#e8e0d0' : 'rgba(139,128,112,0.6)',
              }}
            >
              <span
                style={{
                  fontFamily: 'var(--font-sans)',
                  fontWeight: isActive ? 600 : 400,
                  fontSize: '0.875rem',
                  letterSpacing: '0.02em',
                  color: 'inherit',
                }}
              >
                {label}
              </span>
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.65rem',
                  color: isActive ? '#c94a4a' : 'rgba(139,128,112,0.4)',
                  letterSpacing: '0.04em',
                }}
              >
                {sub}
              </span>
            </button>
          );
        })}
      </div>

      {tab === 'particles' && <ParticleTable />}
      {tab === 'pos' && <PosReference />}
      {tab === 'flashcard' && <FlashcardMode />}
    </div>
  );
}

// ── Particle Table ──────────────────────────────────────────────────────────

function ParticleTable() {
  const [expanded, setExpanded] = useState<string | null>(null);
  const navigate = useNavigate();

  return (
    <div
      className="space-y-2 animate-fade-up"
      style={{ animationDelay: '100ms' }}
    >
      {CASE_PARTICLES.map((p, i) => {
        const isOpen = expanded === p.tag;
        return (
          <div
            key={p.tag}
            style={{
              background: isOpen
                ? 'rgba(255,255,255,0.05)'
                : 'rgba(255,255,255,0.03)',
              border: `1px solid ${
                isOpen ? 'rgba(212,168,83,0.3)' : 'rgba(255,255,255,0.06)'
              }`,
              borderRadius: '12px',
              overflow: 'hidden',
              transition: 'all 0.25s ease',
              boxShadow: isOpen
                ? '0 4px 20px rgba(212,168,83,0.06), inset 0 1px 0 rgba(255,255,255,0.05)'
                : '0 2px 8px rgba(0,0,0,0.2)',
            }}
          >
            {/* Row header */}
            <button
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                textAlign: 'left',
              }}
              onClick={() => setExpanded(isOpen ? null : p.tag)}
            >
              {/* Index */}
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.6rem',
                  color: 'rgba(139,128,112,0.35)',
                  width: '1.5rem',
                  flexShrink: 0,
                }}
              >
                {String(i + 1).padStart(2, '0')}
              </span>

              {/* Tibetan */}
              <span
                className="tibetan"
                style={{
                  fontFamily: 'var(--font-tibetan)',
                  fontWeight: 700,
                  fontSize: '1.05rem',
                  color: '#d97706',
                  minWidth: '4rem',
                  flexShrink: 0,
                  letterSpacing: '0.01em',
                }}
              >
                {p.tibetan}
              </span>

              {/* Name */}
              <span
                style={{
                  fontFamily: 'var(--font-sans)',
                  fontWeight: 600,
                  fontSize: '0.8rem',
                  color: '#e8e0d0',
                  minWidth: '5rem',
                  flexShrink: 0,
                }}
              >
                {p.name}
              </span>

              {/* Function */}
              <span
                style={{
                  fontFamily: 'var(--font-sans)',
                  fontSize: '0.75rem',
                  color: 'rgba(139,128,112,0.7)',
                  flex: 1,
                }}
              >
                {p.function}
                <span
                  style={{
                    color: 'rgba(212,168,83,0.6)',
                    marginLeft: '0.25rem',
                  }}
                >
                  （{p.chinese}）
                </span>
              </span>

              {/* Tag */}
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.6rem',
                  color: 'rgba(139,128,112,0.4)',
                  background: 'rgba(255,255,255,0.04)',
                  padding: '0.15rem 0.4rem',
                  borderRadius: '4px',
                  flexShrink: 0,
                }}
              >
                {p.tag}
              </span>

              {/* Chevron */}
              <span
                style={{
                  color: 'rgba(139,128,112,0.4)',
                  fontSize: '0.6rem',
                  flexShrink: 0,
                  transform: isOpen ? 'rotate(180deg)' : 'none',
                  transition: 'transform 0.2s ease',
                }}
              >
                ▼
              </span>
            </button>

            {/* Expanded detail */}
            {isOpen && (
              <div
                style={{
                  borderTop: '1px solid rgba(255,255,255,0.05)',
                  padding: '0.875rem 1rem 1rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.75rem',
                }}
              >
                {/* English + Pinyin */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                  <div>
                    <div
                      style={{
                        fontFamily: 'var(--font-sans)',
                        fontSize: '0.65rem',
                        fontWeight: 600,
                        letterSpacing: '0.08em',
                        textTransform: 'uppercase',
                        color: 'rgba(139,128,112,0.5)',
                        marginBottom: '0.25rem',
                      }}
                    >
                      English
                    </div>
                    <div
                      style={{
                        fontFamily: 'var(--font-sans)',
                        fontSize: '0.8rem',
                        color: '#e8e0d0',
                      }}
                    >
                      {p.english}
                    </div>
                  </div>
                  <div>
                    <div
                      style={{
                        fontFamily: 'var(--font-sans)',
                        fontSize: '0.65rem',
                        fontWeight: 600,
                        letterSpacing: '0.08em',
                        textTransform: 'uppercase',
                        color: 'rgba(139,128,112,0.5)',
                        marginBottom: '0.25rem',
                      }}
                    >
                      拼音
                    </div>
                    <div
                      style={{
                        fontFamily: 'var(--font-sans)',
                        fontSize: '0.8rem',
                        color: '#d4a853',
                        fontStyle: 'italic',
                      }}
                    >
                      {p.pinyin}
                    </div>
                  </div>
                </div>

                {/* Example */}
                <div
                  style={{
                    background: 'rgba(0,0,0,0.25)',
                    border: '1px solid rgba(255,255,255,0.05)',
                    borderRadius: '10px',
                    padding: '0.75rem 0.875rem',
                    borderLeft: '2px solid #d97706',
                  }}
                >
                  <div
                    style={{
                      fontFamily: 'var(--font-sans)',
                      fontSize: '0.65rem',
                      fontWeight: 600,
                      letterSpacing: '0.08em',
                      textTransform: 'uppercase',
                      color: 'rgba(139,128,112,0.5)',
                      marginBottom: '0.375rem',
                    }}
                  >
                    例句
                  </div>
                  <div
                    className="tibetan"
                    style={{
                      fontFamily: 'var(--font-tibetan)',
                      fontSize: '1.05rem',
                      color: '#e8e0d0',
                      fontWeight: 700,
                      marginBottom: '0.375rem',
                    }}
                  >
                    {p.example}
                  </div>
                  <div
                    style={{
                      fontFamily: 'var(--font-sans)',
                      fontSize: '0.8rem',
                      color: 'rgba(139,128,112,0.8)',
                      marginBottom: '0.25rem',
                    }}
                  >
                    {p.example_meaning}
                  </div>
                  <div
                    style={{
                      fontFamily: 'var(--font-sans)',
                      fontSize: '0.72rem',
                      color: 'rgba(139,128,112,0.45)',
                      fontStyle: 'italic',
                    }}
                  >
                    {p.example_note}
                  </div>
                </div>

                {/* CTA */}
                <button
                  onClick={() => navigate('/', { state: { text: p.example } })}
                  style={{
                    background: 'none',
                    border: '1px solid rgba(201,74,74,0.3)',
                    borderRadius: '8px',
                    padding: '0.4rem 0.875rem',
                    cursor: 'pointer',
                    fontSize: '0.75rem',
                    color: '#c94a4a',
                    fontFamily: 'var(--font-sans)',
                    width: 'fit-content',
                    transition: 'all 0.2s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(201,74,74,0.1)';
                    e.currentTarget.style.borderColor = 'rgba(201,74,74,0.5)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'none';
                    e.currentTarget.style.borderColor = 'rgba(201,74,74,0.3)';
                  }}
                >
                  → 用分析器分析这个例句
                </button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── POS Reference ───────────────────────────────────────────────────────────

function PosReference() {
  return (
    <div
      className="grid grid-cols-1 sm:grid-cols-2 gap-3 animate-fade-up"
      style={{ animationDelay: '100ms' }}
    >
      {POS_REFS.map((p) => (
        <div
          key={p.tag}
          style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: '12px',
            padding: '0.875rem 1rem',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '0.75rem',
          }}
        >
          {/* Accent dot */}
          <div
            style={{
              width: '3px',
              borderRadius: '2px',
              background: p.accent,
              alignSelf: 'stretch',
              flexShrink: 0,
              minHeight: '2rem',
              opacity: 0.7,
            }}
          />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                marginBottom: '0.25rem',
              }}
            >
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.7rem',
                  fontWeight: 700,
                  color: p.accent,
                }}
              >
                {p.tag}
              </span>
              <span
                style={{
                  fontFamily: 'var(--font-sans)',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  color: '#e8e0d0',
                }}
              >
                {p.tag_zh}
              </span>
            </div>
            <div
              style={{
                fontFamily: 'var(--font-sans)',
                fontSize: '0.72rem',
                color: 'rgba(139,128,112,0.6)',
                lineHeight: 1.4,
              }}
            >
              {p.desc}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Flashcard ───────────────────────────────────────────────────────────────

function FlashcardMode() {
  const [idx, setIdx] = useState(() => Math.floor(Math.random() * CASE_PARTICLES.length));
  const [flipped, setFlipped] = useState(false);
  const navigate = useNavigate();

  function next() {
    setFlipped(false);
    setIdx((i) => (i + 1) % CASE_PARTICLES.length);
  }

  function shuffle() {
    setFlipped(false);
    setIdx(Math.floor(Math.random() * CASE_PARTICLES.length));
  }

  const p = CASE_PARTICLES[idx];

  return (
    <div
      className="animate-fade-up"
      style={{ animationDelay: '100ms' }}
    >
      {/* Progress */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '0.5rem',
          marginBottom: '2rem',
        }}
      >
        <div
          style={{
            height: '2px',
            borderRadius: '1px',
            background: 'rgba(255,255,255,0.06)',
            flex: 1,
            maxWidth: '160px',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${((idx + 1) / CASE_PARTICLES.length) * 100}%`,
              background: 'linear-gradient(to right, #c94a4a, #d4a853)',
              borderRadius: '1px',
              transition: 'width 0.4s ease',
            }}
          />
        </div>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.7rem',
            color: 'rgba(139,128,112,0.5)',
          }}
        >
          {idx + 1} / {CASE_PARTICLES.length}
        </span>
      </div>

      {/* 3D Card */}
      <div
        style={{
          perspective: '1200px',
          maxWidth: '460px',
          margin: '0 auto',
        }}
      >
        <div
          onClick={() => setFlipped((f) => !f)}
          style={{
            position: 'relative',
            transformStyle: 'preserve-3d',
            transition: 'transform 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
            transform: flipped ? 'rotateY(180deg)' : 'rotateY(0deg)',
            cursor: 'pointer',
          }}
        >
          {/* Front */}
          <div
            style={{
              backfaceVisibility: 'hidden',
              WebkitBackfaceVisibility: 'hidden',
              borderRadius: '20px',
              padding: '3rem 2.5rem',
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.08)',
              boxShadow: '0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '1rem',
              minHeight: '260px',
            }}
          >
            {/* Decorative ring */}
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%,-50%)',
                width: '180px',
                height: '180px',
                borderRadius: '50%',
                border: '1px solid rgba(212,168,83,0.06)',
                pointerEvents: 'none',
              }}
            />
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%,-50%)',
                width: '140px',
                height: '140px',
                borderRadius: '50%',
                border: '1px solid rgba(212,168,83,0.04)',
                pointerEvents: 'none',
              }}
            />
            {/* Tibetan */}
            <span
              className="tibetan"
              style={{
                fontFamily: 'var(--font-tibetan)',
                fontWeight: 700,
                fontSize: 'clamp(2.5rem, 8vw, 3.5rem)',
                color: '#d97706',
                letterSpacing: '0.05em',
                lineHeight: 1,
                textShadow: '0 0 30px rgba(217,119,6,0.25)',
              }}
            >
              {p.tibetan}
            </span>
            {/* Tag */}
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.65rem',
                color: 'rgba(139,128,112,0.4)',
                background: 'rgba(255,255,255,0.04)',
                padding: '0.2rem 0.6rem',
                borderRadius: '99px',
                letterSpacing: '0.06em',
              }}
            >
              {p.tag}
            </span>
            {/* Tap hint */}
            <span
              style={{
                fontFamily: 'var(--font-sans)',
                fontSize: '0.72rem',
                color: 'rgba(139,128,112,0.35)',
                marginTop: '0.5rem',
                letterSpacing: '0.06em',
              }}
            >
              点击翻转
            </span>
          </div>

          {/* Back */}
          <div
            style={{
              backfaceVisibility: 'hidden',
              WebkitBackfaceVisibility: 'hidden',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              borderRadius: '20px',
              padding: '3rem 2.5rem',
              background: 'rgba(201,74,74,0.08)',
              border: '1px solid rgba(201,74,74,0.2)',
              boxShadow: '0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05)',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.625rem',
              minHeight: '260px',
              transform: 'rotateY(180deg)',
            }}
          >
            <span
              style={{
                fontFamily: 'var(--font-display)',
                fontSize: 'clamp(1.6rem, 5vw, 2rem)',
                fontWeight: 700,
                color: '#e8e0d0',
                lineHeight: 1.1,
              }}
            >
              {p.name}
            </span>
            <span
              style={{
                fontFamily: 'var(--font-sans)',
                fontSize: '0.8rem',
                color: '#d97706',
                fontWeight: 500,
              }}
            >
              {p.function}（{p.chinese}）
            </span>
            <span
              style={{
                fontFamily: 'var(--font-sans)',
                fontSize: '0.75rem',
                color: 'rgba(139,128,112,0.6)',
                fontStyle: 'italic',
              }}
            >
              {p.english}
            </span>
            <div
              style={{
                width: '40px',
                height: '1px',
                background: 'rgba(201,74,74,0.3)',
                margin: '0.5rem 0',
              }}
            />
            <span
              style={{
                fontFamily: 'var(--font-sans)',
                fontSize: '0.72rem',
                color: 'rgba(139,128,112,0.45)',
              }}
            >
              拼音: {p.pinyin}
            </span>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '0.625rem',
          marginTop: '2rem',
          flexWrap: 'wrap',
        }}
      >
        <button
          onClick={shuffle}
          style={{
            padding: '0.5rem 1rem',
            borderRadius: '10px',
            border: '1px solid rgba(255,255,255,0.08)',
            background: 'rgba(255,255,255,0.04)',
            color: 'rgba(139,128,112,0.7)',
            fontSize: '0.8rem',
            fontFamily: 'var(--font-sans)',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = 'rgba(212,168,83,0.3)';
            e.currentTarget.style.color = '#d4a853';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
            e.currentTarget.style.color = 'rgba(139,128,112,0.7)';
          }}
        >
          🔀 随机
        </button>
        <button
          onClick={next}
          style={{
            padding: '0.5rem 1rem',
            borderRadius: '10px',
            border: '1px solid rgba(255,255,255,0.08)',
            background: 'rgba(255,255,255,0.04)',
            color: 'rgba(139,128,112,0.7)',
            fontSize: '0.8rem',
            fontFamily: 'var(--font-sans)',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = 'rgba(201,74,74,0.3)';
            e.currentTarget.style.color = '#c94a4a';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
            e.currentTarget.style.color = 'rgba(139,128,112,0.7)';
          }}
        >
          → 下一个
        </button>
        <button
          onClick={() => navigate('/', { state: { text: p.example } })}
          style={{
            padding: '0.5rem 1rem',
            borderRadius: '10px',
            border: '1px solid rgba(201,74,74,0.3)',
            background: 'rgba(201,74,74,0.08)',
            color: '#c94a4a',
            fontSize: '0.8rem',
            fontFamily: 'var(--font-sans)',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(201,74,74,0.15)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(201,74,74,0.08)';
          }}
        >
          📖 分析例句
        </button>
      </div>
    </div>
  );
}
