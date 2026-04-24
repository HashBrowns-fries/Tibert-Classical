import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CASE_PARTICLES } from '../components/CaseParticleGuide';
import { useLearner } from '../hooks/useLearner';
import type { CaseParticleDrill, SRSGrade } from '../types/api';

// ── Color accent for case particles ─────────────────────────────────────────
const CASE_COLORS: Record<string, string> = {
  'case.gen':  '#d97706',
  'case.agn':  '#dc2626',
  'case.all':  '#2563eb',
  'case.abl':  '#7c3aed',
  'case.ela':  '#db2777',
  'case.ass':  '#059669',
  'case.term': '#0891b2',
  'case.loc':  '#65a30d',
};

function caseColor(tag: string): string {
  return CASE_COLORS[tag] ?? '#9ca3af';
}

type Tab = 'particles' | 'drill' | 'verbs' | 'flashcard';

// ── SRS grade buttons ─────────────────────────────────────────────────────────
function GradeButtons({ onGrade }: { onGrade: (g: SRSGrade) => void }) {
  const grades: { g: SRSGrade; label: string; color: string }[] = [
    { g: 0, label: '✗ 完全不认识', color: '#dc2626' },
    { g: 2, label: '○ 模糊', color: '#f59e0b' },
    { g: 3, label: '◑ 基本认识', color: '#3b82f6' },
    { g: 4, label: '◕ 认识', color: '#22c55e' },
    { g: 5, label: '★ 非常熟练', color: '#10b981' },
  ];
  return (
    <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'center' }}>
      {grades.map(({ g, label, color }) => (
        <button
          key={g}
          onClick={() => onGrade(g)}
          style={{
            padding: '0.4rem 0.875rem',
            borderRadius: '10px',
            border: `1px solid ${color}44`,
            background: `${color}11`,
            color,
            fontSize: '0.75rem',
            fontFamily: 'var(--font-sans)',
            cursor: 'pointer',
            transition: 'all 0.15s ease',
          }}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

// ── Main LearnerPage ───────────────────────────────────────────────────────────
export function LearnerPage() {
  const [tab, setTab] = useState<Tab>('particles');
  const {
    particles, verbs, stats, loading, error,
    drillLoading, currentDrill, drillAnswer, drillFeedback, drillScore,
    fetchParticles, fetchVerbs, generateDrill, submitDrill, recordReview,
    dueItems,
  } = useLearner();

  useEffect(() => { fetchParticles(); }, [fetchParticles]);

  const navigate = useNavigate();

  // SRS summary badge
  const srsSummary = (
    <div style={{
      display: 'flex', gap: '0.75rem', alignItems: 'center',
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.07)',
      borderRadius: '12px', padding: '0.5rem 1rem',
    }}>
      <span style={{ fontSize: '0.7rem', color: 'rgba(139,128,112,0.5)', fontFamily: 'var(--font-sans)' }}>
        间隔重复
      </span>
      <span style={{ fontSize: '0.7rem', color: '#22c55e', fontFamily: 'var(--font-mono)' }}>
        ✓ 已掌握 {stats.mastered}
      </span>
      {stats.due > 0 && (
        <span style={{
          fontSize: '0.7rem', color: '#f59e0b', fontFamily: 'var(--font-mono)',
          background: '#f59e0b22', padding: '0.1rem 0.4rem', borderRadius: '6px',
        }}>
          待复习 {stats.due}
        </span>
      )}
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto px-6 pt-8 pb-12">
      {/* Header */}
      <div className="mb-6 animate-fade-up">
        <div style={{
          fontFamily: 'var(--font-display)',
          fontSize: 'clamp(1.6rem, 4vw, 2.2rem)',
          fontWeight: 700, color: '#e8e0d0',
          lineHeight: 1.1, letterSpacing: '-0.01em',
        }}>
          学习助手
          <span style={{ fontSize: '0.8rem', color: 'rgba(139,128,112,0.5)', marginLeft: '0.75rem', fontFamily: 'var(--font-sans)' }}>
            ་{stats.total.toLocaleString()} 句 · {stats.words.toLocaleString()} 词
          </span>
        </div>
        <div style={{ fontSize: '0.75rem', color: 'rgba(139,128,112,0.5)', marginTop: '0.25rem', fontFamily: 'var(--font-sans)' }}>
          格助词练习 · 动词变位 · 间隔重复
        </div>
        <div style={{ marginTop: '0.75rem', height: '1px', background: 'linear-gradient(to right, rgba(212,168,83,0.4), transparent)', maxWidth: '280px' }} />
      </div>

      {/* SRS summary + tab bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
        {srsSummary}
      </div>

      {/* Tab bar */}
      <div className="flex gap-0 mb-8 animate-fade-up" style={{ animationDelay: '60ms' }}>
        {([
          { key: 'particles', label: '格助词表', sub: '8+2 经典格' },
          { key: 'drill', label: '互动练习', sub: '格助词识别' },
          { key: 'verbs', label: '动词变位', sub: '语料库例句' },
          { key: 'flashcard', label: '闪卡', sub: '间隔重复' },
        ] as const).map(({ key, label, sub }) => {
          const isActive = tab === key;
          return (
            <button
              key={key}
              onClick={() => { setTab(key); if (key === 'verbs') fetchVerbs(); }}
              style={{
                position: 'relative',
                padding: '0.625rem 1.25rem 0.625rem 1.25rem',
                background: 'none', border: 'none', cursor: 'pointer',
                display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '0.1rem',
                borderBottom: isActive ? '2px solid #c94a4a' : '2px solid transparent',
                transition: 'all 0.2s ease', color: isActive ? '#e8e0d0' : 'rgba(139,128,112,0.6)',
              }}
            >
              <span style={{ fontFamily: 'var(--font-sans)', fontWeight: isActive ? 600 : 400, fontSize: '0.875rem', color: 'inherit' }}>
                {label}
                {key === 'drill' && dueItems.length > 0 && (
                  <span style={{
                    marginLeft: '0.3rem', background: '#c94a4a', color: 'white',
                    fontSize: '0.6rem', padding: '0.05rem 0.35rem', borderRadius: '99px',
                  }}>{dueItems.length}</span>
                )}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: isActive ? '#c94a4a' : 'rgba(139,128,112,0.4)' }}>
                {sub}
              </span>
            </button>
          );
        })}
      </div>

      {loading && (
        <div style={{ textAlign: 'center', padding: '2rem', color: 'rgba(139,128,112,0.5)', fontFamily: 'var(--font-sans)', fontSize: '0.85rem' }}>
          加载语料数据…
        </div>
      )}

      {error && (
        <div style={{ marginBottom: '1rem', padding: '0.75rem 1rem', borderRadius: '0.75rem',
          background: 'rgba(220,38,38,0.1)', border: '1px solid rgba(220,38,38,0.2)',
          color: '#f87171', fontSize: '0.85rem', fontFamily: 'var(--font-sans)' }}>
          ⚠️ {error}
        </div>
      )}

      {!loading && tab === 'particles' && <ParticleTable particles={particles} navigate={navigate} />}
      {!loading && tab === 'drill' && (
        <DrillTab
          particles={particles}
          loading={drillLoading}
          currentDrill={currentDrill}
          drillAnswer={drillAnswer}
          drillFeedback={drillFeedback}
          drillScore={drillScore}
          onGenerate={generateDrill}
          onSubmit={submitDrill}
          onGrade={recordReview}
        />
      )}
      {!loading && tab === 'verbs' && <VerbExplorer verbs={verbs} navigate={navigate} />}
      {!loading && tab === 'flashcard' && (
        <SRSCards
          particles={particles}
          dueItems={dueItems}
          onGrade={recordReview}
        />
      )}
    </div>
  );
}

// ── Particle Table ─────────────────────────────────────────────────────────────
function ParticleTable({ particles, navigate }: { particles: CaseParticleDrill[]; navigate: ReturnType<typeof useNavigate> }) {
  const [expanded, setExpanded] = useState<string | null>(null);

  // Use API data if available, fallback to static CASE_PARTICLES
  const displayParticles = particles.length > 0 ? particles : CASE_PARTICLES.map((p) => ({
    tag: p.tag, tibetan: p.tibetan.split(' ')[0], name: p.name, english: p.english,
    chinese: p.chinese, function: p.function, count: 0, examples: [],
  }));

  return (
    <div className="space-y-2 animate-fade-up" style={{ animationDelay: '100ms' }}>
      {displayParticles.map((p, i) => {
        const isOpen = expanded === p.tag;
        const accent = caseColor(p.tag);
        const exampleCount = 'count' in p ? (p as CaseParticleDrill).count : 0;
        return (
          <div
            key={p.tag}
            style={{
              background: isOpen ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.03)',
              border: `1px solid ${isOpen ? accent + '44' : 'rgba(255,255,255,0.06)'}`,
              borderRadius: '12px', overflow: 'hidden',
              transition: 'all 0.25s ease',
              boxShadow: isOpen ? `0 4px 20px ${accent}11, inset 0 1px 0 rgba(255,255,255,0.05)` : '0 2px 8px rgba(0,0,0,0.2)',
            }}
          >
            <button
              style={{
                width: '100%', padding: '0.75rem 1rem', background: 'none',
                border: 'none', cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: '0.75rem', textAlign: 'left',
              }}
              onClick={() => setExpanded(isOpen ? null : p.tag)}
            >
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'rgba(139,128,112,0.35)', width: '1.5rem', flexShrink: 0 }}>
                {String(i + 1).padStart(2, '0')}
              </span>
              <span className="tibetan" style={{
                fontFamily: 'var(--font-tibetan)', fontWeight: 700, fontSize: '1.05rem',
                color: accent, minWidth: '4rem', flexShrink: 0, letterSpacing: '0.01em',
              }}>
                {p.tibetan.split(' / ')[0]}
              </span>
              <span style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.8rem', color: '#e8e0d0', minWidth: '5rem', flexShrink: 0 }}>
                {p.name}
              </span>
              <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', color: 'rgba(139,128,112,0.7)', flex: 1 }}>
                {p.function}
                <span style={{ color: `${accent}99`, marginLeft: '0.25rem' }}>（{p.chinese}）</span>
              </span>
              {exampleCount > 0 && (
                <span style={{
                  fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: `${accent}88`,
                  background: `${accent}11`, padding: '0.1rem 0.4rem', borderRadius: '6px', flexShrink: 0,
                }}>
                  {exampleCount.toLocaleString()} 句
                </span>
              )}
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'rgba(139,128,112,0.4)', background: 'rgba(255,255,255,0.04)', padding: '0.15rem 0.4rem', borderRadius: '4px', flexShrink: 0 }}>
                {p.tag}
              </span>
              <span style={{ color: 'rgba(139,128,112,0.4)', fontSize: '0.6rem', flexShrink: 0, transform: isOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s ease' }}>
                ▼
              </span>
            </button>

            {isOpen && (
              <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', padding: '0.875rem 1rem 1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                  <div>
                    <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.65rem', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'rgba(139,128,112,0.5)', marginBottom: '0.25rem' }}>English</div>
                    <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', color: '#e8e0d0' }}>{p.english}</div>
                  </div>
                  <div>
                    <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.65rem', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'rgba(139,128,112,0.5)', marginBottom: '0.25rem' }}>功能</div>
                    <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', color: '#d4a853', fontStyle: 'italic' }}>{p.function}（{p.chinese}）</div>
                  </div>
                </div>
                <button
                  onClick={() => navigate('/', { state: { text: p.examples[0]?.sentence ?? '' } })}
                  style={{
                    background: 'none', border: `1px solid ${accent}44`, borderRadius: '8px',
                    padding: '0.4rem 0.875rem', cursor: 'pointer', fontSize: '0.75rem',
                    color: accent, fontFamily: 'var(--font-sans)', width: 'fit-content',
                    transition: 'all 0.2s ease',
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

// ── Corpus Drill Tab ────────────────────────────────────────────────────────────
function DrillTab({
  particles, loading, currentDrill, drillAnswer, drillFeedback, drillScore,
  onGenerate, onSubmit, onGrade,
}: {
  particles: CaseParticleDrill[];
  loading: boolean;
  currentDrill: ReturnType<typeof useLearner>['currentDrill'];
  drillAnswer: string;
  drillFeedback: string | null;
  drillScore: number | null;
  onGenerate: ReturnType<typeof useLearner>['generateDrill'];
  onSubmit: ReturnType<typeof useLearner>['submitDrill'];
  onGrade: (grade: SRSGrade, tagOverride?: string) => void;
}) {
  const [selectedTag, setSelectedTag] = useState<string>('');
  const [showAnswer, setShowAnswer] = useState(false);
  const [inputAnswer, setInputAnswer] = useState('');

  function handleStart() {
    setShowAnswer(false);
    setInputAnswer('');
    onGenerate('particle_identify', selectedTag || undefined);
  }

  function handleSubmit() {
    onSubmit(inputAnswer);
    setShowAnswer(true);
  }

  function handleGrade(g: SRSGrade) {
    onGrade(g);
    setShowAnswer(false);
    setInputAnswer('');
  }

  return (
    <div className="animate-fade-up" style={{ animationDelay: '100ms' }}>
      {/* Tag filter */}
      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.7rem', color: 'rgba(139,128,112,0.5)', marginBottom: '0.5rem', letterSpacing: '0.05em' }}>
          选择要练习的格助词（留空则随机）
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
          {particles.map((p) => (
            <button
              key={p.tag}
              onClick={() => setSelectedTag(selectedTag === p.tag ? '' : p.tag)}
              style={{
                padding: '0.3rem 0.75rem', borderRadius: '8px',
                border: `1px solid ${selectedTag === p.tag ? caseColor(p.tag) + 'aa' : 'rgba(255,255,255,0.08)'}`,
                background: selectedTag === p.tag ? `${caseColor(p.tag)}18` : 'rgba(255,255,255,0.03)',
                color: selectedTag === p.tag ? caseColor(p.tag) : 'rgba(139,128,112,0.6)',
                fontSize: '0.72rem', fontFamily: 'var(--font-tibetan)', cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {p.tibetan.split(' / ')[0]}
              <span style={{ marginLeft: '0.3rem', fontFamily: 'var(--font-mono)', fontSize: '0.6rem', opacity: 0.7 }}>
                {p.count > 0 ? p.count.toLocaleString() : ''}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Start button */}
      {!currentDrill && !loading && (
        <button
          onClick={handleStart}
          style={{
            width: '100%', padding: '1.25rem', borderRadius: '16px',
            background: 'rgba(201,74,74,0.12)', border: '1px solid rgba(201,74,74,0.3)',
            color: '#c94a4a', fontSize: '1rem', fontFamily: 'var(--font-display)',
            fontWeight: 700, cursor: 'pointer', transition: 'all 0.2s ease',
            letterSpacing: '0.02em',
          }}
        >
          开始练习 {selectedTag ? `( ${particles.find(p => p.tag === selectedTag)?.tibetan.split(' / ')[0]} )` : '(随机)'}
        </button>
      )}

      {/* Loading */}
      {loading && (
        <div style={{ textAlign: 'center', padding: '2.5rem', color: 'rgba(139,128,112,0.5)', fontFamily: 'var(--font-sans)', fontSize: '0.85rem' }}>
          gemma 正在生成练习…
        </div>
      )}

      {/* Drill card */}
      {currentDrill && !loading && (
        <div style={{
          background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '20px', padding: '2rem',
          boxShadow: '0 8px 40px rgba(0,0,0,0.3)',
        }}>
          {/* Question */}
          <div style={{ marginBottom: '1.5rem' }}>
            <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.7rem', color: 'rgba(139,128,112,0.5)', marginBottom: '0.5rem', letterSpacing: '0.06em' }}>
              题目类型：识别格助词
            </div>
            <div className="tibetan" style={{
              fontFamily: 'var(--font-tibetan)', fontSize: 'clamp(1.4rem, 4vw, 1.8rem)',
              color: '#e8e0d0', fontWeight: 700, lineHeight: 1.6,
              background: 'rgba(0,0,0,0.2)', borderRadius: '12px',
              padding: '1.25rem 1.5rem', textAlign: 'center',
              border: '1px solid rgba(255,255,255,0.06)',
            }}>
              {currentDrill.sentence}
            </div>
          </div>

          {/* Hint */}
          <div style={{
            fontFamily: 'var(--font-sans)', fontSize: '0.8rem', color: '#d4a853',
            marginBottom: '1rem', fontStyle: 'italic',
          }}>
            💡 {currentDrill.hint ?? '找出句子中的格助词，说明它附着在哪个词上，表示什么语法功能'}
          </div>

          {/* Answer input */}
          {!showAnswer && (
            <div>
              <textarea
                value={inputAnswer}
                onChange={(e) => setInputAnswer(e.target.value)}
                placeholder="例如：属格 གི，附着在「ཡུལ」上，表示所属关系「的」"
                style={{
                  width: '100%', minHeight: '80px', padding: '0.875rem',
                  background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '12px', color: '#e8e0d0',
                  fontFamily: 'var(--font-sans)', fontSize: '0.85rem',
                  resize: 'vertical', outline: 'none',
                  boxSizing: 'border-box',
                }}
              />
              <div style={{ display: 'flex', gap: '0.625rem', marginTop: '0.75rem', justifyContent: 'flex-end' }}>
                <button onClick={() => { setShowAnswer(true); onSubmit(inputAnswer); }}
                  style={{ padding: '0.5rem 1.25rem', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.04)', color: 'rgba(139,128,112,0.7)', fontSize: '0.8rem', fontFamily: 'var(--font-sans)', cursor: 'pointer' }}>
                  查看答案
                </button>
                <button onClick={handleSubmit}
                  disabled={!inputAnswer.trim()}
                  style={{ padding: '0.5rem 1.25rem', borderRadius: '10px', border: 'none', background: inputAnswer.trim() ? '#c94a4a' : 'rgba(201,74,74,0.3)', color: 'white', fontSize: '0.8rem', fontFamily: 'var(--font-sans)', cursor: inputAnswer.trim() ? 'pointer' : 'not-allowed', opacity: inputAnswer.trim() ? 1 : 0.5 }}>
                  提交答案
                </button>
              </div>
            </div>
          )}

          {/* Revealed answer */}
          {showAnswer && drillFeedback && (
            <div>
              {/* Correct answer box */}
              <div style={{
                background: `${caseColor(currentDrill.target)}11`,
                border: `1px solid ${caseColor(currentDrill.target)}33`,
                borderRadius: '12px', padding: '1rem 1.25rem', marginBottom: '1rem',
              }}>
                <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.65rem', color: `${caseColor(currentDrill.target)}88`, letterSpacing: '0.08em', marginBottom: '0.375rem' }}>
                  参考答案
                </div>
                <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.9rem', color: '#e8e0d0', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                  {drillAnswer || currentDrill.answer}
                </div>
              </div>

              {/* Score badge */}
              {drillScore !== null && (
                <div style={{
                  display: 'inline-block', marginBottom: '1rem',
                  padding: '0.3rem 0.875rem', borderRadius: '99px',
                  background: drillScore >= 0.7 ? '#22c55e22' : drillScore >= 0.4 ? '#f59e0b22' : '#dc262622',
                  border: `1px solid ${drillScore >= 0.7 ? '#22c55e44' : drillScore >= 0.4 ? '#f59e0b44' : '#dc262644'}`,
                  color: drillScore >= 0.7 ? '#22c55e' : drillScore >= 0.4 ? '#f59e0b' : '#dc2626',
                  fontSize: '0.8rem', fontFamily: 'var(--font-mono)', fontWeight: 700,
                }}>
                  {drillScore >= 0.7 ? '✓ 掌握' : drillScore >= 0.4 ? '◑ 还需复习' : '✗ 不认识'}
                  <span style={{ marginLeft: '0.4rem' }}>{drillScore.toFixed(1)}/1.0</span>
                </div>
              )}

              {/* Gemma feedback */}
              <div style={{
                background: 'rgba(0,0,0,0.25)', borderRadius: '12px',
                padding: '1rem 1.25rem', marginBottom: '1.25rem',
                border: '1px solid rgba(255,255,255,0.06)',
              }}>
                <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.65rem', color: 'rgba(139,128,112,0.5)', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>
                  gemma 反馈
                </div>
                <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.85rem', color: '#e8e0d0', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                  {drillFeedback}
                </div>
              </div>

              {/* Grade buttons */}
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.7rem', color: 'rgba(139,128,112,0.5)', marginBottom: '0.5rem' }}>
                  这道题对你来说有多难？
                </div>
                <GradeButtons onGrade={handleGrade} />
              </div>

              <button onClick={handleStart}
                style={{ padding: '0.5rem 1.25rem', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.04)', color: 'rgba(139,128,112,0.7)', fontSize: '0.8rem', fontFamily: 'var(--font-sans)', cursor: 'pointer' }}>
                下一题 →
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Verb Explorer Tab ───────────────────────────────────────────────────────────
function VerbExplorer({ verbs, navigate }: { verbs: ReturnType<typeof useLearner>['verbs']; navigate: ReturnType<typeof useNavigate> }) {
  const [selectedTag, setSelectedTag] = useState<string | null>(null);

  const VERB_TAG_LABELS: Record<string, { zh: string; color: string }> = {
    'v.past': { zh: '过去时', color: '#ef4444' },
    'v.pres': { zh: '现在时', color: '#3b82f6' },
    'v.fut': { zh: '将来时', color: '#22c55e' },
    'v.invar': { zh: '不变词', color: '#a78bfa' },
    'v.cop': { zh: '系词', color: '#f59e0b' },
    'v.aux': { zh: '助动词', color: '#06b6d4' },
    'v.neg': { zh: '否定动词', color: '#f43f5e' },
    'v.imp': { zh: '命令式', color: '#ec4899' },
  };

  const selectedVerb = verbs.find((v) => v.tag === selectedTag);

  return (
    <div className="animate-fade-up" style={{ animationDelay: '100ms' }}>
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', color: 'rgba(139,128,112,0.6)', marginBottom: '1rem', lineHeight: 1.6 }}>
        以下动词变形示例来自 {verbs.reduce((a, v) => a + v.count, 0).toLocaleString()} 条语料标注。点击一种时态查看语料库中的真实例句。
      </div>

      {/* Verb tag filter */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1.5rem' }}>
        {verbs.map((v) => {
          const info = VERB_TAG_LABELS[v.tag] ?? { zh: v.tag, color: '#9ca3af' };
          const isActive = selectedTag === v.tag;
          return (
            <button
              key={v.tag}
              onClick={() => setSelectedTag(isActive ? null : v.tag)}
              style={{
                padding: '0.4rem 0.875rem', borderRadius: '10px',
                border: `1px solid ${isActive ? info.color + 'aa' : 'rgba(255,255,255,0.08)'}`,
                background: isActive ? `${info.color}18` : 'rgba(255,255,255,0.03)',
                color: isActive ? info.color : 'rgba(139,128,112,0.6)',
                fontSize: '0.72rem', fontFamily: 'var(--font-mono)', cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {info.zh}
              <span style={{ marginLeft: '0.3rem', opacity: 0.6, fontSize: '0.6rem' }}>{v.count}</span>
            </button>
          );
        })}
      </div>

      {/* Verb examples */}
      {selectedVerb && (
        <div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.85rem', color: '#e8e0d0', marginBottom: '0.75rem' }}>
            <span style={{ fontFamily: 'var(--font-mono)', color: (VERB_TAG_LABELS[selectedVerb.tag] ?? { color: '#9ca3af' }).color, fontWeight: 700 }}>
              {selectedVerb.tag}
            </span>
            <span style={{ marginLeft: '0.5rem', color: 'rgba(139,128,112,0.5)' }}>
              — {selectedVerb.count.toLocaleString()} 条例句
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {selectedVerb.examples.slice(0, 20).map((ex, i) => (
              <div key={i} style={{
                background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: '10px', padding: '0.75rem 1rem',
                display: 'flex', alignItems: 'flex-start', gap: '0.75rem',
              }}>
                <span style={{ fontFamily: 'var(--font-tibetan)', fontSize: '0.95rem', color: '#d97706', fontWeight: 700, flexShrink: 0 }}>
                  {ex.form}
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="tibetan" style={{ fontFamily: 'var(--font-tibetan)', fontSize: '0.85rem', color: '#e8e0d0', lineHeight: 1.5, marginBottom: '0.25rem' }}>
                    {ex.sentence}
                  </div>
                  {ex.lexicon_meaning && (
                    <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.7rem', color: 'rgba(139,128,112,0.5)', fontStyle: 'italic' }}>
                      {ex.lexicon_meaning}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => navigate('/', { state: { text: ex.sentence.replace(/\s+/g, '') } })}
                  style={{ background: 'none', border: '1px solid rgba(201,74,74,0.3)', borderRadius: '6px', padding: '0.2rem 0.5rem', cursor: 'pointer', fontSize: '0.65rem', color: '#c94a4a', fontFamily: 'var(--font-sans)', flexShrink: 0 }}
                >
                  分析
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {!selectedVerb && verbs.length === 0 && (
        <div style={{ textAlign: 'center', padding: '2rem', color: 'rgba(139,128,112,0.4)', fontFamily: 'var(--font-sans)', fontSize: '0.85rem' }}>
          加载中…
        </div>
      )}

      {!selectedVerb && verbs.length > 0 && (
        <div style={{ textAlign: 'center', padding: '2rem', color: 'rgba(139,128,112,0.4)', fontFamily: 'var(--font-sans)', fontSize: '0.85rem' }}>
          点击上方一种时态查看语料例句
        </div>
      )}
    </div>
  );
}

// ── SRS Flashcard Mode ─────────────────────────────────────────────────────────
function SRSCards({
  particles, dueItems, onGrade,
}: {
  particles: CaseParticleDrill[];
  dueItems: ReturnType<typeof useLearner>['dueItems'];
  onGrade: (grade: SRSGrade, tagOverride?: string) => void;
}) {
  const [cardIdx, setCardIdx] = useState(0);
  const [flipped, setFlipped] = useState(false);

  // Build cards from due SRS items + random particles
  const allCards = [
    ...dueItems.map((item) => ({
      tag: item.tag,
      tibetan: item.tibetan,
      particle: particles.find((p) => p.tag === item.tag),
      easeFactor: item.easeFactor,
      interval: item.interval,
    })),
    ...particles
      .filter((p) => !dueItems.find((d) => d.tag === p.tag))
      .slice(0, 8)
      .map((p) => ({ tag: p.tag, tibetan: p.tibetan, particle: p, easeFactor: 2.5, interval: 0 })),
  ];

  const card = allCards[cardIdx];

  function next() {
    setFlipped(false);
    setCardIdx((i) => (i + 1) % Math.max(1, allCards.length));
  }

  function shuffle() {
    setFlipped(false);
    setCardIdx(Math.floor(Math.random() * Math.max(1, allCards.length)));
  }

  if (allCards.length === 0 || !card) {
    return (
      <div style={{ textAlign: 'center', padding: '3rem', color: 'rgba(139,128,112,0.5)', fontFamily: 'var(--font-sans)', fontSize: '0.9rem' }}>
        <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>✓</div>
        <div>暂无需要复习的格助词！</div>
        <div style={{ fontSize: '0.75rem', marginTop: '0.25rem', opacity: 0.6 }}>或点击「随机练习」继续学习</div>
        <button onClick={shuffle} style={{ marginTop: '1rem', padding: '0.5rem 1.25rem', borderRadius: '10px', border: '1px solid rgba(201,74,74,0.3)', background: 'rgba(201,74,74,0.1)', color: '#c94a4a', fontSize: '0.8rem', cursor: 'pointer', fontFamily: 'var(--font-sans)' }}>
          随机练习
        </button>
      </div>
    );
  }

  const particle = card.particle;
  const info = particle ?? { tibetan: card.tibetan, tag: card.tag, name: card.tag, english: '', chinese: '', function: '', examples: [] };

  return (
    <div className="animate-fade-up" style={{ animationDelay: '100ms' }}>
      {/* Progress */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', marginBottom: '2rem' }}>
        <div style={{ height: '2px', borderRadius: '1px', background: 'rgba(255,255,255,0.06)', flex: 1, maxWidth: '160px', overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${((cardIdx + 1) / Math.max(1, allCards.length)) * 100}%`, background: 'linear-gradient(to right, #c94a4a, #d4a853)', borderRadius: '1px', transition: 'width 0.4s ease' }} />
        </div>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'rgba(139,128,112,0.5)' }}>
          {cardIdx + 1} / {allCards.length}
        </span>
      </div>

      {/* 3D Card */}
      <div style={{ perspective: '1200px', maxWidth: '460px', margin: '0 auto' }}>
        <div
          onClick={() => setFlipped((f) => !f)}
          style={{
            position: 'relative', transformStyle: 'preserve-3d',
            transition: 'transform 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
            transform: flipped ? 'rotateY(180deg)' : 'rotateY(0deg)',
            cursor: 'pointer',
          }}
        >
          {/* Front */}
          <div style={{
            backfaceVisibility: 'hidden', WebkitBackfaceVisibility: 'hidden',
            borderRadius: '20px', padding: '3rem 2.5rem',
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.08)',
            boxShadow: '0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)',
            textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            gap: '1rem', minHeight: '280px',
          }}>
            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', width: '180px', height: '180px', borderRadius: '50%', border: '1px solid rgba(212,168,83,0.06)', pointerEvents: 'none' }} />
            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', width: '140px', height: '140px', borderRadius: '50%', border: '1px solid rgba(212,168,83,0.04)', pointerEvents: 'none' }} />
            <span className="tibetan" style={{ fontFamily: 'var(--font-tibetan)', fontWeight: 700, fontSize: 'clamp(2.5rem, 8vw, 3.5rem)', color: caseColor(card.tag), letterSpacing: '0.05em', lineHeight: 1, textShadow: `0 0 30px ${caseColor(card.tag)}40` }}>
              {info.tibetan.split(' / ')[0]}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'rgba(139,128,112,0.4)', background: 'rgba(255,255,255,0.04)', padding: '0.2rem 0.6rem', borderRadius: '99px', letterSpacing: '0.06em' }}>
              {card.tag}
            </span>
            <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'rgba(139,128,112,0.35)', marginTop: '0.5rem', letterSpacing: '0.06em' }}>
              点击翻转 · 查看答案
            </span>
          </div>

          {/* Back */}
          <div style={{
            backfaceVisibility: 'hidden', WebkitBackfaceVisibility: 'hidden',
            position: 'absolute', top: 0, left: 0, right: 0,
            borderRadius: '20px', padding: '3rem 2.5rem',
            background: `${caseColor(card.tag)}0d`, border: `1px solid ${caseColor(card.tag)}33`,
            boxShadow: '0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05)',
            textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            gap: '0.625rem', minHeight: '280px', transform: 'rotateY(180deg)',
          }}>
            <span style={{ fontFamily: 'var(--font-display)', fontSize: 'clamp(1.6rem, 5vw, 2rem)', fontWeight: 700, color: '#e8e0d0', lineHeight: 1.1 }}>
              {info.name}
            </span>
            <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', color: caseColor(card.tag), fontWeight: 500 }}>
              {info.function}（{info.chinese}）
            </span>
            <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', color: 'rgba(139,128,112,0.6)', fontStyle: 'italic' }}>
              {info.english}
            </span>
            <div style={{ width: '40px', height: '1px', background: `${caseColor(card.tag)}44`, margin: '0.5rem 0' }} />
            <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'rgba(139,128,112,0.45)' }}>
              间隔：{card.interval === 0 ? '新词' : `${card.interval}天`} · EF={card.easeFactor.toFixed(1)}
            </span>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '0.625rem', marginTop: '2rem', flexWrap: 'wrap' }}>
        <button onClick={shuffle} style={{ padding: '0.5rem 1rem', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.04)', color: 'rgba(139,128,112,0.7)', fontSize: '0.8rem', fontFamily: 'var(--font-sans)', cursor: 'pointer', transition: 'all 0.2s ease' }}>
          🔀 随机
        </button>
        <button onClick={next} style={{ padding: '0.5rem 1rem', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.04)', color: 'rgba(139,128,112,0.7)', fontSize: '0.8rem', fontFamily: 'var(--font-sans)', cursor: 'pointer', transition: 'all 0.2s ease' }}>
          → 下一个
        </button>
      </div>

      {/* Grade buttons */}
      <div style={{ marginTop: '1.5rem' }}>
        <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.7rem', color: 'rgba(139,128,112,0.5)', marginBottom: '0.5rem', textAlign: 'center' }}>
          这个格助词你掌握得如何？
        </div>
        <GradeButtons
          onGrade={(g) => {
            onGrade(g, card.tag);
            next();
          }}
        />
      </div>
    </div>
  );
}
