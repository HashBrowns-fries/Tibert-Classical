import { useState, useCallback, useEffect } from 'react';
import type {
  LearnerParticlesResponse,
  LearnerVerbsResponse,
  LearnerDrillResponse,
  SRSItem,
  SRSGrade,
  CaseParticleDrill,
} from '../types/api';

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8001';
const SRS_KEY = 'tibert_srs_v1';

// ── SM-2 Algorithm ────────────────────────────────────────────────────────────

export function sm2(grade: SRSGrade, item: SRSItem): SRSItem {
  const EF_MIN = 1.3;
  let { easeFactor, interval, repetitions } = item;

  if (grade >= 3) {
    if (repetitions === 0) interval = 1;
    else if (repetitions === 1) interval = 6;
    else interval = Math.round(interval * easeFactor);
    repetitions += 1;
  } else {
    repetitions = 0;
    interval = 1;
  }

  easeFactor = Math.max(
    EF_MIN,
    easeFactor + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
  );

  const nextReview = Date.now() + interval * 24 * 60 * 60 * 1000;
  return { ...item, easeFactor, interval, repetitions, nextReview, lastReview: Date.now() };
}

// ── SRS localStorage helpers ───────────────────────────────────────────────────

function loadSRS(): SRSItem[] {
  try {
    const raw = localStorage.getItem(SRS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveSRS(items: SRSItem[]): void {
  try {
    localStorage.setItem(SRS_KEY, JSON.stringify(items));
  } catch {}
}

// ── Hook ───────────────────────────────────────────────────────────────────────

export function useLearner() {
  const [particles, setParticles] = useState<CaseParticleDrill[]>([]);
  const [verbs, setVerbs] = useState<LearnerVerbsResponse['verbs']>([]);
  const [srsItems, setSrsItems] = useState<SRSItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [drillLoading, setDrillLoading] = useState(false);
  const [currentDrill, setCurrentDrill] = useState<LearnerDrillResponse | null>(null);
  const [drillAnswer, setDrillAnswer] = useState('');
  const [drillFeedback, setDrillFeedback] = useState<string | null>(null);
  const [drillScore, setDrillScore] = useState<number | null>(null);
  const [stats, setStats] = useState({ total: 0, words: 0, mastered: 0, due: 0 });
  const [error, setError] = useState<string | null>(null);

  // Load SRS on mount
  useEffect(() => {
    const items = loadSRS();
    setSrsItems(items);
    const now = Date.now();
    const due = items.filter((i) => i.nextReview <= now).length;
    const mastered = items.filter((i) => i.repetitions >= 3).length;
    setStats((s) => ({ ...s, mastered, due }));
  }, []);

  // Fetch particles data
  const fetchParticles = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/learn/particles`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: LearnerParticlesResponse = await res.json();
      setParticles(data.particles);
      setStats((s) => ({ ...s, total: data.total_sentences, words: data.total_words }));
    } catch (e) {
      console.error('learn/particles failed:', e);
      setError('无法加载格助词数据，请检查后端服务是否启动。');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch verbs data
  const fetchVerbs = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/learn/verbs`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: LearnerVerbsResponse = await res.json();
      setVerbs(data.verbs);
    } catch (e) {
      console.error('learn/verbs failed:', e);
      setError('无法加载动词数据，请检查后端服务是否启动。');
    } finally {
      setLoading(false);
    }
  }, []);

  // Generate a drill (gemma-powered)
  const generateDrill = useCallback(
    async (type: string = 'particle_identify', particleTag?: string) => {
      setDrillLoading(true);
      setDrillAnswer('');
      setDrillFeedback(null);
      setDrillScore(null);
      try {
        const body: Record<string, unknown> = { type };
        if (particleTag) body.particle_tag = particleTag;
        const res = await fetch(`${API}/learn/drill`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: LearnerDrillResponse = await res.json();
        setCurrentDrill(data);
      } catch (e) {
        console.error('learn/drill failed:', e);
        setCurrentDrill(null);
      } finally {
        setDrillLoading(false);
      }
    },
    []
  );

  // Submit answer to gemma grading
  const submitDrill = useCallback(
    async (userAnswer: string) => {
      if (!currentDrill) {
        setDrillFeedback('请先生成练习题');
        return;
      }
      setDrillAnswer(userAnswer);
      try {
        const res = await fetch(`${API}/learn/check`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
body: JSON.stringify({
            drill_type: currentDrill.drill_type,
            question_type: currentDrill.question_type,
            sentence: currentDrill.sentence,
            user_answer: userAnswer,
            answer: currentDrill.answer,
            target: currentDrill.target,
          }),
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setDrillFeedback(data.feedback);
        setDrillScore(data.score);
      } catch (e) {
        console.error('learn/check failed:', e);
        setDrillFeedback('评分失败，请稍后重试。');
        setDrillScore(0);
      }
    },
    [currentDrill]
  );

  // Record SRS update after grading
  // tag: optional override (for flashcards); uses currentDrill.target otherwise
  const recordReview = useCallback(
    (grade: SRSGrade, tagOverride?: string) => {
      const tag = tagOverride ?? currentDrill?.target;
      if (!tag) return;
      // Find the particle data
      const particle = particles.find((p) => p.tag === tag);
      const tibetan = particle?.tibetan?.split(' / ')[0] ?? particle?.tibetan ?? tag;

      const id = `particle:${tag}`;
      const existing = srsItems.find((i) => i.id === id);

      const updated = sm2(grade, existing ?? {
        id,
        tag,
        tibetan,
        easeFactor: 2.5,
        interval: 0,
        repetitions: 0,
        nextReview: 0,
      });

      const next = existing
        ? srsItems.map((i) => (i.id === id ? updated : i))
        : [...srsItems, updated];

      setSrsItems(next);
      saveSRS(next);
      const now = Date.now();
      setStats((s) => ({
        ...s,
        mastered: next.filter((i) => i.repetitions >= 3).length,
        due: next.filter((i) => i.nextReview <= now).length,
      }));
    },
    [currentDrill, particles, srsItems]
  );

  // Get due SRS items
  const dueItems = srsItems.filter((i) => i.nextReview <= Date.now());

  return {
    particles,
    verbs,
    srsItems,
    dueItems,
    stats,
    loading,
    error,
    drillLoading,
    currentDrill,
    drillAnswer,
    drillFeedback,
    drillScore,
    fetchParticles,
    fetchVerbs,
    generateDrill,
    submitDrill,
    recordReview,
  };
}
