// API 类型定义 — 对应 Python Pydantic 模型

export interface TokenResponse {
  token: string;
  pos: string;
  pos_zh: string;
  is_case_particle: boolean;
  case_name?: string;
  case_desc?: string;
  dict_entries?: LookupEntry[];
}

export interface PosResponse {
  original: string;
  syllables: string;
  tokens: TokenResponse[];
  stats: {
    nouns: number;
    verbs: number;
    case_particles: number;
    syllable_count: number;
  };
}

export interface AnalyzeResponse extends PosResponse {
  llm_explanation?: string;
  structure?: string;
  error?: string;
}

export interface CorpusStats {
  total_sentences: number;
  total_collections: number;
  collections: { name: string; count: number }[];
  pos_dataset_stats: Record<string, { sentences: number; max_length: number }>;
}

export interface CorpusSentence {
  id: string;
  collection: string;
  text: string;
  syllables: string[];
}

export interface CorpusSentencesResponse {
  sentences: CorpusSentence[];
  total: number;
  page: number;
  page_size: number;
  collections: string[];
}

export interface LookupEntry {
  word?: string;
  dict_name: string;
  definition: string;
}

export interface VerbEntry {
  head: string;
  headword: string;
  present?: string;
  past?: string;
  future?: string;
  imperative?: string;
  meaning?: string;
}

export interface LookupResponse {
  word: string;
  entries: LookupEntry[];
  verb_entries?: VerbEntry[];
}

// Gemma segmentation types
export interface SegmentResponse {
  text: string;
  syllables: string[];
  method: string;
}

// gemma 分词 + 词典查询 (rag_server /lookup)
export interface GemmaLookupResponse {
  syllables: string[];
  entries: LookupEntry[];
  total: number;
}

// RAG types
export interface RagChunk {
  text: string;
  source: string;
  distance: number;
}

export interface RagResponse {
  question: string;
  answer: string;
  retrieved_chunks: RagChunk[];
  retrieve_time_s: number;
}

export interface RagStats {
  collection: string;
  total_chunks: number;
  index_dir: string;
  embedding_model: string;
  llm_model: string;
}

// POS 标签 → 颜色类别映射
export type PosCategory =
  | 'case'
  | 'noun'
  | 'verb'
  | 'adj'
  | 'adv'
  | 'neg'
  | 'punc'
  | 'other';

export function getPosCategory(pos: string): PosCategory {
  if (pos === 'punc') return 'punc';
  if (pos.startsWith('case')) return 'case';
  if (pos.startsWith('n.')) return 'noun';
  if (pos.startsWith('v.')) return 'verb';
  if (pos === 'adj') return 'adj';
  if (pos.startsWith('adv')) return 'adv';
  if (pos === 'neg') return 'neg';
  return 'other';
}

export const POS_CATEGORY_LABELS: Record<PosCategory, string> = {
  case: '格助词',
  noun: '名词',
  verb: '动词',
  adj: '形容词',
  adv: '副词',
  neg: '否定词',
  punc: '分隔符',
  other: '其他',
};

// ── Learner tool types ──────────────────────────────────────────────────────────

export interface CaseParticleExample {
  particle: string;
  particle_tag: string;
  noun?: string;
  sentence: string;
  context: string;
  collection: string;
}

export interface CaseParticleDrill {
  tag: string;
  tibetan: string;
  name: string;
  english: string;
  chinese: string;
  function: string;
  count: number;
  examples: CaseParticleExample[];
}

export interface LearnerParticlesResponse {
  particles: CaseParticleDrill[];
  total_sentences: number;
  total_words: number;
}

export interface VerbFormExample {
  form: string;
  sentence: string;
  lexicon_meaning: string;
}

export interface VerbDrill {
  tag: string;
  count: number;
  examples: VerbFormExample[];
}

export interface LearnerVerbsResponse {
  verbs: VerbDrill[];
  total_verb_examples: number;
}

export interface LearnerDrillResponse {
  drill_type: string;
  question_type: string;
  sentence: string;
  target: string;
  hint?: string;
  answer: string;
  explanation: string;
}

// ── SRS (Spaced Repetition) state stored in localStorage ───────────────────────

export interface SRSItem {
  id: string;
  tag: string;
  tibetan: string;
  easeFactor: number;   // SM-2 ease factor (default 2.5)
  interval: number;    // days until next review
  repetitions: number; // number of successful reviews
  nextReview: number;  // timestamp (ms) of next review
  lastReview?: number;
}

export type SRSGrade = 0 | 1 | 2 | 3 | 4 | 5; // SM-2 grades
