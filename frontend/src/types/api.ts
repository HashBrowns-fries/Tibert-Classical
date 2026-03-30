// API 类型定义 — 对应 Python Pydantic 模型

export interface TokenResponse {
  token: string;
  pos: string;
  pos_zh: string;
  is_case_particle: boolean;
  case_name?: string;
  case_desc?: string;
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

export interface LookupEntry {
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
