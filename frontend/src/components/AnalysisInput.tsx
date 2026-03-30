import { useState } from 'react';

const EXAMPLES = [
  { label: '去西藏南部', text: 'བོད་གི་ཡུལ་ལྷོ་ལ་སོང་' },
  { label: '上师教言', text: 'སློབ་དཔོན་གྱི་ཆོས་སྐུལ་དེ་ཡིན' },
  { label: '祈祷上师', text: 'བདག་གི་སློབ་དཔོན་ལ་གསོལ་བ་འདེབས་སོ།' },
  { label: '请说法', text: 'ཁྱོད་ཀྱི་ཆོས་གཞུང་གི་སྒ໲་དགོད་ཅིག' },
];

interface AnalysisInputProps {
  onAnalyze: (text: string, useLlm: boolean) => void;
  loading: boolean;
  defaultText?: string;
}

export function AnalysisInput({ onAnalyze, loading, defaultText = '' }: AnalysisInputProps) {
  const [text, setText] = useState(defaultText);
  const [useLlm, setUseLlm] = useState(true);

  function handleAnalyze() {
    if (!text.trim()) return;
    onAnalyze(text.trim(), useLlm);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleAnalyze();
    }
  }

  return (
    <div className="space-y-3">
      {/* 文本输入 */}
      <div>
        <textarea
          className="tibetan w-full rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50 p-4 text-base resize-none focus:outline-none focus:ring-2 focus:ring-indigo-400 dark:focus:ring-indigo-500 transition-shadow"
          rows={8}
          placeholder="输入或粘贴古典藏文..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          style={{ fontFamily: 'var(--font-tibetan)' }}
        />
        <div
          className="text-xs text-gray-400 mt-1 text-right"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          {text.length} 字符 · Ctrl+Enter 分析
        </div>
      </div>

      {/* 示例句 */}
      <div>
        <div
          className="text-xs text-gray-500 dark:text-gray-400 mb-1.5"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          示例句
        </div>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.text}
              onClick={() => {
                setText(ex.text);
                onAnalyze(ex.text, useLlm);
              }}
              className="tibetan text-sm px-3 py-1.5 rounded-full border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 hover:border-indigo-300 dark:hover:border-indigo-700 hover:text-indigo-700 dark:hover:text-indigo-300 transition-colors"
            >
              {ex.label} →
            </button>
          ))}
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex items-center gap-3">
        {/* LLM 开关 */}
        <label className="flex items-center gap-2 cursor-pointer">
          <div
            className="text-xs text-gray-500"
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            🤖 LLM 解释
          </div>
          <button
            role="switch"
            aria-checked={useLlm}
            onClick={() => setUseLlm((v) => !v)}
            className={[
              'relative w-10 h-5 rounded-full transition-colors',
              useLlm ? 'bg-indigo-500' : 'bg-gray-300 dark:bg-gray-600',
            ].join(' ')}
          >
            <span
              className={[
                'absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform',
                useLlm ? 'translate-x-5' : 'translate-x-0.5',
              ].join(' ')}
            />
          </button>
        </label>

        {/* 分析按钮 */}
        <button
          onClick={handleAnalyze}
          disabled={loading || !text.trim()}
          className={[
            'flex-1 py-2.5 rounded-xl font-semibold text-white transition-all',
            'bg-indigo-600 hover:bg-indigo-700 active:scale-[0.98]',
            'disabled:opacity-40 disabled:cursor-not-allowed',
          ].join(' ')}
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          {loading ? '⏳ 分析中...' : '🔍 分析'}
        </button>

        {/* 清空 */}
        <button
          onClick={() => setText('')}
          disabled={!text}
          className="px-4 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          清空
        </button>
      </div>
    </div>
  );
}
