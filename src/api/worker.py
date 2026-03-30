"""
Python Worker — 持久化 POS 推理进程。

通过 stdin 接收 JSON 命令，stdout 返回 JSON 结果。
由 Tauri Rust 后端 spawn 并管理。

Usage:
    python -m src.api.worker
"""

from __future__ import annotations

import sys
import json
import os
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import dependencies as dep


def main():
    """主循环：读取 stdin 命令，输出 JSON 结果。"""
    # 预热：启动时加载模型
    print(json.dumps({"ok": True, "log": "正在加载模型..."}), flush=True)
    try:
        dep.get_tokenizer()
        dep.get_pos_model()
        print(json.dumps({"ok": True, "log": "模型加载完成"}), flush=True)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"模型加载失败: {e}"}), flush=True)
        sys.exit(1)

    # 主循环
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"ok": False, "error": f"无效 JSON: {line}"}), flush=True)
            continue

        cmd = req.get("cmd", "")

        try:
            if cmd == "health":
                print(json.dumps({"ok": True, "data": True}), flush=True)

            elif cmd == "pos":
                text = req.get("text", "")
                tokens, stats, syllables_str = dep.tag_text(text)
                print(json.dumps({
                    "ok": True,
                    "data": {
                        "original": text,
                        "syllables": syllables_str,
                        "tokens": tokens,
                        "stats": stats,
                    }
                }), flush=True)

            elif cmd == "analyze":
                text = req.get("text", "")
                use_llm = req.get("use_llm", True)

                tokens, stats, syllables_str = dep.tag_text(text)

                llm_explanation = None
                structure = None

                if use_llm:
                    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
                    if api_key:
                        prompt = _build_llm_prompt(text, tokens)
                        llm_explanation = _call_qwen_llm(prompt, api_key)

                        parts = []
                        for t in tokens:
                            if t["token"] in ("་", "།"):
                                continue
                            if t["is_case_particle"]:
                                parts.append(f"[{t['case_name']}:{t['token']}]")
                            elif t["pos"].startswith("n."):
                                parts.append(f"[名:{t['token']}]")
                            elif t["pos"].startswith("v."):
                                parts.append(f"[动:{t['token']}]")
                            elif t["pos"] == "adj":
                                parts.append(f"[形:{t['token']}]")
                            else:
                                parts.append(f"[{t['token']}]")
                        structure = " ".join(parts)
                    else:
                        llm_explanation = "⚠️ 未设置 DASHSCOPE_API_KEY，无法生成语法解释"

                print(json.dumps({
                    "ok": True,
                    "data": {
                        "original": text,
                        "syllables": syllables_str,
                        "tokens": tokens,
                        "stats": stats,
                        "llm_explanation": llm_explanation,
                        "structure": structure,
                    }
                }), flush=True)

            elif cmd == "corpus_stats":
                stats = dep.get_corpus_stats()
                print(json.dumps({"ok": True, "data": stats}), flush=True)

            elif cmd == "lookup":
                from src.dict import lookup_word
                word = req.get("word", "")
                dict_names = req.get("dict_names")
                include_verbs = req.get("include_verbs", True)
                result = lookup_word(word, dict_names=dict_names, include_verbs=include_verbs)
                print(json.dumps({"ok": True, "data": result}), flush=True)

            else:
                print(json.dumps({"ok": False, "error": f"未知命令: {cmd}"}), flush=True)

        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)


def _build_llm_prompt(text: str, tagged: list[dict]) -> str:
    """Build rich prompt for Qwen LLM."""
    token_rows = []
    case_particles_found = []
    for t in tagged:
        if t["token"] in ("་", "།"):
            continue
        pos_zh = t["pos_zh"]
        note = ""
        if t["is_case_particle"]:
            note = f"【{t['case_name']}】{t['case_desc']}"
            case_particles_found.append(f"{t['token']}（{t['case_name']}: {t['case_desc']}）")
        token_rows.append(f"  - {t['token']}: {pos_zh}{' ' + note if note else ''}")

    tokens_block = "\n".join(token_rows) if token_rows else "  （无有效音节）"
    case_block = ""
    if case_particles_found:
        case_block = "\n\n本句中的格助词及其功能：\n" + "\n".join(f"  • {cp}" for cp in case_particles_found)

    structure = []
    for t in tagged:
        if t["token"] in ("་", "།"):
            continue
        if t["is_case_particle"]:
            structure.append(f"[{t['case_name']}助词]")
        elif t["pos"].startswith("n."):
            structure.append(f"[名词:{t['token']}]")
        elif t["pos"].startswith("v."):
            structure.append(f"[动词:{t['token']}]")
        elif t["pos"] == "adj":
            structure.append(f"[形容词:{t['token']}]")
        elif t["pos"] == "punc":
            pass
        else:
            structure.append(f"[{t['token']}]")

    return f"""请分析以下古典藏文，提供详细的自然语言语法解释。

【原文】
{text}

【词性标注结果】（由机器自动标注，仅供参考）
{tokens_block}
{case_block}

【句法结构提示】
{" ".join(structure)}

请用中文（白话文）给出完整的语法分析，包括：
1. 句子的基本含义（整句翻译）
2. 每个词的含义
3. 格助词的功能（属格/作格/为格等）
4. 句子的语法结构分析
5. 特殊语法现象说明（如有）
"""


def _call_qwen_llm(prompt: str, api_key: str) -> str:
    """Call Qwen LLM via DashScope."""
    import dashscope
    from dashscope import Generation
    from dashscope.api_entities.dashscope_response import Message

    dashscope.api_key = api_key

    SYSTEM = (
        "你是一位专业的古典藏文（Classical Tibetan）语言学家，精通藏文语法、"
        "词性标注和格助词分析。请用中文（白话文）详细分析用户提供的古典藏文句子。"
    )

    try:
        resp = Generation.call(
            model="qwen-plus-2025-07-28",
            messages=[
                Message(role="system", content=SYSTEM),
                Message(role="user", content=prompt),
            ],
            temperature=0.2,
            top_p=0.9,
            api_key=api_key,
        )
        if resp.status_code != 200:
            return f"⚠️ LLM 调用失败: {resp.code} — {getattr(resp, 'message', '')}"
        return resp.output.text
    except Exception as e:
        return f"⚠️ LLM 调用异常: {e}"


if __name__ == "__main__":
    main()
