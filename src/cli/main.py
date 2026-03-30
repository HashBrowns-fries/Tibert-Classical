"""
TiBERT Classical CLI
====================
统一命令行工具：POS 标注、完整分析、分词、批量处理、REST 服务器启动。

Usage:
    python -m src.cli.main --help
    pip install -e .   # 安装为 tibert 命令
"""

from __future__ import annotations

import os
import sys
import json
import time
import textwrap
import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── App ────────────────────────────────────────────────────────────────────────

__version__ = "1.0.0"
__app_name__ = "tibert"

app = typer.Typer(
    name=__app_name__,
    help="TiBERT Classical Tibetan NLP 工具",
    add_completion=False,
)
console = Console()


# ── POS Tagging (shared logic) ──────────────────────────────────────────────────

def _get_dep():
    """Lazy-load dependencies."""
    from src.api import dependencies as dep
    return dep


def _get_llm_prompt(text: str, tagged: list[dict]) -> str:
    """Build LLM prompt (same as worker.py)."""
    token_rows = []
    case_particles_found = []
    for t in tagged:
        if t["token"] in ("་", "།"):
            continue
        note = ""
        if t["is_case_particle"]:
            note = f"【{t['case_name']}】{t['case_desc']}"
            case_particles_found.append(f"{t['token']}（{t['case_name']}: {t['case_desc']}）")
        token_rows.append(f"  - {t['token']}: {t['pos_zh']}{' ' + note if note else ''}")

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


def _call_qwen_llm(prompt: str) -> str:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        return "⚠️ 未设置 DASHSCOPE_API_KEY"

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


def _format_pos(text: str, tagged: list[dict]) -> None:
    """Print formatted POS result to console."""
    # 分隔音节
    syllables = [t for t in tagged if t["token"] not in ("་", "།")]

    # Token 表格
    table = Table(title=f"词性标注 — {text[:30]}{'…' if len(text)>30 else ''}", show_header=True)
    table.add_column("音节", style="bold")
    table.add_column("标签")
    table.add_column("中文名")
    table.add_column("格助词", style="yellow")

    for t in tagged:
        if t["token"] in ("་", "།"):
            continue
        case = f"★ {t['case_name']}" if t["is_case_particle"] else ""
        table.add_row(t["token"], t["pos"], t["pos_zh"], case)

    console.print(table)

    # 统计
    stats = {
        k: sum(1 for t in tagged if not t["token"] in ("་", "།") and _stat_key(t) == k)
        for k in ["nouns", "verbs", "case_particles"]
    }
    console.print(f"\n  名词: {stats['nouns']}  |  动词: {stats['verbs']}  |  格助词: {stats['case_particles']}  |  音节: {len(syllables)}")


def _stat_key(t: dict) -> str:
    if t["is_case_particle"]:
        return "case_particles"
    if t["pos"].startswith("n."):
        return "nouns"
    if t["pos"].startswith("v."):
        return "verbs"
    return "other"


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command("pos")
def pos_cmd(
    text: str = typer.Argument(..., help="要标注的藏文文本"),
    json_output: bool = typer.Option(False, "--json", "-j", help="输出 JSON 格式"),
) -> None:
    """POS 标注（仅 TiBERT 模型，毫秒级响应）。"""
    dep = _get_dep()
    tokens, stats, syllables = dep.tag_text(text)

    if json_output:
        console.print_json(json.dumps({
            "original": text,
            "syllables": syllables,
            "tokens": tokens,
            "stats": stats,
        }))
    else:
        _format_pos(text, tokens)


@app.command("analyze")
def analyze_cmd(
    text: str = typer.Argument(..., help="要分析的藏文文本"),
    no_llm: bool = typer.Option(False, "--no-llm", help="跳过 LLM 解释（仅 POS）"),
    json_output: bool = typer.Option(False, "--json", "-j", help="输出 JSON 格式"),
) -> None:
    """完整分析：POS 标注 + LLM 语法解释。"""
    dep = _get_dep()
    tokens, stats, syllables = dep.tag_text(text)

    llm_explanation = None
    structure = None
    if not no_llm:
        console.print("[yellow]⏳ 调用 LLM 生成语法解释...[/yellow]")
        prompt = _get_llm_prompt(text, tokens)
        llm_explanation = _call_qwen_llm(prompt)

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

    if json_output:
        console.print_json(json.dumps({
            "original": text,
            "syllables": syllables,
            "tokens": tokens,
            "stats": stats,
            "llm_explanation": llm_explanation,
            "structure": structure,
        }, ensure_ascii=False))
        return

    # 格式输出
    _format_pos(text, tokens)

    if structure:
        console.print(f"\n[bold]句法结构：[/bold] {structure}")

    if llm_explanation:
        console.print(Panel(
            llm_explanation,
            title="🤖 LLM 语法分析",
            border_style="blue",
            padding=(1, 2),
        ))


@app.command("segment")
def segment_cmd(
    text: str = typer.Argument(..., help="要分词的藏文文本"),
) -> None:
    """分词：将藏文切分为音节列表。"""
    dep = _get_dep()
    tok = dep.get_tokenizer()
    syllables = dep.spm_tokenize(text, tok)
    syllables_only = [s for s in syllables if s not in ("་", "།")]

    console.print(f"[bold]原文：[/bold] {text}")
    console.print(f"[bold]音节：[/bold] {' · '.join(syllables_only)}")
    console.print(f"[bold]共 {len(syllables_only)} 音节[/bold]")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="监听地址"),
    port: int = typer.Option(8000, "--port", help="监听端口"),
    reload: bool = typer.Option(False, "--reload", help="启用热重载（开发用）"),
) -> None:
    """启动 FastAPI REST 服务器。"""
    import uvicorn
    from src.api.server import app as fastapi_app

    console.print(f"[green]🚀 启动 TiBERT Classical API 服务器[/green]")
    console.print(f"   地址: http://{host}:{port}")
    console.print(f"   文档: http://{host}:{port}/docs")
    console.print(f"   热重载: {'启用' if reload else '禁用'}")
    console.print()

    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command("batch")
def batch_cmd(
    input_file: Path = typer.Option(..., "--input", "-i", help="输入文件（一行一句）", exists=True, dir_okay=False),
    output_file: Path = typer.Option(..., "--output", "-o", help="输出 JSONL 文件", dir_okay=False),
    no_llm: bool = typer.Option(False, "--no-llm", help="跳过 LLM 解释"),
    api_url: str = typer.Option("http://localhost:8000", "--api-url", help="REST API 地址（batch 模式时忽略，使用本地推理）"),
) -> None:
    """批量分析：读取文件中的句子，逐句分析并输出 JSONL。"""
    dep = _get_dep()

    lines = input_file.read_text(encoding="utf-8").splitlines()
    lines = [l.strip() for l in lines if l.strip()]

    console.print(f"[green]📦 批量分析 {len(lines)} 句...[/green]")

    with output_file.open("w", encoding="utf-8") as out:
        for i, line in enumerate(lines):
            try:
                tokens, stats, syllables = dep.tag_text(line)
                result = {
                    "original": line,
                    "syllables": syllables,
                    "tokens": tokens,
                    "stats": stats,
                }

                if not no_llm:
                    prompt = _get_llm_prompt(line, tokens)
                    result["llm_explanation"] = _call_qwen_llm(prompt)

                out.write(json.dumps(result, ensure_ascii=False) + "\n")

                if (i + 1) % 10 == 0:
                    console.print(f"  已处理 {i+1}/{len(lines)}")

            except Exception as e:
                console.print(f"[red]  错误 (line {i+1}): {e}[/red]")
                out.write(json.dumps({"original": line, "error": str(e)}, ensure_ascii=False) + "\n")

    console.print(f"\n[green]✅ 完成！结果已写入 {output_file}[/green]")


@app.command("lookup")
def lookup_cmd(
    word: Optional[str] = typer.Argument(
        default=None,
        help="要查询的藏文词语或动词词干（Wylie）；使用 --list 列出所有词典",
    ),
    dict_name: str = typer.Option(
        "RangjungYeshe", "--dict", "-d",
        help="词典名称（默认 RangjungYeshe；用 --all 查看所有）",
    ),
    all_dicts: bool = typer.Option(False, "--all", "-a", help="查询所有词典"),
    verbs: bool = typer.Option(False, "--verbs", "-v", help="查询动词词干词典（Wylie 罗马字）"),
    json_output: bool = typer.Option(False, "--json", "-j", help="输出 JSON 格式"),
    list_dicts_flag: bool = typer.Option(False, "--list", "-l", help="列出所有可用词典"),
) -> None:
    """词典查询：查找藏文词语在多个词典中的释义。"""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.dict import lookup_word, list_dicts, get_dict

    # --list: 列出所有词典
    if list_dicts_flag:
        dicts = list_dicts()
        t = Table(title="可用词典", show_header=True)
        t.add_column("名称", style="bold cyan")
        t.add_column("说明")
        for name, desc in dicts.items():
            t.add_row(name, desc)
        console.print(t)
        return

    if not word or not word.strip():
        console.print("[red]错误：请提供要查询的词语，或使用 --list 列出所有词典[/red]")
        raise typer.Exit(code=1)

    if not word.strip():
        console.print("[red]错误：查询词不能为空[/red]")
        raise typer.Exit(code=1)

    dict_names = None
    if all_dicts:
        dict_names = None  # None = all
    elif dict_name:
        dict_names = [dict_name]

    result = lookup_word(word, dict_names=dict_names, include_verbs=verbs)

    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return

    # 格式化输出
    word_display = result["word"]
    entries = result["entries"]
    verb_entries = result.get("verb_entries") or []

    if not entries and not verb_entries:
        console.print(f"[yellow]未在词典中找到: {word_display}[/yellow]")
        return

    console.print(f"\n[bold]查询结果: {word_display}[/bold]")

    # StarDict 词典结果
    if entries:
        t = Table(title="词典释义", show_header=True)
        t.add_column("词典", style="bold cyan", width=18)
        t.add_column("释义", width=80)
        for entry in entries:
            # 截断长释义
            defn = entry["definition"]
            if len(defn) > 300:
                defn = defn[:300] + "…"
            t.add_row(entry["dict_name"], defn)
        console.print(t)

    # 动词词干结果
    if verb_entries:
        t = Table(title="动词词干 (Verb Stems)", show_header=True)
        t.add_column("词干", style="bold yellow")
        t.add_column("现在时", style="green")
        t.add_column("过去时", style="red")
        t.add_column("将来时", style="blue")
        t.add_column("命令式", style="magenta")
        t.add_column("释义", width=40)
        for ve in verb_entries:
            t.add_row(
                ve.get("headword", ""),
                (ve.get("present") or "—")[:40],
                (ve.get("past") or "—")[:40],
                (ve.get("future") or "—")[:40],
                (ve.get("imperative") or "—")[:40],
                (ve.get("meaning") or "—")[:40],
            )
        console.print(t)


@app.command("version")
def version_cmd() -> None:
    """显示版本信息。"""
    console.print(f"[bold]{__app_name__}[/bold] v{__version__}")
    console.print(f"TiBERT Classical Tibetan NLP")


@app.command()
def help_cmd() -> None:
    """显示帮助。"""
    console.print(Panel(
        textwrap.dedent("""\
        TiBERT Classical Tibetan NLP 工具

        Commands:
          tibert pos TEXT       POS 标注（仅模型推理）
          tibert analyze TEXT   完整分析（POS + LLM 解释）
          tibert segment TEXT   分词
          tibert lookup WORD    词典查询（默认 RangjungYeshe）
          tibert lookup WORD --all   查询所有词典
          tibert lookup WORD --verbs    查询动词词干词典（Wylie 罗马字）
          tibert lookup --list     列出所有可用词典
          tibert serve          启动 REST API 服务器
          tibert batch          批量分析
          tibert version        版本信息

        Examples:
          tibert pos "བོད་གི་ཡུལ་ལྷོ་ལ་སོང་"
          tibert analyze "བོད་གི་ཡུལ་ལྷོ་ལ་སོང་"
          tibert lookup བོད --dict RangjungYeshe
          tibert lookup kum --verbs
          tibert serve --port 8000
        """).strip(),
        title="TiBERT Classical CLI",
        border_style="blue",
    ))


if __name__ == "__main__":
    app()
