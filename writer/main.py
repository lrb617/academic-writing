#!/usr/bin/env python3
"""学术写作助手的命令行入口。"""

import os
import sys

# 必须在其他导入之前把 .env 读进来
from pathlib import Path
from dotenv import load_dotenv

cwd_resolved = Path.cwd().resolve()
env_file = cwd_resolved / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file, override=True)

import asyncio
import time
from typing import Optional

from claude_agent_sdk import query, ClaudeAgentOptions
from claude_agent_sdk.types import HookMatcher, StopHookInput, HookContext

from .helpers import (
    create_data_context_message,
    detect_paper_reference,
    ensure_output_folder,
    find_existing_papers,
    get_api_key,
    get_data_files,
    load_system_instructions,
    process_data_files,
    scan_paper_directory,
    setup_claude_skills,
)
from .data_types import TokenUsage
from .runner import generate_paper_workflow


def create_completion_check_stop_hook(auto_continue: bool = True):
    async def completion_check_stop_hook(
        hook_input: StopHookInput,
        matcher: str | None,
        context: HookContext,
    ) -> dict:
        if auto_continue:
            # 强制让 agent 继续，不允许它自己停下来
            return {"continue_": True}

        return {"continue_": False}

    return completion_check_stop_hook


async def main(track_token_usage: bool = False) -> Optional[TokenUsage]:
    """CLI 主循环。"""
    cwd_resolved = Path.cwd().resolve()
    env_file = cwd_resolved / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)

    try:
        get_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    cwd = Path.cwd().resolve()
    package_dir = Path(__file__).parent.absolute()

    setup_claude_skills(package_dir, cwd)

    output_folder = ensure_output_folder(cwd)

    system_instructions = load_system_instructions(cwd)

    # 这里追加的指令只在同一个 CLI 会话中生效
    system_instructions += "\n\n" + f"""
IMPORTANT - WORKING DIRECTORY:
- Your working directory is: {cwd}
- ALWAYS create writing_outputs folder in this directory: {cwd}/writing_outputs/
- NEVER write to /tmp/ or any other temporary directory
- All paper outputs MUST go to: {cwd}/writing_outputs/<timestamp>_<description>/

IMPORTANT - CONVERSATION CONTINUITY:
- The user will provide context in their prompt if they want to continue working on an existing paper
- If the prompt includes [CONTEXT: You are currently working on a paper in: ...], continue editing that paper
- If no such context is provided, this is a NEW paper request - create a new paper directory
- Do NOT assume there's an existing paper unless explicitly told in the prompt context
- Each new chat session should start with a new paper unless context says otherwise
"""

    # 通过环境变量控制 auto_continue，默认开启以确保任务跑完
    auto_continue = os.environ.get("WRITER_AUTO_CONTINUE", "true").lower() in ("true", "1", "yes")

    # 显式构造子进程环境变量：把 .env 的中转站 + 模型设置灌给 claude CLI 子进程，
    # 防止用户级 ~/.claude/settings.json 里的 env 段（如 Kimi 中转站）覆盖。
    model_name = os.environ.get("OPENAI_MODEL", "claude-sonnet-4-6")
    proxy_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    subprocess_env = {
        "ANTHROPIC_MODEL": model_name,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model_name,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model_name,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_name,
    }
    if proxy_url:
        subprocess_env["ANTHROPIC_BASE_URL"] = proxy_url
    if api_key:
        subprocess_env["ANTHROPIC_API_KEY"] = api_key

    options = ClaudeAgentOptions(
        system_prompt=system_instructions,
        model=model_name,
        allowed_tools=["Read", "Write", "Edit", "Bash", "WebSearch", "research_lookup"],
        permission_mode="bypassPermissions",
        setting_sources=["project"],
        cwd=str(cwd),
        env=subprocess_env,
        max_turns=500,
        hooks={
            "Stop": [
                HookMatcher(
                    matcher=None,
                    hooks=[create_completion_check_stop_hook(auto_continue=auto_continue)],
                )
            ]
        },
    )

    current_paper_path = None
    conversation_history = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0

    print("=" * 70)
    print("学术写作工具")
    print("=" * 70)
    print("\n你好！我是你的科学写作助手。")
    print("\n我可以帮你：")
    print("  • 撰写学术论文（IMRaD 结构）")
    print("  • 文献综述与引用管理")
    print("  • 同行评审反馈")
    print("  • 通过 Perplexity Sonar Pro Search 实时检索文献")
    print("  • 原生网页搜索获取最新信息")
    print("  • 文档处理（docx、pdf、pptx、xlsx）")
    print("\n📋 工作流程：")
    print("  1. 我会先给出简要计划，然后立即开始执行")
    print("  2. 过程中会持续更新进度")
    print("  3. 所有输出保存至：writing_outputs/<时间戳_主题>/")
    print("  4. 实时在 progress.md 中记录进度")
    print(f"\n📁 工作目录：{cwd}")
    print(f"📁 输出文件夹：{output_folder}")
    print(f"\n📦 数据文件：")
    print("  • 将文件放入 'data/' 文件夹即可纳入论文")
    print("  • 手稿文件（.tex）→ 复制到 drafts/（编辑模式）")
    print("  • 上下文文件（.md、.docx、.pdf）→ 复制到 sources/（参考）")
    print("  • 数据文件（csv、txt、json 等）→ 复制到论文的 data/ 文件夹")
    print("  • 图片（png、jpg、svg 等）→ 复制到论文的 figures/ 文件夹")
    print("  • 其他文件 → 复制到 sources/（上下文）")
    print("  • 复制后原始文件将自动删除")
    print("\n🤖 智能论文检测：")
    print("  • 自动识别你是否在指代之前的论文/演示文稿")
    print("  • 继续：'continue'、'update'、'edit'、'the paper'、'the presentation' 等")
    print("  • 查找：'look for'、'find'、'show me'、'where is' 等")
    print("  • 或直接提及主题（例如：'find the acoustics paper'）")
    print("  • 输入 'new paper' 可明确开始新论文")
    print("\n输入 'exit' 或 'quit' 结束会话。")
    print("输入 'help' 查看使用提示。")
    print("=" * 70)
    print()

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("\n感谢使用，再见！")
                if track_token_usage:
                    return TokenUsage(
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        cache_creation_input_tokens=total_cache_creation_tokens,
                        cache_read_input_tokens=total_cache_read_tokens,
                    )
                return None

            if user_input.lower() == "help":
                _print_help()
                continue

            if not user_input:
                continue

            existing_papers = find_existing_papers(output_folder)

            new_paper_keywords = [
                "new paper", "start fresh", "start afresh", "create new", "different paper", "another paper",
                "new presentation", "new poster", "different presentation", "another presentation"
            ]
            is_new_paper_request = any(keyword in user_input.lower() for keyword in new_paper_keywords)

            detected_paper_path = None
            if not is_new_paper_request:
                detected_paper_path = detect_paper_reference(user_input, existing_papers)

                if detected_paper_path and str(detected_paper_path) != current_paper_path:
                    current_paper_path = str(detected_paper_path)
                    print(f"\n🔍 Detected reference to existing paper: {detected_paper_path.name}")
                    print(f"📂 Working on: {current_paper_path}")

                    paper_info = scan_paper_directory(detected_paper_path)
                    file_count = sum([
                        1 if paper_info['tex_final'] else 0,
                        1 if paper_info['pdf_final'] else 0,
                        len(paper_info['tex_drafts']),
                        len(paper_info['pdf_drafts']),
                        len(paper_info['figures']),
                        len(paper_info['data']),
                        len(paper_info['sources']),
                        1 if paper_info['bibliography'] else 0,
                        1 if paper_info['progress_log'] else 0,
                        1 if paper_info['summary'] else 0,
                    ])
                    print(f"📄 Found {file_count} file(s) in this directory\n")

                elif detected_paper_path and str(detected_paper_path) == current_paper_path:
                    print(f"📂 Continuing with: {Path(current_paper_path).name}\n")

            data_context = ""
            data_files = get_data_files(cwd)

            # 新论文 + 有数据文件：先建目录再处理文件
            if data_files and not current_paper_path and (is_new_paper_request or not current_paper_path):
                print(f"\n📦 Found {len(data_files)} file(s) in data folder.")
                print("📝 Starting a new paper...")
                print("⏳ Step 1/2: Creating paper directory...\n")

                directory_prompt = f"""Create a new paper directory structure in writing_outputs/ following the standard format:
writing_outputs/YYYYMMDD_HHMMSS_<description>/

Create these folders:
- drafts/
- final/
- references/
- figures/
- data/
- sources/

IMPORTANT:
1. Only create the directory structure and progress.md file
2. Do NOT start writing the paper yet
3. Report back the directory path you created
4. Wait for further instructions

Based on the user request: {user_input}"""

                async for message in query(prompt=directory_prompt, options=options):
                    if track_token_usage and hasattr(message, "usage") and message.usage:
                        usage = message.usage
                        total_input_tokens += getattr(usage, "input_tokens", 0)
                        total_output_tokens += getattr(usage, "output_tokens", 0)
                        total_cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0)
                        total_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0)

                    if hasattr(message, "content") and message.content:
                        for block in message.content:
                            if hasattr(block, "text"):
                                print(block.text, end="", flush=True)

                print("\n")

                # 等一下让文件系统更新
                time.sleep(1)
                try:
                    paper_dirs = [d for d in output_folder.iterdir() if d.is_dir()]
                    if paper_dirs:
                        most_recent = max(paper_dirs, key=lambda d: d.stat().st_mtime)
                        time_since_modification = time.time() - most_recent.stat().st_mtime

                        # 15 秒内的目录认为是刚刚创建的
                        if time_since_modification < 15:
                            current_paper_path = str(most_recent)
                            print(f"✓ Directory created: {most_recent.name}\n")
                except Exception as e:
                    print(f"Warning: Could not detect paper directory: {e}\n")

                # 第二步：处理数据文件
                if current_paper_path:
                    print(f"⏳ Step 2/2: Processing and copying data files...")
                    processed_info = process_data_files(cwd, data_files, current_paper_path)
                    if processed_info:
                        data_context = create_data_context_message(processed_info)
                        manuscript_count = len(processed_info.get('manuscript_files', []))
                        source_count = len(processed_info.get('source_files', []))
                        data_count = len(processed_info.get('data_files', []))
                        image_count = len(processed_info.get('image_files', []))
                        if manuscript_count > 0:
                            print(f"   ✓ Copied {manuscript_count} .tex manuscript(s) to drafts/ [EDITING MODE]")
                        if source_count > 0:
                            print(f"   ✓ Copied {source_count} source/context file(s) to sources/")
                        if data_count > 0:
                            print(f"   ✓ Copied {data_count} data file(s) to data/")
                        if image_count > 0:
                            print(f"   ✓ Copied {image_count} image(s) to figures/")
                        print("   ✓ Deleted original files from data folder\n")
                        print("✅ Files processed. Now starting paper generation...\n")

                contextual_prompt = f"""[CONTEXT: You are working on a paper in: {current_paper_path}]
[FILES HAVE BEEN PROCESSED AND COPIED - see details below]
{data_context}

Now continue with the actual paper generation for the user's request:
{user_input}"""

            elif data_files and current_paper_path and not is_new_paper_request:
                # 已有论文 + 新数据文件：直接处理
                print(f"📦 Found {len(data_files)} file(s) in data folder. Processing...")
                processed_info = process_data_files(cwd, data_files, current_paper_path)
                if processed_info:
                    data_context = create_data_context_message(processed_info)
                    manuscript_count = len(processed_info.get('manuscript_files', []))
                    source_count = len(processed_info.get('source_files', []))
                    data_count = len(processed_info.get('data_files', []))
                    image_count = len(processed_info.get('image_files', []))
                    if manuscript_count > 0:
                        print(f"   ✓ Copied {manuscript_count} .tex manuscript(s) to drafts/ [EDITING MODE]")
                    if source_count > 0:
                        print(f"   ✓ Copied {source_count} source/context file(s) to sources/")
                    if data_count > 0:
                        print(f"   ✓ Copied {data_count} data file(s) to data/")
                    if image_count > 0:
                        print(f"   ✓ Copied {image_count} image(s) to figures/")
                    print("   ✓ Deleted original files from data folder\n")

                contextual_prompt = f"""[CONTEXT: You are currently working on a paper in: {current_paper_path}]
[INSTRUCTION: Continue editing this existing paper. Do NOT create a new paper directory.]
{data_context}
User request: {user_input}"""

            elif is_new_paper_request and not data_files:
                # 新论文且无数据文件
                current_paper_path = None
                print("📝 Starting a new paper...\n")
                contextual_prompt = user_input

            elif current_paper_path and not data_files:
                # 在已有论文上继续工作，给 agent 一份现状概览
                paper_info = scan_paper_directory(Path(current_paper_path))

                context_parts = [
                    f"[CONTEXT: You are currently working on a paper in: {current_paper_path}]",
                    "[INSTRUCTION: Continue working on this existing paper. Do NOT create a new paper directory.]",
                    "\n📁 Current paper contents:"
                ]

                if paper_info['tex_final']:
                    context_parts.append(f"  • Final LaTeX: {Path(paper_info['tex_final']).name}")
                if paper_info['pdf_final']:
                    context_parts.append(f"  • Final PDF: {Path(paper_info['pdf_final']).name}")
                if paper_info['tex_drafts']:
                    context_parts.append(f"  • Draft LaTeX files: {len(paper_info['tex_drafts'])} file(s)")
                    for draft in paper_info['tex_drafts']:
                        context_parts.append(f"    - {Path(draft).name}")
                if paper_info['pdf_drafts']:
                    context_parts.append(f"  • Draft PDF files: {len(paper_info['pdf_drafts'])} file(s)")
                if paper_info['figures']:
                    context_parts.append(f"  • Figures: {len(paper_info['figures'])} file(s)")
                if paper_info['data']:
                    context_parts.append(f"  • Data files: {len(paper_info['data'])} file(s)")
                if paper_info['sources']:
                    context_parts.append(f"  • Source/context files: {len(paper_info['sources'])} file(s)")
                if paper_info['bibliography']:
                    context_parts.append(f"  • Bibliography: {Path(paper_info['bibliography']).name}")
                if paper_info['progress_log']:
                    context_parts.append(f"  • Progress log: progress.md")
                if paper_info['summary']:
                    context_parts.append(f"  • Summary: SUMMARY.md")

                context_parts.append(f"\nUser request: {user_input}")
                contextual_prompt = "\n".join(context_parts)

            else:
                contextual_prompt = user_input

            print()
            async for message in query(prompt=contextual_prompt, options=options):
                if track_token_usage and hasattr(message, "usage") and message.usage:
                    usage = message.usage
                    total_input_tokens += getattr(usage, "input_tokens", 0)
                    total_output_tokens += getattr(usage, "output_tokens", 0)
                    total_cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0)
                    total_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0)

                if hasattr(message, "content") and message.content:
                    for block in message.content:
                        if hasattr(block, "text"):
                            print(block.text, end="", flush=True)

            print()

            # 没有 data_files 时也尝试探测是否有新建目录
            if not current_paper_path and not data_files:
                try:
                    paper_dirs = [d for d in output_folder.iterdir() if d.is_dir()]
                    if paper_dirs:
                        most_recent = max(paper_dirs, key=lambda d: d.stat().st_mtime)
                        time_since_modification = time.time() - most_recent.stat().st_mtime

                        # 10 秒以内才认为是刚创建的
                        if time_since_modification < 10:
                            current_paper_path = str(most_recent)
                            print(f"\n📂 Working on: {most_recent.name}")
                except Exception:
                    pass

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit or continue with a new prompt.")
            continue
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'exit' to quit.")

    if track_token_usage:
        return TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cache_creation_input_tokens=total_cache_creation_tokens,
            cache_read_input_tokens=total_cache_read_tokens,
        )
    return None


def _print_help():
    print("\n" + "=" * 70)
    print("帮助 - 学术写作工具")
    print("=" * 70)
    print("\n📝 What I Can Do:")
    print("  • Create complete scientific papers (LaTeX, Word, Markdown)")
    print("  • Literature reviews with citation management")
    print("  • Peer review feedback on drafts")
    print("  • Real-time research lookup using Perplexity Sonar Pro Search")
    print("  • Native web search for current events and general information")
    print("  • Format citations in any style (APA, IEEE, Nature, etc.)")
    print("  • Document manipulation (docx, pdf, pptx, xlsx)")
    print("\n🔄 How I Work:")
    print("  1. You describe what you need")
    print("  2. I present a brief plan and start execution immediately")
    print("  3. I provide continuous progress updates")
    print("  4. All files organized in writing_outputs/ folder")
    print("\n💡 Example Requests:")
    print("  'Create a NeurIPS paper on transformer attention mechanisms'")
    print("  'Write a literature review on CRISPR gene editing'")
    print("  'Review my methods section in draft.docx'")
    print("  'Research recent advances in quantum computing 2024'")
    print("  'Create a Nature paper on climate change impacts'")
    print("  'Format 20 citations in IEEE style'")
    print("\n📁 File Organization:")
    print("  All work saved to: writing_outputs/<timestamp>_<description>/")
    print("  - drafts/ - Working versions")
    print("  - final/ - Completed documents")
    print("  - references/ - Bibliography files")
    print("  - figures/ - Images and charts")
    print("  - data/ - Data files for the paper")
    print("  - sources/ - Context/reference materials")
    print("  - progress.md - Real-time progress log")
    print("  - SUMMARY.md - Project summary and instructions")
    print("\n📦 Data Files:")
    print("  Place files in the 'data/' folder at project root:")
    print("  • Manuscript files (.tex) → copied to drafts/ for EDITING")
    print("  • Context files (.md, .docx, .pdf) → copied to sources/ for REFERENCE")
    print("  • Data files (csv, txt, json, etc.) → copied to paper's data/")
    print("  • Images (png, jpg, svg, etc.) → copied to paper's figures/")
    print("  • Other files → copied to sources/ for CONTEXT")
    print("  • Files are used as context for the paper")
    print("  • Original files automatically deleted after copying")
    print("\n🎯 Pro Tips:")
    print("  • Be specific about journal/conference (e.g., 'Nature', 'NeurIPS')")
    print("  • Mention citation style if you have a preference")
    print("  • I'll make smart defaults if you don't specify details")
    print("  • Check progress.md for detailed execution logs")
    print("\n🔄 Intelligent Paper Detection:")
    print("  • I automatically detect when you're referring to a previous paper/presentation")
    print("  • Continue working: 'continue the paper', 'update my presentation', 'edit the poster'")
    print("  • Search/find: 'look for the X paper', 'find the presentation about Y'")
    print("  • Or mention the topic: 'show me the acoustics paper'")
    print("  • Keywords like 'continue', 'update', 'edit', 'look for', 'find' trigger detection")
    print("  • I'll find the most relevant paper/presentation based on topic matching")
    print("  • Say 'new paper' or 'start fresh' to explicitly begin a new one")
    print("  • Current working paper/presentation is tracked throughout the session")
    print("=" * 70)



async def run_workflow_mode(
    topic: str,
    knowledge_base_path: Optional[str] = None,
    max_iterations: int = 3,
    score_threshold: float = 7.0,
    track_token_usage: bool = False,
    output_dir: Optional[str] = None,
):
    """运行工作流式模式（含检索、大纲、撰写、审计、评审、修订循环）。"""
    print("=" * 70)
    print("🔬 学术写作工具 - 工作流模式")
    print("=" * 70)
    print(f"\n主题: {topic}")
    print(f"知识库: {knowledge_base_path or '未启用'}")
    print(f"最大迭代: {max_iterations}")
    print(f"通过阈值: {score_threshold}")
    if output_dir:
        print(f"目标目录: {output_dir} (复用已有论文)")
    print("\n工作流阶段:")
    print("  1️⃣  知识库准备（混合检索+Rerank）")
    print("  2️⃣  JSON大纲生成")
    print("  3️⃣  串行撰写链（5个节点）")
    print("  4️⃣  自动化审计（逻辑+引用）")
    print("  5️⃣  顶会同行评审（多维度评分+对抗性问题）")
    print("  6️⃣  闭环修改迭代（条件分支+熔断）")
    print("  7️⃣  LaTeX编译与输出")
    print("=" * 70)
    print()

    total_input_tokens = 0
    total_output_tokens = 0

    async for update in generate_paper_workflow(
        topic=topic,
        knowledge_base_path=knowledge_base_path,
        max_iterations=max_iterations,
        score_threshold=score_threshold,
        track_token_usage=track_token_usage,
        output_dir=output_dir,
    ):
        msg_type = update.get("type", "")

        if msg_type == "workflow_progress":
            stage = update.get("stage", "")
            message = update.get("message", "")
            iteration = update.get("iteration", 0)

            timestamp = update.get("timestamp", "")[11:19] if update.get("timestamp") else ""

            if iteration > 0:
                print(f"[{timestamp}] [{stage.upper()}] [迭代{iteration}] {message}")
            else:
                print(f"[{timestamp}] [{stage.upper()}] {message}")

            scores = update.get("current_scores")
            if scores:
                avg_score = sum(scores.values()) / len(scores)
                score_str = " | ".join([f"{k}={v:.1f}" for k, v in scores.items()])
                print(f"           📊 评分: {score_str} (平均: {avg_score:.1f})")

        elif msg_type == "workflow_result":
            print("\n" + "=" * 70)
            print("✅ 工作流执行完成!")
            print("=" * 70)
            print(f"\n📁 输出目录: {update.get('output_dir', 'N/A')}")
            print(f"📝 论文名称: {update.get('paper_name', 'N/A')}")
            print(f"🔄 迭代次数: {update.get('iteration_count', 0)}")

            final_scores = update.get('final_scores', {})
            if final_scores:
                avg = sum(final_scores.values()) / len(final_scores)
                print(f"\n📊 最终评分:")
                for dim, score in final_scores.items():
                    status = "✓" if score >= score_threshold else "✗"
                    print(f"  {status} {dim}: {score:.1f}/10")
                print(f"\n  综合评分: {avg:.1f}/10")

            token_usage = update.get('token_usage')
            if token_usage:
                print(f"\n📈 Token使用:")
                print(f"  输入: {token_usage.get('input_tokens', 0)}")
                print(f"  输出: {token_usage.get('output_tokens', 0)}")
                print(f"  总计: {token_usage.get('total_tokens', 0)}")

            print("=" * 70)

        elif msg_type == "error":
            print(f"\n❌ 错误: {update.get('message', 'Unknown error')}")

    return None


def _print_workflow_help():
    print("\n" + "=" * 70)
    print("工作流模式帮助")
    print("=" * 70)
    print("\n🔄 工作流模式提供完整的学术写作流水线:")
    print("  • 知识库集成（混合检索 + Rerank）")
    print("  • JSON大纲生成")
    print("  • 5节点串行撰写（保持连贯性）")
    print("  • 自动化审计（逻辑+引用+一致性）")
    print("  • 顶会同行评审（多维度1-10分评分）")
    print("  • 对抗性问题生成")
    print("  • 闭环修改迭代（条件分支+熔断机制）")
    print("\n📖 使用方法:")
    print("  writer workflow \"你的论文主题\"")
    print("\n📖 高级用法:")
    print("  writer workflow \"主题\" --kb ./pdfs --iterations 5 --threshold 8.0")
    print("\n参数说明:")
    print("  --kb, -k          知识库路径（包含PDF文件的目录）")
    print("  --iterations, -i  最大迭代次数（默认: 3）")
    print("  --threshold, -t   通过阈值（默认: 7.0）")
    print("  --output, -o      输出目录")
    print("  --token-usage     跟踪Token使用")
    print("\n💡 示例:")
    print('  writer workflow "Transformer在蛋白质结构预测中的应用"')
    print('  writer workflow "大语言模型的涌现能力研究" --kb ./papers --iterations 5')
    print("=" * 70)


def cli_main():
    """CLI 入口点。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="学术写作工具 - 基于 AI 的论文撰写",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    standard_parser = subparsers.add_parser("standard", help="标准模式（交互式）")
    standard_parser.add_argument("--token-usage", action="store_true", help="跟踪Token使用")

    workflow_parser = subparsers.add_parser("workflow", help="工作流模式（评审迭代）")
    workflow_parser.add_argument("topic", nargs="?", help="论文主题")
    workflow_parser.add_argument("--kb", "--knowledge-base", dest="kb_path",
                                help="知识库路径（PDF目录）")
    workflow_parser.add_argument("--iterations", "-i", type=int, default=3,
                                help="最大迭代次数（默认: 3）")
    workflow_parser.add_argument("--threshold", "-t", type=float, default=7.0,
                                help="通过阈值（默认: 7.0）")
    workflow_parser.add_argument("--output", "-o", help="输出目录")
    workflow_parser.add_argument("--token-usage", action="store_true", help="跟踪Token使用")

    args = parser.parse_args()

    # 没指定子命令则进入标准交互模式
    if args.command is None:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
        return

    if args.command == "standard":
        try:
            asyncio.run(main(track_token_usage=args.token_usage))
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)

    elif args.command == "workflow":
        if not args.topic:
            _print_workflow_help()
            sys.exit(0)

        try:
            asyncio.run(run_workflow_mode(
                topic=args.topic,
                knowledge_base_path=args.kb_path,
                max_iterations=args.iterations,
                score_threshold=args.threshold,
                track_token_usage=args.token_usage,
                output_dir=args.output,
            ))
        except KeyboardInterrupt:
            print("\n\n工作流已中断")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 工作流执行失败: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    cli_main()
