"""异步生成学术文档的对外 API。"""

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

from dotenv import load_dotenv

# 优先把工作目录下的 .env 读进来，否则后续 API key 取不到
_cwd_resolved = Path.cwd().resolve()
_env_file = _cwd_resolved / ".env"
if _env_file.exists():
    load_dotenv(dotenv_path=_env_file, override=True)

from claude_agent_sdk import query as claude_query, ClaudeAgentOptions
from claude_agent_sdk.types import HookMatcher, StopHookInput, HookContext

from .helpers import (
    count_citations_in_bib,
    count_words_in_tex,
    create_data_context_message,
    ensure_output_folder,
    extract_citation_style,
    extract_title_from_tex,
    get_api_key,
    get_data_files,
    load_system_instructions,
    process_data_files,
    scan_paper_directory,
    setup_claude_skills,
)
from .data_types import (
    PaperFiles,
    PaperMetadata,
    PaperResult,
    ProgressUpdate,
    TextUpdate,
    TokenUsage,
    WorkflowConfig,
    WorkflowProgressUpdate,
    WorkflowResult,
)
from .pipeline.controller import PipelineBuilder, WorkflowEngine
from .pipeline.stages import (
    AuditNode,
    NodeInput,
    OutlineNode,
    ReviewNode,
    RevisionNode,
    WritingNode,
)

# 从环境变量读默认模型
_DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "claude-sonnet-4-6")

EFFORT_LEVEL_MODELS = {
    "low": _DEFAULT_MODEL,
    "medium": _DEFAULT_MODEL,
    "high": _DEFAULT_MODEL,
}


def create_completion_check_stop_hook(auto_continue: bool = True):
    async def completion_check_stop_hook(
        hook_input: StopHookInput,
        matcher: str | None,
        context: HookContext,
    ) -> dict:
        if auto_continue:
            # 强制让 agent 继续执行，不允许它自己停下来
            return {"continue_": True}

        return {"continue_": False}

    return completion_check_stop_hook


async def generate_paper(
    query: str,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    effort_level: Literal["low", "medium", "high"] = "medium",
    data_files: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    track_token_usage: bool = False,
    auto_continue: bool = True,
) -> AsyncGenerator[Dict[str, Any], None]:
    """异步生成学术文档，过程中持续 yield 进度更新，最终 yield 一份完整结果。"""
    start_time = time.time()

    # 没传 model 时按 effort_level 取
    if model is None:
        model = EFFORT_LEVEL_MODELS[effort_level]

    if cwd:
        work_dir = Path(cwd).resolve()
    else:
        work_dir = Path.cwd().resolve()

    env_file = work_dir / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)

    try:
        api_key_value = get_api_key(api_key)
    except ValueError as e:
        yield _create_error_result(str(e))
        return

    package_dir = Path(__file__).parent.absolute()

    setup_claude_skills(package_dir, work_dir)

    output_folder = ensure_output_folder(work_dir, output_dir)

    yield ProgressUpdate(
        message="Initializing document generation",
        stage="initialization",
    ).to_dict()

    system_instructions = load_system_instructions(work_dir)

    system_instructions += "\n\n" + f"""
IMPORTANT - WORKING DIRECTORY:
- Your working directory is: {work_dir}
- ALWAYS create writing_outputs folder in this directory: {work_dir}/writing_outputs/
- NEVER write to /tmp/ or any other temporary directory
- All paper outputs MUST go to: {work_dir}/writing_outputs/<timestamp>_<description>/

IMPORTANT - CONVERSATION CONTINUITY:
- This is a NEW paper request - create a new paper directory
- Create a unique timestamped directory in the writing_outputs folder
- Do NOT assume there's an existing paper unless explicitly told in the prompt context
"""

    data_context = ""
    temp_paper_path = None

    if data_files:
        data_file_paths = get_data_files(work_dir, data_files)
        if data_file_paths:
            yield ProgressUpdate(
                message=f"Found {len(data_file_paths)} data file(s) to process",
                stage="initialization",
            ).to_dict()

    # 环境变量可以把 auto_continue 关掉
    env_auto_continue = os.environ.get("WRITER_AUTO_CONTINUE", "").lower()
    if env_auto_continue in ("false", "0", "no"):
        auto_continue = False

    options = ClaudeAgentOptions(
        system_prompt=system_instructions,
        model=model,
        allowed_tools=["Read", "Write", "Edit", "Bash", "WebSearch", "research_lookup"],
        permission_mode="bypassPermissions",
        setting_sources=["project"],
        cwd=str(work_dir),
        env={
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            **({"ANTHROPIC_BASE_URL": os.environ["ANTHROPIC_BASE_URL"]}
               if os.environ.get("ANTHROPIC_BASE_URL") else
               ({"ANTHROPIC_BASE_URL": os.environ["OPENAI_BASE_URL"]}
                if os.environ.get("OPENAI_BASE_URL") else {})),
            **({"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]}
               if os.environ.get("ANTHROPIC_API_KEY") else
               ({"ANTHROPIC_API_KEY": os.environ["OPENAI_API_KEY"]}
                if os.environ.get("OPENAI_API_KEY") else {})),
        },
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

    current_stage = "initialization"
    output_directory = None
    last_message = ""
    tool_call_count = 0
    files_written = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0

    yield ProgressUpdate(
        message="Starting document generation",
        stage="initialization",
        details={"query_length": len(query)},
    ).to_dict()

    try:
        accumulated_text = ""
        async for message in claude_query(prompt=query, options=options):
            if track_token_usage and hasattr(message, "usage") and message.usage:
                usage = message.usage
                total_input_tokens += getattr(usage, "input_tokens", 0)
                total_output_tokens += getattr(usage, "output_tokens", 0)
                total_cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0)
                total_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0)

            if hasattr(message, "content") and message.content:
                for block in message.content:
                    if hasattr(block, "text"):
                        text = block.text
                        accumulated_text += text

                        yield TextUpdate(content=text).to_dict()

                        stage, msg = _analyze_progress(accumulated_text, current_stage)

                        if stage != current_stage and msg and msg != last_message:
                            current_stage = stage
                            last_message = msg

                            yield ProgressUpdate(
                                message=msg,
                                stage=stage,
                            ).to_dict()

                    elif hasattr(block, "type") and block.type == "tool_use":
                        tool_call_count += 1
                        tool_name = getattr(block, "name", "unknown")
                        tool_input = getattr(block, "input", {})

                        if tool_name.lower() == "write":
                            file_path = tool_input.get("file_path", tool_input.get("path", ""))
                            if file_path:
                                files_written.append(file_path)

                        tool_progress = _analyze_tool_use(tool_name, tool_input, current_stage)

                        if tool_progress:
                            stage, msg = tool_progress
                            if msg != last_message:
                                current_stage = stage
                                last_message = msg

                                yield ProgressUpdate(
                                    message=msg,
                                    stage=stage,
                                    details={
                                        "tool": tool_name,
                                        "tool_calls": tool_call_count,
                                        "files_created": len(files_written),
                                    },
                                ).to_dict()

        yield ProgressUpdate(
            message="Scanning output directory",
            stage="complete",
        ).to_dict()

        output_directory = _find_most_recent_output(output_folder, start_time)

        if not output_directory:
            error_result = _create_error_result("Output directory not found after generation")
            if track_token_usage:
                error_result['token_usage'] = TokenUsage(
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cache_creation_input_tokens=total_cache_creation_tokens,
                    cache_read_input_tokens=total_cache_read_tokens,
                ).to_dict()
            yield error_result
            return

        if data_files:
            data_file_paths = get_data_files(work_dir, data_files)
            if data_file_paths:
                processed_info = process_data_files(
                    work_dir,
                    data_file_paths,
                    str(output_directory),
                    delete_originals=False  # 程序化调用时不删除用户原文件
                )
                if processed_info:
                    manuscript_count = len(processed_info.get('manuscript_files', []))
                    message = f"Processed {len(processed_info['all_files'])} file(s)"
                    if manuscript_count > 0:
                        message += f" ({manuscript_count} manuscript(s) copied to drafts/)"
                    yield ProgressUpdate(
                        message=message,
                        stage="complete",
                    ).to_dict()

        file_info = scan_paper_directory(output_directory)

        result = _build_paper_result(output_directory, file_info)

        if track_token_usage:
            result.token_usage = TokenUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_creation_input_tokens=total_cache_creation_tokens,
                cache_read_input_tokens=total_cache_read_tokens,
            )

        yield ProgressUpdate(
            message="Document generation complete",
            stage="complete",
        ).to_dict()

        yield result.to_dict()

    except Exception as e:
        error_result = _create_error_result(f"Error during document generation: {str(e)}")
        if track_token_usage:
            error_result['token_usage'] = TokenUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_creation_input_tokens=total_cache_creation_tokens,
                cache_read_input_tokens=total_cache_read_tokens,
            ).to_dict()
        yield error_result


def _analyze_progress(text: str, current_stage: str) -> tuple:
    text_lower = text.lower()

    stage_order = ["initialization", "planning", "research", "writing", "compilation", "complete"]
    current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0

    # 只在出现明显信号时才切换大阶段，避免误判
    if current_idx < stage_order.index("compilation"):
        if "pdflatex" in text_lower or "latexmk" in text_lower or "compiling" in text_lower:
            return "compilation", "Compiling document"

    if current_idx < stage_order.index("complete"):
        if "successfully compiled" in text_lower or "pdf generated" in text_lower:
            return "complete", "Finalizing output"

    return current_stage, None


def _detect_document_type(file_path: str) -> str:
    path_lower = file_path.lower()
    if "slide" in path_lower or "presentation" in path_lower or "beamer" in path_lower:
        return "slides"
    elif "poster" in path_lower:
        return "poster"
    elif "report" in path_lower:
        return "report"
    elif "grant" in path_lower or "proposal" in path_lower:
        return "grant"
    return "document"


def _get_section_from_filename(filename: str) -> str:
    name_lower = filename.lower().replace('.tex', '').replace('.md', '')

    section_mappings = {
        'abstract': 'abstract',
        'intro': 'introduction',
        'introduction': 'introduction',
        'method': 'methods',
        'methods': 'methods',
        'methodology': 'methodology',
        'result': 'results',
        'results': 'results',
        'discussion': 'discussion',
        'conclusion': 'conclusion',
        'conclusions': 'conclusions',
        'background': 'background',
        'related': 'related work',
        'experiment': 'experiments',
        'experiments': 'experiments',
        'evaluation': 'evaluation',
        'appendix': 'appendix',
        'supplement': 'supplementary material',
    }

    for key, section in section_mappings.items():
        if key in name_lower:
            return section
    return None


def _analyze_tool_use(tool_name: str, tool_input: Dict[str, Any], current_stage: str) -> tuple:
    stage_order = ["initialization", "planning", "research", "writing", "compilation", "complete"]
    current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0

    file_path = tool_input.get("file_path", tool_input.get("path", ""))
    command = tool_input.get("command", "")
    filename = Path(file_path).name if file_path else ""
    doc_type = _detect_document_type(file_path)

    if tool_name.lower() == "read":
        if ".bib" in file_path:
            return ("writing", f"Reading bibliography: {filename}")
        elif ".tex" in file_path:
            section = _get_section_from_filename(filename)
            if section:
                return ("writing", f"Reading {section} section")
            return ("writing", f"Reading {filename}")
        elif ".pdf" in file_path:
            return ("research", f"Analyzing PDF: {filename}")
        elif ".csv" in file_path:
            return ("research", f"Loading data from {filename}")
        elif ".json" in file_path:
            return ("research", f"Reading configuration: {filename}")
        elif ".md" in file_path:
            return ("planning", f"Reading {filename}")
        elif file_path:
            return (current_stage, f"Reading {filename}")
        return None

    elif tool_name.lower() == "write":
        if ".bib" in file_path:
            return ("writing", f"Creating bibliography with references")
        elif ".tex" in file_path:
            section = _get_section_from_filename(filename)
            if section:
                return ("writing", f"Writing {section} section")
            elif "main" in filename.lower():
                return ("writing", f"Creating main {doc_type} structure")
            elif current_idx < stage_order.index("writing"):
                return ("writing", f"Writing {doc_type}: {filename}")
            else:
                return ("compilation", f"Updating {filename}")
        elif ".md" in file_path:
            if "progress" in filename.lower():
                return ("writing", "Updating progress log")
            elif "readme" in filename.lower():
                return ("complete", "Creating documentation")
            return ("writing", f"Writing {filename}")
        elif ".sty" in file_path:
            return ("writing", f"Creating style file: {filename}")
        elif ".cls" in file_path:
            return ("writing", f"Creating document class: {filename}")
        elif file_path:
            return (current_stage, f"Creating {filename}")
        return None

    elif tool_name.lower() == "edit":
        if ".tex" in file_path:
            section = _get_section_from_filename(filename)
            if section:
                return ("writing", f"Refining {section} section")
            return ("writing", f"Editing {filename}")
        elif ".bib" in file_path:
            return ("writing", "Updating bibliography")
        elif file_path:
            return (current_stage, f"Editing {filename}")
        return None

    elif tool_name.lower() == "bash":
        if "pdflatex" in command:
            if "-output-directory" in command:
                return ("compilation", "Compiling PDF with output directory")
            return ("compilation", "Compiling LaTeX to PDF")
        elif "latexmk" in command:
            return ("compilation", "Running full LaTeX compilation pipeline")
        elif "bibtex" in command:
            return ("compilation", "Processing bibliography citations")
        elif "makeindex" in command:
            return ("compilation", "Building document index")
        elif "mkdir" in command:
            if "writing_outputs" in command or "output" in command.lower():
                return ("initialization", "Creating output directory")
            elif "figures" in command.lower():
                return ("initialization", "Setting up figures directory")
            elif "drafts" in command.lower():
                return ("initialization", "Setting up drafts directory")
            return ("initialization", "Creating directory structure")
        elif "cp " in command:
            if ".pdf" in command:
                return ("complete", "Copying final PDF to output")
            elif ".tex" in command:
                return ("complete", "Archiving LaTeX source")
            return ("complete", "Organizing files")
        elif "mv " in command:
            return ("complete", "Moving files to final location")
        elif "ls " in command or "cat " in command:
            # 这类只读命令不上报进度
            return None
        elif command:
            cmd_preview = command.split()[0] if command.split() else command[:30]
            return (current_stage, f"Running {cmd_preview}")
        return None

    elif "research" in tool_name.lower() or "lookup" in tool_name.lower():
        query_text = tool_input.get("query", "")
        if query_text:
            truncated = query_text[:50] + "..." if len(query_text) > 50 else query_text
            return ("research", f"Searching: {truncated}")
        return ("research", "Searching literature databases")

    elif "search" in tool_name.lower() or "web" in tool_name.lower():
        query_text = tool_input.get("query", tool_input.get("search_term", ""))
        if query_text:
            truncated = query_text[:40] + "..." if len(query_text) > 40 else query_text
            return ("research", f"Web search: {truncated}")
        return ("research", "Searching online resources")

    return None


def _find_most_recent_output(output_folder: Path, start_time: float) -> Optional[Path]:
    try:
        output_dirs = [d for d in output_folder.iterdir() if d.is_dir()]
        if not output_dirs:
            return None

        # 留 5 秒缓冲，避免文件系统时间差导致漏掉刚刚创建的目录
        recent_dirs = [
            d for d in output_dirs
            if d.stat().st_mtime >= start_time - 5
        ]

        if not recent_dirs:
            recent_dirs = output_dirs

        most_recent = max(recent_dirs, key=lambda d: d.stat().st_mtime)
        return most_recent
    except Exception:
        return None


def _build_paper_result(paper_dir: Path, file_info: Dict[str, Any]) -> PaperResult:
    tex_file = file_info['tex_final'] or (file_info['tex_drafts'][0] if file_info['tex_drafts'] else None)

    title = extract_title_from_tex(tex_file)
    word_count = count_words_in_tex(tex_file)

    topic = ""
    parts = paper_dir.name.split('_', 2)
    if len(parts) >= 3:
        topic = parts[2].replace('_', ' ')

    metadata = PaperMetadata(
        title=title,
        created_at=datetime.fromtimestamp(paper_dir.stat().st_ctime).isoformat() + "Z",
        topic=topic,
        word_count=word_count,
    )

    files = PaperFiles(
        pdf_final=file_info['pdf_final'],
        tex_final=file_info['tex_final'],
        pdf_drafts=file_info['pdf_drafts'],
        tex_drafts=file_info['tex_drafts'],
        bibliography=file_info['bibliography'],
        figures=file_info['figures'],
        data=file_info['data'],
        progress_log=file_info['progress_log'],
        summary=file_info['summary'],
    )

    citation_count = count_citations_in_bib(file_info['bibliography'])
    citation_style = extract_citation_style(file_info['bibliography'])

    citations = {
        'count': citation_count,
        'style': citation_style,
        'file': file_info['bibliography'],
    }

    status = "success"
    compilation_success = file_info['pdf_final'] is not None

    if not compilation_success:
        if file_info['tex_final']:
            # tex 写出来了但 PDF 没编出来，算部分成功
            status = "partial"
        else:
            status = "failed"

    result = PaperResult(
        status=status,
        paper_directory=str(paper_dir),
        paper_name=paper_dir.name,
        metadata=metadata,
        files=files,
        citations=citations,
        figures_count=len(file_info['figures']),
        compilation_success=compilation_success,
        errors=[],
    )

    return result


def _create_error_result(error_message: str) -> Dict[str, Any]:
    result = PaperResult(
        status="failed",
        paper_directory="",
        paper_name="",
        errors=[error_message],
    )
    return result.to_dict()


async def generate_paper_workflow(
    topic: str,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    effort_level: Literal["low", "medium", "high"] = "high",
    knowledge_base_path: Optional[str] = None,
    max_iterations: int = 3,
    score_threshold: float = 7.0,
    cwd: Optional[str] = None,
    track_token_usage: bool = False,
) -> AsyncGenerator[Dict[str, Any], None]:
    """通过工作流式流水线生成论文（含检索、大纲、撰写、审计、评审、修订循环）。"""
    start_time = time.time()

    # 解析工作目录
    if cwd:
        work_dir = Path(cwd).resolve()
    else:
        work_dir = Path.cwd().resolve()

    # 加载环境变量
    env_file = work_dir / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)

    # 获取API key
    try:
        api_key_value = get_api_key(api_key)
    except ValueError as e:
        yield WorkflowProgressUpdate(
            stage="initialization",
            message=f"Error: {str(e)}"
        ).to_dict()
        return

    # 创建配置
    config = WorkflowConfig(
        max_iterations=max_iterations,
        score_threshold=score_threshold,
        use_knowledge_base=bool(knowledge_base_path),
        retrieval_method="hybrid",
        use_rerank=True,
        top_k=10,
        latex_format_required=True,
        output_format="latex",
        save_intermediate=True
    )

    # 初始化工作流引擎
    engine = WorkflowEngine(
        work_dir=work_dir,
        config=config,
        api_key=api_key_value,
        output_dir=Path(output_dir) if output_dir else None,
    )

    # 解析知识库路径
    kb_path = Path(knowledge_base_path) if knowledge_base_path else None

    # 执行工作流
    token_usage_stats = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0
    }

    async for update in engine.execute(
        topic=topic,
        paper_type="research_paper",
        knowledge_base_path=kb_path
    ):
        # 转换更新格式
        if update.get("type") == "progress":
            progress_update = WorkflowProgressUpdate(
                stage=update.get("stage", ""),
                message=update.get("message", ""),
                iteration=update.get("details", {}).get("iteration", 0),
                current_node=update.get("details", {}).get("node_id"),
                current_scores=update.get("details", {}).get("scores"),
                details=update.get("details")
            )
            yield progress_update.to_dict()

        elif update.get("type") == "error":
            yield WorkflowProgressUpdate(
                stage=update.get("stage", ""),
                message=f"Error: {update.get('message', '')}"
            ).to_dict()

        elif update.get("type") == "result":
            # 最终结果
            final_result = WorkflowResult(
                status=update.get("status", "success"),
                output_dir=update.get("output_dir", ""),
                paper_name=update.get("paper_name", ""),
                final_content=update.get("content_preview", ""),
                iteration_count=update.get("iteration_count", 0),
                final_scores=update.get("final_scores", {}),
                node_states=[],
                review_history=[]
            )

            if track_token_usage:
                final_result.token_usage = TokenUsage(**token_usage_stats)

            yield final_result.to_dict()


async def run_workflow_step(
    step_name: str,
    input_data: Dict[str, Any],
    config: Optional[WorkflowConfig] = None,
    work_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """单独跑工作流的某一个步骤，便于调试。"""
    # 根据步骤名称创建对应节点
    node = None
    if step_name == "outline":
        node = OutlineNode("outline", config={})
    elif step_name.startswith("writing_"):
        section_order = int(step_name.split("_")[1])
        sections = ["introduction", "methods", "results", "discussion", "conclusion"]
        node = WritingNode(
            step_name,
            config={
                "section_type": sections[section_order - 1],
                "section_order": section_order
            }
        )
    elif step_name == "audit":
        node = AuditNode("audit", config={})
    elif step_name == "review":
        node = ReviewNode("review", config={})
    elif step_name == "revision":
        node = RevisionNode("revision", config={})
    else:
        return {"status": "error", "message": f"Unknown step: {step_name}"}

    # 创建输入
    node_input = NodeInput(
        content=input_data.get("content", ""),
        context=input_data.get("context", {}),
        previous_outputs=input_data.get("previous_outputs", []),
        metadata=input_data.get("metadata", {})
    )

    # 执行节点
    result = await node.execute(node_input)

    return result.to_dict()
