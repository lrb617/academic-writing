"""通用辅助函数：API key、数据文件处理、目录扫描、文本统计等。"""

import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


def setup_claude_skills(package_dir: Path, work_dir: Path) -> None:
    """把包内 .claude/ 目录复制到用户工作目录。"""
    source_claude = package_dir / ".claude"
    dest_claude = work_dir / ".claude"

    # 仅当目标目录不存在时才复制，避免覆盖用户自己的配置
    if source_claude.exists() and not dest_claude.exists():
        try:
            shutil.copytree(source_claude, dest_claude)
        except Exception as e:
            pass


def get_api_key(api_key: Optional[str] = None) -> str:
    """获取 API key（优先级：参数 > OPENAI_API_KEY > ANTHROPIC_API_KEY）。"""
    if api_key:
        return api_key

    env_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not env_key:
        raise ValueError(
            "API key not found. Either pass api_key parameter or set "
            "OPENAI_API_KEY (for OpenAI-compatible endpoints) or "
            "ANTHROPIC_API_KEY environment variable."
        )
    return env_key


def load_system_instructions(work_dir: Path) -> str:
    """从 .claude/WRITER.md 中读取系统提示词。"""
    instructions_file = work_dir / ".claude" / "WRITER.md"

    if instructions_file.exists():
        with open(instructions_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # 找不到 WRITER.md 时给一个最简的默认提示
        return (
            "You are a scientific writing assistant. Follow best practices for "
            "scientific communication and always present a plan before execution."
        )


def ensure_output_folder(cwd: Path, custom_dir: Optional[str] = None) -> Path:
    """确保输出文件夹存在，返回其路径。"""
    if custom_dir:
        output_folder = Path(custom_dir).resolve()
    else:
        output_folder = cwd / "writing_outputs"

    output_folder.mkdir(exist_ok=True, parents=True)
    return output_folder


def get_image_extensions() -> set:
    return {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.svg', '.webp', '.ico'}


def get_manuscript_extensions() -> set:
    return {'.tex'}


def get_source_extensions() -> set:
    return {'.md', '.docx', '.pdf'}


def get_data_extensions() -> set:
    return {'.csv', '.json', '.txt', '.xlsx', '.xls', '.tsv', '.xml', '.yaml', '.yml', '.sql'}


def get_data_files(cwd: Path, data_files: Optional[List[str]] = None) -> List[Path]:
    """获取数据文件列表（来自参数或 data/ 文件夹）。"""
    if data_files:
        return [Path(f).resolve() for f in data_files]

    data_folder = cwd / "data"
    if not data_folder.exists():
        return []

    files = []
    for file_path in data_folder.iterdir():
        if file_path.is_file():
            files.append(file_path)

    return files


def extract_images_from_docx(docx_path: Path, figures_output: Path) -> List[Dict[str, Any]]:
    """从 .docx 文件中抽出 word/media/ 下的图片，复制到 figures 目录。"""
    extracted_images = []
    image_extensions = get_image_extensions()

    try:
        with zipfile.ZipFile(docx_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()

            # docx 本质是 zip，图片都放在 word/media/ 目录里
            media_files = [f for f in all_files if f.startswith('word/media/')]

            for media_file in media_files:
                file_name = Path(media_file).name
                file_ext = Path(media_file).suffix.lower()

                if file_ext in image_extensions:
                    output_path = figures_output / file_name

                    with zip_ref.open(media_file) as source:
                        with open(output_path, 'wb') as target:
                            target.write(source.read())

                    extracted_images.append({
                        'name': file_name,
                        'path': str(output_path),
                        'source_docx': docx_path.name
                    })

    except zipfile.BadZipFile:
        print(f"Warning: {docx_path.name} is not a valid .docx file (ZIP archive)")
    except Exception as e:
        print(f"Warning: Could not extract images from {docx_path.name}: {str(e)}")

    return extracted_images


def process_data_files(
    cwd: Path,
    data_files: List[Path],
    paper_output_path: str,
    delete_originals: bool = True
) -> Optional[Dict[str, Any]]:
    """根据文件类型把数据文件分发到论文目录的不同子文件夹。"""
    if not data_files:
        return None

    paper_output = Path(paper_output_path)
    data_output = paper_output / "data"
    figures_output = paper_output / "figures"
    drafts_output = paper_output / "drafts"
    sources_output = paper_output / "sources"

    data_output.mkdir(parents=True, exist_ok=True)
    figures_output.mkdir(parents=True, exist_ok=True)
    drafts_output.mkdir(parents=True, exist_ok=True)
    sources_output.mkdir(parents=True, exist_ok=True)

    image_extensions = get_image_extensions()
    manuscript_extensions = get_manuscript_extensions()
    source_extensions = get_source_extensions()
    data_extensions = get_data_extensions()

    processed_info = {
        'data_files': [],
        'image_files': [],
        'manuscript_files': [],
        'source_files': [],
        'all_files': []
    }

    for file_path in data_files:
        file_ext = file_path.suffix.lower()
        file_name = file_path.name

        if file_ext in manuscript_extensions:
            # .tex 视为待编辑稿件，统一进 drafts/
            destination = drafts_output / file_name
            file_type = 'manuscript'
            processed_info['manuscript_files'].append({
                'name': file_name,
                'path': str(destination),
                'original': str(file_path),
                'extension': file_ext
            })
        elif file_ext in image_extensions:
            destination = figures_output / file_name
            file_type = 'image'
            processed_info['image_files'].append({
                'name': file_name,
                'path': str(destination),
                'original': str(file_path)
            })
        elif file_ext in data_extensions:
            destination = data_output / file_name
            file_type = 'data'
            processed_info['data_files'].append({
                'name': file_name,
                'path': str(destination),
                'original': str(file_path)
            })
        else:
            # 其余文件统一进 sources/
            destination = sources_output / file_name
            file_type = 'source'
            processed_info['source_files'].append({
                'name': file_name,
                'path': str(destination),
                'original': str(file_path),
                'extension': file_ext
            })

        try:
            shutil.copy2(file_path, destination)
            processed_info['all_files'].append({
                'name': file_name,
                'type': file_type,
                'destination': str(destination)
            })

            # docx 内嵌图片也一并抽出来
            if file_ext == '.docx':
                extracted_images = extract_images_from_docx(file_path, figures_output)
                if extracted_images:
                    for img_info in extracted_images:
                        processed_info['image_files'].append(img_info)

            if delete_originals:
                file_path.unlink()

        except Exception as e:
            print(f"Warning: Could not process {file_name}: {str(e)}")

    return processed_info


def create_data_context_message(processed_info: Optional[Dict[str, Any]]) -> str:
    """生成传给 LLM 的"已有数据文件"上下文片段。"""
    if not processed_info or not processed_info['all_files']:
        return ""

    context_parts = ["\n[DATA FILES AVAILABLE]"]

    # 出现 .tex 稿件视为编辑模式，需要特别提示
    if processed_info.get('manuscript_files'):
        context_parts.append("\n⚠️  EDITING MODE - Manuscript files (.tex) detected!")
        context_parts.append("\nManuscript files (in drafts/ folder for editing):")
        for file_info in processed_info['manuscript_files']:
            context_parts.append(f"  - {file_info['name']} ({file_info['extension']}): {file_info['path']}")
        context_parts.append("\n🔧 TASK: This is an EDITING task, not creating from scratch.")
        context_parts.append("   → Read the existing manuscript from drafts/")
        context_parts.append("   → Apply the requested changes/improvements")
        context_parts.append("   → Create new version following version numbering protocol")
        context_parts.append("   → Document changes in revision_notes.md")

    if processed_info.get('source_files'):
        context_parts.append("\nSource/Context files (in sources/ folder for reference):")
        for file_info in processed_info['source_files']:
            ext = file_info.get('extension', '')
            context_parts.append(f"  - {file_info['name']} ({ext}): {file_info['path']}")
        context_parts.append("\nNote: These files are available as reference/context material.")

    if processed_info.get('data_files'):
        context_parts.append("\nData files (in data/ folder):")
        for file_info in processed_info['data_files']:
            context_parts.append(f"  - {file_info['name']}: {file_info['path']}")

    if processed_info.get('image_files'):
        direct_images = [img for img in processed_info['image_files'] if 'source_docx' not in img]
        extracted_images = [img for img in processed_info['image_files'] if 'source_docx' in img]

        context_parts.append("\nImage files (in figures/ folder):")

        if direct_images:
            context_parts.append("  Directly provided:")
            for file_info in direct_images:
                context_parts.append(f"    - {file_info['name']}: {file_info['path']}")

        if extracted_images:
            from collections import defaultdict
            images_by_docx = defaultdict(list)
            for img in extracted_images:
                images_by_docx[img['source_docx']].append(img)

            context_parts.append("  Extracted from .docx files:")
            for docx_name, images in images_by_docx.items():
                img_names = ', '.join([img['name'] for img in images])
                context_parts.append(f"    - From {docx_name}: {img_names}")

        context_parts.append("\nNote: These images can be referenced as figures in the paper.")

    context_parts.append("[END DATA FILES]\n")

    return "\n".join(context_parts)


def find_existing_papers(output_folder: Path) -> List[Dict[str, Any]]:
    """获取已有论文目录列表（按修改时间排序）。"""
    papers = []
    if not output_folder.exists():
        return papers

    for paper_dir in output_folder.iterdir():
        if paper_dir.is_dir():
            papers.append({
                'path': paper_dir,
                'name': paper_dir.name,
                'mtime': paper_dir.stat().st_mtime
            })

    papers.sort(key=lambda x: x['mtime'], reverse=True)
    return papers


def detect_paper_reference(user_input: str, existing_papers: List[Dict[str, Any]]) -> Optional[Path]:
    """根据用户输入推断是否在指代某篇已有论文。"""
    if not existing_papers:
        return None

    user_input_lower = user_input.lower()

    continuation_keywords = [
        "continue", "update", "edit", "revise", "modify", "change",
        "add to", "fix", "improve", "review", "the paper", "this paper",
        "my paper", "current paper", "previous paper", "last paper",
        "poster", "the poster", "my poster", "presentation", "the presentation",
        "my presentation", "previous presentation", "last presentation",
        "compile", "generate pdf"
    ]

    search_keywords = [
        "look for", "find", "search for", "where is", "which paper",
        "show me", "open", "locate", "get"
    ]

    new_paper_keywords = [
        "new paper", "start fresh", "start afresh", "create new",
        "different paper", "another paper", "write a new",
        "new presentation", "new poster", "different presentation", "another presentation"
    ]

    if any(keyword in user_input_lower for keyword in new_paper_keywords):
        return None

    has_continuation_keyword = any(keyword in user_input_lower for keyword in continuation_keywords)
    has_search_keyword = any(keyword in user_input_lower for keyword in search_keywords)

    best_match = None
    best_match_score = 0

    for paper in existing_papers:
        paper_name = paper['name'].lower()
        # 目录名格式：YYYYMMDD_HHMMSS_topic
        parts = paper_name.split('_', 2)
        if len(parts) >= 3:
            topic = parts[2].replace('_', ' ')
            topic_words = topic.split()
            matches = sum(1 for word in topic_words if len(word) > 3 and word in user_input_lower)

            if matches > best_match_score:
                best_match_score = matches
                best_match = paper['path']

            # 命中两个以上的关键词时直接返回
            if matches >= 2 and (has_search_keyword or has_continuation_keyword):
                return paper['path']

    if has_search_keyword and best_match_score > 0:
        return best_match

    # 仅出现"继续"类关键词、又没有具体匹配，则取最近的一篇
    if has_continuation_keyword and existing_papers:
        return existing_papers[0]['path']

    return None


def scan_paper_directory(paper_dir: Path) -> Dict[str, Any]:
    """扫描论文目录，整理出所有相关文件信息。"""
    result = {
        'pdf_final': None,
        'tex_final': None,
        'pdf_drafts': [],
        'tex_drafts': [],
        'bibliography': None,
        'figures': [],
        'data': [],
        'sources': [],
        'progress_log': None,
        'summary': None,
    }

    if not paper_dir.exists():
        return result

    final_dir = paper_dir / "final"
    if final_dir.exists():
        for file in final_dir.iterdir():
            if file.is_file():
                if file.suffix == '.pdf':
                    result['pdf_final'] = str(file)
                elif file.suffix == '.tex':
                    result['tex_final'] = str(file)

    drafts_dir = paper_dir / "drafts"
    if drafts_dir.exists():
        for file in sorted(drafts_dir.iterdir()):
            if file.is_file():
                if file.suffix == '.pdf':
                    result['pdf_drafts'].append(str(file))
                elif file.suffix == '.tex':
                    result['tex_drafts'].append(str(file))

    references_dir = paper_dir / "references"
    if references_dir.exists():
        bib_file = references_dir / "references.bib"
        if bib_file.exists():
            result['bibliography'] = str(bib_file)

    figures_dir = paper_dir / "figures"
    if figures_dir.exists():
        for file in sorted(figures_dir.iterdir()):
            if file.is_file():
                result['figures'].append(str(file))

    data_dir = paper_dir / "data"
    if data_dir.exists():
        for file in sorted(data_dir.iterdir()):
            if file.is_file():
                result['data'].append(str(file))

    sources_dir = paper_dir / "sources"
    if sources_dir.exists():
        for file in sorted(sources_dir.iterdir()):
            if file.is_file():
                result['sources'].append(str(file))

    progress_file = paper_dir / "progress.md"
    if progress_file.exists():
        result['progress_log'] = str(progress_file)

    summary_file = paper_dir / "SUMMARY.md"
    if summary_file.exists():
        result['summary'] = str(summary_file)

    return result


def count_citations_in_bib(bib_file: Optional[str]) -> int:
    """统计 .bib 文件里的引用条目数。"""
    if not bib_file or not Path(bib_file).exists():
        return 0

    try:
        with open(bib_file, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = re.findall(r'@\w+\s*{', content)
            return len(matches)
    except Exception:
        return 0


def extract_citation_style(bib_file: Optional[str]) -> str:
    """简单返回引用风格名称。"""
    return "BibTeX"


def count_words_in_tex(tex_file: Optional[str]) -> Optional[int]:
    """估算 LaTeX 文件中的词数。"""
    if not tex_file or not Path(tex_file).exists():
        return None

    try:
        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()

            # 去掉 LaTeX 命令、注释和特殊符号后再分词
            content = re.sub(r'\\[a-zA-Z]+(\[.*?\])?(\{.*?\})?', '', content)
            content = re.sub(r'%.*', '', content)
            content = re.sub(r'[{}$\\]', '', content)

            words = content.split()
            return len(words)
    except Exception:
        return None


def extract_title_from_tex(tex_file: Optional[str]) -> Optional[str]:
    """从 LaTeX 文件中提取 \\title{...} 的标题。"""
    if not tex_file or not Path(tex_file).exists():
        return None

    try:
        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()

            match = re.search(r'\\title\s*\{([^}]+)\}', content)
            if match:
                title = match.group(1)
                title = re.sub(r'\\[a-zA-Z]+(\[.*?\])?(\{.*?\})?', '', title)
                return title.strip()
    except Exception:
        pass

    return None
