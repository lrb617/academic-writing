"""学术写作流水线的所有阶段节点：基类、大纲、撰写、审计、评审、条件、修改、工具。"""

import base64
import io
import json
import os
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 文本输出中提取第一个有效的 JSON 对象。

    依次尝试：```json``` 代码块 → 任意 ``` ``` 代码块 → 文本里第一个 `{...}` 跨度。
    全部失败返回 None（调用方负责降级）。
    """
    fenced = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
    for chunk in fenced:
        try:
            return json.loads(chunk.strip())
        except json.JSONDecodeError:
            continue

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _extract_score_from_text(text: str) -> Optional[float]:
    """正则兜底：从任意文本里抽取首个 score 数值（1-10 之间）。

    用于 _extract_json_block 解析失败时（例如 LLM 在 justification 字符串里混了
    未转义的半角双引号、或整体输出非合法 JSON）的最后一道防线——只要 score 这个
    最关键字段还能识别出来，就不必降级到 5.0。
    """
    match = re.search(r'["\']?score["\']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return None
    if value < 0 or value > 10:
        return None
    return value


def _build_subprocess_env() -> Dict[str, str]:
    """构造传给 claude_query 子进程的 env，锁定模型 + 代理 URL + API Key。

    与 writer/main.py、writer/runner.py 中给顶层 Agent 注入的 env 保持一致，
    避免被用户级 ~/.claude/settings.json 中的 env 块污染（例如默认改成 kimi）。
    """
    model = os.environ.get("OPENAI_MODEL", "claude-sonnet-4-6")
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

    env: Dict[str, str] = {
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
    }
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    return env


class NodeStatus(Enum):
    """节点执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeType(Enum):
    """节点类型"""
    OUTLINE = "outline"
    WRITING = "writing"
    AUDIT = "audit"
    REVIEW = "review"
    CONDITION = "condition"
    REVISION = "revision"
    TOOL = "tool"


@dataclass
class NodeInput:
    """节点输入数据"""
    content: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    previous_outputs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_context_value(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "context": self.context,
            "previous_outputs": self.previous_outputs,
            "metadata": self.metadata,
        }


@dataclass
class NodeOutput:
    """节点输出数据"""
    content: str = ""
    status: NodeStatus = NodeStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    # 审计/评审专用字段
    scores: Optional[Dict[str, float]] = None
    adversarial_questions: Optional[List[str]] = None
    revision_suggestions: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "status": self.status.value,
            "metadata": self.metadata,
            "errors": self.errors,
            "scores": self.scores,
            "adversarial_questions": self.adversarial_questions,
            "revision_suggestions": self.revision_suggestions,
        }


class BaseNode(ABC):
    """工作流节点基类"""

    def __init__(self, node_id: str, node_type: NodeType, config: Dict[str, Any] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.config = config or {}
        self.status = NodeStatus.PENDING

        self.execution_time: Optional[float] = None
        self.token_usage: Dict[str, int] = field(default_factory=dict)

    @abstractmethod
    async def execute(self, input_data: NodeInput) -> NodeOutput:
        """执行节点逻辑，子类必须实现"""
        pass

    def validate_input(self, input_data: NodeInput) -> bool:
        return True

    def get_dependencies(self) -> List[str]:
        return self.config.get("dependencies", [])

    def _create_success_output(
        self,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> NodeOutput:
        return NodeOutput(
            content=content,
            status=NodeStatus.COMPLETED,
            metadata=metadata or {}
        )

    def _create_error_output(
        self,
        error_message: str,
        metadata: Dict[str, Any] = None
    ) -> NodeOutput:
        return NodeOutput(
            content="",
            status=NodeStatus.FAILED,
            errors=[error_message],
            metadata=metadata or {}
        )

    def get_required_model(self) -> str:
        return self.config.get("model", "claude-sonnet-4-6")


class OutlineNode(BaseNode):
    """大纲生成节点"""

    def __init__(self, node_id: str, config: Dict[str, Any] = None):
        super().__init__(node_id, NodeType.OUTLINE, config)
        self.output_format = config.get("output_format", "json")

    async def execute(self, input_data: NodeInput) -> NodeOutput:
        self.status = NodeStatus.RUNNING

        try:
            topic = input_data.get_context_value("topic", "")
            paper_type = input_data.get_context_value("paper_type", "research_paper")
            format_requirement = input_data.get_context_value("format_requirement", "")

            prompt = self._build_prompt(topic, paper_type, format_requirement)

            outline_content = await self._call_llm(prompt)

            try:
                outline_json = json.loads(outline_content)
            except json.JSONDecodeError:
                outline_json = self._extract_json(outline_content)

            if not self._validate_outline(outline_json):
                return self._create_error_output("生成的大纲结构不完整")

            self.status = NodeStatus.COMPLETED

            return NodeOutput(
                content=json.dumps(outline_json, ensure_ascii=False, indent=2),
                status=NodeStatus.COMPLETED,
                metadata={
                    "outline_structure": outline_json,
                    "sections_count": len(outline_json.get("sections", [])),
                    "title": outline_json.get("title", ""),
                    "node_id": self.node_id,
                }
            )

        except Exception as e:
            self.status = NodeStatus.FAILED
            return self._create_error_output(f"大纲生成失败: {str(e)}")

    def _build_prompt(self, topic: str, paper_type: str, format_requirement: str) -> str:
        return f"""你是一位资深学术写作专家。请为以下主题生成论文大纲，必须以JSON格式输出。

主题：{topic}
论文类型：{paper_type}
格式要求：{format_requirement}

请生成以下结构的JSON大纲：
{{
    "title": "论文标题",
    "abstract_summary": "摘要要点（100字以内）",
    "sections": [
        {{
            "section_id": "sec_1",
            "section_name": "Introduction",
            "subsections": [
                {{
                    "subsection_id": "sec_1_1",
                    "name": "Background",
                    "key_points": ["要点1", "要点2"],
                    "citations_needed": 3
                }}
            ],
            "estimated_length": "800 words",
            "main_contribution": "本节主要贡献"
        }},
        {{
            "section_id": "sec_2",
            "section_name": "Methods",
            "subsections": [
                {{
                    "subsection_id": "sec_2_1",
                    "name": "Experimental Setup",
                    "key_points": ["实验设计", "数据收集"],
                    "citations_needed": 2
                }}
            ],
            "estimated_length": "1000 words",
            "main_contribution": "方法论创新点"
        }},
        {{
            "section_id": "sec_3",
            "section_name": "Results",
            "subsections": [
                {{
                    "subsection_id": "sec_3_1",
                    "name": "Main Findings",
                    "key_points": ["主要发现1", "主要发现2"],
                    "figures_needed": 2
                }}
            ],
            "estimated_length": "1200 words",
            "main_contribution": "核心实验结果"
        }},
        {{
            "section_id": "sec_4",
            "section_name": "Discussion",
            "subsections": [
                {{
                    "subsection_id": "sec_4_1",
                    "name": "Implications",
                    "key_points": ["理论意义", "实践应用"],
                    "citations_needed": 4
                }}
            ],
            "estimated_length": "1000 words",
            "main_contribution": "深度分析与讨论"
        }},
        {{
            "section_id": "sec_5",
            "section_name": "Conclusion",
            "subsections": [],
            "estimated_length": "400 words",
            "main_contribution": "总结与展望"
        }}
    ],
    "key_contributions": ["贡献1", "贡献2", "贡献3"],
    "methodology_overview": "研究方法概述（200字）",
    "expected_results": "预期主要结果（200字）",
    "total_word_estimate": 4400,
    "required_figures": 4,
    "required_tables": 2
}}

要求：
1. 严格遵循IMRaD结构（Introduction, Methods, Results, Discussion, Conclusion）
2. 每个section必须有明确的key_points（至少2个要点）
3. estimated_length用于后续字数分配
4. citations_needed标注每节需要引用的文献数量
5. 确保JSON格式合法，可以被Python json.loads解析
6. 所有字段必须存在，不能省略
7. 标题应该具体、学术化，避免过于宽泛
"""

    async def _call_llm(self, prompt: str) -> str:
        from claude_agent_sdk import query as claude_query, ClaudeAgentOptions

        options = ClaudeAgentOptions(
            system_prompt="你是一位专业的学术论文大纲生成专家。只输出JSON格式，不要添加任何额外说明。",
            model=self.get_required_model(),
            max_turns=50
        )

        content = ""
        async for message in claude_query(prompt=prompt, options=options):
            if hasattr(message, "content") and message.content:
                for block in message.content:
                    if hasattr(block, "text"):
                        content += block.text

        return content

    def _extract_json(self, text: str) -> Dict[str, Any]:
        json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(json_pattern, text)

        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

        return {
            "title": "Generated Paper",
            "sections": [],
            "key_contributions": [],
            "error": "Failed to parse JSON"
        }

    def _validate_outline(self, outline: Dict[str, Any]) -> bool:
        required_fields = ["title", "sections"]
        for fld in required_fields:
            if fld not in outline:
                return False

        if not isinstance(outline.get("sections"), list):
            return False

        for section in outline.get("sections", []):
            if not all(k in section for k in ["section_id", "section_name"]):
                return False

        return True


class WritingNode(BaseNode):
    """串行撰写节点"""

    SECTION_CONFIG = {
        "introduction": {
            "name": "Introduction",
            "description": "研究背景、问题陈述、研究目标",
            "key_elements": ["background", "problem_statement", "research_questions", "contributions"],
            "typical_length": "800-1000 words"
        },
        "methods": {
            "name": "Methods",
            "description": "研究方法、实验设计、数据分析",
            "key_elements": ["experimental_setup", "data_collection", "analysis_methods", "validation"],
            "typical_length": "1000-1200 words"
        },
        "results": {
            "name": "Results",
            "description": "实验结果、数据展示",
            "key_elements": ["main_findings", "statistical_results", "comparisons", "visualizations"],
            "typical_length": "1200-1500 words"
        },
        "discussion": {
            "name": "Discussion",
            "description": "结果讨论、与相关工作对比",
            "key_elements": ["interpretation", "comparison", "limitations", "implications"],
            "typical_length": "1000-1200 words"
        },
        "conclusion": {
            "name": "Conclusion",
            "description": "总结、未来工作",
            "key_elements": ["summary", "key_findings", "future_work"],
            "typical_length": "400-500 words"
        }
    }

    def __init__(self, node_id: str, config: Dict[str, Any] = None):
        super().__init__(node_id, NodeType.WRITING, config)
        self.section_type = config.get("section_type", "introduction")
        self.section_order = config.get("section_order", 1)
        self.total_sections = config.get("total_sections", 5)
        self.latex_format = config.get("latex_format", True)

    async def execute(self, input_data: NodeInput) -> NodeOutput:
        self.status = NodeStatus.RUNNING

        try:
            topic = input_data.get_context_value("topic", "")
            outline = input_data.get_context_value("outline", {})
            previous_outputs = input_data.previous_outputs

            section_config = self.SECTION_CONFIG.get(self.section_type, {})

            coherence_context = self._build_coherence_context(previous_outputs)

            section_outline = self._get_section_outline(outline)

            prompt = self._build_prompt(
                topic=topic,
                section_config=section_config,
                section_outline=section_outline,
                coherence_context=coherence_context,
                previous_outputs=previous_outputs
            )

            content = await self._call_llm(prompt)

            if self.latex_format:
                content = self._ensure_latex_format(content)

            self.status = NodeStatus.COMPLETED

            return NodeOutput(
                content=content,
                status=NodeStatus.COMPLETED,
                metadata={
                    "section_type": self.section_type,
                    "section_order": self.section_order,
                    "word_count": len(content.split()),
                    "latex_formatted": self.latex_format,
                    "node_id": self.node_id,
                }
            )

        except Exception as e:
            self.status = NodeStatus.FAILED
            return self._create_error_output(f"第{self.section_order}部分撰写失败: {str(e)}")

    def _build_coherence_context(self, previous_outputs: List[Dict[str, Any]]) -> str:
        if not previous_outputs:
            return ""

        context_parts = ["===== 前文内容摘要 ====="]

        for i, output in enumerate(previous_outputs, 1):
            section_name = output.get("section", f"Part {i}")
            content = output.get("content", "")

            summary = content[:300] + "..." if len(content) > 300 else content

            context_parts.append(f"\n【{section_name}】")
            context_parts.append(summary)

        context_parts.append("\n===== 前文结束 =====")

        return "\n".join(context_parts)

    def _get_section_outline(self, outline: Dict[str, Any]) -> Dict[str, Any]:
        sections = outline.get("sections", [])

        if self.section_order <= len(sections):
            return sections[self.section_order - 1]

        return {}

    def _build_prompt(
        self,
        topic: str,
        section_config: Dict[str, Any],
        section_outline: Dict[str, Any],
        coherence_context: str,
        previous_outputs: List[Dict[str, Any]]
    ) -> str:
        section_name = section_config.get("name", self.section_type)
        section_desc = section_config.get("description", "")
        key_elements = section_config.get("key_elements", [])
        typical_length = section_config.get("typical_length", "1000 words")

        key_points = []
        for subsec in section_outline.get("subsections", []):
            key_points.extend(subsec.get("key_points", []))

        latex_requirement = """
必须使用标准LaTeX语法：
- 数学公式使用 $...$ 或 $$...$$
- 表格使用 \\begin{table}...\\end{table}
- 图表引用使用 \\ref{}
- 章节标题使用 \\section{}, \\subsection{}
- 引用使用 \\cite{}
""" if self.latex_format else ""

        coherence_requirement = """
连贯性要求：
1. 请参考前文的逻辑，确保术语和变量定义一致
2. 如果前文定义了特定概念或符号，请保持一致使用
3. 确保本节内容与前文有自然的过渡和衔接
4. 如果本节提到前文的概念，请使用相同的术语
""" if previous_outputs else ""

        return f"""你是一位资深学术写作专家。请撰写论文的【{section_name}】部分。

论文主题：{topic}

本节描述：{section_desc}

预期长度：{typical_length}

本节必须包含的关键要素：
{chr(10).join(f"- {elem}" for elem in key_elements)}

大纲关键点：
{chr(10).join(f"- {point}" for point in key_points) if key_points else "（请根据主题自行组织）"}

{latex_requirement}

{coherence_requirement}

{coherence_context}

撰写要求：
1. 内容必须是完整的学术段落，不是 bullet points
2. 使用正式的学术写作风格
3. 确保逻辑严密、论证充分
4. 适当引用相关文献（使用[Author, Year]格式，后续会转换为BibTeX）
5. 包含过渡句，与前后文自然衔接
6. 本节应该是自洽的，但也要注意与全文的连贯性

请直接输出{section_name}部分的完整内容，不要包含其他说明。
"""

    async def _call_llm(self, prompt: str) -> str:
        from claude_agent_sdk import query as claude_query, ClaudeAgentOptions

        options = ClaudeAgentOptions(
            system_prompt="你是一位专业的学术写作专家。输出完整的学术段落，使用LaTeX格式，保持术语一致性。",
            model=self.get_required_model(),
            max_turns=100
        )

        content = ""
        async for message in claude_query(prompt=prompt, options=options):
            if hasattr(message, "content") and message.content:
                for block in message.content:
                    if hasattr(block, "text"):
                        content += block.text

        return content

    def _ensure_latex_format(self, content: str) -> str:
        if not content.strip().startswith("\\"):
            section_name = self.SECTION_CONFIG.get(self.section_type, {}).get("name", self.section_type)
            content = f"\\section{{{section_name}}}\n\n{content}"

        return content


class AuditNode(BaseNode):
    """审计节点"""

    AUDIT_TYPES = {
        "logic": {
            "name": "逻辑审计",
            "checks": [
                "论证是否自洽",
                "假设是否合理",
                "推理过程是否有漏洞",
                "结论是否由证据支持"
            ]
        },
        "citation": {
            "name": "引用审计",
            "checks": [
                "引用格式是否统一",
                "引用与正文是否对应",
                "关键论点是否有引用支持",
                "是否存在过度引用或引用不足"
            ]
        },
        "consistency": {
            "name": "一致性审计",
            "checks": [
                "术语使用是否一致",
                "变量定义是否前后统一",
                "数据引用是否一致",
                "缩写词使用是否规范"
            ]
        },
        "completeness": {
            "name": "完整性审计",
            "checks": [
                "IMRaD结构是否完整",
                "必要章节是否缺失",
                "图表引用是否完整",
                "参考文献是否完整"
            ]
        }
    }

    def __init__(self, node_id: str, config: Dict[str, Any] = None):
        super().__init__(node_id, NodeType.AUDIT, config)
        self.audit_types = config.get("audit_types", ["logic", "citation", "consistency"])

    async def execute(self, input_data: NodeInput) -> NodeOutput:
        self.status = NodeStatus.RUNNING

        try:
            content = input_data.content
            if not content:
                content = input_data.get_context_value("full_draft", "")

            if not content:
                return self._create_error_output("没有可审计的内容")

            audit_results = {}
            issues_found = []

            for audit_type in self.audit_types:
                if audit_type in self.AUDIT_TYPES:
                    result = await self._perform_audit(content, audit_type)
                    audit_results[audit_type] = result
                    issues_found.extend(result.get("issues", []))

            overall_score = self._calculate_audit_score(audit_results)
            passed = overall_score >= 0.7

            self.status = NodeStatus.COMPLETED

            return NodeOutput(
                content=self._format_audit_report(audit_results),
                status=NodeStatus.COMPLETED,
                metadata={
                    "audit_results": audit_results,
                    "overall_score": overall_score,
                    "passed": passed,
                    "issues_count": len(issues_found),
                    "issues": issues_found,
                    "node_id": self.node_id,
                }
            )

        except Exception as e:
            self.status = NodeStatus.FAILED
            return self._create_error_output(f"审计失败: {str(e)}")

    async def _perform_audit(self, content: str, audit_type: str) -> Dict[str, Any]:
        audit_config = self.AUDIT_TYPES.get(audit_type, {})
        audit_name = audit_config.get("name", audit_type)
        checks = audit_config.get("checks", [])

        prompt = f"""你是一位严格的学术论文审计员。请对以下论文内容进行【{audit_name}】。

审计检查项：
{chr(10).join(f"- {check}" for check in checks)}

论文内容：
{content[:5000]}  # 限制长度，避免超出token限制

请以JSON格式输出审计结果：
{{
    "audit_type": "{audit_type}",
    "score": 0.85,  // 0-1分
    "passed": true,  // 是否通过
    "issues": [
        {{
            "severity": "major" | "minor" | "suggestion",
            "location": "具体位置（如：Introduction第2段）",
            "description": "问题描述",
            "suggestion": "修改建议"
        }}
    ],
    "summary": "审计摘要（100字以内）"
}}

要求：
1. 严格按照JSON格式输出
2. 每个问题必须包含具体位置
3. severity分级：major（严重）、minor（轻微）、suggestion（建议）
4. 评分客观公正
"""

        from claude_agent_sdk import query as claude_query, ClaudeAgentOptions

        options = ClaudeAgentOptions(
            system_prompt="你是一位严格的学术论文审计专家。输出JSON格式的审计报告。",
            model=self.get_required_model(),
            max_turns=50
        )

        result_text = ""
        async for message in claude_query(prompt=prompt, options=options):
            if hasattr(message, "content") and message.content:
                for block in message.content:
                    if hasattr(block, "text"):
                        result_text += block.text

        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {
                "audit_type": audit_type,
                "score": 0.5,
                "passed": False,
                "issues": [{"severity": "error", "description": "解析审计结果失败"}],
                "summary": "审计结果解析失败"
            }

    def _calculate_audit_score(self, audit_results: Dict[str, Any]) -> float:
        scores = []
        for result in audit_results.values():
            if isinstance(result, dict):
                score = result.get("score", 0)
                scores.append(score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def _format_audit_report(self, audit_results: Dict[str, Any]) -> str:
        report_parts = ["# 学术论文审计报告\n"]

        for audit_type, result in audit_results.items():
            audit_name = self.AUDIT_TYPES.get(audit_type, {}).get("name", audit_type)
            score = result.get("score", 0)
            passed = result.get("passed", False)
            summary = result.get("summary", "")

            report_parts.append(f"\n## {audit_name}")
            report_parts.append(f"- 评分: {score:.2f}/1.0")
            report_parts.append(f"- 状态: {'通过 ✓' if passed else '未通过 ✗'}")
            report_parts.append(f"- 摘要: {summary}")

            issues = result.get("issues", [])
            if issues:
                report_parts.append("\n### 发现的问题：")
                for issue in issues:
                    severity = issue.get("severity", "unknown")
                    severity_emoji = {"major": "🔴", "minor": "🟡", "suggestion": "🔵"}.get(severity, "⚪")
                    report_parts.append(f"\n{severity_emoji} [{severity.upper()}]")
                    report_parts.append(f"位置: {issue.get('location', '未知')}")
                    report_parts.append(f"问题: {issue.get('description', '未描述')}")
                    report_parts.append(f"建议: {issue.get('suggestion', '无')}")

        return "\n".join(report_parts)


class ReviewNode(BaseNode):
    """同行评审节点"""

    REVIEW_DIMENSIONS = {
        "innovation": {
            "name": "创新性 (Innovation)",
            "description": "研究的新颖程度和原创性",
            "criteria": [
                "问题定义是否新颖",
                "方法是否有创新",
                "与已有工作的区别是否明确"
            ]
        },
        "rigor": {
            "name": "严谨性 (Rigor)",
            "description": "研究方法的严密性和实验设计的合理性",
            "criteria": [
                "实验设计是否严谨",
                "数据分析是否充分",
                "控制变量是否合理"
            ]
        },
        "significance": {
            "name": "显著性 (Significance)",
            "description": "研究结果的重要性和影响力",
            "criteria": [
                "结果是否有重要意义",
                "对领域发展的贡献",
                "实际应用价值"
            ]
        },
        "clarity": {
            "name": "清晰度 (Clarity)",
            "description": "论文写作的清晰度和可读性",
            "criteria": [
                "逻辑结构是否清晰",
                "表达是否准确",
                "图表是否易懂"
            ]
        },
        "methodology": {
            "name": "方法论 (Methodology)",
            "description": "研究方法的先进性和适用性",
            "criteria": [
                "方法是否适合问题",
                "技术细节是否充分",
                "复现性如何"
            ]
        }
    }

    VENUE_STYLES = {
        "neurips": "NeurIPS风格 - 注重理论深度和创新性",
        "icml": "ICML风格 - 注重实验验证和方法实用性",
        "iclr": "ICLR风格 - 注重深度学习理论和可复现性",
        "acl": "ACL风格 - 注重语言学和计算结合",
        "cvpr": "CVPR风格 - 注重实验效果和视觉展示",
        "sigir": "SIGIR风格 - 注重检索效果评估",
        "kdd": "KDD风格 - 注重数据挖掘应用价值",
        "top_conference": "综合顶会风格 - 高标准全面评审"
    }

    def __init__(self, node_id: str, config: Dict[str, Any] = None):
        super().__init__(node_id, NodeType.REVIEW, config)
        self.dimensions = config.get("dimensions", list(self.REVIEW_DIMENSIONS.keys()))
        self.adversarial_count = config.get("adversarial_questions_count", 3)
        self.venue_style = config.get("venue_style", "top_conference")

    async def execute(self, input_data: NodeInput) -> NodeOutput:
        self.status = NodeStatus.RUNNING

        try:
            content = input_data.content
            if not content:
                content = input_data.get_context_value("full_draft", "")

            if not content:
                return self._create_error_output("没有可评审的内容")

            audit_result = input_data.get_context_value("audit_result", {})

            scores = await self._calculate_scores(content)

            adversarial_questions = await self._generate_adversarial_questions(content)

            revision_suggestions = await self._generate_suggestions(content, scores)

            overall_score = sum(scores.values()) / len(scores) if scores else 0

            self.status = NodeStatus.COMPLETED

            return NodeOutput(
                content=self._format_review_report(scores, adversarial_questions, revision_suggestions),
                status=NodeStatus.COMPLETED,
                scores=scores,
                adversarial_questions=adversarial_questions,
                revision_suggestions=revision_suggestions,
                metadata={
                    "overall_score": overall_score,
                    "scores": scores,
                    "adversarial_questions": adversarial_questions,
                    "revision_suggestions": revision_suggestions,
                    "venue_style": self.venue_style,
                    "passed": overall_score >= 7.0,
                    "node_id": self.node_id,
                }
            )

        except Exception as e:
            self.status = NodeStatus.FAILED
            return self._create_error_output(f"评审失败: {str(e)}")

    async def _calculate_scores(self, content: str) -> Dict[str, float]:
        scores = {}

        venue_style_desc = self.VENUE_STYLES.get(self.venue_style, "顶会评审标准")

        for dim_key in self.dimensions:
            dim_config = self.REVIEW_DIMENSIONS.get(dim_key, {})
            dim_name = dim_config.get("name", dim_key)
            dim_desc = dim_config.get("description", "")
            criteria = dim_config.get("criteria", [])

            prompt = f"""你是一位{venue_style_desc}的审稿人。请对以下论文的【{dim_name}】进行评分。

评分维度：{dim_desc}

评审标准：
{chr(10).join(f"- {c}" for c in criteria)}

评分标准（1-10分）：
- 10分：卓越，该维度达到顶会最佳论文水平
- 8-9分：优秀，该维度无明显缺陷
- 6-7分：良好，有小问题但不影响整体
- 4-5分：一般，有明显改进空间
- 1-3分：较差，需要重大修改

论文内容：
{content[:4000]}

⚠️ 输出格式严格要求（违反任何一条都会导致评分失败）：
1. 只输出一个 JSON 对象，前后不要任何说明文字、不要 Markdown 代码围栏。
2. `score` 字段必须为 1-10 之间的数字（可保留一位小数），**禁止为 null**。
3. JSON 字符串值的内部如需引用，请使用中文「」或日文『』，**禁止使用未转义的半角双引号 `"`**（必要时用 `\\"` 转义）。

请按下列格式输出（仅此一种）：
{{"score": 7.5, "justification": "评分理由（100-200字，引用请用「」）"}}
"""

            from claude_agent_sdk import query as claude_query, ClaudeAgentOptions

            options = ClaudeAgentOptions(
                system_prompt="你是一位严格的顶会审稿人。客观评分，详细说明理由。仅输出 JSON。",
                model=self.get_required_model(),
                max_turns=30,
                setting_sources=["project"],
                env=_build_subprocess_env(),
            )

            async def _query_once(p: str) -> str:
                buf = ""
                async for message in claude_query(prompt=p, options=options):
                    if hasattr(message, "content") and message.content:
                        for block in message.content:
                            if hasattr(block, "text"):
                                buf += block.text
                return buf

            result_text = await _query_once(prompt)
            score_value = self._parse_score_from_response(result_text)

            if score_value is None:
                retry_prompt = (
                    f"上一次返回无法识别为合法评分。现在请重新评估【{dim_name}】，"
                    f"**只输出一个 1-10 之间的数字（可保留一位小数）**，不要 JSON、"
                    f"不要任何说明文字。\n\n论文内容：\n{content[:4000]}"
                )
                retry_text = await _query_once(retry_prompt)
                retry_match = re.search(r'([0-9]+(?:\.[0-9]+)?)', retry_text)
                if retry_match:
                    try:
                        candidate = float(retry_match.group(1))
                        if 0 <= candidate <= 10:
                            score_value = candidate
                    except (TypeError, ValueError):
                        pass
                if score_value is None:
                    print(
                        f"[ReviewNode] WARN: 维度 {dim_key} 重试后仍无法解析分数，降级到 5.0。\n"
                        f"首轮返回前 300 字：\n{result_text[:300]}\n"
                        f"重试返回前 300 字：\n{retry_text[:300]}",
                        file=sys.stderr,
                    )
                    score_value = 5.0

            scores[dim_key] = score_value

        return scores

    @staticmethod
    def _parse_score_from_response(result_text: str) -> Optional[float]:
        """从 LLM 响应里抽取 score 数值，按 JSON → 正则两级回退。"""
        parsed = _extract_json_block(result_text)
        if parsed is not None:
            raw_score = parsed.get("score")
            if raw_score is not None:
                try:
                    value = float(raw_score)
                    if 0 <= value <= 10:
                        return value
                except (TypeError, ValueError):
                    pass
        return _extract_score_from_text(result_text)

    async def _generate_adversarial_questions(self, content: str) -> List[str]:
        prompt = f"""你是一位严格的顶会审稿人。请仔细阅读以下论文，提出{self.adversarial_count}个"刁钻"的问题。

论文内容：
{content[:5000]}

要求：
1. 问题必须切中要害，揭示论文的潜在弱点
2. 问题应该具有挑战性，作者不容易回答
3. 问题应该有助于提升论文质量
4. 使用专业的学术语言

请以下列格式输出问题（每个问题100-200字）：

对抗性问题1：
[问题内容]

对抗性问题2：
[问题内容]

对抗性问题3：
[问题内容]

注意：这些问题将在下一轮修改中要求作者必须回答。
"""

        from claude_agent_sdk import query as claude_query, ClaudeAgentOptions

        options = ClaudeAgentOptions(
            system_prompt="你是一位挑剔的审稿人，善于发现论文的弱点。提出尖锐但有建设性的问题。",
            model=self.get_required_model(),
            max_turns=50,
            setting_sources=["project"],
            env=_build_subprocess_env(),
        )

        result_text = ""
        async for message in claude_query(prompt=prompt, options=options):
            if hasattr(message, "content") and message.content:
                for block in message.content:
                    if hasattr(block, "text"):
                        result_text += block.text

        questions = re.findall(r'对抗性问题\d+：\s*\n?(.+?)(?=\n对抗性问题|\Z)', result_text, re.DOTALL)

        if not questions:
            lines = result_text.split('\n')
            questions = [line.strip() for line in lines if line.strip() and ('?' in line or '？' in line)]

        return questions[:self.adversarial_count] if questions else [
            "论文的创新点与现有工作相比有何本质区别？",
            "实验结果是否在不同的数据集上都能得到验证？",
            "论文的方法在更大规模的数据上是否仍然有效？"
        ]

    async def _generate_suggestions(self, content: str, scores: Dict[str, float]) -> List[str]:
        weak_dimensions = [dim for dim, score in scores.items() if score < 7.0]

        if not weak_dimensions:
            return ["论文整体质量良好，建议进行细微润色即可。"]

        prompt = f"""请针对论文中得分较低的维度提供具体的修改建议。

论文内容：
{content[:3000]}

需要改进的维度：{', '.join(weak_dimensions)}

请提供3-5条具体的修改建议，每条建议应该：
1. 明确指出具体位置（如：Introduction第2段）
2. 说明问题所在
3. 提供具体的修改方案
"""

        from claude_agent_sdk import query as claude_query, ClaudeAgentOptions

        options = ClaudeAgentOptions(
            system_prompt="你是一位建设性的审稿人。提供具体、可操作的修改建议。",
            model=self.get_required_model(),
            max_turns=30,
            setting_sources=["project"],
            env=_build_subprocess_env(),
        )

        result_text = ""
        async for message in claude_query(prompt=prompt, options=options):
            if hasattr(message, "content") and message.content:
                for block in message.content:
                    if hasattr(block, "text"):
                        result_text += block.text

        suggestions = [s.strip() for s in result_text.split('\n') if s.strip() and len(s.strip()) > 20]
        return suggestions[:5]

    def _format_review_report(
        self,
        scores: Dict[str, float],
        adversarial_questions: List[str],
        suggestions: List[str]
    ) -> str:
        report_parts = ["# 顶会同行评审报告\n"]

        report_parts.append("## 多维度评分表\n")
        report_parts.append("| 维度 | 分数 | 说明 |")
        report_parts.append("|------|------|------|")

        overall = 0
        for dim_key, score in scores.items():
            dim_config = self.REVIEW_DIMENSIONS.get(dim_key, {})
            dim_name = dim_config.get("name", dim_key)
            status = "✓" if score >= 7 else "⚠" if score >= 5 else "✗"
            report_parts.append(f"| {dim_name} | {score}/10 {status} | {self._get_score_level(score)} |")
            overall += score

        avg_score = overall / len(scores) if scores else 0
        report_parts.append(f"\n**综合评分: {avg_score:.1f}/10**")
        report_parts.append(f"**评审结果: {'通过 ✓' if avg_score >= 7.0 else '需修改 ✗'}**\n")

        report_parts.append("\n## 对抗性问题（必须在修改中回答）\n")
        for i, question in enumerate(adversarial_questions, 1):
            report_parts.append(f"**问题{i}：** {question}\n")

        report_parts.append("\n## 修改建议\n")
        for i, suggestion in enumerate(suggestions, 1):
            report_parts.append(f"{i}. {suggestion}\n")

        return "\n".join(report_parts)

    def _get_score_level(self, score: float) -> str:
        if score >= 9:
            return "卓越"
        elif score >= 8:
            return "优秀"
        elif score >= 7:
            return "良好"
        elif score >= 5:
            return "一般"
        else:
            return "需改进"


class ConditionNode(BaseNode):
    """条件分支节点"""

    def __init__(self, node_id: str, config: Dict[str, Any] = None):
        super().__init__(node_id, NodeType.CONDITION, config)
        self.condition_type = config.get("condition_type", "score_threshold")
        self.threshold = config.get("threshold", 7.0)

    async def execute(self, input_data: NodeInput) -> NodeOutput:
        self.status = NodeStatus.RUNNING

        try:
            scores = input_data.get_context_value("scores", {})

            if not scores and input_data.metadata:
                scores = input_data.metadata.get("scores", {})

            if scores:
                overall_score = sum(scores.values()) / len(scores)
            else:
                overall_score = 0

            passed = overall_score >= self.threshold

            dimension_status = {
                dim: score >= self.threshold
                for dim, score in scores.items()
            }

            self.status = NodeStatus.COMPLETED

            return NodeOutput(
                content=self._format_decision(passed, overall_score, scores),
                status=NodeStatus.COMPLETED,
                metadata={
                    "passed": passed,
                    "overall_score": overall_score,
                    "threshold": self.threshold,
                    "scores": scores,
                    "dimension_status": dimension_status,
                    "next_node": "finalize" if passed else "revision",
                    "node_id": self.node_id,
                }
            )

        except Exception as e:
            self.status = NodeStatus.FAILED
            return self._create_error_output(f"条件判断失败: {str(e)}")

    def _format_decision(self, passed: bool, overall_score: float, scores: Dict[str, float]) -> str:
        status = "通过" if passed else "未通过"
        emoji = "✓" if passed else "✗"

        report = f"""# 条件判断结果

**状态: {status} {emoji}**

- 综合评分: {overall_score:.2f}/10
- 阈值: {self.threshold}
- 差额: {abs(overall_score - self.threshold):.2f}

## 各维度评分
"""
        for dim, score in scores.items():
            dim_passed = "✓" if score >= self.threshold else "✗"
            report += f"- {dim}: {score}/10 {dim_passed}\n"

        if passed:
            report += "\n**决策: 评分达到阈值，进入最终输出阶段。**"
        else:
            report += f"\n**决策: 评分未达到阈值，进入修改阶段。**"
            report += f"\n需要提升: {self.threshold - overall_score:.2f}分"

        return report


class RevisionNode(BaseNode):
    """二次修改节点"""

    def __init__(self, node_id: str, config: Dict[str, Any] = None):
        super().__init__(node_id, NodeType.REVISION, config)
        self.revision_strategy = config.get("revision_strategy", "targeted")

    async def execute(self, input_data: NodeInput) -> NodeOutput:
        self.status = NodeStatus.RUNNING

        try:
            content = input_data.content
            if not content:
                content = input_data.get_context_value("full_draft", "")

            review_result = input_data.get_context_value("review_result", {})
            adversarial_questions = input_data.get_context_value("adversarial_questions", [])
            revision_suggestions = input_data.get_context_value("revision_suggestions", [])

            iteration = input_data.get_context_value("iteration", 1)

            audit_issues = []
            audit_result = input_data.get_context_value("audit_result", {})
            if isinstance(audit_result, dict):
                for audit_type, result in audit_result.items():
                    if isinstance(result, dict):
                        issues = result.get("issues", [])
                        for issue in issues:
                            if issue.get("severity") in ["major", "minor"]:
                                audit_issues.append(issue)

            prompt = self._build_revision_prompt(
                content=content,
                review_result=review_result,
                adversarial_questions=adversarial_questions,
                revision_suggestions=revision_suggestions,
                audit_issues=audit_issues,
                iteration=iteration
            )

            revised_content = await self._call_llm(prompt)

            revised_content = self._ensure_format_consistency(revised_content, content)

            self.status = NodeStatus.COMPLETED

            return NodeOutput(
                content=revised_content,
                status=NodeStatus.COMPLETED,
                metadata={
                    "revision_iteration": iteration,
                    "original_length": len(content),
                    "revised_length": len(revised_content),
                    "issues_addressed": len(adversarial_questions) + len(revision_suggestions),
                    "node_id": self.node_id,
                }
            )

        except Exception as e:
            self.status = NodeStatus.FAILED
            return self._create_error_output(f"修改失败: {str(e)}")

    def _build_revision_prompt(
        self,
        content: str,
        review_result: Dict[str, Any],
        adversarial_questions: List[str],
        revision_suggestions: List[str],
        audit_issues: List[Dict],
        iteration: int
    ) -> str:
        scores = review_result.get("scores", {}) if isinstance(review_result, dict) else {}

        prompt = f"""你是一位资深学术论文修改专家。这是第{iteration}轮修改，请对论文进行靶向重构。

## 原始论文
{content[:6000]}

## 需要回答的对抗性问题
"""
        for i, question in enumerate(adversarial_questions, 1):
            prompt += f"{i}. {question}\n"

        prompt += f"""
## 评审评分
"""
        for dim, score in scores.items():
            prompt += f"- {dim}: {score}/10\n"

        if revision_suggestions:
            prompt += f"""
## 修改建议
"""
            for i, suggestion in enumerate(revision_suggestions, 1):
                prompt += f"{i}. {suggestion}\n"

        if audit_issues:
            prompt += f"""
## 审计发现的问题
"""
            for i, issue in enumerate(audit_issues[:5], 1):
                prompt += f"{i}. [{issue.get('severity', 'unknown')}] {issue.get('description', '')}\n"

        prompt += f"""
## 修改要求
1. **必须回答所有对抗性问题**，将答案融入论文相关部分
2. **针对性改进得分较低的维度**
3. **修复审计发现的所有问题**
4. 保持LaTeX格式和原有结构
5. 保持术语和符号定义的一致性
6. 修改后的内容应该能够提升评审分数
7. 输出完整的修改后论文

## 输出格式
直接输出修改后的完整论文内容（LaTeX格式），不要添加额外说明。
"""
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        from claude_agent_sdk import query as claude_query, ClaudeAgentOptions

        options = ClaudeAgentOptions(
            system_prompt="你是一位专业的学术论文修改专家。全面修改论文，回答所有问题，保持格式一致。",
            model=self.get_required_model(),
            max_turns=150
        )

        content = ""
        async for message in claude_query(prompt=prompt, options=options):
            if hasattr(message, "content") and message.content:
                for block in message.content:
                    if hasattr(block, "text"):
                        content += block.text

        return content

    def _ensure_format_consistency(self, revised: str, original: str) -> str:
        if "\\section" in original and "\\section" not in revised:
            revised = self._restore_latex_structure(revised)

        return revised

    def _restore_latex_structure(self, content: str) -> str:
        if "\\section" not in content:
            sections = ["Introduction", "Methods", "Results", "Discussion", "Conclusion"]
            structured = []
            lines = content.split('\n')
            current_section = 0

            for line in lines:
                if any(s.lower() in line.lower() for s in sections):
                    if current_section < len(sections):
                        line = f"\\section{{{sections[current_section]}}}\n\n{line}"
                        current_section += 1
                structured.append(line)

            return '\n'.join(structured)

        return content


class ToolNode(BaseNode):
    """工具节点"""

    def __init__(self, node_id: str, config: Dict[str, Any] = None):
        super().__init__(node_id, NodeType.TOOL, config)
        self.tool_type = config.get("tool_type", "python_visualization")

    async def execute(self, input_data: NodeInput) -> NodeOutput:
        self.status = NodeStatus.RUNNING

        try:
            if self.tool_type == "python_visualization":
                return await self._execute_python_viz(input_data)
            elif self.tool_type == "latex_compile":
                return await self._execute_latex_compile(input_data)
            else:
                return self._create_error_output(f"未知的工具类型: {self.tool_type}")

        except Exception as e:
            self.status = NodeStatus.FAILED
            return self._create_error_output(f"工具执行失败: {str(e)}")

    async def _execute_python_viz(self, input_data: NodeInput) -> NodeOutput:
        data = input_data.get_context_value("experiment_data", [])
        viz_type = input_data.get_context_value("viz_type", "line")
        output_format = input_data.get_context_value("viz_output_format", "base64")

        code = self._build_viz_code(data, viz_type)

        try:
            # 注意：生产环境应使用更安全的沙箱
            local_vars = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)

            output_path = local_vars.get("output_path", "")

            if output_format == "base64" and output_path:
                with open(output_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")

                self.status = NodeStatus.COMPLETED
                return NodeOutput(
                    content=img_base64,
                    status=NodeStatus.COMPLETED,
                    metadata={
                        "tool_type": "python_visualization",
                        "viz_type": viz_type,
                        "output_format": "base64",
                        "output_path": output_path,
                    }
                )
            else:
                self.status = NodeStatus.COMPLETED
                return NodeOutput(
                    content=output_path,
                    status=NodeStatus.COMPLETED,
                    metadata={
                        "tool_type": "python_visualization",
                        "output_path": output_path,
                    }
                )

        except Exception as e:
            return self._create_error_output(f"可视化生成失败: {str(e)}")

    def _build_viz_code(self, data: Any, viz_type: str) -> str:
        base_code = """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# 数据（实际使用时会替换）
data = {data}

# 创建图表
fig, ax = plt.subplots()
"""

        if viz_type == "line":
            plot_code = """
ax.plot(data['x'], data['y'], marker='o', linewidth=2)
ax.set_xlabel(data.get('xlabel', 'X'))
ax.set_ylabel(data.get('ylabel', 'Y'))
ax.set_title(data.get('title', 'Line Plot'))
"""
        elif viz_type == "bar":
            plot_code = """
ax.bar(data['x'], data['y'])
ax.set_xlabel(data.get('xlabel', 'X'))
ax.set_ylabel(data.get('ylabel', 'Y'))
ax.set_title(data.get('title', 'Bar Plot'))
"""
        elif viz_type == "heatmap":
            plot_code = """
sns.heatmap(data['matrix'], annot=True, cmap='YlOrRd', ax=ax)
ax.set_title(data.get('title', 'Heatmap'))
"""
        else:
            plot_code = """
ax.plot(data['x'], data['y'])
ax.set_title(data.get('title', 'Plot'))
"""

        save_code = """
# 保存
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'figure_viz.png')
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
"""

        return base_code + plot_code + save_code

    async def _execute_latex_compile(self, input_data: NodeInput) -> NodeOutput:
        content = input_data.content
        output_dir = input_data.get_context_value("output_dir", "./")

        tex_path = Path(output_dir) / "manuscript.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(content)

        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
                capture_output=True,
                text=True,
                timeout=120
            )

            subprocess.run(
                ["bibtex", str(tex_path.with_suffix(""))],
                capture_output=True,
                text=True,
                timeout=60
            )

            # 第二次编译以解析交叉引用
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
                capture_output=True,
                text=True,
                timeout=120
            )

            pdf_path = Path(output_dir) / "manuscript.pdf"

            if pdf_path.exists():
                self.status = NodeStatus.COMPLETED
                return NodeOutput(
                    content=str(pdf_path),
                    status=NodeStatus.COMPLETED,
                    metadata={
                        "tool_type": "latex_compile",
                        "tex_path": str(tex_path),
                        "pdf_path": str(pdf_path),
                        "compilation_log": result.stdout if result.returncode == 0 else result.stderr,
                    }
                )
            else:
                return self._create_error_output("PDF生成失败")

        except subprocess.TimeoutExpired:
            return self._create_error_output("LaTeX编译超时")
        except Exception as e:
            return self._create_error_output(f"LaTeX编译错误: {str(e)}")
