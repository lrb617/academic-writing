"""工作流控制：上下文管理、流水线编排、迭代控制、引擎执行。"""

import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from .stages import (
    AuditNode,
    BaseNode,
    ConditionNode,
    NodeInput,
    NodeStatus,
    NodeType,
    OutlineNode,
    ReviewNode,
    RevisionNode,
    ToolNode,
    WritingNode,
)


# ---------------------------------------------------------------------------
# 上下文
# ---------------------------------------------------------------------------


@dataclass
class WorkflowContext:
    """工作流上下文，在所有节点间传递状态。"""

    topic: str = ""
    paper_type: str = "research_paper"
    output_dir: Optional[Path] = None

    config: Dict[str, Any] = field(default_factory=dict)

    knowledge_base: Optional[Any] = None
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)

    outline: Dict[str, Any] = field(default_factory=dict)

    section_1: str = ""  # Introduction
    section_2: str = ""  # Methods
    section_3: str = ""  # Results
    section_4: str = ""  # Discussion
    section_5: str = ""  # Conclusion

    full_draft: str = ""

    audit_history: List[Dict[str, Any]] = field(default_factory=list)
    review_history: List[Dict[str, Any]] = field(default_factory=list)
    revision_history: List[Dict[str, Any]] = field(default_factory=list)

    current_scores: Dict[str, float] = field(default_factory=dict)

    adversarial_questions: List[str] = field(default_factory=list)

    variables: Dict[str, Any] = field(default_factory=dict)

    tool_outputs: Dict[str, Any] = field(default_factory=dict)

    format_requirement: str = "必须使用标准 LaTeX 语法输出公式和表格"

    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self._extra.get(key)

    def __setitem__(self, key: str, value: Any):
        if hasattr(self, key) and not key.startswith('_'):
            setattr(self, key, value)
        else:
            self._extra[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "topic": self.topic,
            "paper_type": self.paper_type,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "outline": self.outline,
            "full_draft": self.full_draft,
            "current_scores": self.current_scores,
            "adversarial_questions": self.adversarial_questions,
            "variables": self.variables,
            "format_requirement": self.format_requirement,
        }
        result.update(self._extra)
        return result

    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def get_section_content(self, section_name: str) -> str:
        section_map = {
            "introduction": self.section_1,
            "methods": self.section_2,
            "results": self.section_3,
            "discussion": self.section_4,
            "conclusion": self.section_5,
        }
        return section_map.get(section_name.lower(), "")

    def set_variable(self, key: str, value: Any):
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)


class ContextManager:
    """上下文管理器。"""

    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.contexts: Dict[str, WorkflowContext] = {}

    def create_context(
        self,
        topic: str,
        paper_type: str,
        output_dir: Path,
        config: Any
    ) -> WorkflowContext:
        context = WorkflowContext(
            topic=topic,
            paper_type=paper_type,
            output_dir=output_dir,
            config=config.to_dict() if hasattr(config, 'to_dict') else config
        )

        context.set_variable("format_requirement", context.format_requirement)
        context.set_variable("loop_count", 0)
        context.set_variable("revision_count", 0)

        self.contexts[topic] = context
        return context

    def get_context(self, topic: str) -> Optional[WorkflowContext]:
        return self.contexts.get(topic)

    def update_context(self, topic: str, updates: Dict[str, Any]):
        context = self.contexts.get(topic)
        if context:
            for key, value in updates.items():
                if hasattr(context, key):
                    setattr(context, key, value)

    def save_context(self, topic: str, filename: str = "workflow_context.json"):
        context = self.contexts.get(topic)
        if context and context.output_dir:
            save_path = context.output_dir / filename
            context.save(save_path)


# ---------------------------------------------------------------------------
# 迭代控制
# ---------------------------------------------------------------------------


class IterationStatus(Enum):
    """迭代状态"""
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    MAX_ITERATIONS = "max_iterations"
    EARLY_STOP = "early_stop"


@dataclass
class IterationResult:
    """迭代结果"""
    status: IterationStatus
    final_content: str = ""
    iteration_count: int = 0
    final_scores: Dict[str, float] = None
    stop_reason: str = ""


class IteratorController:
    """迭代控制器，管理撰写-校验-评审-修改的闭环。"""

    def __init__(
        self,
        max_iterations: int = 3,
        score_threshold: float = 7.0,
        early_stop_threshold: float = 0.1
    ):
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.early_stop_threshold = early_stop_threshold

        self.score_history: List[Dict[str, float]] = []
        self.iteration_count = 0

    def should_continue(
        self,
        current_iteration: int,
        current_scores: Dict[str, float]
    ) -> bool:
        if current_iteration >= self.max_iterations:
            return False

        if self._has_passed_threshold(current_scores):
            return False

        if self._should_fuse(current_scores):
            return False

        return True

    def _has_passed_threshold(self, scores: Dict[str, float]) -> bool:
        if not scores:
            return False

        avg_score = sum(scores.values()) / len(scores)
        if avg_score >= self.score_threshold:
            return True

        if all(score >= self.score_threshold for score in scores.values()):
            return True

        return False

    def _should_fuse(self, current_scores: Dict[str, float]) -> bool:
        if len(self.score_history) < 2:
            return False

        last_scores = self.score_history[-1]
        if not last_scores or not current_scores:
            return False

        last_avg = sum(last_scores.values()) / len(last_scores) if last_scores else 0
        current_avg = sum(current_scores.values()) / len(current_scores) if current_scores else 0

        change = abs(current_avg - last_avg)

        return change < self.early_stop_threshold

    def record_iteration(self, scores: Dict[str, float]):
        self.score_history.append(scores.copy())
        self.iteration_count += 1

    def get_iteration_summary(self) -> Dict[str, Any]:
        if not self.score_history:
            return {
                "iteration_count": 0,
                "score_progression": [],
                "final_status": "not_started"
            }

        score_progression = []
        for i, scores in enumerate(self.score_history, 1):
            avg = sum(scores.values()) / len(scores) if scores else 0
            score_progression.append({
                "iteration": i,
                "average_score": round(avg, 2),
                "scores": scores
            })

        final_scores = self.score_history[-1]
        final_avg = sum(final_scores.values()) / len(final_scores) if final_scores else 0

        if final_avg >= self.score_threshold:
            final_status = "passed"
        elif self.iteration_count >= self.max_iterations:
            final_status = "max_iterations_reached"
        else:
            final_status = "early_stopped"

        return {
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "score_threshold": self.score_threshold,
            "score_progression": score_progression,
            "final_status": final_status,
            "final_average_score": round(final_avg, 2)
        }

    def reset(self):
        self.score_history = []
        self.iteration_count = 0


class LoopCounter:
    """循环计数器，用于在 prompt 中实现 Loop_Count 累加。"""

    def __init__(self, initial: int = 0, max_count: int = 3):
        self.count = initial
        self.max_count = max_count

    def increment(self) -> int:
        self.count += 1
        return self.count

    def get_count(self) -> int:
        return self.count

    def has_reached_max(self) -> bool:
        return self.count >= self.max_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loop_count": self.count,
            "max_loops": self.max_count,
            "remaining": self.max_count - self.count,
            "has_reached_max": self.has_reached_max()
        }


# ---------------------------------------------------------------------------
# 流水线构建
# ---------------------------------------------------------------------------


class NodeConnectionType(Enum):
    """节点连接类型"""
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    LOOP_BACK = "loop_back"


@dataclass
class NodeConnection:
    """节点连接定义"""
    from_node: str
    to_node: str
    connection_type: NodeConnectionType = NodeConnectionType.SEQUENTIAL
    condition: Optional[str] = None


@dataclass
class PipelineNode:
    """流水线节点定义"""
    node_id: str
    node_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


class PipelineBuilder:
    """流水线构建器，用于构建工作流的有向无环图。"""

    def __init__(self):
        self.nodes: Dict[str, PipelineNode] = {}
        self.connections: List[NodeConnection] = []
        self.entry_point: Optional[str] = None

    def add_node(
        self,
        node_id: str,
        node_type: str,
        config: Dict[str, Any] = None,
        dependencies: List[str] = None
    ) -> "PipelineBuilder":
        self.nodes[node_id] = PipelineNode(
            node_id=node_id,
            node_type=node_type,
            config=config or {},
            dependencies=dependencies or []
        )
        return self

    def connect(
        self,
        from_node: str,
        to_node: str,
        connection_type: NodeConnectionType = NodeConnectionType.SEQUENTIAL,
        condition: Optional[str] = None
    ) -> "PipelineBuilder":
        self.connections.append(NodeConnection(
            from_node=from_node,
            to_node=to_node,
            connection_type=connection_type,
            condition=condition
        ))
        return self

    def set_entry_point(self, node_id: str) -> "PipelineBuilder":
        self.entry_point = node_id
        return self

    def build_serial_writing_pipeline(self) -> "PipelineBuilder":
        self.add_node("outline", "outline", config={"output_format": "json"})

        sections = ["introduction", "methods", "results", "discussion", "conclusion"]
        for i, section in enumerate(sections, 1):
            prev_node = "outline" if i == 1 else f"writing_{i-1}"
            self.add_node(
                f"writing_{i}",
                "writing",
                config={"section_type": section, "section_order": i},
                dependencies=[prev_node]
            )
            self.connect(prev_node, f"writing_{i}")

        self.set_entry_point("outline")
        return self

    def build_full_workflow_pipeline(self, max_iterations: int = 3) -> "PipelineBuilder":
        self.build_serial_writing_pipeline()

        self.add_node("audit", "audit", dependencies=["writing_5"])
        self.connect("writing_5", "audit")

        self.add_node("review", "review", dependencies=["audit"])
        self.connect("audit", "review")

        self.add_node("condition", "condition", dependencies=["review"])
        self.connect("review", "condition")

        self.add_node("revision", "revision", dependencies=["condition"])
        self.connect(
            "condition",
            "revision",
            connection_type=NodeConnectionType.CONDITIONAL,
            condition="score < threshold"
        )

        self.connect(
            "revision",
            "audit",
            connection_type=NodeConnectionType.LOOP_BACK
        )

        self.add_node("finalize", "finalize", dependencies=["condition"])
        self.connect(
            "condition",
            "finalize",
            connection_type=NodeConnectionType.CONDITIONAL,
            condition="score >= threshold"
        )

        return self

    def get_execution_order(self) -> List[str]:
        in_degree = {node_id: 0 for node_id in self.nodes}
        graph = {node_id: [] for node_id in self.nodes}

        for conn in self.connections:
            if conn.connection_type == NodeConnectionType.SEQUENTIAL:
                graph[conn.from_node].append(conn.to_node)
                in_degree[conn.to_node] += 1

        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            node_id = queue.pop(0)
            order.append(node_id)

            for neighbor in graph[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {
                node_id: {
                    "node_type": node.node_type,
                    "config": node.config,
                    "dependencies": node.dependencies
                }
                for node_id, node in self.nodes.items()
            },
            "connections": [
                {
                    "from": conn.from_node,
                    "to": conn.to_node,
                    "type": conn.connection_type.value,
                    "condition": conn.condition
                }
                for conn in self.connections
            ],
            "entry_point": self.entry_point,
            "execution_order": self.get_execution_order()
        }


# ---------------------------------------------------------------------------
# 工作流引擎
# ---------------------------------------------------------------------------


class WorkflowStage(Enum):
    """工作流执行阶段"""
    INITIALIZATION = "initialization"
    KNOWLEDGE_BASE_PREP = "knowledge_base_prep"
    OUTLINE_GENERATION = "outline_generation"
    SERIAL_WRITING = "serial_writing"
    AUDIT = "audit"
    PEER_REVIEW = "peer_review"
    CONDITION_CHECK = "condition_check"
    REVISION = "revision"
    FINALIZATION = "finalization"
    COMPLETED = "completed"


@dataclass
class WorkflowConfig:
    """工作流配置"""
    max_iterations: int = 3
    score_threshold: float = 7.0  # 评审通过阈值

    use_knowledge_base: bool = True
    retrieval_method: str = "hybrid"  # hybrid, vector, keyword
    use_rerank: bool = True
    top_k: int = 10

    writing_sections: List[str] = field(default_factory=lambda: [
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion"
    ])

    review_dimensions: List[str] = field(default_factory=lambda: [
        "innovation",
        "rigor",
        "significance",
        "clarity",
        "methodology"
    ])

    latex_format_required: bool = True

    output_format: str = "latex"  # latex, markdown, docx
    save_intermediate: bool = True


@dataclass
class WorkflowState:
    """工作流状态"""
    stage: WorkflowStage = WorkflowStage.INITIALIZATION
    current_node_id: Optional[str] = None
    completed_nodes: List[str] = field(default_factory=list)
    failed_nodes: List[str] = field(default_factory=list)
    iteration_count: int = 0
    current_scores: Dict[str, float] = field(default_factory=dict)
    final_output: Optional[str] = None
    errors: List[str] = field(default_factory=list)


class WorkflowEngine:
    """工作流引擎"""

    def __init__(
        self,
        work_dir: Path,
        config: Optional[WorkflowConfig] = None,
        api_key: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        self.work_dir = Path(work_dir)
        self.config = config or WorkflowConfig()
        self.api_key = api_key

        self.state = WorkflowState()
        self.context_manager = ContextManager(work_dir)
        self.iterator = IteratorController(
            max_iterations=self.config.max_iterations,
            score_threshold=self.config.score_threshold
        )

        self.nodes: Dict[str, BaseNode] = {}
        self._register_default_nodes()

        self.output_dir = self._create_output_dir(output_dir)

    def _register_default_nodes(self):
        _model = os.environ.get("OPENAI_MODEL", "claude-sonnet-4-6")
        self.nodes["outline"] = OutlineNode(
            node_id="outline",
            config={"output_format": "json", "model": _model}
        )

        sections = self.config.writing_sections
        for i, section in enumerate(sections, 1):
            self.nodes[f"writing_{i}"] = WritingNode(
                node_id=f"writing_{i}",
                config={
                    "section_type": section,
                    "section_order": i,
                    "total_sections": len(sections),
                    "model": _model,
                    "latex_format": self.config.latex_format_required
                }
            )

        self.nodes["audit"] = AuditNode(
            node_id="audit",
            config={"audit_types": ["logic", "citation", "consistency"]}
        )

        self.nodes["review"] = ReviewNode(
            node_id="review",
            config={
                "dimensions": self.config.review_dimensions,
                "adversarial_questions_count": 3,
                "venue_style": "top_conference"
            }
        )

        self.nodes["condition"] = ConditionNode(
            node_id="condition",
            config={
                "condition_type": "score_threshold",
                "threshold": self.config.score_threshold
            }
        )

        self.nodes["revision"] = RevisionNode(
            node_id="revision",
            config={"revision_strategy": "targeted"}
        )

        self.nodes["latex_compile"] = ToolNode(
            node_id="latex_compile",
            config={"tool_type": "latex_compile"}
        )

        self.nodes["python_viz"] = ToolNode(
            node_id="python_viz",
            config={"tool_type": "python_visualization"}
        )

    def _create_output_dir(self, override: Optional[Path] = None) -> Path:
        if override is not None:
            output_dir = Path(override).expanduser().resolve()
            if not output_dir.is_absolute():
                output_dir = (self.work_dir / output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.work_dir / "writing_outputs" / f"{timestamp}_workflow_paper"
            output_dir.mkdir(parents=True, exist_ok=True)

        for subdir in ["drafts", "final", "references", "figures", "data", "sources", "reviews"]:
            (output_dir / subdir).mkdir(exist_ok=True)

        return output_dir

    async def execute(
        self,
        topic: str,
        paper_type: str = "research_paper",
        knowledge_base_path: Optional[Path] = None,
        data_files: Optional[List[Path]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """执行完整工作流。"""
        try:
            self.state.stage = WorkflowStage.INITIALIZATION
            yield self._create_progress_update("初始化工作流", {"stage": "initialization"})

            workflow_context = self.context_manager.create_context(
                topic=topic,
                paper_type=paper_type,
                output_dir=self.output_dir,
                config=self.config
            )

            if self.config.use_knowledge_base and knowledge_base_path:
                self.state.stage = WorkflowStage.KNOWLEDGE_BASE_PREP
                yield self._create_progress_update("准备知识库", {"stage": "knowledge_base_prep"})

                yield self._create_progress_update(
                    "知识库检索功能未启用，跳过此阶段",
                    {"stage": "knowledge_base_prep", "skipped": True}
                )

            self.state.stage = WorkflowStage.OUTLINE_GENERATION
            yield self._create_progress_update("生成论文大纲（JSON模式）", {"stage": "outline_generation"})

            outline_result = await self.nodes["outline"].execute(
                self._create_node_input(workflow_context)
            )

            if outline_result.status != NodeStatus.COMPLETED:
                yield self._create_error_update("大纲生成失败", outline_result.errors)
                return

            workflow_context["outline"] = outline_result.metadata.get("outline_structure", {})
            self.state.completed_nodes.append("outline")

            outline_path = self.output_dir / "outline.json"
            with open(outline_path, 'w', encoding='utf-8') as f:
                json.dump(workflow_context["outline"], f, ensure_ascii=False, indent=2)

            yield self._create_progress_update(
                f"大纲生成完成，共{outline_result.metadata.get('sections_count', 0)}个章节",
                {"outline_path": str(outline_path)}
            )

            self.state.stage = WorkflowStage.SERIAL_WRITING

            full_content = []
            for i in range(1, 6):
                node_id = f"writing_{i}"
                section_type = self.config.writing_sections[i-1]

                yield self._create_progress_update(
                    f"撰写第{i}/5部分: {section_type}",
                    {"stage": "serial_writing", "current_section": section_type, "progress": f"{i}/5"}
                )

                node_input = self._create_node_input(
                    workflow_context,
                    previous_outputs=full_content,
                    section_type=section_type
                )

                writing_result = await self.nodes[node_id].execute(node_input)

                if writing_result.status != NodeStatus.COMPLETED:
                    yield self._create_error_update(f"第{i}部分撰写失败", writing_result.errors)
                    return

                full_content.append({
                    "section": section_type,
                    "content": writing_result.content
                })

                workflow_context[f"section_{i}"] = writing_result.content
                self.state.completed_nodes.append(node_id)

                if self.config.save_intermediate:
                    section_path = self.output_dir / "drafts" / f"section_{i}_{section_type}.tex"
                    with open(section_path, 'w', encoding='utf-8') as f:
                        f.write(writing_result.content)

            full_text = "\n\n".join([s["content"] for s in full_content])
            workflow_context["full_draft"] = full_text

            final_content = ""
            async for iter_update in self._execute_iteration_loop_with_yield(
                full_text, workflow_context
            ):
                if isinstance(iter_update, dict) and iter_update.get("type") == "progress":
                    yield iter_update
                elif isinstance(iter_update, str):
                    final_content = iter_update

            self.state.stage = WorkflowStage.FINALIZATION
            yield self._create_progress_update("最终编译与格式化", {"stage": "finalization"})

            if self.config.output_format == "latex":
                compile_result = await self._compile_latex(final_content, workflow_context)
                if compile_result:
                    yield self._create_progress_update(
                        "PDF编译完成",
                        {"pdf_path": compile_result}
                    )

            final_path = self.output_dir / "final" / "manuscript.tex"
            with open(final_path, 'w', encoding='utf-8') as f:
                f.write(final_content)

            self.state.final_output = final_content
            self.state.stage = WorkflowStage.COMPLETED

            yield self._create_result_update(final_content)

        except Exception as e:
            self.state.errors.append(str(e))
            yield self._create_error_update("工作流执行失败", [str(e)])

    async def _execute_iteration_loop_with_yield(
        self,
        initial_content: str,
        workflow_context: Dict[str, Any]
    ):
        current_content = initial_content

        while self.iterator.should_continue(self.state.iteration_count, self.state.current_scores):
            self.state.iteration_count += 1
            loop_num = self.state.iteration_count

            yield self._create_progress_update(
                f"开始第{loop_num}/{self.config.max_iterations}轮迭代",
                {"iteration": loop_num, "stage": "iteration_loop"}
            )

            self.state.stage = WorkflowStage.AUDIT
            yield self._create_progress_update(
                "执行自动化审计（逻辑+引用）",
                {"stage": "audit", "iteration": loop_num}
            )

            audit_input = self._create_node_input(
                workflow_context,
                content=current_content
            )
            audit_result = await self.nodes["audit"].execute(audit_input)

            if audit_result.status != NodeStatus.COMPLETED:
                yield self._create_error_update("审计失败", audit_result.errors)
                break

            workflow_context[f"audit_iteration_{loop_num}"] = audit_result.metadata

            self.state.stage = WorkflowStage.PEER_REVIEW
            yield self._create_progress_update(
                "模拟顶会同行评审（多维度评分+对抗性问题）",
                {"stage": "peer_review", "iteration": loop_num}
            )

            review_input = self._create_node_input(
                workflow_context,
                content=current_content,
                audit_result=audit_result.metadata
            )
            review_result = await self.nodes["review"].execute(review_input)

            if review_result.status != NodeStatus.COMPLETED:
                yield self._create_error_update("评审失败", review_result.errors)
                break

            self.state.current_scores = review_result.scores or {}
            workflow_context[f"review_iteration_{loop_num}"] = review_result.metadata
            workflow_context[f"scores_iteration_{loop_num}"] = review_result.scores
            workflow_context[f"adversarial_questions_iteration_{loop_num}"] = review_result.adversarial_questions

            review_path = self.output_dir / "reviews" / f"review_iteration_{loop_num}.json"
            with open(review_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "scores": review_result.scores,
                    "adversarial_questions": review_result.adversarial_questions,
                    "revision_suggestions": review_result.revision_suggestions,
                    "timestamp": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)

            yield self._create_progress_update(
                f"评审完成 - 平均分: {self._calculate_average_score(review_result.scores):.1f}/10",
                {
                    "scores": review_result.scores,
                    "adversarial_questions": review_result.adversarial_questions,
                    "stage": "peer_review"
                }
            )

            self.state.stage = WorkflowStage.CONDITION_CHECK
            condition_input = self._create_node_input(
                workflow_context,
                scores=review_result.scores,
                threshold=self.config.score_threshold
            )
            condition_result = await self.nodes["condition"].execute(condition_input)

            if condition_result.metadata.get("passed", False):
                yield self._create_progress_update(
                    f"评分达到阈值({self.config.score_threshold})，通过评审！",
                    {"stage": "condition_check", "passed": True}
                )
                break
            else:
                yield self._create_progress_update(
                    f"评分未达阈值，进入修改阶段",
                    {"stage": "condition_check", "passed": False}
                )

            self.state.stage = WorkflowStage.REVISION
            yield self._create_progress_update(
                "执行靶向重构（吸收评审意见与对抗性问题）",
                {"stage": "revision", "iteration": loop_num}
            )

            revision_input = self._create_node_input(
                workflow_context,
                content=current_content,
                review_result=review_result.metadata,
                adversarial_questions=review_result.adversarial_questions,
                revision_suggestions=review_result.revision_suggestions,
                iteration=loop_num
            )
            revision_result = await self.nodes["revision"].execute(revision_input)

            if revision_result.status != NodeStatus.COMPLETED:
                yield self._create_error_update("修改失败", revision_result.errors)
                break

            current_content = revision_result.content
            workflow_context[f"revision_iteration_{loop_num}"] = revision_result.content

            revision_path = self.output_dir / "drafts" / f"revision_v{loop_num}.tex"
            with open(revision_path, 'w', encoding='utf-8') as f:
                f.write(current_content)

            yield self._create_progress_update(
                f"第{loop_num}轮修改完成",
                {"stage": "revision", "revision_path": str(revision_path)}
            )

        else:
            yield self._create_progress_update(
                f"达到最大迭代次数({self.config.max_iterations})，使用当前最佳版本",
                {"stage": "iteration_complete", "reason": "max_iterations_reached"}
            )

        yield current_content

    def _create_node_input(self, context, **kwargs) -> Any:
        if hasattr(context, 'to_dict'):
            context_dict = context.to_dict()
        else:
            context_dict = dict(context) if context else {}

        node_input_fields = {'content', 'previous_outputs', 'metadata'}
        for key, value in kwargs.items():
            if key not in node_input_fields:
                context_dict[key] = value

        return NodeInput(
            content=kwargs.get("content", ""),
            context=context_dict,
            previous_outputs=kwargs.get("previous_outputs", []),
            metadata=kwargs.get("metadata", {})
        )

    def _create_progress_update(self, message: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
        return {
            "type": "progress",
            "timestamp": datetime.now().isoformat(),
            "stage": self.state.stage.value,
            "message": message,
            "iteration": self.state.iteration_count,
            "details": details or {}
        }

    def _create_error_update(self, message: str, errors: List[str]) -> Dict[str, Any]:
        return {
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "stage": self.state.stage.value,
            "message": message,
            "errors": errors
        }

    def _create_result_update(self, final_content: str) -> Dict[str, Any]:
        return {
            "type": "result",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "output_dir": str(self.output_dir),
            "final_scores": self.state.current_scores,
            "iteration_count": self.state.iteration_count,
            "content_preview": final_content[:1000] if final_content else ""
        }

    def _calculate_average_score(self, scores: Dict[str, float]) -> float:
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)

    async def _compile_latex(self, content: str, context: Dict[str, Any]) -> Optional[str]:
        try:
            tex_path = self.output_dir / "manuscript.tex"
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(content)

            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(self.output_dir / "final"), str(tex_path)],
                capture_output=True,
                text=True,
                timeout=120
            )

            pdf_path = self.output_dir / "final" / "manuscript.pdf"
            if pdf_path.exists():
                return str(pdf_path)

            return None
        except Exception:
            return None
