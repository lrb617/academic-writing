"""数据模型定义。"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


@dataclass
class ProgressUpdate:
    """文档生成过程中的进度更新。"""
    type: str = "progress"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    message: str = ""
    stage: str = "initialization"
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result.get('details') is None:
            del result['details']
        return result


@dataclass
class TextUpdate:
    """实时输出的文本片段。"""
    type: str = "text"
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperMetadata:
    """论文元数据。"""
    title: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    topic: str = ""
    word_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperFiles:
    """生成的所有文件路径。"""
    pdf_final: Optional[str] = None
    tex_final: Optional[str] = None
    pdf_drafts: List[str] = field(default_factory=list)
    tex_drafts: List[str] = field(default_factory=list)
    bibliography: Optional[str] = None
    figures: List[str] = field(default_factory=list)
    data: List[str] = field(default_factory=list)
    progress_log: Optional[str] = None
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenUsage:
    """Token 使用统计。"""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['total_tokens'] = self.total_tokens
        return result


@dataclass
class PaperResult:
    """论文生成的最终结果。"""
    type: str = "result"
    status: str = "success"
    paper_directory: str = ""
    paper_name: str = ""
    metadata: PaperMetadata = field(default_factory=PaperMetadata)
    files: PaperFiles = field(default_factory=PaperFiles)
    citations: Dict[str, Any] = field(default_factory=dict)
    figures_count: int = 0
    compilation_success: bool = False
    errors: List[str] = field(default_factory=list)
    token_usage: Optional[TokenUsage] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if isinstance(self.metadata, PaperMetadata):
            result['metadata'] = self.metadata.to_dict()
        if isinstance(self.files, PaperFiles):
            result['files'] = self.files.to_dict()
        if isinstance(self.token_usage, TokenUsage):
            result['token_usage'] = self.token_usage.to_dict()
        elif self.token_usage is None:
            del result['token_usage']
        return result


class WorkflowStage(Enum):
    """工作流执行阶段。"""
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


class NodeStatus(Enum):
    """节点执行状态。"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeType(Enum):
    """节点类型。"""
    OUTLINE = "outline"
    WRITING = "writing"
    AUDIT = "audit"
    REVIEW = "review"
    CONDITION = "condition"
    REVISION = "revision"
    TOOL = "tool"


@dataclass
class WorkflowNodeState:
    """工作流节点状态。"""
    node_id: str
    node_type: str
    status: NodeStatus = NodeStatus.PENDING
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    output_preview: str = ""
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "output_preview": self.output_preview,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class WorkflowConfig:
    """工作流配置。"""
    max_iterations: int = 3
    score_threshold: float = 7.0

    use_knowledge_base: bool = True
    retrieval_method: str = "hybrid"
    use_rerank: bool = True
    top_k: int = 10

    writing_sections: List[str] = field(default_factory=lambda: [
        "introduction", "methods", "results", "discussion", "conclusion"
    ])

    review_dimensions: List[str] = field(default_factory=lambda: [
        "innovation", "rigor", "significance", "clarity", "methodology"
    ])

    latex_format_required: bool = True

    output_format: str = "latex"
    save_intermediate: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DimensionScore:
    """评审维度评分。"""
    dimension: str
    score: float
    max_score: float = 10.0
    justification: str = ""
    confidence: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": self.score,
            "max_score": self.max_score,
            "justification": self.justification,
            "confidence": self.confidence
        }


@dataclass
class ReviewResult:
    """评审结果。"""
    overall_score: float
    dimension_scores: Dict[str, float]
    adversarial_questions: List[str]
    revision_suggestions: List[str]
    passed: bool
    iteration: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "adversarial_questions": self.adversarial_questions,
            "revision_suggestions": self.revision_suggestions,
            "passed": self.passed,
            "iteration": self.iteration
        }


@dataclass
class WorkflowResult:
    """工作流执行结果。"""
    type: str = "workflow_result"
    status: str = "success"
    output_dir: str = ""
    paper_name: str = ""
    final_content: str = ""
    iteration_count: int = 0
    final_scores: Dict[str, float] = field(default_factory=dict)
    node_states: List[Dict[str, Any]] = field(default_factory=list)
    review_history: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    token_usage: Optional[TokenUsage] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.token_usage:
            result['token_usage'] = self.token_usage.to_dict()
        else:
            del result['token_usage']
        return result


@dataclass
class WorkflowProgressUpdate:
    """工作流进度更新。"""
    type: str = "workflow_progress"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    stage: str = ""
    message: str = ""
    iteration: int = 0
    current_node: Optional[str] = None
    current_scores: Optional[Dict[str, float]] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result.get('details') is None:
            del result['details']
        if result.get('current_scores') is None:
            del result['current_scores']
        if result.get('current_node') is None:
            del result['current_node']
        return result
