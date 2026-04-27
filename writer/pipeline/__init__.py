"""学术写作流水线：节点定义与工作流控制。"""

from .controller import (
    ContextManager,
    IteratorController,
    PipelineBuilder,
    WorkflowContext,
    WorkflowEngine,
)
from .stages import (
    AuditNode,
    BaseNode,
    ConditionNode,
    NodeInput,
    NodeOutput,
    NodeStatus,
    NodeType,
    OutlineNode,
    ReviewNode,
    RevisionNode,
    ToolNode,
    WritingNode,
)

__all__ = [
    "BaseNode",
    "NodeStatus",
    "NodeType",
    "NodeInput",
    "NodeOutput",
    "OutlineNode",
    "WritingNode",
    "AuditNode",
    "ReviewNode",
    "ConditionNode",
    "RevisionNode",
    "ToolNode",
    "WorkflowEngine",
    "WorkflowContext",
    "ContextManager",
    "PipelineBuilder",
    "IteratorController",
]
