"""病例安全工作流的稳定数据模型。

所有时间字段均使用 ISO 8601 字符串，避免持久化时出现不可序列化对象。
模型只保存事实、证据和版本，不直接删除或改写原始来源。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import types
from typing import Any, Union, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class Span:
    """来源文本中的字符区间，便于医学审阅者回到原文核对。"""

    start: int
    end: int
    text: str


@dataclass
class Product:
    """药品目录中的已知产品。"""

    id: str
    name: str
    ingredient: str | None = None
    aliases: list[str] = field(default_factory=list)


@dataclass
class Subject:
    """患者或叙述中出现的其他自然人。"""

    id: str
    case_id: str
    label: str
    identity_status: str = "anonymous_unresolved"
    patient_key: str | None = None
    source_record_ids: list[str] = field(default_factory=list)
    merged_into: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceRecord:
    """不可变的原始来源记录。

    更正通过 RecordCorrection 追加，而不是就地修改本对象。
    """

    id: str
    case_id: str
    text: str
    source: str | None = None
    kind: str | None = None
    channel: str | None = None
    author: str | None = None
    patient_key: str | None = None
    derived_from: list[str] = field(default_factory=list)
    duplicate_of: str | None = None
    received_at: str | None = None
    recorded_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordCorrection:
    """一次可审计的记录更正。"""

    id: str
    record_id: str
    case_id: str
    reason: str
    corrected_by: str
    corrected_at: str
    new_text: str | None = None
    metadata_patch: dict[str, Any] = field(default_factory=dict)
    revision: int = 1


@dataclass
class Mention:
    """从文本中抽出的实体、事件或时间表达。"""

    id: str
    case_id: str
    record_id: str
    revision: int
    kind: str
    text: str
    span: Span
    subject_id: str | None = None
    canonical: str | None = None
    concept_id: str | None = None
    time_id: str | None = None
    polarity: str = "positive"
    certainty: str = "stated"
    identity_status: str = "not_applicable"
    verification_status: str = "extracted"
    is_current: bool = True
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """患者-药物-症状-时间-证据之间的可复核关系。"""

    id: str
    case_id: str
    kind: str
    explanation: str
    subject_id: str | None = None
    drug_mention_id: str | None = None
    symptom_mention_id: str | None = None
    time_mention_id: str | None = None
    lab_id: str | None = None
    evidence_mention_ids: list[str] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)
    status: str = "supported"
    confidence: float = 0.7
    verification_status: str = "pending_review"
    is_current: bool = True
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class LabResult:
    """后补实验室结果。severity_rank 越大越严重。"""

    id: str
    case_id: str
    record_id: str
    revision: int
    test_name: str
    severity_rank: int
    severity_label: str
    span: Span | None = None
    test_concept: str | None = None
    value: float | None = None
    unit: str | None = None
    flag: str | None = None
    observed_time_id: str | None = None
    observed_time_text: str | None = None
    subject_id: str | None = None
    verification_status: str = "extracted"
    is_current: bool = True
    evidence_mention_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineEvent:
    """时间线上的一个可回溯事件。"""

    order_key: float | None
    title: str
    event_type: str
    record_id: str
    mention_id: str | None = None
    time_text: str | None = None
    time_status: str = "resolved"
    subject_id: str | None = None
    detail: str | None = None


@dataclass
class JudgmentVersion:
    """某一时刻不可变的医学/信号判断快照。"""

    version: int
    case_id: str
    created_at: str
    trigger: str
    severity_rank: int
    severity_label: str
    signal_status: str
    summary: str
    rationale: list[str]
    timeline: list[TimelineEvent]
    relationship_ids: list[str]
    positive_symptoms: list[dict[str, Any]]
    negated_findings: list[dict[str, Any]]
    pending_items: list[dict[str, Any]]
    independent_case_count: int
    provisional_case_count: int
    pending_identity_report_count: int
    relationship_snapshots: list[dict[str, Any]] = field(default_factory=list)
    based_on_correction_id: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    lab_ids: list[str] = field(default_factory=list)


@dataclass
class FollowUpRequest:
    """已签发的随访请求。

    标题、说明、证据和签发时间等原始字段签发后不应被更正流程改写；
    新情况通过 ReconfirmationItem 或新的请求处理。
    """

    id: str
    case_id: str
    title: str
    description: str
    issued_by: str
    issued_at: str
    due_at: str | None = None
    channel: str | None = None
    question_codes: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    status: str = "issued"
    original_snapshot: dict[str, Any] = field(default_factory=dict)
    response_entries: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RiskStatementVersion:
    version: int
    text: str
    severity_rank: int
    changed_by: str
    changed_at: str
    reason: str
    decision_id: str | None = None


@dataclass
class RiskStatement:
    id: str
    case_id: str
    product_id: str | None
    product_name: str
    symptom_concept: str | None
    current_version: int
    status: str = "draft"
    versions: list[RiskStatementVersion] = field(default_factory=list)
    pending_reconfirmation_ids: list[str] = field(default_factory=list)


@dataclass
class SignalDecision:
    id: str
    case_id: str
    decision: str
    decided_by: str
    decided_at: str
    role: str = "safety_signal_lead"
    rationale: str = ""
    follow_up_id: str | None = None
    risk_statement_id: str | None = None
    reconfirmation_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconfirmationItem:
    """更正后必须由负责人重新确认的随访或风险说明。"""

    id: str
    case_id: str
    target_type: str
    target_id: str
    reason: str
    correction_id: str
    status: str = "pending"
    created_at: str = ""
    resolved_at: str | None = None
    resolved_by: str | None = None
    resolution: str | None = None


@dataclass
class ReviewItem:
    id: str
    case_id: str
    target_type: str
    target_id: str
    label: str
    reason: str
    status: str = "open"
    created_at: str = ""
    resolved_at: str | None = None
    reviewer: str | None = None
    resolution_note: str | None = None


@dataclass
class AuditEvent:
    seq: int
    at: str
    actor: str
    action: str
    case_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseAggregate:
    case_id: str
    created_at: str
    updated_at: str
    status: str = "active"
    merged_into_case_id: str | None = None
    patients: list[Subject] = field(default_factory=list)
    records: list[SourceRecord] = field(default_factory=list)
    corrections: list[RecordCorrection] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    labs: list[LabResult] = field(default_factory=list)
    judgments: list[JudgmentVersion] = field(default_factory=list)
    follow_ups: list[FollowUpRequest] = field(default_factory=list)
    risk_statements: list[RiskStatement] = field(default_factory=list)
    decisions: list[SignalDecision] = field(default_factory=list)
    reconfirmations: list[ReconfirmationItem] = field(default_factory=list)
    review_items: list[ReviewItem] = field(default_factory=list)

    def current_judgment(self) -> JudgmentVersion | None:
        return self.judgments[-1] if self.judgments else None


def to_jsonable(obj: Any) -> Any:
    """将 dataclass/列表/字典转换为 JSON 原生结构。"""

    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj


def model_to_dict(obj: Any) -> dict[str, Any]:
    if not hasattr(obj, "__dataclass_fields__"):
        raise TypeError("model_to_dict only supports dataclass instances")
    return asdict(obj)


def _coerce_type(annotation: Any, value: Any) -> Any:
    if value is None:
        return None

    if isinstance(annotation, str):
        # 正常情况下 get_type_hints 已把字符串解析成类型；保留兜底。
        return value

    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        for arg in args:
            if hasattr(arg, "__dataclass_fields__") and isinstance(value, dict):
                return model_from_dict(arg, value)
            if get_origin(arg) in (list, list) and isinstance(value, list):
                return _coerce_type(arg, value)
        return value

    if origin in (list, list):
        item_type = get_args(annotation)[0]
        if isinstance(value, list):
            return [_coerce_type(item_type, item) for item in value]

    if hasattr(annotation, "__dataclass_fields__") and isinstance(value, dict):
        return model_from_dict(annotation, value)

    return value


def model_from_dict(cls: type[Any], data: dict[str, Any]) -> Any:
    """从 asdict 生成的 JSON 友好字典重建数据类。"""

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        name = field_info.name
        if name not in data:
            continue
        kwargs[name] = _coerce_type(hints.get(name, field_info.type), data[name])
    return cls(**kwargs)
