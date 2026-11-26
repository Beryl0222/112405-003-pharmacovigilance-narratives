"""病例安全叙述 NLP 与随访证据闭环工作流。"""

from .models import (
    CaseAggregate,
    FollowUpRequest,
    JudgmentVersion,
    Product,
    RecordCorrection,
    ReconfirmationItem,
    Relationship,
    ReviewItem,
    RiskStatement,
    SourceRecord,
    Subject,
)
from .nlp import extract_narrative
from .pipeline import NLPWorkflow, PharmacovigilanceWorkflow, create_workbench
from .workflow import CaseWorkflow, SafetyWorkflow
from .workbench import CaseWorkbench

__all__ = [
    "CaseWorkbench",
    "NLPWorkflow",
    "PharmacovigilanceWorkflow",
    "SafetyWorkflow",
    "CaseWorkflow",
    "create_workbench",
    "CaseAggregate",
    "FollowUpRequest",
    "JudgmentVersion",
    "Product",
    "RecordCorrection",
    "ReconfirmationItem",
    "Relationship",
    "ReviewItem",
    "RiskStatement",
    "SourceRecord",
    "Subject",
    "extract_narrative",
    "analyze_case_payload",
    "analyze_text",
    "analyze_narrative",
    "process_case_payload",
]

__version__ = "0.2.0"


def analyze_case_payload(payload: dict) -> dict:
    """一次性分析病例 JSON，返回可 JSON 序列化报告。"""

    wb = CaseWorkbench(products=payload.get("products", []))
    case = wb.ingest_payload(payload)
    if isinstance(case, list):
        return {"cases": [wb.case_report(item.case_id) for item in case], "signal_board": wb.signal_board()}
    return {"case": wb.case_report(case.case_id), "signal_board": wb.signal_board()}


def analyze_text(text: str, *, case_id: str = "inline-case", products: list[dict] | None = None) -> dict:
    """分析单段中文叙述的便捷函数。"""

    wb = CaseWorkbench(products=products or [])
    wb.ingest_text(text, case_id=case_id, record_id="inline-record")
    return wb.case_report(case_id)


analyze_narrative = analyze_text
process_case_payload = analyze_case_payload
