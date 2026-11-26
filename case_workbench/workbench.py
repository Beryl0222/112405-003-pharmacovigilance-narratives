"""病例安全 NLP 工作流的应用服务层。

该模块负责持续接收记录、维护证据图、生成不可变判断版本、
处理更正影响，以及随访和风险说明的版本化。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .identity import duplicate_confidence, normalize_text
from .models import (
    AuditEvent,
    CaseAggregate,
    FollowUpRequest,
    JudgmentVersion,
    LabResult,
    Mention,
    Product,
    RecordCorrection,
    ReconfirmationItem,
    Relationship,
    ReviewItem,
    RiskStatement,
    RiskStatementVersion,
    SignalDecision,
    SourceRecord,
    Subject,
    TimelineEvent,
    model_from_dict,
)
from .nlp import extract_narrative


PENDING_RELATION_STATUSES = {"pending_drug", "pending_drug_identity", "pending_time"}
TIME_PRECISION_PENDING_STATUS = "relative_order_known_time_pending"
REVIEW_RELATION_STATUSES = PENDING_RELATION_STATUSES | {
    "confounded_by_prior_dose_reduction",
    TIME_PRECISION_PENDING_STATUS,
}


class CaseWorkbench:
    """内存态病例工作台；可用 to_dict/save 持久化。"""

    def __init__(
        self,
        products: Iterable[Product | dict[str, Any]] | None = None,
        now_provider: Any | None = None,
    ) -> None:
        self.products: list[Product] = [self._as_product(p) for p in (products or [])]
        self.cases: dict[str, CaseAggregate] = {}
        self.audit_events: list[AuditEvent] = []
        self._now_provider = now_provider

    # ------------------------------------------------------------------ basics

    @staticmethod
    def _as_product(value: Product | dict[str, Any]) -> Product:
        if isinstance(value, Product):
            return value
        return Product(
            id=str(value["id"]),
            name=str(value["name"]),
            ingredient=value.get("ingredient"),
            aliases=list(value.get("aliases", [])),
        )

    def add_products(self, products: Iterable[Product | dict[str, Any]]) -> None:
        existing = {p.id for p in self.products}
        for product in products:
            item = self._as_product(product)
            if item.id not in existing:
                self.products.append(item)
                existing.add(item.id)

    def _now(self) -> str:
        if self._now_provider is not None:
            return self._now_provider()
        return datetime.now(timezone.utc).isoformat()

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:10]}"

    def _audit(self, actor: str, action: str, case_id: str | None = None, **payload: Any) -> None:
        self.audit_events.append(
            AuditEvent(
                seq=len(self.audit_events) + 1,
                at=self._now(),
                actor=actor,
                action=action,
                case_id=case_id,
                payload=payload,
            )
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "products": [p.__dict__ for p in self.products],
            "cases": {cid: _dataclass_to_dict(case) for cid, case in self.cases.items()},
            "audit_events": [_dataclass_to_dict(event) for event in self.audit_events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseWorkbench":
        wb = cls(products=[model_from_dict(Product, p) for p in data.get("products", [])])
        for cid, case_data in data.get("cases", {}).items():
            wb.cases[cid] = model_from_dict(CaseAggregate, case_data)
        wb.audit_events = [model_from_dict(AuditEvent, e) for e in data.get("audit_events", [])]
        return wb

    @classmethod
    def load(cls, path: str | Path) -> "CaseWorkbench":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -------------------------------------------------------------- ingestion

    def ingest_path(self, path: str | Path) -> CaseAggregate | list[CaseAggregate]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.ingest_payload(data)

    def ingest_payload(
        self, data: dict[str, Any] | list[dict[str, Any]], case_id: str | None = None
    ) -> CaseAggregate | list[CaseAggregate]:
        if isinstance(data, list):
            results = []
            for item in data:
                results.append(self.ingest_record(item, case_id=case_id or item.get("case_id")))
            return results

        if data.get("products"):
            self.add_products(data["products"])

        if isinstance(data.get("records"), list):
            cid = case_id or data.get("case_id") or self._new_id("CASE")
            result = None
            for record in data["records"]:
                result = self.ingest_record(record, case_id=record.get("case_id") or cid)
            return result or self.get_case(cid)

        if "text" in data and "id" in data:
            cid = case_id or data.get("case_id") or self._new_id("CASE")
            return self.ingest_record(data, case_id=cid)

        raise ValueError("无法识别的输入：需要 records 列表，或包含 id/text 的单条记录")

    def ingest_text(
        self,
        text: str,
        *,
        case_id: str | None = None,
        record_id: str | None = None,
        source: str = "manual-text",
        **record_fields: Any,
    ) -> CaseAggregate:
        cid = case_id or self._new_id("CASE")
        rid = record_id or self._new_id("REC")
        payload = {"id": rid, "text": text, "source": source, **record_fields}
        return self.ingest_record(payload, case_id=cid)

    def get_case(self, case_id: str) -> CaseAggregate:
        if case_id not in self.cases:
            raise KeyError(f"找不到病例 {case_id}")
        return self.cases[case_id]

    def review_case(self, case_id: str) -> CaseAggregate:
        """供医学审阅者读取完整证据图。"""

        return self.get_case(case_id)

    def review_graph(self, case_id: str) -> dict[str, Any]:
        """返回患者-药物-症状-时间-来源的连接视图。"""

        report = self.case_report(case_id)
        keys = [
            "case_id",
            "disclaimer",
            "status",
            "merged_into_case_id",
            "patients",
            "records",
            "mentions",
            "relationships",
            "labs",
            "current_judgment",
            "review_queue",
            "provenance",
        ]
        return {key: report[key] for key in keys if key in report}

    def ingest_record(self, record: dict[str, Any] | SourceRecord, case_id: str | None = None) -> CaseAggregate:
        data = record.__dict__ if isinstance(record, SourceRecord) else dict(record)
        record_id = str(data.get("id") or self._new_id("REC"))
        text = str(data.get("text") or "")
        if not text.strip():
            raise ValueError(f"记录 {record_id} 的 text 不能为空")

        cid = case_id or data.get("case_id") or self._new_id("CASE")
        now = self._now()
        case = self.cases.get(cid)
        if case is None:
            case = CaseAggregate(case_id=cid, created_at=now, updated_at=now)
            self.cases[cid] = case

        existing = next((r for r in case.records if r.id == record_id), None)
        if existing is not None:
            if existing.text == text:
                return case
            # 持续摄入时也可能收到同 ID 的更正版正文。拒绝覆盖原文，转为追加更正事件。
            correction_reason = str(
                data.get("correction_reason")
                or data.get("reason")
                or "同一记录编号送达了不同正文；为避免覆盖原始证据，按更正版本处理"
            )
            corrected_by = str(data.get("corrected_by") or data.get("updated_by") or "ingestion-pipeline")
            self.correct_record(
                cid,
                record_id,
                new_text=text,
                reason=correction_reason,
                corrected_by=corrected_by,
            )
            return self.get_case(cid)

        derived_from = list(data.get("derived_from", []) or [])
        if data.get("quoted_from"):
            derived_from.append(str(data["quoted_from"]))
        if isinstance(data.get("derived_from"), str):
            derived_from = [data["derived_from"]]

        source_record = SourceRecord(
            id=record_id,
            case_id=cid,
            text=text,
            source=data.get("source"),
            kind=data.get("kind") or data.get("source"),
            channel=data.get("channel"),
            author=data.get("author"),
            patient_key=data.get("patient_key") or data.get("anonymous_key"),
            derived_from=derived_from,
            duplicate_of=data.get("duplicate_of"),
            received_at=data.get("received_at") or now,
            recorded_at=data.get("recorded_at"),
            metadata={k: v for k, v in data.items() if k not in {
                "id", "text", "source", "kind", "channel", "author", "patient_key",
                "anonymous_key", "derived_from", "quoted_from", "duplicate_of",
                "received_at", "recorded_at", "case_id",
            }},
        )
        case.records.append(source_record)
        case.updated_at = now

        index_subject = self._ensure_index_subject(case, source_record.patient_key)
        extraction = extract_narrative(cid, record_id, text, self.products, revision=0)

        local_subjects: dict[str, str] = {"index": index_subject.id}
        for mention in extraction.mentions:
            ref = mention.subject_id
            if ref and ref != "index" and ref.startswith("other:"):
                if ref not in local_subjects:
                    subject = Subject(
                        id=self._new_id("P-OTHER"),
                        case_id=cid,
                        label="叙述中的不明身份第三方",
                        identity_status="anonymous_unresolved",
                        source_record_ids=[record_id],
                        attributes={"local_ref": ref},
                    )
                    case.patients.append(subject)
                    local_subjects[ref] = subject.id
                    self._add_review_item(
                        case,
                        target_type="subject",
                        target_id=subject.id,
                        label="不明身份报告对象",
                        reason="叙述出现“有人/另一名”等第三方表达，不能默认并入主患者，需核实是否为重复病例。",
                    )
                mention.subject_id = local_subjects[ref]
            elif ref == "index":
                mention.subject_id = index_subject.id

            if mention.subject_id:
                subject = self._find_subject_by_id(mention.subject_id)
                if subject and record_id not in subject.source_record_ids:
                    subject.source_record_ids.append(record_id)

        for lab in extraction.labs:
            ref = lab.subject_id or "index"
            if ref.startswith("other:") and ref not in local_subjects:
                subject = Subject(
                    id=self._new_id("P-OTHER"),
                    case_id=cid,
                    label="叙述中的不明身份第三方",
                    identity_status="anonymous_unresolved",
                    source_record_ids=[record_id],
                )
                case.patients.append(subject)
                local_subjects[ref] = subject.id
            lab.subject_id = local_subjects.get(ref, index_subject.id)

        case.mentions.extend(extraction.mentions)
        for lab in extraction.labs:
            case.labs.append(lab)

        for index, draft in enumerate(extraction.relations):
            subject_id = local_subjects.get(draft.subject_ref, index_subject.id)
            relationship = Relationship(
                id=f"REL:{record_id}:r0:{index}",
                case_id=cid,
                kind=draft.kind,
                explanation=draft.explanation,
                subject_id=subject_id,
                drug_mention_id=draft.drug_id,
                symptom_mention_id=draft.symptom_id,
                evidence_mention_ids=draft.evidence_ids,
                record_ids=[record_id],
                status=draft.status,
                confidence=draft.confidence,
                verification_status="pending_review"
                if draft.status in REVIEW_RELATION_STATUSES
                else "extracted",
                attributes=draft.attributes,
            )
            case.relationships.append(relationship)
            if relationship.verification_status == "pending_review":
                self._add_review_item(
                    case,
                    target_type="relationship",
                    target_id=relationship.id,
                    label=self._relation_label(relationship.kind, relationship.status),
                    reason=draft.explanation,
                )

        # 对明确待核的实体表达生成审阅任务。
        for mention in extraction.mentions:
            if mention.verification_status == "pending_verification":
                label = {
                    "drug": "药物身份待核",
                    "time": "时间表达待核",
                    "symptom": "症状确定性待核",
                }.get(mention.kind, f"{mention.kind}待核")
                self._add_review_item(
                    case,
                    target_type="mention",
                    target_id=mention.id,
                    label=label,
                    reason=f"原文“{mention.text}”身份或确定性不足，不能压缩为已确认关系。",
                )

        self._enrich_case_drug_context(case)
        self._relink_quotes()
        self._auto_link_duplicates()
        self._reconcile_timeline_relationships(case)
        previous_judgment = case.current_judgment()
        judgment = self._reanalyze(cid, trigger="record_ingested")
        if judgment.lab_ids and (
            previous_judgment is None or judgment.severity_rank > previous_judgment.severity_rank
        ):
            for statement in case.risk_statements:
                if statement.status in {"active", "draft"}:
                    self._add_review_item(
                        case,
                        target_type="risk_statement",
                        target_id=statement.id,
                        label="实验室结果可能要求调整风险说明",
                        reason=(
                            f"新实验室结果将严重程度提升至 {judgment.severity_label}；"
                            "原风险说明仍保留，需安全信号负责人确认是否调整。"
                        ),
                    )
            self._audit(
                "nlp-pipeline",
                "lab_severity_updated",
                cid,
                record_id=record_id,
                severity_rank=judgment.severity_rank,
            )
        self._audit("ingestion-pipeline", "record_ingested", cid, record_id=record_id, source=source_record.source)
        return case

    def _ensure_index_subject(self, case: CaseAggregate, patient_key: str | None) -> Subject:
        index_id = f"P:{case.case_id}:index"
        subject = next((p for p in case.patients if p.id == index_id), None)
        if subject is None:
            subject = Subject(
                id=index_id,
                case_id=case.case_id,
                label="病例主患者（匿名）",
                identity_status="key_linked" if patient_key else "case_scoped_anonymous",
                patient_key=patient_key,
            )
            case.patients.append(subject)
        elif patient_key and not subject.patient_key:
            subject.patient_key = patient_key
            subject.identity_status = "key_linked"
        return subject

    def _find_subject_by_id(self, subject_id: str | None) -> Subject | None:
        if not subject_id:
            return None
        for case in self.cases.values():
            subject = next((p for p in case.patients if p.id == subject_id), None)
            if subject:
                return subject
        return None

    def _find_subject_by_key(self, patient_key: str) -> Subject | None:
        for case in self.cases.values():
            subject = next((p for p in case.patients if p.patient_key == patient_key), None)
            if subject:
                return subject
        return None

    def _add_review_item(
        self, case: CaseAggregate, *, target_type: str, target_id: str, label: str, reason: str
    ) -> ReviewItem:
        existing = next(
            (
                item
                for item in case.review_items
                if item.target_type == target_type and item.target_id == target_id and item.status == "open"
            ),
            None,
        )
        if existing:
            return existing
        item = ReviewItem(
            id=self._new_id("REV"),
            case_id=case.case_id,
            target_type=target_type,
            target_id=target_id,
            label=label,
            reason=reason,
            created_at=self._now(),
        )
        case.review_items.append(item)
        return item

    @staticmethod
    def _relation_label(kind: str, status: str) -> str:
        labels = {
            "drug_symptom_link": "药物-症状关系",
            "dose_reduction_precedes_withdrawal": "减量早于停药",
            "withdrawal_to_improvement": "停药后改善",
            "negated_symptom": "否定症状",
        }
        base = labels.get(kind, kind)
        if status == "confounded_by_prior_dose_reduction":
            return f"{base}（存在混杂，需复核时间顺序）"
        if status.startswith("pending"):
            return f"{base}（待核）"
        return base

    # ------------------------------------------------------------ provenance

    def _relink_quotes(self) -> None:
        """把引号中的患者原话与原始叙述来源连接，避免形成第二个病例。"""

        all_records = [(case, record) for case in self.cases.values() for record in case.records]
        for case, record in all_records:
            quote_mentions = [
                m for m in case.mentions
                if m.record_id == record.id and m.kind == "quote" and m.is_current
            ]
            for quote in quote_mentions:
                quoted = quote.attributes.get("quoted_text") or quote.canonical or ""
                best_id = None
                best_score = 0.0
                for other_case, other in all_records:
                    if other.id == record.id:
                        continue
                    if normalize_text(quoted) and normalize_text(quoted) in normalize_text(other.text):
                        best_id, best_score = other.id, 1.0
                        break
                    score = duplicate_confidence(quoted, other.text)
                    if score > best_score:
                        best_id, best_score = other.id, score
                if best_id and best_score >= 0.72 and best_id not in record.derived_from:
                    record.derived_from.append(best_id)
                    self._audit(
                        "nlp-pipeline",
                        "quote_provenance_linked",
                        case.case_id,
                        record_id=record.id,
                        source_record_id=best_id,
                        confidence=best_score,
                    )
        self._link_provenance_identities()

    def _find_record_anywhere(self, record_id: str) -> tuple[CaseAggregate | None, SourceRecord | None]:
        for case in self.cases.values():
            record = next((r for r in case.records if r.id == record_id), None)
            if record:
                return case, record
        return None, None

    def _link_provenance_identities(self) -> None:
        """引用/重复来源链指向同一病例主体，避免引用者被算成新的独立病例。"""

        for case in list(self.cases.values()):
            child_subject = next((p for p in case.patients if p.id == f"P:{case.case_id}:index"), None)
            if child_subject is None or child_subject.merged_into:
                continue
            for record in case.records:
                linked_ids = list(record.derived_from)
                if record.duplicate_of:
                    linked_ids.append(record.duplicate_of)
                for parent_id in linked_ids:
                    parent_case, _ = self._find_record_anywhere(parent_id)
                    if parent_case is None or parent_case.case_id == case.case_id:
                        continue
                    parent_subject = next(
                        (p for p in parent_case.patients if p.id == f"P:{parent_case.case_id}:index"), None
                    )
                    if parent_subject is None:
                        continue
                    if (
                        child_subject.patient_key
                        and parent_subject.patient_key
                        and child_subject.patient_key != parent_subject.patient_key
                    ):
                        continue
                    child_subject.merged_into = parent_subject.id
                    if child_subject.identity_status in {"case_scoped_anonymous", "key_linked"}:
                        child_subject.identity_status = "provenance_linked"
                    if child_subject.patient_key and not parent_subject.patient_key:
                        parent_subject.patient_key = child_subject.patient_key
                    return

    def _auto_link_duplicates(self) -> None:
        """对极高相似度跨渠道记录建立临时身份根；仍生成审阅任务，不物理删除病例。"""

        matches = self.find_duplicate_sources(threshold=0.9)
        for match in matches:
            if match["case_a"] == match["case_b"]:
                continue
            case_a = self.cases.get(match["case_a"])
            case_b = self.cases.get(match["case_b"])
            if not case_a or not case_b:
                continue

            pairs: list[tuple[Subject, Subject]] = []
            index_a = next((p for p in case_a.patients if p.id == f"P:{case_a.case_id}:index"), None)
            index_b = next((p for p in case_b.patients if p.id == f"P:{case_b.case_id}:index"), None)
            if index_a and index_b:
                pairs.append((index_a, index_b))

            other_a = [p for p in case_a.patients if match["record_a"] in p.source_record_ids and p.identity_status == "anonymous_unresolved"]
            other_b = [p for p in case_b.patients if match["record_b"] in p.source_record_ids and p.identity_status == "anonymous_unresolved"]
            if len(other_a) == 1 and len(other_b) == 1:
                pairs.append((other_a[0], other_b[0]))

            for source, target in pairs:
                if source.id == target.id or source.merged_into:
                    continue
                if source.patient_key and target.patient_key and source.patient_key != target.patient_key:
                    continue
                source.attributes["pre_merge_identity_status"] = source.identity_status
                # 以较早创建的病例作为身份根。
                if case_b.created_at < case_a.created_at and source is index_a:
                    source, target = target, source
                source.merged_into = target.id
                if source.identity_status != "anonymous_unresolved":
                    source.identity_status = "provisional_duplicate"
                self._add_review_item(
                    case_a if source.case_id == case_a.case_id else case_b,
                    target_type="subject",
                    target_id=source.id,
                    label="高置信跨渠道重复身份，待医学审阅确认",
                    reason=(
                        f"记录 {match['record_a']} 与 {match['record_b']} 的文本相似度为 "
                        f"{match['confidence']}，已临时归并身份根；确认前仍保留两条来源。"
                    ),
                )
                self._audit(
                    "identity-pipeline",
                    "provisional_duplicate_linked",
                    source.case_id,
                    source_subject_id=source.id,
                    target_subject_id=target.id,
                    confidence=match["confidence"],
                )

    def provenance_graph(self, case_id: str) -> dict[str, Any]:
        case = self.get_case(case_id)
        nodes = [
            {"id": r.id, "source": r.source, "kind": r.kind, "channel": r.channel, "text": r.text}
            for r in case.records
        ]
        edges: list[dict[str, str]] = []
        for record in case.records:
            for parent in record.derived_from:
                edges.append({"from": parent, "to": record.id, "type": "derived_from_or_quote"})
            if record.duplicate_of:
                edges.append({"from": record.duplicate_of, "to": record.id, "type": "duplicate_of"})
        families = self._source_families(case)
        return {"nodes": nodes, "edges": edges, "source_families": families}

    def _source_families(self, case: CaseAggregate) -> list[dict[str, Any]]:
        parent = {r.id: r.id for r in case.records}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        ids = {r.id for r in case.records}
        for record in case.records:
            for linked in record.derived_from:
                if linked in ids:
                    union(record.id, linked)
            if record.duplicate_of and record.duplicate_of in ids:
                union(record.id, record.duplicate_of)

        groups: dict[str, list[str]] = {}
        for record_id in parent:
            groups.setdefault(find(record_id), []).append(record_id)
        return [{"root": root, "record_ids": sorted(members)} for root, members in groups.items()]

    def source_independence(self, case_id: str) -> dict[str, Any]:
        graph = self.provenance_graph(case_id)
        families = graph["source_families"]
        return {
            "record_count": len(graph["nodes"]),
            "independent_source_family_count": len(families),
            "families": families,
            "note": "引用/转发形成的是同一来源链，不产生新的独立病例。",
        }

    def find_duplicate_sources(self, threshold: float = 0.78) -> list[dict[str, Any]]:
        records = [(case.case_id, r) for case in self.cases.values() for r in case.records]
        matches = []
        for i, (cid_a, a) in enumerate(records):
            for cid_b, b in records[i + 1 :]:
                score = duplicate_confidence(a.text, b.text)
                explicit = bool(a.duplicate_of == b.id or b.duplicate_of == a.id)
                if explicit or score >= threshold:
                    matches.append(
                        {
                            "record_a": a.id,
                            "case_a": cid_a,
                            "record_b": b.id,
                            "case_b": cid_b,
                            "confidence": 1.0 if explicit else score,
                            "explicit_link": explicit,
                            "recommended_action": "merge_identity" if explicit or score >= 0.9 else "review_before_merge",
                        }
                    )
        return matches

    def find_duplicate_cases(self, threshold: float = 0.72) -> list[dict[str, Any]]:
        """按跨渠道文本相似度汇总可能重复的病例；默认只建议，不自动合并身份。"""

        source_matches = self.find_duplicate_sources(threshold=threshold)
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for match in source_matches:
            if match["case_a"] == match["case_b"]:
                continue
            key = tuple(sorted((match["case_a"], match["case_b"])))
            item = grouped.setdefault(
                key,
                {
                    "case_a": key[0],
                    "case_b": key[1],
                    "confidence": 0.0,
                    "source_matches": [],
                    "recommended_action": "review_before_merge",
                },
            )
            item["confidence"] = max(item["confidence"], match["confidence"])
            item["source_matches"].append(match)
            if match["explicit_link"] or item["confidence"] >= 0.9:
                item["recommended_action"] = "merge_identity"
        return sorted(grouped.values(), key=lambda x: -x["confidence"])

    def mark_duplicate_case(
        self,
        duplicate_case_id: str,
        primary_case_id: str,
        *,
        actor: str = "medical-reviewer",
        reason: str = "不同渠道识别为同一匿名病例",
    ) -> CaseAggregate:
        duplicate = self.get_case(duplicate_case_id)
        primary = self.get_case(primary_case_id)
        if duplicate_case_id == primary_case_id:
            raise ValueError("不能把病例标记为自身重复")
        duplicate.status = "duplicate"
        duplicate.merged_into_case_id = primary_case_id
        target_index = f"P:{primary_case_id}:index"
        for subject in duplicate.patients:
            if subject.identity_status != "anonymous_unresolved":
                subject.identity_status = "duplicate"
                subject.merged_into = target_index
        duplicate.updated_at = self._now()
        self._audit(actor, "case_marked_duplicate", duplicate_case_id, primary_case_id=primary_case_id, reason=reason)
        return duplicate

    def merge_patients(
        self,
        source_case_id: str,
        source_patient_id: str,
        target_patient_id: str,
        *,
        target_case_id: str | None = None,
        actor: str = "medical-reviewer",
        reason: str = "审阅确认身份相同",
    ) -> Subject:
        source_case = self.get_case(source_case_id)
        source = next((p for p in source_case.patients if p.id == source_patient_id), None)
        if source is None:
            raise KeyError(f"找不到主体 {source_patient_id}")
        target = self._find_subject_by_id(target_patient_id)
        if target is None:
            raise KeyError(f"找不到目标主体 {target_patient_id}")
        source.attributes["pre_merge_identity_status"] = source.identity_status
        source.identity_status = "duplicate"
        source.merged_into = target_patient_id
        if source.patient_key and not target.patient_key:
            target.patient_key = source.patient_key
        for rid in source.source_record_ids:
            if rid not in target.source_record_ids:
                target.source_record_ids.append(rid)
        self._audit(
            actor,
            "patient_merged",
            source_case_id,
            source_patient_id=source_patient_id,
            target_patient_id=target_patient_id,
            target_case_id=target.case_id if target else target_case_id,
            reason=reason,
        )
        self._reanalyze(source_case_id, trigger="identity_merged")
        if target.case_id != source_case_id:
            self._reanalyze(target.case_id, trigger="identity_merged")
        return target

    def confirm_subject(
        self, case_id: str, subject_id: str, reviewer: str, *, note: str = "身份已经审阅确认"
    ) -> Subject:
        case = self.get_case(case_id)
        subject = next((p for p in case.patients if p.id == subject_id), None)
        if subject is None:
            raise KeyError(f"找不到主体 {subject_id}")
        subject.identity_status = "confirmed"
        for item in case.review_items:
            if item.target_type == "subject" and item.target_id == subject_id and item.status == "open":
                item.status = "resolved"
                item.resolved_at = self._now()
                item.reviewer = reviewer
                item.resolution_note = note
        self._audit(reviewer, "subject_confirmed", case_id, subject_id=subject_id)
        self._reanalyze(case_id, trigger="identity_confirmed")
        return subject

    def reject_identity_merge(
        self, case_id: str, subject_id: str, reviewer: str, *, note: str = "审阅认为是不同病例"
    ) -> Subject:
        """撤销临时/人工身份归并，使该身份根重新独立计数。"""

        case = self.get_case(case_id)
        subject = next((p for p in case.patients if p.id == subject_id), None)
        if subject is None:
            raise KeyError(f"找不到主体 {subject_id}")
        previous = subject.attributes.get("pre_merge_identity_status")
        subject.merged_into = None
        subject.identity_status = previous or (
            "anonymous_unresolved" if "不明身份" in subject.label else "case_scoped_anonymous"
        )
        for item in case.review_items:
            if item.target_type == "subject" and item.target_id == subject_id and item.status == "open":
                item.status = "resolved"
                item.resolved_at = self._now()
                item.reviewer = reviewer
                item.resolution_note = f"merge_rejected: {note}"
        self._audit(reviewer, "identity_merge_rejected", case_id, subject_id=subject_id, note=note)
        self._reanalyze(case_id, trigger="identity_merge_rejected")
        return subject

    # --------------------------------------------------------------- review

    def review_queue(self, case_id: str | None = None) -> list[ReviewItem]:
        if case_id:
            return [i for i in self.get_case(case_id).review_items if i.status == "open"]
        return [item for case in self.cases.values() for item in case.review_items if item.status == "open"]

    def relationship_evidence(self, case_id: str, relationship_id: str) -> dict[str, Any]:
        """返回关系、字符区间证据和来源记录，供医学审阅界面展示。"""

        case = self.get_case(case_id)
        relationship = next((r for r in case.relationships if r.id == relationship_id), None)
        if relationship is None:
            raise KeyError(f"找不到关系 {relationship_id}")
        mention_ids = set(relationship.evidence_mention_ids)
        mentions = [_dataclass_to_dict(m) for m in case.mentions if m.id in mention_ids]
        record_ids = set(relationship.record_ids) | {m["record_id"] for m in mentions}
        records = [_dataclass_to_dict(r) for r in case.records if r.id in record_ids]
        return {
            "relationship": _dataclass_to_dict(relationship),
            "mentions": mentions,
            "source_records": records,
            "provenance": self.provenance_graph(case_id),
        }

    def verify_mention(
        self,
        case_id: str,
        mention_id: str,
        reviewer: str,
        *,
        status: str = "confirmed",
        canonical: str | None = None,
        concept_id: str | None = None,
        note: str = "",
    ) -> Mention:
        case = self.get_case(case_id)
        mention = next((m for m in case.mentions if m.id == mention_id), None)
        if mention is None:
            raise KeyError(f"找不到证据 {mention_id}")
        if status not in {"confirmed", "rejected", "corrected"}:
            raise ValueError("status 必须是 confirmed/rejected/corrected")
        if status == "rejected":
            mention.verification_status = "rejected"
            mention.is_current = False
        else:
            mention.verification_status = "confirmed" if status == "confirmed" else "corrected"
            if canonical:
                mention.canonical = canonical
            if concept_id:
                mention.concept_id = concept_id
        self._resolve_target_review_items(case, "mention", mention_id, reviewer, note or status)
        self._audit(reviewer, "mention_verified", case_id, mention_id=mention_id, status=status)
        self._reanalyze(case_id, trigger="mention_verified")
        return mention

    def verify_relationship(
        self,
        case_id: str,
        relationship_id: str,
        reviewer: str,
        *,
        status: str = "confirmed",
        relationship_status: str | None = None,
        note: str = "",
    ) -> Relationship:
        case = self.get_case(case_id)
        relationship = next((r for r in case.relationships if r.id == relationship_id), None)
        if relationship is None:
            raise KeyError(f"找不到关系 {relationship_id}")
        if status == "rejected":
            relationship.verification_status = "rejected"
            relationship.is_current = False
        else:
            relationship.verification_status = "confirmed"
            if relationship_status:
                relationship.status = relationship_status
        self._resolve_target_review_items(case, "relationship", relationship_id, reviewer, note or status)
        self._audit(reviewer, "relationship_verified", case_id, relationship_id=relationship_id, status=status)
        self._reanalyze(case_id, trigger="relationship_verified")
        return relationship

    def _resolve_target_review_items(
        self, case: CaseAggregate, target_type: str, target_id: str, reviewer: str, note: str
    ) -> None:
        for item in case.review_items:
            if item.target_type == target_type and item.target_id == target_id and item.status == "open":
                item.status = "resolved"
                item.resolved_at = self._now()
                item.reviewer = reviewer
                item.resolution_note = note

    # ------------------------------------------------------------ correction

    def correct_record(
        self,
        case_id: str,
        record_id: str,
        *,
        reason: str,
        corrected_by: str,
        new_text: str | None = None,
        metadata_patch: dict[str, Any] | None = None,
    ) -> RecordCorrection:
        case = self.get_case(case_id)
        record = next((r for r in case.records if r.id == record_id), None)
        if record is None:
            raise KeyError(f"找不到原始记录 {record_id}")
        if new_text is None and not metadata_patch:
            raise ValueError("更正必须提供 new_text 或 metadata_patch")

        revision = sum(1 for c in case.corrections if c.record_id == record_id) + 1
        correction = RecordCorrection(
            id=self._new_id("COR"),
            record_id=record_id,
            case_id=case_id,
            reason=reason,
            corrected_by=corrected_by,
            corrected_at=self._now(),
            new_text=new_text,
            metadata_patch=metadata_patch or {},
            revision=revision,
        )
        case.corrections.append(correction)

        if metadata_patch:
            record.metadata.update(metadata_patch)
            for key in ("source", "channel", "author", "patient_key", "duplicate_of"):
                if key in metadata_patch:
                    setattr(record, key, metadata_patch[key])

        if new_text is not None:
            # 旧抽取结果保留用于历史判断，但不再作为当前事实。
            for mention in case.mentions:
                if mention.record_id == record_id and mention.is_current:
                    mention.is_current = False
            for relationship in case.relationships:
                if relationship.record_ids and record_id in relationship.record_ids and relationship.is_current:
                    relationship.is_current = False
            for lab in case.labs:
                if lab.record_id == record_id and lab.is_current:
                    lab.is_current = False

            extraction = extract_narrative(case_id, record_id, new_text, self.products, revision=revision)
            index_subject = self._ensure_index_subject(case, record.patient_key)
            local_subjects = {"index": index_subject.id}
            for mention in extraction.mentions:
                ref = mention.subject_id
                if ref and ref.startswith("other:"):
                    if ref not in local_subjects:
                        subject = Subject(
                            id=self._new_id("P-OTHER"),
                            case_id=case_id,
                            label="叙述中的不明身份第三方",
                            identity_status="anonymous_unresolved",
                            source_record_ids=[record_id],
                        )
                        case.patients.append(subject)
                        local_subjects[ref] = subject.id
                    mention.subject_id = local_subjects[ref]
                elif ref == "index":
                    mention.subject_id = index_subject.id
            for lab in extraction.labs:
                ref = lab.subject_id or "index"
                if ref.startswith("other:") and ref not in local_subjects:
                    subject = Subject(
                        id=self._new_id("P-OTHER"),
                        case_id=case_id,
                        label="叙述中的不明身份第三方",
                        identity_status="anonymous_unresolved",
                        source_record_ids=[record_id],
                    )
                    case.patients.append(subject)
                    local_subjects[ref] = subject.id
                lab.subject_id = local_subjects.get(ref, index_subject.id)
            case.mentions.extend(extraction.mentions)
            case.labs.extend(extraction.labs)
            for index, draft in enumerate(extraction.relations):
                subject_id = local_subjects.get(draft.subject_ref, index_subject.id)
                case.relationships.append(
                    Relationship(
                        id=f"REL:{record_id}:r{revision}:{index}",
                        case_id=case_id,
                        kind=draft.kind,
                        explanation=draft.explanation,
                        subject_id=subject_id,
                        drug_mention_id=draft.drug_id,
                        symptom_mention_id=draft.symptom_id,
                        evidence_mention_ids=draft.evidence_ids,
                        record_ids=[record_id],
                        status=draft.status,
                        confidence=draft.confidence,
                        verification_status="pending_review"
                        if draft.status in REVIEW_RELATION_STATUSES
                        else "extracted",
                        attributes=draft.attributes,
                    )
                )

        # 更正不能删除已签发请求；只生成必须重新确认的影响项。
        for follow_up in case.follow_ups:
            self._create_reconfirmation(
                case,
                target_type="follow_up",
                target_id=follow_up.id,
                correction_id=correction.id,
                reason=f"记录 {record_id} 已更正：{reason}。原随访请求内容保留，但其问题和优先级需重新确认。",
            )
        for statement in case.risk_statements:
            if statement.status != "superseded":
                statement.status = "needs_reconfirmation"
                item = self._create_reconfirmation(
                    case,
                    target_type="risk_statement",
                    target_id=statement.id,
                    correction_id=correction.id,
                    reason=f"记录 {record_id} 已更正：{reason}。风险说明的旧版本保留，结论需重新确认或调整。",
                )
                if item.id not in statement.pending_reconfirmation_ids:
                    statement.pending_reconfirmation_ids.append(item.id)

        case.updated_at = self._now()
        self._enrich_case_drug_context(case)
        self._reconcile_timeline_relationships(case)
        self._reanalyze(case_id, trigger="record_corrected", correction_id=correction.id)
        self._audit(
            corrected_by,
            "record_corrected",
            case_id,
            record_id=record_id,
            correction_id=correction.id,
            reason=reason,
        )
        return correction

    def _create_reconfirmation(
        self,
        case: CaseAggregate,
        *,
        target_type: str,
        target_id: str,
        correction_id: str,
        reason: str,
    ) -> ReconfirmationItem:
        existing = next(
            (
                item
                for item in case.reconfirmations
                if item.target_type == target_type
                and item.target_id == target_id
                and item.status == "pending"
                and item.correction_id == correction_id
            ),
            None,
        )
        if existing:
            return existing
        item = ReconfirmationItem(
            id=self._new_id("RECONF"),
            case_id=case.case_id,
            target_type=target_type,
            target_id=target_id,
            correction_id=correction_id,
            reason=reason,
            created_at=self._now(),
        )
        case.reconfirmations.append(item)
        return item

    def pending_reconfirmations(self, case_id: str | None = None) -> list[ReconfirmationItem]:
        if case_id:
            return [i for i in self.get_case(case_id).reconfirmations if i.status == "pending"]
        return [i for case in self.cases.values() for i in case.reconfirmations if i.status == "pending"]

    def resolve_reconfirmation(
        self,
        item_id: str,
        reviewer: str,
        *,
        resolution: str = "reaffirmed",
        note: str = "",
    ) -> ReconfirmationItem:
        for case in self.cases.values():
            item = next((i for i in case.reconfirmations if i.id == item_id), None)
            if item:
                item.status = "resolved"
                item.resolved_at = self._now()
                item.resolved_by = reviewer
                item.resolution = resolution if not note else f"{resolution}: {note}"
                if item.target_type == "risk_statement":
                    statement = next((s for s in case.risk_statements if s.id == item.target_id), None)
                    if statement and statement.status == "needs_reconfirmation" and resolution == "reaffirmed":
                        statement.status = "active"
                    if statement and item.id in statement.pending_reconfirmation_ids:
                        statement.pending_reconfirmation_ids.remove(item.id)
                self._audit(reviewer, "reconfirmation_resolved", case.case_id, item_id=item_id, resolution=resolution)
                return item
        raise KeyError(f"找不到重新确认项 {item_id}")

    # ---------------------------------------------------------- follow-ups

    def issue_follow_up(
        self,
        case_id: str,
        *,
        title: str,
        description: str,
        issued_by: str,
        due_at: str | None = None,
        channel: str | None = None,
        question_codes: list[str] | None = None,
        evidence_ids: list[str] | None = None,
    ) -> FollowUpRequest:
        case = self.get_case(case_id)
        now = self._now()
        payload = {
            "title": title,
            "description": description,
            "due_at": due_at,
            "channel": channel,
            "question_codes": list(question_codes or []),
            "evidence_ids": list(evidence_ids or []),
        }
        request = FollowUpRequest(
            id=self._new_id("FU"),
            case_id=case_id,
            title=title,
            description=description,
            issued_by=issued_by,
            issued_at=now,
            due_at=due_at,
            channel=channel,
            question_codes=list(question_codes or []),
            evidence_ids=list(evidence_ids or []),
            original_snapshot=payload,
        )
        case.follow_ups.append(request)
        case.updated_at = now
        self._audit(issued_by, "follow_up_issued", case_id, follow_up_id=request.id, title=title)
        return request

    def append_follow_up_response(
        self, case_id: str, follow_up_id: str, response: dict[str, Any], recorded_by: str
    ) -> FollowUpRequest:
        case = self.get_case(case_id)
        request = next((f for f in case.follow_ups if f.id == follow_up_id), None)
        if request is None:
            raise KeyError(f"找不到随访请求 {follow_up_id}")
        entry = {"at": self._now(), "recorded_by": recorded_by, "response": response}
        request.response_entries.append(entry)
        if response.get("complete") is True:
            request.status = "completed"
        self._audit(recorded_by, "follow_up_response_appended", case_id, follow_up_id=follow_up_id)
        return request

    # ------------------------------------------------------ risk statements

    def create_risk_statement(
        self,
        case_id: str,
        text: str,
        *,
        changed_by: str,
        product_id: str | None = None,
        product_name: str | None = None,
        symptom_concept: str | None = None,
        severity_rank: int = 3,
        reason: str = "初始风险说明",
        status: str = "active",
    ) -> RiskStatement:
        case = self.get_case(case_id)
        name = product_name
        if not name and product_id:
            product = next((p for p in self.products if p.id == product_id), None)
            name = product.name if product else product_id
        statement = RiskStatement(
            id=self._new_id("RS"),
            case_id=case_id,
            product_id=product_id,
            product_name=name or "待核药品",
            symptom_concept=symptom_concept,
            current_version=1,
            status=status,
        )
        statement.versions.append(
            RiskStatementVersion(
                version=1,
                text=text,
                severity_rank=severity_rank,
                changed_by=changed_by,
                changed_at=self._now(),
                reason=reason,
            )
        )
        case.risk_statements.append(statement)
        self._audit(changed_by, "risk_statement_created", case_id, risk_statement_id=statement.id)
        return statement

    def adjust_risk_statement(
        self,
        statement_id: str,
        new_text: str,
        *,
        changed_by: str,
        reason: str,
        severity_rank: int | None = None,
        decision_id: str | None = None,
        reconfirmation_id: str | None = None,
    ) -> RiskStatement:
        for case in self.cases.values():
            statement = next((s for s in case.risk_statements if s.id == statement_id), None)
            if statement:
                current = statement.versions[-1]
                version = statement.current_version + 1
                statement.versions.append(
                    RiskStatementVersion(
                        version=version,
                        text=new_text,
                        severity_rank=current.severity_rank if severity_rank is None else severity_rank,
                        changed_by=changed_by,
                        changed_at=self._now(),
                        reason=reason,
                        decision_id=decision_id,
                    )
                )
                statement.current_version = version
                statement.status = "active"
                if reconfirmation_id:
                    self.resolve_reconfirmation(
                        reconfirmation_id,
                        changed_by,
                        resolution="adjusted",
                        note="已发布新版本风险说明",
                    )
                self._audit(
                    changed_by,
                    "risk_statement_adjusted",
                    case.case_id,
                    risk_statement_id=statement_id,
                    version=version,
                    decision_id=decision_id,
                )
                return statement
        raise KeyError(f"找不到风险说明 {statement_id}")

    def record_signal_decision(
        self,
        case_id: str,
        decision: str,
        decided_by: str,
        *,
        role: str = "safety_signal_lead",
        rationale: str = "",
        follow_up: dict[str, Any] | None = None,
        risk_statement_id: str | None = None,
        new_risk_text: str | None = None,
        risk_severity_rank: int | None = None,
        reconfirmation_id: str | None = None,
    ) -> SignalDecision:
        allowed = {"continue_follow_up", "request_lab", "adjust_risk_statement", "monitor", "close_or_archive"}
        if decision not in allowed:
            raise ValueError(f"decision 必须是 {sorted(allowed)}")
        case = self.get_case(case_id)
        signal_decision = SignalDecision(
            id=self._new_id("DEC"),
            case_id=case_id,
            decision=decision,
            decided_by=decided_by,
            decided_at=self._now(),
            role=role,
            rationale=rationale,
            risk_statement_id=risk_statement_id,
            reconfirmation_id=reconfirmation_id,
        )

        if decision == "continue_follow_up":
            params = follow_up or {
                "title": "继续追访并核实待核项",
                "description": "请核实用药身份、具体时间顺序、剂量变化、实验室结果及是否存在重复渠道报告。",
            }
            request = self.issue_follow_up(case_id, issued_by=decided_by, **params)
            signal_decision.follow_up_id = request.id

        if decision == "adjust_risk_statement":
            if not risk_statement_id and not new_risk_text:
                raise ValueError("调整风险说明需要 risk_statement_id 或新的 new_risk_text")
            if risk_statement_id:
                if new_risk_text:
                    statement = self.adjust_risk_statement(
                        risk_statement_id,
                        new_risk_text,
                        changed_by=decided_by,
                        reason=rationale or "安全信号负责人决定调整",
                        severity_rank=risk_severity_rank,
                        reconfirmation_id=reconfirmation_id,
                    )
                    signal_decision.risk_statement_id = statement.id
                else:
                    signal_decision.risk_statement_id = risk_statement_id
                    if reconfirmation_id:
                        self.resolve_reconfirmation(
                            reconfirmation_id, decided_by, resolution="scheduled_adjustment", note=rationale
                        )
            else:
                statement = self.create_risk_statement(
                    case_id,
                    new_risk_text or "",
                    changed_by=decided_by,
                    severity_rank=risk_severity_rank or 3,
                    reason=rationale or "安全信号负责人决定新建风险说明",
                )
                signal_decision.risk_statement_id = statement.id

        if reconfirmation_id and decision != "adjust_risk_statement":
            self.resolve_reconfirmation(
                reconfirmation_id, decided_by, resolution=decision, note=rationale
            )

        case.decisions.append(signal_decision)
        self._audit(decided_by, "signal_decision_recorded", case_id, decision_id=signal_decision.id, decision=decision)
        return signal_decision

    # ------------------------------------------------------------- analysis

    def _enrich_case_drug_context(self, case: CaseAggregate) -> None:
        """对未重述药名的记录，尝试关联同病例唯一目录药品，但保留待确认状态。"""

        mention_by_id = {m.id: m for m in case.mentions}
        for relationship in case.relationships:
            if not relationship.is_current:
                continue
            drug = mention_by_id.get(relationship.drug_mention_id or "")
            if drug is not None and drug.concept_id:
                continue

            candidates = [
                m
                for m in case.mentions
                if m.is_current
                and m.kind == "drug"
                and m.concept_id
                and m.subject_id == relationship.subject_id
            ]
            product_ids = {m.concept_id for m in candidates}
            if len(product_ids) != 1 or not candidates:
                continue
            inferred_drug = candidates[0]
            relationship.attributes["case_inferred_drug_mention_id"] = inferred_drug.id
            relationship.attributes["case_inferred_product_id"] = inferred_drug.concept_id
            relationship.attributes["context_inference"] = "same_subject_unique_catalog_product"

            if relationship.kind == "drug_symptom_link" and (
                relationship.drug_mention_id is None or relationship.status == "pending_drug"
            ):
                relationship.drug_mention_id = inferred_drug.id
                relationship.status = "pending_drug_identity"
                relationship.verification_status = "pending_review"
                suffix = "病例上下文中该主体仅有一个目录药品，已临时关联，但该记录未明确重述药名，仍需审阅确认。"
                if suffix not in relationship.explanation:
                    relationship.explanation += suffix
                if inferred_drug.id not in relationship.evidence_mention_ids:
                    relationship.evidence_mention_ids.append(inferred_drug.id)
            elif relationship.kind in {"withdrawal_to_improvement", "dose_reduction_precedes_withdrawal"}:
                relationship.drug_mention_id = inferred_drug.id
                if inferred_drug.id not in relationship.evidence_mention_ids:
                    relationship.evidence_mention_ids.append(inferred_drug.id)

    def _reconcile_timeline_relationships(self, case: CaseAggregate) -> None:
        """跨记录融合时间线，防止把分散叙述压缩成单一“停药后好转”标签。"""

        mentions = {m.id: m for m in case.mentions if m.is_current}
        dose_events = [m for m in mentions.values() if m.kind == "dose_change" and m.polarity != "negated"]
        withdrawal_events = [m for m in mentions.values() if m.kind == "withdrawal" and m.polarity != "negated"]

        def order(mention: Mention) -> float | None:
            return self._time_order(case, mention.time_id)

        def product_for(mention: Mention) -> str | None:
            drug_id = mention.attributes.get("drug_mention_id")
            drug = mentions.get(drug_id or "")
            return drug.concept_id if drug and drug.concept_id else None

        prior_pairs: list[tuple[Mention, Mention, str]] = []
        for dose in dose_events:
            for withdrawal in withdrawal_events:
                if dose.subject_id != withdrawal.subject_id:
                    continue
                dose_product, withdrawal_product = product_for(dose), product_for(withdrawal)
                if dose_product and withdrawal_product and dose_product != withdrawal_product:
                    continue
                d_order, w_order = order(dose), order(withdrawal)
                explicit_before = False
                if dose.record_id == withdrawal.record_id:
                    record = next((r for r in case.records if r.id == dose.record_id), None)
                    context = record.text if record else ""
                    explicit_before = "停药前" in context or "停用前" in context
                before = False
                reason = ""
                if d_order is not None and w_order is not None and d_order < w_order:
                    before, reason = True, "time_order"
                elif dose.span.start < withdrawal.span.start and dose.record_id == withdrawal.record_id:
                    before, reason = True, "narrative_order"
                elif explicit_before:
                    before, reason = True, "explicit_before_marker"
                if before:
                    prior_pairs.append((dose, withdrawal, reason))

        for dose, withdrawal, reason in prior_pairs:
            existing = next(
                (
                    r
                    for r in case.relationships
                    if r.is_current
                    and r.kind == "dose_reduction_precedes_withdrawal"
                    and dose.id in r.evidence_mention_ids
                    and withdrawal.id in r.evidence_mention_ids
                ),
                None,
            )
            if existing is None:
                record_ids = sorted({dose.record_id, withdrawal.record_id})
                relationship = Relationship(
                    id=f"REL:CROSS:DW:{dose.id}:{withdrawal.id}",
                    case_id=case.case_id,
                    kind="dose_reduction_precedes_withdrawal",
                    explanation=(
                        "跨记录时间线显示停药前已有剂量变化；不能把后续变化简单解释为仅由停药导致。"
                    ),
                    subject_id=dose.subject_id,
                    drug_mention_id=withdrawal.attributes.get("drug_mention_id")
                    or dose.attributes.get("drug_mention_id"),
                    evidence_mention_ids=[
                        i for i in [dose.id, withdrawal.id, dose.time_id, withdrawal.time_id] if i
                    ],
                    record_ids=record_ids,
                    status="supported",
                    confidence=0.82 if reason != "narrative_order" else 0.6,
                    verification_status="pending_review",
                    attributes={"cross_record": dose.record_id != withdrawal.record_id, "order_basis": reason},
                )
                case.relationships.append(relationship)
                self._add_review_item(
                    case,
                    target_type="relationship",
                    target_id=relationship.id,
                    label="减量早于停药（跨记录时间线）",
                    reason=relationship.explanation,
                )

        def equivalent_withdrawal(left: Mention | None, right: Mention | None) -> bool:
            if left is None or right is None or left.subject_id != right.subject_id:
                return False
            if left.id == right.id:
                return True
            left_product, right_product = product_for(left), product_for(right)
            if left_product and right_product and left_product != right_product:
                return False
            left_order, right_order = order(left), order(right)
            if left_order is not None and right_order is not None and left_order != right_order:
                return False
            # 同一病例主体、同一药品语境且无冲突日期的“停药”叙述，按同一停药事件处理。
            return True

        # 用最新跨记录证据重新评估“停药后改善”。旧判断已在 JudgmentVersion 快照中保留。
        for relationship in case.relationships:
            if not relationship.is_current or relationship.kind != "withdrawal_to_improvement":
                continue
            withdrawal_id = next(
                (eid for eid in relationship.evidence_mention_ids if mentions.get(eid) and mentions[eid].kind == "withdrawal"),
                None,
            )
            withdrawal = mentions.get(withdrawal_id or "")
            confounding_pair = None
            if withdrawal is not None:
                confounding_pair = next(
                    (
                        (dose, w, reason)
                        for dose, w, reason in prior_pairs
                        if dose.subject_id == relationship.subject_id
                        and equivalent_withdrawal(w, withdrawal)
                    ),
                    None,
                )
            if confounding_pair is None and withdrawal is not None:
                # 也接受已经由单记录规则生成的减量先于停药关系。
                dose_relation = next(
                    (
                        r
                        for r in case.relationships
                        if r.is_current
                        and r.kind == "dose_reduction_precedes_withdrawal"
                        and withdrawal.id in r.evidence_mention_ids
                        and r.subject_id == relationship.subject_id
                    ),
                    None,
                )
                if dose_relation:
                    relationship.attributes.setdefault("dose_relation_id", dose_relation.id)
                    confounding_pair = ("relation", withdrawal, "existing_relation")

            if confounding_pair is not None:
                if relationship.status != "confounded_by_prior_dose_reduction":
                    relationship.attributes["pre_reconciliation_status"] = relationship.status
                    relationship.attributes["pre_reconciliation_explanation"] = relationship.explanation
                relationship.status = "confounded_by_prior_dose_reduction"
                relationship.verification_status = "pending_review"
                relationship.attributes["cross_record_reconciled"] = True
                dose_relation_id = relationship.attributes.get("dose_relation_id")
                if dose_relation_id and dose_relation_id not in relationship.evidence_mention_ids:
                    relationship.evidence_mention_ids.append(dose_relation_id)
                suffix = "安全分析必须同时呈现减量、停药和改善，不能压缩成单一“停药后好转”标签。"
                if suffix not in relationship.explanation:
                    relationship.explanation += suffix
                self._add_review_item(
                    case,
                    target_type="relationship",
                    target_id=relationship.id,
                    label="停药后改善存在既往减量混杂",
                    reason=relationship.explanation,
                )
            elif relationship.attributes.get("cross_record_reconciled"):
                relationship.status = relationship.attributes.get(
                    "pre_reconciliation_status", "positive_dechallenge_candidate"
                )
                original_explanation = relationship.attributes.get("pre_reconciliation_explanation")
                if original_explanation:
                    relationship.explanation = original_explanation
                relationship.attributes["cross_record_reconciled"] = False
                relationship.verification_status = "extracted"

    def _record_order(self, case: CaseAggregate, record_id: str) -> int:
        return next((i for i, r in enumerate(case.records) if r.id == record_id), 999)

    def _time_order(self, case: CaseAggregate, mention_id: str | None) -> float | None:
        if not mention_id:
            return None
        mention = next((m for m in case.mentions if m.id == mention_id), None)
        if not mention:
            return None
        value = mention.attributes.get("order_key")
        return float(value) if isinstance(value, (int, float)) else None

    def _build_timeline(self, case: CaseAggregate) -> list[TimelineEvent]:
        events: list[TimelineEvent] = []
        event_kinds = {"exposure", "dose_change", "withdrawal", "symptom", "outcome"}
        current_mentions = [m for m in case.mentions if m.is_current and m.kind in event_kinds]
        for mention in current_mentions:
            order = self._time_order(case, mention.time_id)
            time_mention = next((m for m in case.mentions if m.id == mention.time_id), None)
            title = self._timeline_title(mention)
            detail = None
            if mention.kind == "symptom":
                polarity = "否定" if mention.polarity == "negated" else "阳性"
                change = mention.attributes.get("change")
                detail = f"{polarity}；变化={change or '未说明'}"
            if mention.kind == "dose_change":
                detail = "自行调整" if mention.attributes.get("self_adjusted") else "剂量变化"
            events.append(
                TimelineEvent(
                    order_key=order,
                    title=title,
                    event_type=mention.kind,
                    record_id=mention.record_id,
                    mention_id=mention.id,
                    time_text=time_mention.text if time_mention else None,
                    time_status=(time_mention.verification_status if time_mention else "pending_time"),
                    subject_id=mention.subject_id,
                    detail=detail,
                )
            )

        for lab in case.labs:
            if not lab.is_current:
                continue
            order = self._time_order(case, lab.observed_time_id)
            events.append(
                TimelineEvent(
                    order_key=order,
                    title=f"实验室结果：{lab.test_name}（{lab.severity_label}）",
                    event_type="lab_result",
                    record_id=lab.record_id,
                    mention_id=lab.id,
                    time_text=lab.observed_time_text,
                    time_status="resolved" if order is not None else "pending_time",
                    detail=f"{lab.flag or ''} {lab.value or ''}{lab.unit or ''}".strip(),
                )
            )

        mention_by_id = {m.id: m for m in case.mentions}

        def sort_key(event: TimelineEvent) -> tuple[float, int, int, str]:
            record_order = self._record_order(case, event.record_id)
            mention = mention_by_id.get(event.mention_id or "")
            span_start = mention.span.start if mention is not None and hasattr(mention, "span") else 0
            if event.order_key is None:
                return (9000 + record_order, 0, span_start, event.mention_id or "")
            return (event.order_key, record_order, span_start, event.mention_id or "")

        return sorted(events, key=sort_key)

    @staticmethod
    def _timeline_title(mention: Mention) -> str:
        if mention.kind == "symptom":
            prefix = "未出现/否定：" if mention.polarity == "negated" else ""
            return f"{prefix}{mention.canonical or mention.text}"
        return {
            "exposure": f"用药/暴露：{mention.text}",
            "dose_change": f"{'否定/未发生：' if mention.polarity == 'negated' else ''}剂量变化：{mention.text}",
            "withdrawal": f"{'否定/未发生：' if mention.polarity == 'negated' else ''}停药：{mention.text}",
            "outcome": f"临床变化：{mention.canonical or mention.text}",
        }.get(mention.kind, mention.text)

    def identity_root(self, subject: Subject, _seen: set[str] | None = None) -> str:
        seen = _seen or set()
        if subject.id in seen:
            return subject.id
        seen.add(subject.id)
        if subject.merged_into:
            target = self._find_subject_by_id(subject.merged_into)
            if target:
                return self.identity_root(target, seen)
            return subject.merged_into
        if subject.patient_key:
            return f"key:{subject.patient_key}"
        return subject.id

    def _identity_counts(self, case: CaseAggregate) -> dict[str, int]:
        roots: dict[str, Subject] = {}
        for subject in case.patients:
            root = self.identity_root(subject)
            target = self._find_subject_by_id(root) or subject
            roots[root] = target

        independent = provisional = pending_identity = confirmed = 0
        for root_id, target in roots.items():
            if target.identity_status == "anonymous_unresolved":
                pending_identity += 1
                continue
            if target.identity_status == "duplicate":
                continue
            independent += 1
            if target.identity_status == "confirmed":
                confirmed += 1
            else:
                provisional += 1
        return {
            "independent_case_count": independent,
            "provisional_case_count": provisional,
            "reviewer_confirmed_case_count": confirmed,
            "pending_identity_report_count": pending_identity,
        }

    def _active_relationships(self, case: CaseAggregate) -> list[Relationship]:
        mentions = {m.id: m for m in case.mentions}
        results = []
        for relationship in case.relationships:
            if not relationship.is_current or relationship.verification_status == "rejected":
                continue
            invalid_evidence = False
            for mention_id in (relationship.symptom_mention_id, relationship.drug_mention_id):
                mention = mentions.get(mention_id or "")
                if mention and (not mention.is_current or mention.verification_status == "rejected"):
                    invalid_evidence = True
                    break
            if not invalid_evidence:
                results.append(relationship)
        return results

    def _reanalyze(
        self, case_id: str, *, trigger: str, correction_id: str | None = None
    ) -> JudgmentVersion:
        case = self.get_case(case_id)
        timeline = self._build_timeline(case)
        symptoms = [
            m for m in case.mentions
            if m.is_current and m.kind == "symptom" and m.verification_status != "rejected"
        ]
        positive = [m for m in symptoms if m.polarity == "positive"]
        negated = [m for m in symptoms if m.polarity == "negated"]
        current_labs = [l for l in case.labs if l.is_current]
        current_relationships = self._active_relationships(case)

        symptom_severity = max((int(m.attributes.get("severity_rank") or 0) for m in positive), default=0)
        lab_severity = max((lab.severity_rank for lab in current_labs), default=0)
        severity_rank = max(symptom_severity, lab_severity)
        severity_reasons = []
        if symptom_severity:
            severity_reasons.append(f"当前阳性症状最高分级 {symptom_severity}")
        if lab_severity:
            severity_reasons.append(f"实验室结果将严重程度判断提升/支持至 {lab_severity}")
        if not severity_reasons:
            severity_reasons.append("尚无阳性症状或异常实验室结果支持分级")

        pending: list[dict[str, Any]] = []
        for mention in case.mentions:
            if not mention.is_current:
                continue
            if mention.verification_status == "pending_verification" or (
                mention.kind == "time" and mention.verification_status.startswith("pending")
            ):
                pending.append(
                    {
                        "type": "mention",
                        "id": mention.id,
                        "kind": mention.kind,
                        "text": mention.text,
                        "reason": mention.identity_status if mention.kind == "drug" else mention.verification_status,
                    }
                )
        for relationship in current_relationships:
            if relationship.status in PENDING_RELATION_STATUSES:
                pending.append(
                    {
                        "type": "relationship",
                        "id": relationship.id,
                        "kind": relationship.kind,
                        "status": relationship.status,
                        "reason": relationship.explanation,
                    }
                )
            elif relationship.status == TIME_PRECISION_PENDING_STATUS:
                pending.append(
                    {
                        "type": "time_precision",
                        "id": relationship.id,
                        "kind": relationship.kind,
                        "status": relationship.status,
                        "reason": relationship.explanation,
                    }
                )
            if relationship.status == "confounded_by_prior_dose_reduction":
                pending.append(
                    {
                        "type": "temporal_confounding",
                        "id": relationship.id,
                        "kind": relationship.kind,
                        "status": relationship.status,
                        "reason": relationship.explanation,
                    }
                )
        for subject in case.patients:
            if subject.identity_status == "anonymous_unresolved":
                pending.append(
                    {
                        "type": "subject_identity",
                        "id": subject.id,
                        "kind": "subject",
                        "text": subject.label,
                        "reason": "叙述对象身份不明，需核实是否为重复或独立病例。",
                    }
                )

        rationale: list[str] = []
        for event in timeline:
            if event.event_type in {"dose_change", "withdrawal"}:
                rationale.append(f"时间线包含 {event.title}（来源 {event.record_id}，时间：{event.time_text or '待核'}）。")
        for relationship in current_relationships:
            if relationship.kind in {
                "dose_reduction_precedes_withdrawal",
                "withdrawal_to_improvement",
                "drug_symptom_link",
            }:
                rationale.append(relationship.explanation)
        for finding in negated:
            rationale.append(
                f"“{finding.text}”为否定/未诉表现（线索：{finding.attributes.get('negation_marker')}），不计入风险升级。"
            )
        rationale.extend(severity_reasons)

        has_pending_core = any(
            r.kind == "drug_symptom_link" and r.status in PENDING_RELATION_STATUSES
            for r in current_relationships
        )
        has_resolved_core = any(
            r.kind == "drug_symptom_link"
            and r.status not in PENDING_RELATION_STATUSES
            and r.verification_status != "rejected"
            for r in current_relationships
        )
        has_confounding = any(
            r.status == "confounded_by_prior_dose_reduction" for r in current_relationships
        )
        has_improvement_link = any(r.kind == "withdrawal_to_improvement" for r in current_relationships)
        if not positive and not current_labs and not has_improvement_link:
            signal_status = "no_positive_finding"
            summary = "当前没有可计入的阳性症状或实验室异常；否定词项未被当作风险证据。"
        elif has_confounding and has_resolved_core:
            signal_status = "causal_ambiguity_needs_review"
            summary = "时间线显示减量、停药和改善交错，不能压缩为单一“停药后好转”因果标签，需医学审阅；待核身份/时间项已单列。"
        elif has_pending_core and not has_resolved_core:
            signal_status = "needs_verification"
            summary = "已发现用药-不良事件线索，但药物或时间身份存在待核项，暂不能形成确定信号。"
        elif has_pending_core:
            signal_status = "candidate_with_pending_verification"
            summary = "已有可支持的用药-不良事件关系，但仍有药物或时间精度待核项；可先补证并控制结论强度。"
        elif has_confounding:
            signal_status = "causal_ambiguity_needs_review"
            summary = "时间线显示减量、停药和改善交错，不能压缩为单一“停药后好转”因果标签，需医学审阅。"
        elif severity_rank >= 4:
            signal_status = "priority_signal_candidate"
            summary = "存在严重阳性表现或实验室异常，且核心关系有证据支持，建议安全信号负责人优先评估。"
        else:
            signal_status = "possible_signal_candidate"
            summary = "存在用药后不良事件候选关系，建议继续补证并按流程复核。"

        counts = self._identity_counts(case)
        version = JudgmentVersion(
            version=len(case.judgments) + 1,
            case_id=case_id,
            created_at=self._now(),
            trigger=trigger,
            severity_rank=severity_rank,
            severity_label=_severity_label(severity_rank),
            signal_status=signal_status,
            summary=summary,
            rationale=list(dict.fromkeys(rationale)),
            timeline=timeline,
            relationship_ids=[r.id for r in current_relationships],
            relationship_snapshots=[_dataclass_to_dict(r) for r in current_relationships],
            positive_symptoms=[self._symptom_brief(m) for m in positive],
            negated_findings=[self._symptom_brief(m) for m in negated],
            pending_items=pending,
            independent_case_count=counts["independent_case_count"],
            provisional_case_count=counts["provisional_case_count"],
            pending_identity_report_count=counts["pending_identity_report_count"],
            based_on_correction_id=correction_id,
            evidence_ids=[m.id for m in case.mentions if m.is_current],
            lab_ids=[l.id for l in current_labs],
        )
        case.judgments.append(version)
        case.updated_at = version.created_at
        return version

    @staticmethod
    def _symptom_brief(mention: Mention) -> dict[str, Any]:
        return {
            "mention_id": mention.id,
            "record_id": mention.record_id,
            "text": mention.text,
            "canonical": mention.canonical,
            "concept_id": mention.concept_id,
            "polarity": mention.polarity,
            "certainty": mention.certainty,
            "severity_rank": mention.attributes.get("severity_rank"),
            "change": mention.attributes.get("change"),
            "time_id": mention.time_id,
            "verification_status": mention.verification_status,
        }

    def explain_judgment(self, case_id: str, version: int | None = None) -> dict[str, Any]:
        case = self.get_case(case_id)
        if not case.judgments:
            raise KeyError(f"病例 {case_id} 尚无判断")
        judgment = case.judgments[-1] if version is None else next(
            (j for j in case.judgments if j.version == version), None
        )
        if judgment is None:
            raise KeyError(f"找不到判断版本 v{version}")
        data = _dataclass_to_dict(judgment)
        data["evidence"] = [
            _dataclass_to_dict(m)
            for m in case.mentions
            if m.id in set(judgment.evidence_ids) or any(e in judgment.evidence_ids for e in [m.id])
        ]
        data["relationships"] = judgment.relationship_snapshots or [
            _dataclass_to_dict(r) for r in case.relationships if r.id in set(judgment.relationship_ids)
        ]
        data["judgment_history_versions"] = [
            {"version": j.version, "created_at": j.created_at, "trigger": j.trigger,
             "severity_rank": j.severity_rank, "signal_status": j.signal_status,
             "based_on_correction_id": j.based_on_correction_id}
            for j in case.judgments
        ]
        data["open_reconfirmations"] = [
            _dataclass_to_dict(i) for i in case.reconfirmations if i.status == "pending"
        ]
        return data

    def audit_trail(self, case_id: str | None = None) -> list[dict[str, Any]]:
        events = self.audit_events if case_id is None else [e for e in self.audit_events if e.case_id == case_id]
        return [_dataclass_to_dict(event) for event in events]

    def case_history(self, case_id: str) -> dict[str, Any]:
        case = self.get_case(case_id)
        return {
            "case_id": case_id,
            "corrections": [_dataclass_to_dict(c) for c in case.corrections],
            "judgment_versions": [
                {
                    "version": j.version,
                    "created_at": j.created_at,
                    "trigger": j.trigger,
                    "severity_rank": j.severity_rank,
                    "signal_status": j.signal_status,
                    "summary": j.summary,
                    "based_on_correction_id": j.based_on_correction_id,
                }
                for j in case.judgments
            ],
            "decisions": [_dataclass_to_dict(d) for d in case.decisions],
            "reconfirmations": [_dataclass_to_dict(i) for i in case.reconfirmations],
            "audit_trail": self.audit_trail(case_id),
        }

    def case_report(self, case_id: str) -> dict[str, Any]:
        case = self.get_case(case_id)
        current = case.current_judgment()
        return {
            "case_id": case.case_id,
            "disclaimer": "合成演练工作流：输出仅用于流程复核，不用于临床决策。",
            "status": case.status,
            "merged_into_case_id": case.merged_into_case_id,
            "current_judgment": _dataclass_to_dict(current) if current else None,
            "patients": [_dataclass_to_dict(p) for p in case.patients],
            "records": [_dataclass_to_dict(r) for r in case.records],
            "corrections": [_dataclass_to_dict(c) for c in case.corrections],
            "mentions": [_dataclass_to_dict(m) for m in case.mentions],
            "relationships": [_dataclass_to_dict(r) for r in case.relationships],
            "labs": [_dataclass_to_dict(l) for l in case.labs],
            "review_queue": [_dataclass_to_dict(i) for i in case.review_items if i.status == "open"],
            "follow_ups": [_dataclass_to_dict(f) for f in case.follow_ups],
            "risk_statements": [_dataclass_to_dict(s) for s in case.risk_statements],
            "decisions": [_dataclass_to_dict(d) for d in case.decisions],
            "reconfirmations": [_dataclass_to_dict(i) for i in case.reconfirmations],
            "provenance": self.provenance_graph(case_id),
            "identity_counts": self._identity_counts(case),
        }

    # ----------------------------------------------------------- signal board

    def signal_board(self) -> dict[str, Any]:
        entries: dict[str, dict[str, Any]] = {}
        negated_count = 0

        for case in self.cases.values():
            if case.status == "duplicate":
                continue
            mention_by_id = {m.id: m for m in case.mentions}

            for relationship in case.relationships:
                if not relationship.is_current or relationship.verification_status == "rejected":
                    continue
                if relationship.kind == "negated_symptom":
                    negated_count += 1
                    continue
                if relationship.kind != "drug_symptom_link":
                    continue

                symptom = mention_by_id.get(relationship.symptom_mention_id or "")
                drug = mention_by_id.get(relationship.drug_mention_id or "")
                if not symptom or symptom.polarity == "negated":
                    continue
                if not symptom.is_current or symptom.verification_status == "rejected":
                    continue
                if drug and (not drug.is_current or drug.verification_status == "rejected"):
                    continue

                inferred_product_id = relationship.attributes.get("case_inferred_product_id")
                product_id = (drug.concept_id if drug and drug.concept_id else None) or inferred_product_id
                product_name = drug.canonical if drug and drug.concept_id else None
                if not product_name and product_id:
                    product = next((p for p in self.products if p.id == product_id), None)
                    product_name = product.name if product else product_id

                no_usable_drug = (
                    relationship.status == "pending_drug"
                    or (drug is None and not inferred_product_id)
                    or (drug is not None and not drug.concept_id and not inferred_product_id)
                )
                drug_identity_pending = (
                    relationship.status in {"pending_drug", "pending_drug_identity"}
                    or bool(relationship.attributes.get("context_inference"))
                    or (drug is not None and drug.identity_status in {"unresolved", "inferred_reference"})
                )
                product_key = "PENDING_DRUG" if no_usable_drug or not product_id else product_id
                symptom_key = symptom.concept_id or symptom.canonical or symptom.text
                key = f"{product_key}|{symptom_key}"

                entry = entries.setdefault(
                    key,
                    {
                        "key": key,
                        "product_id": None if product_key == "PENDING_DRUG" else product_id,
                        "product_name": "药物身份待核" if product_key == "PENDING_DRUG" else (product_name or product_id),
                        "symptom_concept": symptom.concept_id,
                        "symptom_name": symptom.canonical,
                        "independent_case_count": 0,
                        "provisional_case_count": 0,
                        "reviewer_confirmed_case_count": 0,
                        "pending_identity_report_count": 0,
                        "patient_roots": set(),
                        "pending_identity_roots": set(),
                        "case_ids": set(),
                        "record_ids": set(),
                        "relationship_ids": set(),
                        "max_severity_rank": 0,
                        "max_lab_severity_rank": 0,
                        "pending_time_relationship_count": 0,
                        "temporal_confounding_count": 0,
                        "confounded_relationship_ids": set(),
                        "drug_identity_pending": False,
                        "drug_identity_pending_relationship_count": 0,
                        "lab_only": False,
                    },
                )
                entry["drug_identity_pending"] = entry["drug_identity_pending"] or drug_identity_pending
                if drug_identity_pending:
                    entry["drug_identity_pending_relationship_count"] += 1

                subject = next((p for p in case.patients if p.id == relationship.subject_id), None)
                if subject is None:
                    continue
                root = self.identity_root(subject)
                root_subject = self._find_subject_by_id(root) or subject
                if root_subject.identity_status == "anonymous_unresolved":
                    entry["pending_identity_roots"].add(root)
                elif root_subject.identity_status != "duplicate":
                    entry["patient_roots"].add(root)
                entry["case_ids"].add(case.case_id)
                entry["record_ids"].update(relationship.record_ids)
                entry["relationship_ids"].add(relationship.id)
                entry["max_severity_rank"] = max(
                    entry["max_severity_rank"], int(symptom.attributes.get("severity_rank") or 0)
                )
                if relationship.status in {"pending_time", TIME_PRECISION_PENDING_STATUS}:
                    entry["pending_time_relationship_count"] += 1

            # 实验室严重程度按同一身份根并入信号优先级；若尚无药物-症状条目，也保留化验信号。
            for lab in case.labs:
                if not lab.is_current:
                    continue
                lab_subject = next((p for p in case.patients if p.id == lab.subject_id), None)
                if lab_subject is None:
                    lab_subject = next((p for p in case.patients if p.id == f"P:{case.case_id}:index"), None)
                lab_root = self.identity_root(lab_subject) if lab_subject else f"P:{case.case_id}:index"
                matching_entries = [
                    entry
                    for entry in entries.values()
                    if case.case_id in entry["case_ids"]
                    and (lab_root in entry["patient_roots"] or lab_root in entry["pending_identity_roots"])
                ]

                if not matching_entries:
                    drug_candidates = [
                        m for m in case.mentions
                        if m.is_current and m.kind == "drug" and m.concept_id
                        and (m.subject_id == (lab_subject.id if lab_subject else None))
                    ]
                    product_id = drug_candidates[0].concept_id if len({m.concept_id for m in drug_candidates}) == 1 else None
                    product_key = product_id or "PENDING_DRUG"
                    symptom_key = f"lab:{lab.test_concept or 'unspecified'}"
                    key = f"{product_key}|{symptom_key}"
                    entry = entries.setdefault(
                        key,
                        {
                            "key": key,
                            "product_id": product_id,
                            "product_name": drug_candidates[0].canonical if product_id else "药物身份待核",
                            "symptom_concept": symptom_key,
                            "symptom_name": f"实验室异常：{lab.test_name}",
                            "independent_case_count": 0,
                            "provisional_case_count": 0,
                            "reviewer_confirmed_case_count": 0,
                            "pending_identity_report_count": 0,
                            "patient_roots": set(),
                            "pending_identity_roots": set(),
                            "case_ids": set(),
                            "record_ids": set(),
                            "relationship_ids": set(),
                            "max_severity_rank": 0,
                            "max_lab_severity_rank": 0,
                            "pending_time_relationship_count": 0,
                            "temporal_confounding_count": 0,
                            "confounded_relationship_ids": set(),
                            "drug_identity_pending": product_id is None,
                            "drug_identity_pending_relationship_count": 0,
                            "lab_only": True,
                        },
                    )
                    entry["case_ids"].add(case.case_id)
                    if lab_subject and lab_subject.identity_status == "anonymous_unresolved":
                        entry["pending_identity_roots"].add(lab_root)
                    else:
                        entry["patient_roots"].add(lab_root)
                    matching_entries = [entry]

                for entry in matching_entries:
                    entry["max_lab_severity_rank"] = max(entry["max_lab_severity_rank"], lab.severity_rank)
                    entry["max_severity_rank"] = max(entry["max_severity_rank"], lab.severity_rank)
                    entry["record_ids"].add(lab.record_id)

            # 将减量先于停药导致的去激发混杂关联到同病例、同主体、同症状概念的信号。
            for relationship in case.relationships:
                if not relationship.is_current:
                    continue
                if relationship.status != "confounded_by_prior_dose_reduction":
                    continue
                target_symptom = mention_by_id.get(relationship.symptom_mention_id or "")
                target_concept = target_symptom.concept_id if target_symptom else None
                drug = mention_by_id.get(relationship.drug_mention_id or "")
                product_id = (drug.concept_id if drug and drug.concept_id else None) or relationship.attributes.get(
                    "case_inferred_product_id"
                )
                subject = next((p for p in case.patients if p.id == relationship.subject_id), None)
                root = self.identity_root(subject) if subject else None

                for entry in entries.values():
                    if case.case_id not in entry["case_ids"]:
                        continue
                    if root and root not in entry["patient_roots"] and root not in entry["pending_identity_roots"]:
                        continue
                    if product_id and entry["product_id"] and product_id != entry["product_id"]:
                        continue
                    if target_concept and entry["symptom_concept"] and target_concept != entry["symptom_concept"]:
                        continue
                    entry["confounded_relationship_ids"].add(relationship.id)

        serialized = []
        for entry in entries.values():
            item = dict(entry)
            item["independent_case_count"] = len(entry["patient_roots"])
            # provisional/confirmed 计数需要按根主体状态计算，不能在关系循环中重复递增。
            provisional = confirmed = 0
            for root in entry["patient_roots"]:
                subject = self._find_subject_by_id(root)
                if subject and subject.identity_status == "confirmed":
                    confirmed += 1
                else:
                    provisional += 1
            item["reviewer_confirmed_case_count"] = confirmed
            item["provisional_case_count"] = provisional
            item["pending_identity_report_count"] = len(entry["pending_identity_roots"])
            item["temporal_confounding_count"] = len(entry["confounded_relationship_ids"])
            for set_field in (
                "patient_roots",
                "pending_identity_roots",
                "case_ids",
                "record_ids",
                "relationship_ids",
                "confounded_relationship_ids",
            ):
                item[set_field] = sorted(item[set_field])
            item["severity_label"] = _severity_label(item["max_severity_rank"])
            serialized.append(item)
        serialized.sort(key=lambda x: (-x["max_severity_rank"], -x["independent_case_count"], x["key"]))

        open_reconf = self.pending_reconfirmations()
        return {
            "signals": serialized,
            "negated_finding_count": negated_count,
            "open_reconfirmation_count": len(open_reconf),
            "duplicate_case_count": sum(1 for c in self.cases.values() if c.status == "duplicate"),
            "disclaimer": "合成演练数据，不用于临床决策。",
            "principle": "否定词项不计升级；身份或时间不明单列待核；重复病例归并身份根后只计一次。",
        }


# 便于不同业务命名习惯调用的方法别名。
CaseWorkbench.receive_record = CaseWorkbench.ingest_record
CaseWorkbench.process_record = CaseWorkbench.ingest_record
CaseWorkbench.correct_case = CaseWorkbench.correct_record
CaseWorkbench.add_follow_up = CaseWorkbench.issue_follow_up
CaseWorkbench.add_risk_statement = CaseWorkbench.create_risk_statement
CaseWorkbench.decide_signal = CaseWorkbench.record_signal_decision
CaseWorkbench.get_judgment_history = CaseWorkbench.case_history
CaseWorkbench.get_review_graph = CaseWorkbench.review_graph
CaseWorkbench.find_duplicates = CaseWorkbench.find_duplicate_cases


def _severity_label(rank: int) -> str:
    return {0: "not_graded", 1: "mild", 2: "mild_to_moderate", 3: "moderate", 4: "severe", 5: "critical"}.get(rank, "not_graded")


def _dataclass_to_dict(obj: Any) -> Any:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj
