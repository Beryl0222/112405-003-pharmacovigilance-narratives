import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_workbench import CaseWorkbench  # noqa: E402
from case_workbench.nlp import extract_narrative  # noqa: E402

PRODUCTS = [{"id": "DR-2", "name": "宁安片", "ingredient": "合成成分甲"}]


class SeedNarrativeTest(unittest.TestCase):
    def setUp(self):
        self.wb = CaseWorkbench(products=PRODUCTS)
        self.case = self.wb.ingest_path(ROOT / "data" / "case.json")
        self.judgment = self.case.current_judgment()

    def test_materials_load_into_reviewable_graph(self):
        self.assertEqual(self.case.case_id, "pv-4")
        self.assertEqual([r.id for r in self.case.records], ["PV-4A", "PV-4B", "PV-4C"])
        self.assertTrue(any(m.kind == "drug" and m.concept_id == "DR-2" for m in self.case.mentions))
        self.assertTrue(any(m.kind == "symptom" and m.concept_id == "rash" for m in self.case.mentions))

    def test_timeline_keeps_reduction_before_stop_and_improvement_after_stop(self):
        events = self.judgment.timeline
        by_type = {}
        for event in events:
            by_type.setdefault(event.event_type, []).append(event)

        reduction = min(by_type["dose_change"], key=lambda e: e.order_key)
        stop_b = next(e for e in by_type["withdrawal"] if e.record_id == "PV-4B")
        improvement = next(e for e in by_type["outcome"] if e.record_id == "PV-4B")
        self.assertLess(reduction.order_key, stop_b.order_key)
        self.assertLessEqual(stop_b.order_key, improvement.order_key)
        self.assertEqual(reduction.time_text, "周一")
        self.assertEqual(stop_b.time_text, "周三")

        confounded = [
            r for r in self.case.relationships
            if r.status == "confounded_by_prior_dose_reduction"
        ]
        self.assertEqual(len(confounded), 1)
        self.assertIn("不能只保留", confounded[0].explanation)
        self.assertEqual(self.judgment.signal_status, "causal_ambiguity_needs_review")

    def test_negated_chest_tightness_is_retained_but_not_counted_as_positive(self):
        negated = self.judgment.negated_findings
        self.assertTrue(any(item["concept_id"] == "chest_tightness" for item in negated))
        self.assertFalse(any(item["concept_id"] == "chest_tightness" for item in self.judgment.positive_symptoms))
        self.assertLessEqual(self.judgment.severity_rank, 2)
        board = self.wb.signal_board()
        self.assertGreaterEqual(board["negated_finding_count"], 1)
        serialized = json.dumps(board, ensure_ascii=False)
        self.assertNotIn("dyspnea", serialized)

    def test_forwarded_someone_report_is_pending_identity_not_merged_into_index(self):
        identity = self.wb._identity_counts(self.case)
        self.assertEqual(identity["independent_case_count"], 1)
        self.assertEqual(identity["pending_identity_report_count"], 1)
        self.assertTrue(any(item["type"] == "subject_identity" for item in self.judgment.pending_items))
        signal = next(s for s in self.wb.signal_board()["signals"] if s["product_id"] == "DR-2")
        self.assertEqual(signal["independent_case_count"], 1)
        self.assertEqual(signal["pending_identity_report_count"], 1)
        self.assertGreaterEqual(signal["pending_time_relationship_count"], 1)


class NegationAndUncertaintyTest(unittest.TestCase):
    def test_negated_severe_symptom_does_not_escalate(self):
        wb = CaseWorkbench(products=PRODUCTS)
        case = wb.ingest_text(
            "患者服用宁安片后未出现呼吸困难，也没有胸闷。",
            case_id="neg-case",
            record_id="NEG-1",
        )
        judgment = case.current_judgment()
        concepts = {item["concept_id"] for item in judgment.negated_findings}
        self.assertIn("dyspnea", concepts)
        self.assertIn("chest_tightness", concepts)
        self.assertEqual(judgment.positive_symptoms, [])
        self.assertEqual(judgment.severity_rank, 0)
        self.assertEqual(judgment.signal_status, "no_positive_finding")

    def test_unknown_drug_and_time_are_explicit_pending(self):
        wb = CaseWorkbench()
        case = wb.ingest_text(
            "有人最近使用一种不明药物后出现呼吸困难，具体用药时间不详。",
            case_id="unknown-case",
            record_id="UNK-1",
        )
        judgment = case.current_judgment()
        pending_types = {(item["type"], item.get("status")) for item in judgment.pending_items}
        self.assertTrue(any(type_ == "mention" for type_, _ in pending_types))
        self.assertTrue(any(status in {"pending_drug_identity", "pending_time"} for _, status in pending_types))
        self.assertIn(judgment.signal_status, {"needs_verification", "candidate_with_pending_verification"})
        self.assertTrue(wb.review_queue("unknown-case"))

        board = wb.signal_board()
        self.assertTrue(any(s["product_id"] is None and s["symptom_concept"] == "dyspnea" for s in board["signals"]))


class IdentityAndQuoteTest(unittest.TestCase):
    def test_quote_does_not_create_second_independent_case(self):
        wb = CaseWorkbench(products=PRODUCTS)
        wb.ingest_record(
            {
                "id": "CALL-1",
                "case_id": "q-1",
                "text": "患者说：我周二吃宁安片后起皮疹，周三停药后好转。",
                "source": "patient-call",
                "patient_key": "anon-hash-1",
            }
        )
        note = wb.ingest_record(
            {
                "id": "NOTE-1",
                "case_id": "q-1",
                "text": "医生补记：患者原话“我周二吃宁安片后起皮疹，周三停药后好转。”",
                "source": "clinic-note",
                "patient_key": "anon-hash-1",
            }
        )
        self.assertIn("CALL-1", next(r for r in note.records if r.id == "NOTE-1").derived_from)
        independence = wb.source_independence("q-1")
        self.assertEqual(independence["independent_source_family_count"], 1)
        counts = wb._identity_counts(note)
        self.assertEqual(counts["independent_case_count"], 1)

    def test_same_anonymous_key_across_channels_counts_once_on_board(self):
        wb = CaseWorkbench(products=PRODUCTS)
        wb.ingest_record(
            {
                "id": "A-1",
                "case_id": "channel-a",
                "text": "患者周二服用宁安片后出现皮疹。",
                "patient_key": "same-anon-key",
            }
        )
        wb.ingest_record(
            {
                "id": "B-1",
                "case_id": "channel-b",
                "text": "患者周二服用宁安片后出现皮疹。",
                "patient_key": "same-anon-key",
                "channel": "follow-up-app",
            }
        )
        board = wb.signal_board()
        signal = next(s for s in board["signals"] if s["symptom_concept"] == "rash")
        self.assertEqual(signal["independent_case_count"], 1)
        self.assertEqual(set(signal["case_ids"]), {"channel-a", "channel-b"})

    def test_exact_cross_channel_text_is_provisionally_deduplicated(self):
        wb = CaseWorkbench(products=PRODUCTS)
        text = "患者周二服用宁安片后出现皮疹。"
        wb.ingest_record({"id": "W1", "case_id": "wechat", "text": text})
        wb.ingest_record({"id": "M1", "case_id": "phone", "text": text})
        signal = next(s for s in wb.signal_board()["signals"] if s["symptom_concept"] == "rash")
        self.assertEqual(signal["independent_case_count"], 1)
        self.assertTrue(wb.review_queue())
        merged_subject = next(
            p for case in wb.cases.values() for p in case.patients if p.identity_status == "provisional_duplicate"
        )
        wb.reject_identity_merge(merged_subject.case_id, merged_subject.id, "reviewer", note="来源特征不同")
        signal = next(s for s in wb.signal_board()["signals"] if s["symptom_concept"] == "rash")
        self.assertEqual(signal["independent_case_count"], 2)

    def test_manual_duplicate_case_marking_keeps_history_but_excludes_duplicate(self):
        wb = CaseWorkbench(products=PRODUCTS)
        wb.ingest_text("患者服用宁安片后出现皮疹。", case_id="first")
        wb.ingest_text("另一个渠道重复收到：患者服用宁安片后出现皮疹。", case_id="second")
        wb.mark_duplicate_case("second", "first", actor="reviewer")
        board = wb.signal_board()
        self.assertEqual(board["duplicate_case_count"], 1)
        signal = next(s for s in board["signals"] if s["symptom_concept"] == "rash")
        self.assertEqual(signal["independent_case_count"], 1)
        self.assertEqual(signal["case_ids"], ["first"])
        self.assertEqual(wb.get_case("second").status, "duplicate")


class LabCorrectionAndFollowUpTest(unittest.TestCase):
    def test_lab_upgrades_severity_but_issued_followup_remains_immutable(self):
        wb = CaseWorkbench(products=PRODUCTS)
        case = wb.ingest_text("患者服用宁安片后出现皮疹。", case_id="pv-lab", record_id="R1")
        follow_up = wb.issue_follow_up(
            "pv-lab",
            title="核实皮疹和停药时间",
            description="请回电核实皮疹范围、是否有胸闷/呼吸困难及具体停药时间。",
            issued_by="signal-lead",
            due_at="2026-11-27",
            question_codes=["timing", "severity"],
        )
        original_snapshot = dict(follow_up.original_snapshot)
        self.assertEqual(case.current_judgment().severity_rank, 2)

        wb.ingest_record(
            {
                "id": "LAB-1",
                "case_id": "pv-lab",
                "text": "实验室随访：血小板18×10^9/L，报告为危急值。",
                "source": "lab",
            }
        )
        case = wb.get_case("pv-lab")
        self.assertEqual(case.current_judgment().severity_rank, 5)
        self.assertEqual(case.current_judgment().severity_label, "critical")
        board_signal = next(s for s in wb.signal_board()["signals"] if s["symptom_concept"] == "rash")
        self.assertEqual(board_signal["max_severity_rank"], 5)
        self.assertEqual(board_signal["max_lab_severity_rank"], 5)
        self.assertIn("LAB-1", board_signal["record_ids"])
        stored = next(f for f in case.follow_ups if f.id == follow_up.id)
        self.assertEqual(stored.title, "核实皮疹和停药时间")
        self.assertEqual(stored.original_snapshot, original_snapshot)
        self.assertEqual(stored.status, "issued")

    def test_correction_preserves_judgment_history_and_creates_reconfirmation_items(self):
        wb = CaseWorkbench(products=PRODUCTS)
        case = wb.ingest_path(ROOT / "data" / "case.json")
        old_version = case.current_judgment().version
        follow_up = wb.issue_follow_up(
            "pv-4",
            title="继续追访时间线",
            description="核实减量、停药和皮疹变化的准确日期。",
            issued_by="lead",
        )
        statement = wb.create_risk_statement(
            "pv-4",
            "宁安片与皮疹相关，建议继续监测。",
            changed_by="lead",
            product_id="DR-2",
            symptom_concept="rash",
            severity_rank=2,
        )

        correction = wb.correct_record(
            "pv-4",
            "PV-4B",
            new_text="家属再次确认：患者停药前没有减量，是周三停药后皮疹逐渐消退。",
            reason="家属回电更正此前周一减量说法",
            corrected_by="reviewer-li",
        )
        case = wb.get_case("pv-4")
        self.assertEqual(case.current_judgment().version, old_version + 1)
        old_explanation = wb.explain_judgment("pv-4", version=old_version)
        self.assertTrue(
            any("混杂" in item for item in old_explanation["rationale"])
            or any(r["status"] == "confounded_by_prior_dose_reduction" for r in old_explanation["relationships"])
        )
        self.assertTrue(any(h["trigger"] == "record_ingested" for h in old_explanation["judgment_history_versions"]))

        current_relationship_statuses = [r.status for r in case.relationships if r.is_current]
        self.assertIn("positive_dechallenge_candidate", current_relationship_statuses)
        self.assertNotIn("confounded_by_prior_dose_reduction", current_relationship_statuses)

        pending = wb.pending_reconfirmations("pv-4")
        targets = {(item.target_type, item.target_id) for item in pending}
        self.assertIn(("follow_up", follow_up.id), targets)
        self.assertIn(("risk_statement", statement.id), targets)

        stored_follow_up = next(f for f in case.follow_ups if f.id == follow_up.id)
        self.assertEqual(stored_follow_up.title, "继续追访时间线")
        self.assertIn("核实减量", stored_follow_up.description)
        stored_statement = next(s for s in case.risk_statements if s.id == statement.id)
        self.assertEqual(stored_statement.status, "needs_reconfirmation")
        self.assertEqual(len(stored_statement.versions), 1)

        risk_item = next(i for i in pending if i.target_type == "risk_statement")
        decision = wb.record_signal_decision(
            "pv-4",
            "adjust_risk_statement",
            "signal-lead-wang",
            rationale="更正后时间线支持停药后改善，但仍需补核身份",
            risk_statement_id=statement.id,
            new_risk_text="宁安片与皮疹的时间关系值得监测；旧减量混杂已被更正。",
            reconfirmation_id=risk_item.id,
            risk_severity_rank=2,
        )
        self.assertTrue(decision.id)
        updated_statement = next(s for s in wb.get_case("pv-4").risk_statements if s.id == statement.id)
        self.assertEqual(len(updated_statement.versions), 2)
        self.assertEqual(updated_statement.versions[0].version, 1)
        self.assertNotEqual(updated_statement.versions[0].text, updated_statement.versions[-1].text)
        self.assertEqual(updated_statement.status, "active")

        follow_item = next(i for i in wb.pending_reconfirmations("pv-4") if i.target_type == "follow_up")
        wb.resolve_reconfirmation(follow_item.id, "lead", resolution="reaffirmed")
        self.assertEqual(wb.pending_reconfirmations("pv-4"), [])

    def test_state_roundtrip_preserves_history_and_models(self):
        wb = CaseWorkbench(products=PRODUCTS)
        wb.ingest_path(ROOT / "data" / "case.json")
        wb.issue_follow_up("pv-4", title="T", description="D", issued_by="lead")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            wb.save(path)
            loaded = CaseWorkbench.load(path)
            self.assertEqual(loaded.get_case("pv-4").current_judgment().version, 3)
            self.assertEqual(len(loaded.get_case("pv-4").follow_ups), 1)
            report = loaded.case_report("pv-4")
            self.assertTrue(report["timeline"] if "timeline" in report else report["current_judgment"]["timeline"])


class CrossRecordFusionTest(unittest.TestCase):
    def test_prior_reduction_in_second_record_reinterprets_improvement(self):
        wb = CaseWorkbench(products=PRODUCTS)
        wb.ingest_record(
            {"id": "R1", "case_id": "split", "text": "患者周三停用宁安片后皮疹逐渐消退。"}
        )
        first_version = wb.get_case("split").current_judgment().version
        wb.ingest_record(
            {"id": "R2", "case_id": "split", "text": "家属补充：患者周一已经自行减半剂量。"}
        )
        case = wb.get_case("split")
        self.assertTrue(
            any(r.kind == "dose_reduction_precedes_withdrawal" and r.record_ids == ["R1", "R2"]
                for r in case.relationships)
        )
        improvement = next(r for r in case.relationships if r.kind == "withdrawal_to_improvement" and r.is_current)
        self.assertEqual(improvement.status, "confounded_by_prior_dose_reduction")
        self.assertEqual(case.current_judgment().signal_status, "causal_ambiguity_needs_review")

        old = wb.explain_judgment("split", version=first_version)
        old_improvement = next(r for r in old["relationships"] if r["kind"] == "withdrawal_to_improvement")
        self.assertEqual(old_improvement["status"], "positive_dechallenge_candidate")
        current = wb.explain_judgment("split")
        self.assertTrue(any(r["status"] == "confounded_by_prior_dose_reduction" for r in current["relationships"]))

    def test_cross_case_quote_links_same_identity_for_signal_count(self):
        wb = CaseWorkbench(products=PRODUCTS)
        wb.ingest_record(
            {"id": "P-1", "case_id": "patient-channel", "text": "我吃宁安片后起皮疹。"}
        )
        wb.ingest_record(
            {
                "id": "D-1",
                "case_id": "doctor-channel",
                "text": "医生引用患者原话：“我吃宁安片后起皮疹。”",
            }
        )
        signal = next(s for s in wb.signal_board()["signals"] if s["symptom_concept"] == "rash")
        self.assertEqual(signal["independent_case_count"], 1)
        self.assertEqual(sorted(signal["case_ids"]), ["doctor-channel", "patient-channel"])


class RuleExtractorUnitTest(unittest.TestCase):
    def test_weekday_order_and_negation_span(self):
        extraction = extract_narrative(
            "c",
            "r",
            "患者周一减量，周二服药后起疹，周三停药，当晚仍有新疹，未诉呼吸困难。",
            products=[],
        )
        times = [m for m in extraction.mentions if m.kind == "time"]
        weekday_values = [m.attributes.get("weekday") for m in times if m.attributes.get("weekday")]
        self.assertIn(1, weekday_values)
        self.assertIn(2, weekday_values)
        self.assertIn(3, weekday_values)
        severe = next(m for m in extraction.mentions if m.concept_id == "dyspnea")
        self.assertEqual(severe.polarity, "negated")
        self.assertEqual(severe.attributes["severity_rank"], 0)


if __name__ == "__main__":
    unittest.main()
