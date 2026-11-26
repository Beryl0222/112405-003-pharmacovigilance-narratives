"""面向中文药物安全叙述的透明规则抽取器。

这里不依赖第三方 NLP 包。规则刻意保留字符区间和判定线索，
使医学审阅者可以从每个结论回到原文字句。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import LabResult, Mention, Product, Span


PART_OF_DAY = {
    "凌晨": 1,
    "早上": 2,
    "早晨": 2,
    "上午": 3,
    "中午": 4,
    "下午": 5,
    "傍晚": 6,
    "晚上": 7,
    "晚间": 7,
    "晚": 7,
    "夜间": 8,
    "夜里": 8,
}

WEEKDAY = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7, "天": 7}

# canonical, concept_id, base severity, aliases
SYMPTOMS: list[tuple[str, str, int, list[str]]] = [
    ("呼吸困难", "dyspnea", 4, ["呼吸困难", "喘不上气", "喘憋", "气短"]),
    ("喉头水肿", "laryngeal_edema", 5, ["喉头水肿", "喉咙水肿", "喉头紧"]),
    ("过敏性休克", "anaphylactic_shock", 5, ["过敏性休克", "休克"]),
    ("意识障碍", "consciousness_disorder", 5, ["意识障碍", "昏迷", "意识不清"]),
    ("晕厥", "syncope", 4, ["晕厥", "昏倒", "晕倒"]),
    ("低血压", "hypotension", 4, ["低血压", "血压下降"]),
    ("胸闷", "chest_tightness", 3, ["胸闷", "胸部憋闷"]),
    ("心悸", "palpitation", 3, ["心悸", "心慌"]),
    ("哮喘", "asthma", 3, ["哮喘发作", "哮喘"]),
    ("肝功能异常", "liver_injury", 3, ["肝功能异常", "肝损伤", "转氨酶升高", "转氨酶明显升高"]),
    ("血小板减少", "thrombocytopenia", 3, ["血小板减少", "血小板下降"]),
    ("出血", "bleeding", 4, ["消化道出血", "颅内出血", "出血", "瘀斑", "牙龈出血"]),
    ("皮疹", "rash", 2, ["皮疹", "新发皮疹", "新疹", "疹子", "红疹", "起疹", "出疹", "起了疹子"]),
    ("瘙痒", "pruritus", 1, ["皮肤瘙痒", "瘙痒", "发痒"]),
    ("红肿", "swelling_redness", 2, ["红肿", "肿胀"]),
    ("发热", "fever", 1, ["发热", "发烧", "高热", "高烧"]),
    ("寒战", "chills", 2, ["寒战", "发冷"]),
    ("恶心", "nausea", 1, ["恶心", "想吐"]),
    ("呕吐", "vomiting", 2, ["呕吐", "吐了"]),
    ("腹泻", "diarrhea", 1, ["腹泻", "拉肚子"]),
    ("腹痛", "abdominal_pain", 2, ["腹痛", "肚子痛"]),
    ("头痛", "headache", 1, ["头痛", "头疼"]),
    ("头晕", "dizziness", 1, ["头晕", "眩晕"]),
    ("嗜睡", "somnolence", 1, ["嗜睡", "犯困"]),
]

NEGATION_MARKERS = [
    "并没有出现",
    "没有出现",
    "未曾出现",
    "未出现",
    "未发生",
    "未观察到",
    "未发现",
    "未诉",
    "并无",
    "否认",
    "未见",
    "没有",
    "未",
    "无",
]

UNCERTAINTY_MARKERS = [
    "不能排除",
    "待核实",
    "尚未确认",
    "不确定",
    "疑似",
    "可能",
    "也许",
    "或许",
    "据称",
    "据说",
    "不详",
    "不明",
]

CHANGE_MARKERS: list[tuple[str, str]] = [
    ("completely_resolved", "完全消退"),
    ("completely_resolved", "完全消失"),
    ("resolved", "消退"),
    ("resolved", "消失"),
    ("resolved", "痊愈"),
    ("improving", "逐渐消退"),
    ("improving", "好转"),
    ("improving", "缓解"),
    ("improving", "减轻"),
    ("improving", "改善"),
    ("worsening", "加重"),
    ("worsening", "加剧"),
    ("worsening", "增多"),
    ("worsening", "扩散"),
    ("worsening", "恶化"),
    ("persistent", "仍有"),
    ("persistent", "仍然有"),
    ("persistent", "依旧有"),
    ("persistent", "持续"),
    ("persistent", "不退"),
    ("persistent", "未退"),
]

SUBJECT_OTHER_RE = re.compile(
    r"另(?:一|外一)?(?:名|位)?(?:患者|病人|人|用户|网友|群友)|有人|某人|一名(?:网友|群友|用户|陌生人)"
)
QUOTE_RE = re.compile(r"[“\"]([^”\"]{2,})[”\"]")
GENERIC_DRUG_PATTERNS = [
    re.compile(r"具体药名|药物名称不详|药名不详|不明药物|不详药物|何种药物|哪种药物"),
    re.compile(r"[A-Za-z0-9一-龥]{1,12}?(?:口服液|注射液|注射剂|胶囊|颗粒|软膏|乳膏|药丸|丸剂|片剂|片)"),
    re.compile(r"(?:该|此|这种|那个|上述)?药物"),
    re.compile(r"该药|此药"),
    re.compile(r"[一-龥]{2,8}(?<![停吃服用买卖开给换])药(?!名)"),
]
LEADING_DRUG_CHARS = "服用吃用给开买过了的在把将和与及或"

LAB_TESTS: list[tuple[str, list[str], str]] = [
    ("platelet", ["血小板计数", "血小板", "PLT", "plt"], "血小板"),
    ("troponin", ["肌钙蛋白I", "肌钙蛋白T", "肌钙蛋白", "cTnI", "cTnT"], "肌钙蛋白"),
    ("alt", ["丙氨酸氨基转移酶", "谷丙转氨酶", "ALT", "alt"], "丙氨酸氨基转移酶"),
    ("ast", ["天门冬氨酸氨基转移酶", "谷草转氨酶", "AST", "ast"], "天门冬氨酸氨基转移酶"),
    ("bilirubin", ["总胆红素", "胆红素", "TBIL"], "总胆红素"),
    ("hemoglobin", ["血红蛋白", "HGB", "Hb"], "血红蛋白"),
    ("wbc", ["白细胞计数", "白细胞", "WBC"], "白细胞"),
    ("creatinine", ["肌酐", "Cr", "CR"], "肌酐"),
]

SEVERITY_WORDS = [
    ("critical", ["危急值", "危及生命", "危重", "极重度"]),
    ("severe", ["严重异常", "重度异常", "重度", "严重", "剧烈", "显著升高", "明显升高", "大幅下降", "明显下降"]),
    ("moderate", ["中度", "升高", "增高", "偏高", "降低", "偏低", "下降", "异常", "阳性"]),
    ("mild", ["轻度", "轻微", "稍高", "略高"]),
]


@dataclass
class DraftRelation:
    kind: str
    explanation: str
    subject_ref: str
    status: str = "supported"
    confidence: float = 0.7
    drug_id: str | None = None
    symptom_id: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeExtraction:
    mentions: list[Mention]
    labs: list[LabResult]
    relations: list[DraftRelation]


def _mention_id(record_id: str, revision: int, kind: str, start: int, end: int) -> str:
    return f"M:{record_id}:r{revision}:{kind}:{start}:{end}"


def _lab_id(record_id: str, revision: int, start: int) -> str:
    return f"L:{record_id}:r{revision}:{start}"


def iter_clauses(text: str) -> list[tuple[int, int, str]]:
    """按中英文标点切分小句，保留绝对字符偏移。"""

    clauses: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"[，。；;！？!?\n]+", text):
        end = match.start()
        if end > start:
            clauses.append((start, end, text[start:end]))
        start = match.end()
    if start < len(text):
        clauses.append((start, len(text), text[start:]))
    return clauses


def _clause_at(clauses: list[tuple[int, int, str]], pos: int) -> tuple[int, int, str]:
    for clause in clauses:
        if clause[0] <= pos <= clause[1]:
            return clause
    # pos 可能正好在标点上；回退到前一小句。
    for clause in reversed(clauses):
        if clause[1] <= pos:
            return clause
    return (0, len(""), "") if not clauses else clauses[0]


def _is_negated(text: str, start: int, end: int, clause: tuple[int, int, str]) -> tuple[bool, str | None]:
    c_start, c_end, c_text = clause
    local_start = start - c_start
    local_end = end - c_start
    prefix = c_text[:local_start]

    best_marker: str | None = None
    best_end = -1
    for marker in NEGATION_MARKERS:
        idx = prefix.rfind(marker)
        if idx >= 0:
            marker_end = idx + len(marker)
            gap = prefix[marker_end:local_start]
            # 否定词和症状之间只能有很短的时态/连接字；“无明显诱因出现”不会被误判。
            if len(gap) <= 3 and "诱因" not in gap and not re.search(r"[但然可不]", gap):
                if marker_end > best_end:
                    best_marker = marker
                    best_end = marker_end

    if best_marker:
        between = prefix[best_end:local_start]
        if len(between) <= 3:
            return True, best_marker

    suffix = c_text[local_end:]
    for marker in NEGATION_MARKERS[:-2]:  # 后置否定通常是“未出现/未发生”等
        idx = suffix.find(marker)
        if 0 <= idx <= 3:
            return True, marker
    return False, None


SYMPTOM_UNCERTAINTY_MARKERS = [
    "不能排除",
    "待核实",
    "尚未确认",
    "不确定",
    "疑似",
    "可能",
    "也许",
    "或许",
]


def _certainty(clause_text: str, start: int, end: int, c_start: int) -> tuple[str, list[str]]:
    local_start = start - c_start
    local_end = end - c_start
    window = clause_text[max(0, local_start - 10) : min(len(clause_text), local_end + 10)]
    markers = [marker for marker in SYMPTOM_UNCERTAINTY_MARKERS if marker in window]
    if markers:
        return "uncertain", markers
    if any(marker in window for marker in ["家属称", "患者称", "诉", "转述", "听说", "群消息"]):
        return "reported", []
    return "stated", []


def _severity_from_context(clause_text: str, base: int) -> tuple[int, str | None]:
    rank = base
    reason = None
    for label, words in SEVERITY_WORDS:
        if any(word in clause_text for word in words):
            if label == "critical":
                rank, reason = 5, "critical_word"
            elif label == "severe":
                rank, reason = max(rank, 4), "severe_word"
            elif label == "moderate":
                rank, reason = max(rank, 3), "abnormal_word"
            elif label == "mild":
                rank, reason = min(rank, 1), "mild_word"
            break
    return rank, reason


def parse_time_mentions(
    case_id: str, record_id: str, revision: int, text: str, clauses: list[tuple[int, int, str]]
) -> list[Mention]:
    mentions: list[Mention] = []
    occupied: list[tuple[int, int]] = []
    latest_weekday: int | None = None

    def overlaps(start: int, end: int) -> bool:
        return any(not (end <= a or start >= b) for a, b in occupied)

    def add(start: int, end: int, canonical: str, attrs: dict[str, Any], status: str = "resolved") -> None:
        if overlaps(start, end):
            return
        occupied.append((start, end))
        mentions.append(
            Mention(
                id=_mention_id(record_id, revision, "time", start, end),
                case_id=case_id,
                record_id=record_id,
                revision=revision,
                kind="time",
                text=text[start:end],
                span=Span(start, end, text[start:end]),
                canonical=canonical,
                identity_status="temporal_expression",
                verification_status=status,
                attributes=attrs,
            )
        )

    for match in re.finditer(r"(?:周|星期|礼拜)([一二三四五六日天])", text):
        day = WEEKDAY[match.group(1)]
        latest_weekday = day
        c = _clause_at(clauses, match.start())[2]
        part_rank = next((rank for word, rank in PART_OF_DAY.items() if word in c), 5)
        add(
            match.start(),
            match.end(),
            f"weekday-{day}",
            {"weekday": day, "part_rank": part_rank, "order_key": day * 10 + part_rank},
        )

    for match in re.finditer(r"(\d{1,2})月(\d{1,2})[日号]", text):
        month, day = int(match.group(1)), int(match.group(2))
        add(
            match.start(),
            match.end(),
            f"date-{month:02d}-{day:02d}",
            {"month": month, "day": day, "order_key": 1000 + month * 40 + day},
        )

    for match in re.finditer(r"(\d+)\s*天前|前天|昨天|昨日", text):
        raw = match.group(0)
        if raw == "前天":
            offset = 2
        elif raw in ("昨天", "昨日"):
            offset = 1
        else:
            offset = int(match.group(1))
        add(
            match.start(),
            match.end(),
            f"days-ago-{offset}",
            {"offset_days": offset, "order_key": 100 - offset},
        )

    for match in re.finditer(r"当晚|当天晚上|当日晚上|次日|第二天", text):
        raw = match.group(0)
        if raw in ("当晚", "当天晚上", "当日晚上"):
            attrs: dict[str, Any] = {
                "same_day_as_previous": True,
                "part_rank": PART_OF_DAY["晚上"],
                "order_key": (latest_weekday or 0) * 10 + PART_OF_DAY["晚上"],
            }
            canonical = "same-night"
        else:
            attrs = {"next_day": True, "part_rank": 5, "order_key": (latest_weekday or 0) * 10 + 15}
            canonical = "next-day"
        add(match.start(), match.end(), canonical, attrs)

    part_pattern = "|".join(sorted(PART_OF_DAY, key=len, reverse=True))
    for match in re.finditer(part_pattern, text):
        word = match.group(0)
        day = latest_weekday
        add(
            match.start(),
            match.end(),
            f"part-{word}",
            {"part_of_day": word, "part_rank": PART_OF_DAY[word], "weekday": day,
             "order_key": (day * 10 if day else 50) + PART_OF_DAY[word]},
            status="resolved" if day else "pending_calendar",
        )

    for match in re.finditer(r"最近|近日|日前|此前|后来|随后|之前|之后", text):
        add(
            match.start(),
            match.end(),
            f"vague-{match.group(0)}",
            {"vague": True, "order_key": None},
            status="pending_time" if match.group(0) in ("最近", "近日", "日前") else "relative",
        )

    return mentions


def _nearest_time_id(
    start: int,
    end: int,
    clause: tuple[int, int, str],
    time_mentions: list[Mention],
) -> str | None:
    c_start, c_end, _ = clause
    in_clause = [m for m in time_mentions if m.span.start >= c_start and m.span.end <= c_end]
    if in_clause:
        in_clause.sort(key=lambda m: abs((m.span.start + m.span.end) / 2 - (start + end) / 2))
        return in_clause[0].id
    before = [m for m in time_mentions if m.span.end <= start]
    if before:
        return max(before, key=lambda m: m.span.end).id
    after = [m for m in time_mentions if m.span.start >= end]
    if after:
        return min(after, key=lambda m: m.span.start).id
    return None


def _subject_ref(clause: tuple[int, int, str], record_id: str) -> str:
    c_start, _, c_text = clause
    match = SUBJECT_OTHER_RE.search(c_text)
    if match:
        return f"other:{record_id}:{c_start + match.start()}"
    return "index"


def extract_narrative(
    case_id: str,
    record_id: str,
    text: str,
    products: list[Product] | None = None,
    revision: int = 0,
) -> NarrativeExtraction:
    """从单条中文叙述中抽取证据、事件和候选关系。"""

    products = products or []
    clauses = iter_clauses(text)
    mentions: list[Mention] = []
    labs: list[LabResult] = []
    relations: list[DraftRelation] = []

    time_mentions = parse_time_mentions(case_id, record_id, revision, text, clauses)
    mentions.extend(time_mentions)

    # 引用原文：医生/审阅记录中的引号内容只能作为来源链，不能另建病例主体。
    for match in QUOTE_RE.finditer(text):
        mentions.append(
            Mention(
                id=_mention_id(record_id, revision, "quote", match.start(), match.end()),
                case_id=case_id,
                record_id=record_id,
                revision=revision,
                kind="quote",
                text=match.group(0),
                span=Span(match.start(), match.end(), match.group(0)),
                canonical=match.group(1),
                identity_status="quoted_source",
                verification_status="source_quote",
                attributes={"quoted_text": match.group(1)},
            )
        )

    drug_mentions: list[Mention] = []
    occupied_drug: list[tuple[int, int]] = []

    def add_drug(start: int, end: int, raw: str, product: Product | None, inferred: bool = False) -> Mention:
        if product is not None:
            identity_status = "catalog_matched"
            verification = "extracted"
            canonical = product.name
            concept_id = product.id
        elif inferred:
            identity_status = "inferred_reference"
            verification = "pending_verification"
            canonical = raw
            concept_id = None
        else:
            identity_status = "unresolved"
            verification = "pending_verification"
            canonical = raw
            concept_id = None
        clause = _clause_at(clauses, start)
        mention = Mention(
            id=_mention_id(record_id, revision, "drug", start, end),
            case_id=case_id,
            record_id=record_id,
            revision=revision,
            kind="drug",
            text=raw,
            span=Span(start, end, raw),
            subject_id=_subject_ref(clause, record_id),
            canonical=canonical,
            concept_id=concept_id,
            time_id=_nearest_time_id(start, end, clause, time_mentions),
            identity_status=identity_status,
            verification_status=verification,
            attributes={"inferred": inferred},
        )
        mentions.append(mention)
        drug_mentions.append(mention)
        occupied_drug.append((start, end))
        return mention

    # 先匹配药品目录中的确定名称和别名。
    catalog_terms: list[tuple[str, Product]] = []
    for product in products:
        catalog_terms.append((product.name, product))
        for alias in product.aliases:
            catalog_terms.append((alias, product))
    for name, product in sorted(catalog_terms, key=lambda item: len(item[0]), reverse=True):
        if not name:
            continue
        for match in re.finditer(re.escape(name), text):
            s, e = match.span()
            if any(not (e <= a or s >= b) for a, b in occupied_drug):
                continue
            add_drug(s, e, name, product)

    # 再匹配未知但具有药品形态的表达；这些表达会明确进入待核状态。
    for pattern in GENERIC_DRUG_PATTERNS:
        for match in pattern.finditer(text):
            s, e = match.span()
            if any(not (e <= a or s >= b) for a, b in occupied_drug):
                continue
            raw = match.group(0)
            # 通用中文正则可能把“有人用了一种药”中的主语/动词一起吃进来；
            # 在最后一个常见叙述/动词边界后截取真正的药品短语。
            cut_positions = [raw.rfind(ch) for ch in "服用吃打输了过人是在把将给开买"]
            cut = max(cut_positions)
            if cut >= 0:
                s += cut + 1
                raw = raw[cut + 1 :]
            while raw and raw[0] in LEADING_DRUG_CHARS:
                raw = raw[1:]
                s += 1
            while raw and raw[-1] in "的了在把将":
                raw = raw[:-1]
                e -= 1
            e = s + len(raw)
            if len(raw) < 2 or raw in ("药物", "该药", "此药"):
                is_anaphor = raw in ("该药", "此药")
                mention = add_drug(s, e, raw, None, inferred=is_anaphor)
                if raw == "药物":
                    mention.attributes["generic"] = True
                continue
            add_drug(s, e, raw, None)

    symptom_mentions: list[Mention] = []
    occupied_symptoms: list[tuple[int, int]] = []
    all_terms: list[tuple[str, str, str, int]] = []
    for canonical, concept, severity, aliases in SYMPTOMS:
        for alias in aliases:
            all_terms.append((alias, canonical, concept, severity))
    for term, canonical, concept, base_severity in sorted(all_terms, key=lambda x: len(x[0]), reverse=True):
        for match in re.finditer(re.escape(term), text):
            s, e = match.span()
            if any(not (e <= a or s >= b) for a, b in occupied_symptoms):
                continue
            occupied_symptoms.append((s, e))
            clause = _clause_at(clauses, s)
            negated, negation_marker = _is_negated(text, s, e, clause)
            certainty, uncertainty_markers = _certainty(clause[2], s, e, clause[0])
            severity, severity_reason = _severity_from_context(clause[2], base_severity)

            change = None
            change_candidates: list[tuple[int, int, str]] = []
            for change_type, marker in CHANGE_MARKERS:
                idx = clause[2].find(marker)
                if idx >= 0:
                    marker_start = clause[0] + idx
                    marker_end = marker_start + len(marker)
                    distance = min(abs(marker_start - s), abs(marker_end - e))
                    # 变化词应位于症状附近；同小句即可，优先距离近且更长的短语。
                    if distance <= 12:
                        change_candidates.append((distance, -len(marker), change_type))
            if change_candidates:
                change = min(change_candidates)[2]
            if change is None and term in ("起疹", "出疹", "新发皮疹", "新疹"):
                change = "onset"

            verification = "extracted"
            if certainty == "uncertain":
                verification = "pending_verification"
            if negated:
                verification = "negated"

            mention = Mention(
                id=_mention_id(record_id, revision, "symptom", s, e),
                case_id=case_id,
                record_id=record_id,
                revision=revision,
                kind="symptom",
                text=term,
                span=Span(s, e, term),
                subject_id=_subject_ref(clause, record_id),
                canonical=canonical,
                concept_id=concept,
                time_id=_nearest_time_id(s, e, clause, time_mentions),
                polarity="negated" if negated else "positive",
                certainty=certainty,
                verification_status=verification,
                attributes={
                    "base_severity_rank": base_severity,
                    "severity_rank": 0 if negated else severity,
                    "severity_reason": severity_reason,
                    "negation_marker": negation_marker,
                    "uncertainty_markers": uncertainty_markers,
                    "change": change,
                    "clause": clause[2],
                },
            )
            mentions.append(mention)
            symptom_mentions.append(mention)

    # 临床整体变化，如“好转”。即使没有具体症状词，也参与停药后去激发判断。
    outcome_markers = sorted(
        [
            (marker, change_type)
            for change_type, marker in CHANGE_MARKERS
            if change_type in ("improving", "resolved", "completely_resolved")
        ],
        key=lambda item: len(item[0]),
        reverse=True,
    )
    occupied_outcomes: list[tuple[int, int]] = []
    for marker, change_type in outcome_markers:
        for match in re.finditer(re.escape(marker), text):
            s, e = match.span()
            if any(not (e <= a or s >= b) for a, b in occupied_outcomes):
                continue
            occupied_outcomes.append((s, e))
            clause = _clause_at(clauses, s)
            # 若该变化词已经作为症状属性使用，仍保留 outcome 事件，便于时间线展示。
            mentions.append(
                Mention(
                    id=_mention_id(record_id, revision, "outcome", s, e),
                    case_id=case_id,
                    record_id=record_id,
                    revision=revision,
                    kind="outcome",
                    text=marker,
                    span=Span(s, e, marker),
                    subject_id=_subject_ref(clause, record_id),
                    canonical="clinical_improvement" if change_type == "improving" else change_type,
                    time_id=_nearest_time_id(s, e, clause, time_mentions),
                    verification_status="extracted",
                    attributes={"change": change_type},
                )
            )

    event_specs = [
        ("withdrawal", re.compile(r"停药|停止用药|停止使用|停服|停用")),
        ("dose_change", re.compile(r"减半剂量|剂量减半|减半|自行减量|私自减量|擅自减量|减少剂量|调低剂量|加量|增加剂量|减量")),
        ("exposure", re.compile(r"服用|口服|输注|注射|吃过|吃了|使用|用药")),
    ]
    event_mentions: list[Mention] = []
    for kind, pattern in event_specs:
        for match in pattern.finditer(text):
            s, e = match.span()
            clause = _clause_at(clauses, s)
            # “停用”可能后面紧跟药名；关联同小句中最近的药品。
            nearby_drugs = [
                d for d in drug_mentions
                if d.subject_id == _subject_ref(clause, record_id)
                and not (d.span.end < clause[0] or d.span.start > clause[1])
            ]
            if not nearby_drugs and drug_mentions:
                nearby_drugs = sorted(
                    drug_mentions, key=lambda d: min(abs(d.span.start - s), abs(d.span.end - e))
                )[:1]
            drug = min(nearby_drugs, key=lambda d: abs(d.span.start - s)) if nearby_drugs else None
            negated, negation_marker = _is_negated(text, s, e, clause)
            attrs: dict[str, Any] = {}
            if drug:
                attrs["drug_mention_id"] = drug.id
            if negated:
                attrs["negation_marker"] = negation_marker
            if kind == "dose_change":
                raw = match.group(0)
                if "减半" in raw or "一半" in clause[2]:
                    attrs["fraction"] = 0.5
                if any(word in clause[2] for word in ["自行", "私自", "擅自"]):
                    attrs["self_adjusted"] = True
                attrs["direction"] = "increase" if any(word in raw for word in ["加量", "增加"]) else "decrease"
            mention = Mention(
                id=_mention_id(record_id, revision, kind, s, e),
                case_id=case_id,
                record_id=record_id,
                revision=revision,
                kind=kind,
                text=match.group(0),
                span=Span(s, e, match.group(0)),
                subject_id=_subject_ref(clause, record_id),
                time_id=_nearest_time_id(s, e, clause, time_mentions),
                polarity="negated" if negated else "positive",
                verification_status="negated" if negated else "extracted",
                attributes=attrs,
            )
            mentions.append(mention)
            event_mentions.append(mention)

    # 实验室结果。
    labs.extend(_extract_labs(case_id, record_id, revision, text, clauses, time_mentions))

    # 标记位于引号内的实体/事件：它们可作为证据，但来源链属于被引用者。
    quote_spans = [(m.span.start, m.span.end) for m in mentions if m.kind == "quote"]
    for mention in mentions:
        if mention.kind == "quote":
            continue
        if any(q_start <= mention.span.start and mention.span.end <= q_end for q_start, q_end in quote_spans):
            mention.attributes["inside_quote"] = True

    # 候选关系主要在同一主体、同一记录内建立；跨记录关系由 Workbench 汇总。
    relations.extend(
        _draft_relations(
            record_id=record_id,
            text=text,
            clauses=clauses,
            mentions=mentions,
            drug_mentions=drug_mentions,
            symptom_mentions=symptom_mentions,
            event_mentions=event_mentions,
        )
    )

    return NarrativeExtraction(mentions=mentions, labs=labs, relations=relations)


def _extract_labs(
    case_id: str,
    record_id: str,
    revision: int,
    text: str,
    clauses: list[tuple[int, int, str]],
    time_mentions: list[Mention],
) -> list[LabResult]:
    results: list[LabResult] = []
    occupied: list[tuple[int, int]] = []

    for concept, names, canonical in LAB_TESTS:
        for name in sorted(names, key=len, reverse=True):
            for match in re.finditer(re.escape(name), text):
                s, e = match.span()
                if any(not (e <= a or s >= b) for a, b in occupied):
                    continue
                clause = _clause_at(clauses, s)
                c_start, c_end, c_text = clause
                clause_index = next((i for i, item in enumerate(clauses) if item[0] == c_start), -1)
                context_text = c_text
                if 0 <= clause_index + 1 < len(clauses):
                    context_text += "，" + clauses[clause_index + 1][2]
                window = text[e:min(len(text), e + 30)]
                number_match = re.search(r"(\d+(?:\.\d+)?)", window)
                value = float(number_match.group(1)) if number_match else None
                unit = None
                if number_match:
                    after_number = window[number_match.end() : number_match.end() + 18]
                    unit_match = re.match(r"\s*([×x*]?10\^?9/L|g/L|U/L|μmol/L|umol/L|mg/dL|%|/L)?", after_number)
                    if unit_match and unit_match.group(1):
                        unit = unit_match.group(1).replace("x", "×").replace("*", "×")

                rank, reason = _severity_from_context(context_text, 2)
                flag = None
                for candidate in ["危急值", "阳性", "显著升高", "明显升高", "升高", "增高", "偏高", "大幅下降", "明显下降", "降低", "偏低", "异常"]:
                    if candidate in context_text:
                        flag = candidate
                        break

                if value is not None:
                    inferred_rank = _numeric_lab_severity(concept, value)
                    if inferred_rank is not None:
                        rank = max(rank, inferred_rank)
                        reason = reason or "numeric_threshold"

                # 没有任何异常线索时，只作为普通化验值，不提升严重程度。
                if flag is None and reason is None and rank == 2:
                    rank = 1
                    reason = "normal_or_unflagged"

                occupied.append((s, e))
                lab = LabResult(
                    id=_lab_id(record_id, revision, s),
                    case_id=case_id,
                    record_id=record_id,
                    revision=revision,
                    test_name=canonical,
                    test_concept=concept,
                    severity_rank=rank,
                    severity_label=_severity_label(rank),
                    span=Span(s, e, name),
                    value=value,
                    unit=unit,
                    flag=flag,
                    observed_time_id=_nearest_time_id(s, e, clause, time_mentions),
                    observed_time_text=None,
                    subject_id=_subject_ref(clause, record_id),
                    verification_status="extracted" if value is not None or flag else "pending_verification",
                    attributes={"reason": reason, "raw_clause": c_text},
                )
                results.append(lab)

    # 未识别具体项目但明确给出“危急值/严重异常”的实验室随访。
    if not results and re.search(r"实验室|化验|检验|复查", text):
        for match in re.finditer(r"危急值|严重异常|重度异常|明显异常", text):
            s, e = match.span()
            clause = _clause_at(clauses, s)
            rank = 5 if "危急" in match.group(0) or "严重" in match.group(0) or "重度" in match.group(0) else 4
            results.append(
                LabResult(
                    id=_lab_id(record_id, revision, s),
                    case_id=case_id,
                    record_id=record_id,
                    revision=revision,
                    test_name="未明确实验室项目",
                    test_concept="unspecified_lab",
                    severity_rank=rank,
                    severity_label=_severity_label(rank),
                    span=Span(s, e, match.group(0)),
                    flag=match.group(0),
                    observed_time_id=_nearest_time_id(s, e, clause, time_mentions),
                    subject_id=_subject_ref(clause, record_id),
                    verification_status="pending_verification",
                    attributes={"reason": "explicit_severity_word", "raw_clause": clause[2]},
                )
            )
            break

    return results


def _numeric_lab_severity(concept: str, value: float) -> int | None:
    if concept == "platelet":
        if value < 20:
            return 5
        if value < 50:
            return 4
        if value < 100:
            return 3
    if concept == "hemoglobin":
        if value < 70:
            return 5
        if value < 90:
            return 4
        if value < 110:
            return 3
    if concept == "wbc":
        if value < 1 or value > 30:
            return 5
        if value < 2 or value > 15:
            return 4
    if concept in ("alt", "ast"):
        if value >= 500:
            return 4
        if value >= 120:
            return 3
    return None


def _severity_label(rank: int) -> str:
    return {0: "not_graded", 1: "mild", 2: "mild_to_moderate", 3: "moderate", 4: "severe", 5: "critical"}.get(rank, "not_graded")


def _draft_relations(
    record_id: str,
    text: str,
    clauses: list[tuple[int, int, str]],
    mentions: list[Mention],
    drug_mentions: list[Mention],
    symptom_mentions: list[Mention],
    event_mentions: list[Mention],
) -> list[DraftRelation]:
    relations: list[DraftRelation] = []

    def time_order(mention_id: str | None) -> float | None:
        if not mention_id:
            return None
        mention = next((m for m in mentions if m.id == mention_id), None)
        if not mention:
            return None
        return mention.attributes.get("order_key")

    # 药物-症状关系：否定症状仍保留证据，但状态为 negated_finding，不参与升级。
    for symptom in symptom_mentions:
        if symptom.polarity == "negated":
            relations.append(
                DraftRelation(
                    kind="negated_symptom",
                    explanation=f"原文以“{symptom.attributes.get('negation_marker')}”否定{symptom.canonical}，不得作为阳性风险升级依据。",
                    subject_ref=symptom.subject_id or "index",
                    status="negated",
                    confidence=0.95,
                    symptom_id=symptom.id,
                    evidence_ids=[symptom.id],
                )
            )
            continue

        same_subject_drugs = [d for d in drug_mentions if d.subject_id == symptom.subject_id]
        drug = None
        if same_subject_drugs:
            drug = min(same_subject_drugs, key=lambda d: abs(d.span.start - symptom.span.start))
        status = "supported"
        explanation = f"叙述将{symptom.canonical}与用药语境放在同一主体下。"
        confidence = 0.65
        if drug is None:
            status = "pending_drug"
            explanation = f"{symptom.canonical}缺少可关联的药物表达，药物关系待核。"
            confidence = 0.3
        elif drug.identity_status == "unresolved":
            status = "pending_drug_identity"
            explanation = f"药物表达“{drug.text}”身份不明，{symptom.canonical}与具体产品的关系待核。"
            confidence = 0.35
        elif drug.identity_status == "inferred_reference":
            status = "pending_drug_identity"
            explanation = f"“{drug.text}”为指代，虽可尝试关联唯一用药，但需审阅者确认药物身份。"
            confidence = 0.5

        s_time = time_order(symptom.time_id)
        d_time = time_order(drug.time_id) if drug else None
        if status == "supported" and s_time is None and d_time is None:
            # 检查是否有明确“服药后出现”等顺序连接。
            between = text[drug.span.end : symptom.span.start] if drug and drug.span.end <= symptom.span.start else ""
            if "后" in between or "后" in text[max(0, symptom.span.start - 12): symptom.span.start]:
                status = "relative_order_known_time_pending"
                explanation += "原文有“后”等相对顺序线索，但缺少可核对的具体日期/时间，时间精度仍待补核。"
            else:
                status = "pending_time"
                explanation += "药物和症状均缺少可解析时间或先后连接，时间关系待核。"
                confidence = min(confidence, 0.35)
        elif status == "supported":
            time_mention = next((m for m in mentions if m.id in {symptom.time_id, drug.time_id if drug else None}), None)
            if time_mention and str(time_mention.verification_status).startswith("pending"):
                status = "pending_time"
                explanation += "时间表达本身不完整或不明确，需补核具体时间。"

        relations.append(
            DraftRelation(
                kind="drug_symptom_link",
                explanation=explanation,
                subject_ref=symptom.subject_id or "index",
                status=status,
                confidence=confidence,
                drug_id=drug.id if drug else None,
                symptom_id=symptom.id,
                evidence_ids=[i for i in [drug.id if drug else None, symptom.id, symptom.time_id] if i],
                attributes={"record_id": record_id},
            )
        )

    # 减量先于停药。
    reductions = [m for m in event_mentions if m.kind == "dose_change" and m.polarity != "negated"]
    withdrawals = [m for m in event_mentions if m.kind == "withdrawal" and m.polarity != "negated"]
    outcomes = [m for m in mentions if m.kind == "outcome"]
    positive_symptoms = [m for m in symptom_mentions if m.polarity == "positive"]

    for reduction in reductions:
        same_subject_withdrawals = [w for w in withdrawals if w.subject_id == reduction.subject_id]
        for withdrawal in same_subject_withdrawals:
            r_time, w_time = time_order(reduction.time_id), time_order(withdrawal.time_id)
            context_start = max(0, min(reduction.span.start, withdrawal.span.start) - 12)
            context_end = min(len(text), max(reduction.span.end, withdrawal.span.end) + 8)
            context = text[context_start:context_end]
            explicit_before = bool(re.search(r"(?:停药|停用|停止用药|停服)前", context))
            time_before = r_time is not None and w_time is not None and r_time < w_time
            textual_before = reduction.span.start < withdrawal.span.start

            # 当“停药前减量”中“停药”在字面上先出现时，必须有“前”或时间证据，
            # 不能仅凭两个事件的字面位置推断先后。
            should_link = False
            if textual_before and not (r_time is not None and w_time is not None and r_time > w_time):
                should_link = True
            elif withdrawal.span.start < reduction.span.start and (explicit_before or time_before):
                should_link = True
            elif time_before:
                should_link = True
            if not should_link:
                continue

            relations.append(
                DraftRelation(
                    kind="dose_reduction_precedes_withdrawal",
                    explanation="患者在停药前已有剂量变化；不能把后续变化简单解释为仅由停药导致。",
                    subject_ref=reduction.subject_id or "index",
                    status="supported",
                    confidence=0.92 if explicit_before or time_before else 0.65,
                    drug_id=reduction.attributes.get("drug_mention_id") or withdrawal.attributes.get("drug_mention_id"),
                    evidence_ids=[i for i in [reduction.id, withdrawal.id, reduction.time_id, withdrawal.time_id] if i],
                    attributes={
                        "self_adjusted": reduction.attributes.get("self_adjusted", False),
                        "explicit_before_marker": explicit_before,
                    },
                )
            )

    # 停药后改善，但如果此前有减量，则标记为混杂/待审阅，而不是压缩成单一“停药后好转”。
    for withdrawal in withdrawals:
        later_outcomes = [
            o for o in outcomes
            if o.subject_id == withdrawal.subject_id and o.span.start > withdrawal.span.start
        ]
        later_improving_symptoms = [
            s for s in positive_symptoms
            if s.subject_id == withdrawal.subject_id
            and s.span.start > withdrawal.span.start
            and s.attributes.get("change") in ("improving", "resolved", "completely_resolved")
        ]
        target = None
        if later_improving_symptoms:
            target = min(later_improving_symptoms, key=lambda s: s.span.start)
        elif later_outcomes:
            target = min(later_outcomes, key=lambda o: o.span.start)
        if target is None:
            continue

        prior_reduction = False
        for reduction in reductions:
            if reduction.subject_id != withdrawal.subject_id:
                continue
            if reduction.span.start < withdrawal.span.start:
                prior_reduction = True
                break
            context = text[
                max(0, min(reduction.span.start, withdrawal.span.start) - 12) :
                min(len(text), max(reduction.span.end, withdrawal.span.end) + 8)
            ]
            r_time = time_order(reduction.time_id)
            w_time = time_order(withdrawal.time_id)
            if re.search(r"(?:停药|停用|停止用药|停服)前", context):
                prior_reduction = True
                break
            if r_time is not None and w_time is not None and r_time < w_time:
                prior_reduction = True
                break
        after_marker = "后" in text[withdrawal.span.end : target.span.start + 1] or "后" in text[withdrawal.span.start : withdrawal.span.end + 3]
        if target.span.start > withdrawal.span.start and (after_marker or True):
            if prior_reduction:
                status = "confounded_by_prior_dose_reduction"
                explanation = "改善发生在停药后，但停药前已自行/提前减量；停药去激发与减量效应混杂，不能只保留“停药后好转”标签。"
                confidence = 0.86
            else:
                status = "positive_dechallenge_candidate"
                explanation = "原文显示停药后症状或临床状态改善，可作为去激发阳性候选，仍需排除自然病程等因素。"
                confidence = 0.7
            relations.append(
                DraftRelation(
                    kind="withdrawal_to_improvement",
                    explanation=explanation,
                    subject_ref=withdrawal.subject_id or "index",
                    status=status,
                    confidence=confidence,
                    drug_id=withdrawal.attributes.get("drug_mention_id"),
                    symptom_id=target.id if target.kind == "symptom" else None,
                    evidence_ids=[i for i in [withdrawal.id, target.id, withdrawal.time_id, target.time_id] if i],
                )
            )

    return relations
