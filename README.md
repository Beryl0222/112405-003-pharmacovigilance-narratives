# 药物不良反应叙述与随访证据闭环

> 本仓库中的病例、编号、药品和机构均为合成演练资料，只用于本地 NLP 与流程联调，**不用于临床决策**。

项目使用 Python 3.11+，没有第三方运行依赖。它把医生补记、患者/家属来电、转发消息和实验室随访作为持续到达的来源记录处理，保留：

- 患者/叙述主体身份及跨渠道重复关系；
- 药品、症状、剂量/停药/改善事件和中文时间表达；
- 否定、不确定、时间不明和药品身份不明；
- 每个结论对应的原文字符区间、来源记录和证据链；
- 减量、停药、症状变化之间的时间顺序与混杂关系；
- 判断版本、更正历史、已签发随访和风险说明版本；
- 更正后必须重新确认的随访/风险说明清单。

## 运行验证

```sh
python3 -m unittest discover -s tests -v
```

处理合成病例并输出完整 JSON 报告：

```sh
python3 -m case_workbench data/case.json --output /tmp/case-report.json
```

只查看安全信号看板：

```sh
python3 -m case_workbench data/case.json --board-only
```

持续接收新记录并把状态保存到本地 JSON：

```sh
python3 -m case_workbench data/case.json --state /tmp/workbench-state.json
python3 -m case_workbench --state /tmp/workbench-state.json \
  --text "复查提示血小板18，报告为危急值。" --case-id pv-4
```

也可以安装命令入口：

```sh
pip install -e .
case-workbench data/case.json --board-only
```

## Python 工作流

```python
from case_workbench import CaseWorkbench

wb = CaseWorkbench(products=[
    {"id": "DR-2", "name": "宁安片", "ingredient": "合成成分甲"}
])

case = wb.ingest_path("data/case.json")
judgment = case.current_judgment()

print(judgment.signal_status)       # causal_ambiguity_needs_review
print(judgment.summary)
for event in judgment.timeline:
    print(event.order_key, event.event_type, event.title, event.record_id)

# 医学审阅者处理待核关系/身份
for item in wb.review_queue("pv-4"):
    print(item.id, item.label, item.reason)

# 负责人签发随访
follow_up = wb.issue_follow_up(
    "pv-4",
    title="核实减量、停药和皮疹变化",
    description="请核对周一减量、周三停药及当晚仍有新疹的时间线。",
    issued_by="signal-lead",
)

# 发布风险说明
statement = wb.create_risk_statement(
    "pv-4",
    "宁安片与皮疹存在时间关联，需继续核实减量/停药混杂。",
    changed_by="signal-lead",
    product_id="DR-2",
    symptom_concept="rash",
    severity_rank=2,
)

# 后续更正：原始记录不会被覆盖，只追加更正版本
correction = wb.correct_record(
    "pv-4",
    "PV-4B",
    new_text="家属再次确认：停药前没有减量，周三停药后皮疹逐渐消退。",
    reason="家属回电更正",
    corrected_by="reviewer-li",
)

# 查询旧判断如何形成
old = wb.explain_judgment("pv-4", version=3)
print(old["rationale"], old["relationships"], old["evidence"])

# 更正后哪些对象仍待重新确认
print(wb.pending_reconfirmations("pv-4"))

# 安全信号负责人决定调整风险说明
item = next(i for i in wb.pending_reconfirmations("pv-4")
            if i.target_type == "risk_statement")
wb.record_signal_decision(
    "pv-4",
    "adjust_risk_statement",
    "signal-lead-wang",
    risk_statement_id=statement.id,
    new_risk_text="更正后去激发候选更明确，但仍需核实跨渠道身份。",
    reconfirmation_id=item.id,
    rationale="时间线更正后调整措辞",
)

wb.save("/tmp/workbench-state.json")
restored = CaseWorkbench.load("/tmp/workbench-state.json")
```

## 关键设计

### 1. 原文不可变，更正以事件追加

`SourceRecord.text` 保存到达时的原文。`correct_record(..., new_text=...)` 追加 `RecordCorrection`，旧抽取结果和旧 `JudgmentVersion` 不删除。当前图使用新文本重新分析，旧判断中的关系快照、时间线、理由和证据仍可通过版本号查询。

### 2. 时间顺序不压缩成单个标签

抽取器识别周一至周日、日期、天数、当晚/次日、部分日时间，以及“停药前/停药后”等关系。若已有减量、随后停药和改善，系统会生成：

- `dose_reduction_precedes_withdrawal`
- `withdrawal_to_improvement`，状态为 `confounded_by_prior_dose_reduction`

即使两个事实分散在不同记录中，跨记录时间线核对也会重新评估“停药后好转”关系；旧判断仍保留当时的 `positive_dechallenge_candidate` 快照。

### 3. 否定词项保留证据，但不升级风险

“未出现呼吸困难”“未诉胸闷”“无呼吸困难”等会生成 `polarity="negated"` 的症状证据，并进入 `negated_findings`。它们可被审阅和审计，但不会进入阳性症状、严重程度最大值或信号看板计数。

### 4. 身份、药品和时间不明均显式待核

- “有人/另一名网友”等主体进入 `anonymous_unresolved`，不会默认并入主患者。
- “一种药/不明药物/该药”等进入药品身份待核。
- 只有“服药后”等相对线索时，保留相对顺序，同时把具体日期/时间精度列为待核。
- 完全没有时间线索时，药物-症状关系为 `pending_time`。

### 5. 引用和转发不产生第二个独立病例

引号内容会建立 `quote` 证据并与来源文本匹配；显式 `derived_from`、`quoted_from`、`duplicate_of` 也会进入来源图。跨病例的高置信引用会连接身份根，信号看板按身份根去重。

不同渠道的同一匿名 key 会自然归并；完全相同/极高相似度文本会建立“临时重复身份”并生成审阅任务，不会物理删除来源。审阅者若认为只是相似而非同一人，可以调用 `reject_identity_merge(...)` 恢复独立计数；也可以手动处理：

```python
wb.find_duplicate_cases()
wb.mark_duplicate_case("channel-b", "channel-a", actor="reviewer", reason="...")
```

### 6. 实验室结果可改严重程度，已签发请求不被改写

实验室叙述如“血小板 18×10^9/L，危急值”或“肌钙蛋白明显升高”会生成 `LabResult`，新判断版本可取症状和化验分级的最大值。`FollowUpRequest` 的标题、说明、证据和签发时间保存在 `original_snapshot`；更正只会生成 `ReconfirmationItem`，不会就地改写原始请求。

### 7. 信号看板按独立身份根计数

`wb.signal_board()` 按药品概念和症状概念汇总，输出：

- `independent_case_count`
- `provisional_case_count`
- `reviewer_confirmed_case_count`
- `pending_identity_report_count`
- `pending_time_relationship_count`
- `temporal_confounding_count`
- `drug_identity_pending`
- `negated_finding_count`

无法识别具体药品的报告进入 `PENDING_DRUG` 组，避免把未知药品错误计入某个已上市产品。

## 主要模块

- `case_workbench/nlp.py`：中文规则 NLP、字符区间、否定/时间/药品/症状/化验抽取。
- `case_workbench/models.py`：数据模型和 JSON 序列化/反序列化。
- `case_workbench/workbench.py`：摄入、身份归并、审阅、判断版本、更正影响、随访、风险说明和信号看板。
- `case_workbench/identity.py`：跨渠道文本相似度。
- `case_workbench/cli.py` / `__main__.py`：命令行入口。
- `tests/test_workbench.py`：覆盖上述闭环的单元测试。

## 数据与隐私说明

`data/case.json` 是合成资料。真实使用时不应把可识别患者信息提交到代码仓库；持久化 JSON 应放在受控位置并按机构政策管理。
