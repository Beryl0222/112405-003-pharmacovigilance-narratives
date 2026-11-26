"""命令行入口：可持续摄入 JSON/文本并输出审阅报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workbench import CaseWorkbench


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中文药物安全叙述证据闭环工作台")
    parser.add_argument("inputs", nargs="*", help="data/case.json 等 JSON 输入文件")
    parser.add_argument("--text", action="append", default=[], help="直接摄入一段文本，可重复传入")
    parser.add_argument("--case-id", default="cli-case", help="--text 输入使用的病例编号")
    parser.add_argument("--state", help="可选的 JSON 状态文件；存在则载入，运行后写回")
    parser.add_argument("--output", "-o", help="把完整报告写入 JSON 文件")
    parser.add_argument("--board-only", action="store_true", help="只输出安全信号看板")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.state and Path(args.state).exists():
        wb = CaseWorkbench.load(args.state)
    else:
        wb = CaseWorkbench()

    touched_cases: list[str] = []
    for input_path in args.inputs:
        result = wb.ingest_path(input_path)
        if isinstance(result, list):
            touched_cases.extend(item.case_id for item in result)
        else:
            touched_cases.append(result.case_id)

    for index, text in enumerate(args.text):
        case_id = args.case_id if len(args.text) == 1 else f"{args.case_id}-{index + 1}"
        case = wb.ingest_text(text, case_id=case_id)
        touched_cases.append(case.case_id)

    if args.state:
        wb.save(args.state)

    if args.board_only:
        report = wb.signal_board()
    else:
        report = {
            "cases": [wb.case_report(cid) for cid in dict.fromkeys(touched_cases)],
            "signal_board": wb.signal_board(),
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
