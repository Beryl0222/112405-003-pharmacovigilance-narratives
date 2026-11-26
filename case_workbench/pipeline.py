"""面向调用方的工作流别名。

主实现位于 :mod:`case_workbench.workbench`；此模块提供更贴近业务命名的入口，
方便不同调用习惯导入。
"""

from __future__ import annotations

from .workbench import CaseWorkbench


class PharmacovigilanceWorkflow(CaseWorkbench):
    """药物安全病例工作流。"""


class NLPWorkflow(CaseWorkbench):
    """中文病例 NLP + 证据闭环工作流。"""


def create_workbench(products: list[dict] | None = None) -> CaseWorkbench:
    return CaseWorkbench(products=products or [])


__all__ = ["CaseWorkbench", "PharmacovigilanceWorkflow", "NLPWorkflow", "create_workbench"]
