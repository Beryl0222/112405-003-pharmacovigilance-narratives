"""兼容常见命名的工作流入口。"""

from .workbench import CaseWorkbench


class SafetyWorkflow(CaseWorkbench):
    """安全信号工作台别名。"""


class CaseWorkflow(CaseWorkbench):
    """病例审阅工作流别名。"""


__all__ = ["CaseWorkbench", "SafetyWorkflow", "CaseWorkflow"]
