"""harness_adapters — one HarnessAdapter per harness for bench_fleet.py's
shared driver: how each harness is invoked, connects to the MCP endpoint, and
its own retry quirks. The driver owns provisioning and the retry/log/sync loop.
"""

from .base import HarnessAdapter, HarnessLaunchError, Job, QUOTA_EXIT_CODE, run_and_tee
from .claude_code import ClaudeCodeAdapter
from .codex import CodexAdapter
from .nanobot import NanobotAdapter
from .own import OwnHarnessAdapter

__all__ = [
    "HarnessAdapter", "HarnessLaunchError", "Job", "QUOTA_EXIT_CODE", "run_and_tee",
    "ClaudeCodeAdapter", "CodexAdapter", "NanobotAdapter",
    "OwnHarnessAdapter",
]
