# core/registry.py
import importlib
import inspect
import pkgutil
from typing import Dict, Type

from heuristics.base import BaseHeuristic


def discover_llm_archives(pkg_name: str = "heuristics.llm_archives") -> Dict[str, Type[BaseHeuristic]]:
    found: Dict[str, Type[BaseHeuristic]] = {}

    try:
        pkg = importlib.import_module(pkg_name)
    except ModuleNotFoundError:
        return found

    for m in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        module = importlib.import_module(m.name)

        for _, obj in vars(module).items():
            if not inspect.isclass(obj):
                continue
            if obj is BaseHeuristic:
                continue
            if not issubclass(obj, BaseHeuristic):
                continue

            heur_name = getattr(obj, "name", None)
            if not heur_name:
                continue

            found[str(heur_name)] = obj

    return found


def _try_register_llm_based(reg: Dict[str, Type[BaseHeuristic]]) -> None:
    """
    Register the live LLM-based heuristic entry used by debug_start=0.
    If import fails, print a helpful message (otherwise it silently disappears).
    """
    try:
        # 你把 wrapper 类放在这里（推荐）
        from heuristics.llm_entry import LLMBasedHeuristic
        reg["llm_based"] = LLMBasedHeuristic
    except Exception as e:
        # 不要静默失败，否则你又会回到“找不到 llm_based”的状态
        print(f"[Registry] llm_based not registered (import failed): {e}")


def build_registry(
    handcrafted: Dict[str, Type[BaseHeuristic]],
    include_archives: bool = True
) -> Dict[str, Type[BaseHeuristic]]:
    reg: Dict[str, Type[BaseHeuristic]] = dict(handcrafted)

    # NEW: always try to add live llm_based
    _try_register_llm_based(reg)

    if include_archives:
        importlib.invalidate_caches()
        reg.update(discover_llm_archives())
    return reg
