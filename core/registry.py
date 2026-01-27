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


def build_registry(handcrafted: Dict[str, Type[BaseHeuristic]], include_archives: bool = True) -> Dict[str, Type[BaseHeuristic]]:
    reg: Dict[str, Type[BaseHeuristic]] = dict(handcrafted)
    if include_archives:
        importlib.invalidate_caches()
        reg.update(discover_llm_archives())
    return reg
