import os
import time
import importlib.util
from typing import Dict, Optional

from heuristics.base import BaseHeuristic
from heuristics.floor_building import FloorBuilding

from demo.llm_based_function_demo import generate_heuristic, write_heuristic


class LLMBasedHeuristicDemo(BaseHeuristic):
    """
    Demo-only LLM heuristic wrapper.
    caps:
      - buffer: "full" | "first"
      - unstack: "off" | "on"   (stage-1: prompt only)
    """
    name = "llm_demo"

    def __init__(self, *, soft: bool, expose_physics_obs: bool):
        super().__init__()
        self.soft = bool(soft)
        self.expose_physics_obs = bool(expose_physics_obs)

        self.api_url = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")

        self._caps: Dict[str, str] = {"buffer": "full", "unstack": "off"}

        # 生成临时模块放 demo 目录下（和主工程隔离）
        self.out_path = os.path.join(os.path.dirname(__file__), "_llm_demo_generated.py")

        self._last_code: Optional[str] = None
        self.current_code: Optional[str] = None
        self._impl: Optional[BaseHeuristic] = None
        self._feedback_history = []

        self._generate(initial=True)

    # ---------- caps ----------
    def set_capability(self, key: str, value: str):
        self._caps[str(key)] = str(value)

    def get_capabilities(self) -> Dict[str, str]:
        return dict(self._caps)

    # ---------- controls ----------
    def reset_to_naive(self):
        self._last_code = None
        self.current_code = None
        self._feedback_history = []
        self._generate(initial=True)

    def get_current_code(self) -> Optional[str]:
        return self.current_code

    def regenerate(self, feedback: str):
        self._feedback_history.append(feedback)
        self._generate(initial=False, feedback=feedback)

    # ---------- core ----------
    def __call__(self, obs):
        if self._impl is None:
            return FloorBuilding()(obs)

        # 关键：不在 wrapper 里硬改 logits（否则就丢了 x/y/rot）
        # buffer=first 应该由“LLM 生成的 llm_policy”里严格遵守（prompt + 运行期 clamp）
        return self._impl(obs)

    # ---------- internal ----------
    def _generate(self, initial: bool, feedback: str = None):
        if not self.api_key:
            print("[LLM] Missing LLM_API_KEY (env var). Fallback to FloorBuilding.")
            self._impl = FloorBuilding()
            return

        code = generate_heuristic(
            self.api_url,
            self.api_key,
            self.model,
            previous_code=self._last_code,
            feedback=feedback,
            feedback_history=self._feedback_history,
            caps=self.get_capabilities(),
        )
        if not code:
            print("[LLM] Generation failed, fallback to FloorBuilding.")
            self._impl = FloorBuilding()
            return

        write_heuristic(self.out_path, code)
        impl = self._load_generated_class()
        if impl is None:
            print("[LLM] Import failed, fallback to FloorBuilding.")
            self._impl = FloorBuilding()
            return

        self._last_code = code
        self.current_code = code
        self._impl = impl

    def _load_generated_class(self):
        module_name = f"llm_demo_generated_{int(time.time())}"
        spec = importlib.util.spec_from_file_location(module_name, self.out_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print("[LLM] Import exception:", e)
            return None
        cls = getattr(module, "GeneratedHeuristic", None)
        if cls is None:
            return None
        try:
            return cls()
        except Exception as e:
            print("[LLM] Instantiation exception:", e)
            return None
