import os
import time
import importlib.util
import numpy as np

from heuristics.base import BaseHeuristic
from heuristics.floor_building import FloorBuilding
from heuristics.llm_based.llm_based_function import generate_heuristic, write_heuristic


class LLMBasedHeuristic(BaseHeuristic):
    name = "llm_based"

    def __init__(self):
        super().__init__()

        # API config
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.api_key = "sk-69b1b30bca134d86a286fc4e69cfdd86"
        self.model = "deepseek-chat"

        # 指定“临时生成的 heuristic 文件”的保存路径
        # generated heuristic file
        self.out_path = os.path.join(
            os.path.dirname(__file__), "llm_based", "llm_based_heuristic.py"
        )

        self._last_code = None
        self._impl = None


        self._generate(initial=True)


    def regenerate(self, feedback: str):
        self._generate(initial=False, feedback=feedback)

    def __call__(self, obs):
        if self._impl is None:
            return FloorBuilding()(obs)
        return self._impl(obs)

    def _generate(self, initial: bool, feedback: str = None):
        #检查 API key
        if not self.api_key:
            print("[LLM] Missing LLM_API_KEY, fallback to FloorBuilding.")
            self._impl = FloorBuilding()
            return

        #调用 LLM 生成代码
        code = generate_heuristic(
            self.api_url, self.api_key, self.model, self._last_code, feedback
        )
        if not code:
            print("[LLM] Generation failed, fallback to FloorBuilding.")
            self._impl = FloorBuilding()
            return

        #把代码写成文件
        write_heuristic(self.out_path, code)
        impl = self._load_generated_class()
        if impl is None:
            print("[LLM] Import failed, fallback to FloorBuilding.")
            self._impl = FloorBuilding()
            return

        #保存成功的结果 方便调用
        self._last_code = code
        self._impl = impl

    #把这个文件加载成模块并实例化成对象  后面__call__ 时直接使用这个对象
    def _load_generated_class(self):
        module_name = f"llm_generated_{int(time.time())}"
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



