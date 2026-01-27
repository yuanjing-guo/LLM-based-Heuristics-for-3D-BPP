import os
import io
import glob
import queue
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Form
# ✅ 关键：确保从“仓库根目录”启动 uvicorn 时，import demo.* 和 import heuristics.* 都OK
# 你只要按下面“启动方式”从仓库根目录运行，就不需要改 sys.path。

# 你之前已经做了 demo 强隔离：LLMBasedHeuristicDemo + demo runner
from demo.llm_entry_demo import LLMBasedHeuristicDemo  # 你已有/我下面也给你版本
# 如果你没有这个 runner_demo，就用我下面给你的 demo/runner.py
from demo.runner import run_episode_demo

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

VIDEO_DIR = Path("video_demo")
VIDEO_DIR.mkdir(exist_ok=True)

RUNS_DIR = Path("runs_demo") / "latest"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# 静态资源 & 视频目录
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/video_demo", StaticFiles(directory=str(VIDEO_DIR)), name="video_demo")

# --------- 日志队列（SSE）---------
log_q: "queue.Queue[str]" = queue.Queue()
lock = threading.Lock()

def log(msg: str):
    log_q.put(str(msg))

# --------- 全局 demo heuristic（本机展示足够）---------
# 你也可以在 UI 里加按钮切 soft / physics_obs，这里先固定：
demo = LLMBasedHeuristicDemo(soft=False, expose_physics_obs=True)

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/api/caps")
def get_caps():
    return JSONResponse(demo.get_capabilities())

@app.post("/api/caps/buffer/{mode}")
def set_buffer(mode: str):
    mode = (mode or "").strip().lower()
    if mode not in ("first", "full"):
        return JSONResponse({"ok": False, "error": "mode must be first|full"})
    demo.set_capability("buffer", mode)
    log(f"[Caps] buffer={mode}")
    return JSONResponse({"ok": True})

@app.post("/api/caps/unstack/{mode}")
def set_unstack(mode: str):
    mode = (mode or "").strip().lower()
    if mode not in ("on", "off"):
        return JSONResponse({"ok": False, "error": "mode must be on|off"})
    demo.set_capability("unstack", mode)
    log(f"[Caps] unstack={mode}")
    return JSONResponse({"ok": True})

@app.post("/api/reset")
def reset_llm():
    demo.reset_to_naive()
    log("[Reset] reset_to_naive()")
    return JSONResponse({"ok": True})

@app.post("/api/gen")
def gen(feedback: str = Form(...)):
    feedback = (feedback or "").strip()
    if not feedback:
        log("[Gen] ERROR: feedback is empty (did you send as Form?)")
        return JSONResponse({"ok": False, "error": "feedback is empty"})

    log(f"[Gen] feedback={feedback}")

    try:
        with lock:
            demo.regenerate(feedback)
    except Exception as e:
        log(f"[Gen] ERROR: {repr(e)}")
        return JSONResponse({"ok": False, "error": repr(e)})

    log("[Gen] done")
    return JSONResponse({"ok": True})
@app.post("/api/eval")
def eval(seed: int = 0, max_steps: int = 200, video: bool = True):
    log(f"[Eval] seed={seed} max_steps={max_steps} video={video}")
    try:
        with lock:
            util = run_episode_demo(
                demo,
                max_steps=max_steps,
                seed=seed,
                save_video=bool(video),
                soft=demo.soft,
                expose_physics_obs=demo.expose_physics_obs,
                video_dir=str(VIDEO_DIR),
                run_context_meta={
                    "seed": seed,
                    "max_steps": max_steps,
                    "soft": demo.soft,
                    "expose_physics_obs": demo.expose_physics_obs,
                    "caps": demo.get_capabilities(),
                },
            )
    except Exception as e:
        log(f"[Eval] ERROR: {repr(e)}")
        return JSONResponse({"ok": False, "error": repr(e)})

    log(f"[Eval] util={util:.4f}")
    return JSONResponse({"ok": True, "util": util})
@app.get("/api/logs")
def stream_logs():
    def event_gen():
        yield "data: [SSE] connected\n\n"
        while True:
            msg = log_q.get()
            yield f"data: {msg}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/api/videos")
def list_videos():
    vids = sorted(glob.glob(str(VIDEO_DIR / "*.mp4")), key=os.path.getmtime, reverse=True)
    return JSONResponse({"videos": [os.path.basename(v) for v in vids]})

@app.get("/api/logs")
def stream_logs():
    def event_gen():
        while True:
            msg = log_q.get()
            # SSE 格式：每条消息一行 data:
            yield f"data: {msg}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")
