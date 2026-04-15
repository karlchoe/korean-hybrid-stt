"""
실행: python main.py
사전 준비:
  1) pip install -r requirements.txt
  2) vLLM 서버 실행:
       vllm serve Qwen/Qwen3-ASR-0.6B --port 8000
  3) config.py 설정:
       AUDIO_SOURCE = "loopback"  # 브라우저/유튜브 오디오
       OUTPUT_FILE  = "transcript.txt"
       HOTWORDS, LANGUAGE 등
"""

import signal
import sys
import threading
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from config import Config
from asr_pipeline import HybridASRPipeline

console = Console()
cfg: Config = None  # main() 에서 초기화

# ── 상태 저장 ──────────────────────────────────────────────
state = {
    "realtime": "",
    "finals": [],    # (ts, text, is_err) 튜플 목록
    "status": "✅ 대기 중",
}
lock = threading.Lock()


# ── Rich 화면 렌더러 ───────────────────────────────────────

def make_display() -> Layout:
    with lock:
        realtime = state["realtime"]
        finals   = state["finals"]
        status   = state["status"]

    # 확정 텍스트 (최근 10개) — 오류는 빨간색 표시
    history_lines = []
    for ts, text, is_err in finals[-10:]:
        if is_err:
            history_lines.append(f"[dim]{ts}[/dim]  [red]{text}[/red]")
        else:
            history_lines.append(f"[dim]{ts}[/dim]  {text}")
    history_str = "\n".join(history_lines) if history_lines else "[dim]아직 확정된 발화 없음[/dim]"

    layout = Layout()
    layout.split_column(
        Layout(name="status", size=3),
        Layout(name="realtime", size=5),
        Layout(name="final"),
    )

    layout["status"].update(
        Panel(Text(status, style="bold yellow"), title="상태")
    )
    layout["realtime"].update(
        Panel(
            Text(realtime or "[dim]말씀하세요...[/dim]", style="bold cyan"),
            title="[cyan]실시간 (Qwen3-ASR)",
        )
    )
    layout["final"].update(
        Panel(
            history_str,
            title=f"[green]확정 텍스트 (Cohere Transcribe)  "
                  f"[dim]→ {cfg.OUTPUT_FILE} / {cfg.OUTPUT_FILE.rsplit('.', 1)[0]}.err.txt",
        )
    )
    return layout


# ── 콜백 ──────────────────────────────────────────────────

def on_realtime(text: str):
    with lock:
        state["realtime"] = text


def on_final(text: str, is_err: bool, reason: str):
    ts = datetime.now().strftime("%H:%M:%S")
    with lock:
        state["finals"].append((ts, text, is_err))
        state["realtime"] = ""

    if is_err:
        console.print(f"\n[bold red][{ts}] 오류({reason}):[/bold red] {text}\n")
    else:
        console.print(f"\n[bold green][{ts}] 확정:[/bold green] {text}\n")


def on_status(msg: str):
    with lock:
        state["status"] = msg


# ── 메인 ──────────────────────────────────────────────────

def main():
    global cfg
    cfg = Config()

    src = "루프백(브라우저)" if cfg.AUDIO_SOURCE == "loopback" else "마이크"
    console.print(f"[bold]오디오 소스: {src}  →  {cfg.OUTPUT_FILE}[/bold]")
    console.print("[bold]Cohere Transcribe 모델 로딩 중... (최초 1회, 수 분 소요)[/bold]")
    pipeline = HybridASRPipeline(
        cfg=cfg,
        on_realtime=on_realtime,
        on_final=on_final,
        on_status=on_status,
    )

    # Ctrl+C 종료 처리
    def _shutdown(sig, frame):
        console.print("\n[yellow]종료 중...[/yellow]")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    pipeline.start()
    console.print("[bold green]파이프라인 시작됨. Ctrl+C 로 종료.[/bold green]\n")

    # Rich Live 디스플레이 (0.5초 갱신)
    with Live(make_display(), refresh_per_second=2, console=console) as live:
        while True:
            live.update(make_display())
            threading.Event().wait(0.5)


if __name__ == "__main__":
    main()
