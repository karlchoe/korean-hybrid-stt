"""
실행: python main.py
사전 준비:
  1) pip install -r requirements.txt
  2) vLLM 서버: vllm serve Qwen/Qwen3-ASR-0.6B --port 8000
  3) config.py: AUDIO_SOURCE="loopback", OUTPUT_FILE 등 설정

조작:
  - 드래그: 창 위치 이동
  - 더블클릭: 종료
  - Ctrl+C (터미널): 종료
"""

import sys
import signal
import threading
from typing import Optional

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QObject, QPoint, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush

from config import Config
from asr_pipeline import HybridASRPipeline


# ──────────────────────────────────────────────────────────
# Signal Bridge — 파이프라인 스레드 → Qt UI 스레드 (스레드 안전)
# ──────────────────────────────────────────────────────────

class SignalBridge(QObject):
    realtime = pyqtSignal(str)
    final    = pyqtSignal(str, bool, str)   # text, is_err, reason
    status   = pyqtSignal(str)


# ──────────────────────────────────────────────────────────
# PyQt6 오버레이 창
# ──────────────────────────────────────────────────────────

class OverlayWindow(QWidget):
    MAX_FINALS = 5      # 표시할 최근 확정 텍스트 수
    BG_COLOR   = QColor(18, 18, 18, 210)   # 반투명 배경
    BORDER_COLOR = QColor(70, 70, 70, 255)

    def __init__(self, cfg: Config, bridge: SignalBridge):
        super().__init__()
        self.cfg = cfg
        self._drag_pos: Optional[QPoint] = None
        self._finals: list[tuple[str, bool]] = []

        self._init_window()
        self._init_ui()
        self._connect(bridge)

    # ── 초기화 ────────────────────────────────────────────

    def _init_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool          # 작업 표시줄에 표시 안 함
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(620, 270)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        # ── 상태바 ──
        self._lbl_status = QLabel("⏳ 초기화 중...")
        self._lbl_status.setStyleSheet(
            "color: #f0c040; font-size: 12px; font-weight: bold;"
        )

        # ── 실시간 자막 (Qwen3) ──
        self._lbl_realtime = QLabel("말씀하세요...")
        self._lbl_realtime.setWordWrap(True)
        self._lbl_realtime.setStyleSheet(
            "color: #3dd6c8; font-size: 16px;"
        )
        self._lbl_realtime.setMinimumHeight(40)

        # ── 구분선 ──
        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #404040;")

        # ── 확정 텍스트 (Cohere) ──
        self._lbl_final = QLabel("")
        self._lbl_final.setWordWrap(True)
        self._lbl_final.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self._lbl_final.setStyleSheet(
            "color: #e8e8e8; font-size: 14px; line-height: 160%;"
        )
        self._lbl_final.setMinimumHeight(80)

        # ── 파일 상태 ──
        base = self.cfg.OUTPUT_FILE.rsplit(".", 1)[0]
        self._lbl_file = QLabel(
            f"📄  {self.cfg.OUTPUT_FILE}   |   ⚠  {base}.err.txt"
            f"   |   ✕ 더블클릭 종료"
        )
        self._lbl_file.setStyleSheet("color: #606060; font-size: 11px;")

        layout.addWidget(self._lbl_status)
        layout.addWidget(self._lbl_realtime)
        layout.addWidget(sep)
        layout.addWidget(self._lbl_final)
        layout.addStretch()
        layout.addWidget(self._lbl_file)

    def _connect(self, bridge: SignalBridge):
        bridge.realtime.connect(self._on_realtime)
        bridge.final.connect(self._on_final)
        bridge.status.connect(self._on_status)

    # ── 시그널 핸들러 ──────────────────────────────────────

    def _on_status(self, msg: str):
        self._lbl_status.setText(msg)

    def _on_realtime(self, text: str):
        self._lbl_realtime.setText(text or "...")

    def _on_final(self, text: str, is_err: bool, reason: str):
        self._lbl_realtime.setText("...")
        self._finals.append((text, is_err))
        if len(self._finals) > self.MAX_FINALS:
            self._finals.pop(0)
        self._refresh_finals()

    def _refresh_finals(self):
        parts = []
        for text, is_err in self._finals:
            if is_err:
                parts.append(f'<span style="color:#ff6060;">{text}</span>')
            else:
                parts.append(text)
        self._lbl_final.setText("<br>".join(parts))

    # ── 반투명 배경 ────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(self.BG_COLOR))
        p.setPen(QPen(self.BORDER_COLOR, 1))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 10, 10)

    # ── 드래그 이동 ────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event):
        QApplication.quit()


# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────

def main():
    cfg = Config()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    bridge = SignalBridge()
    window = OverlayWindow(cfg, bridge)

    # 화면 우측 하단에 초기 배치
    screen = app.primaryScreen().availableGeometry()
    window.move(
        screen.width()  - window.width()  - 20,
        screen.height() - window.height() - 60,
    )
    window.show()

    # 파이프라인 콜백 → Signal emit (스레드 안전)
    pipeline = HybridASRPipeline(
        cfg=cfg,
        on_realtime=lambda text:         bridge.realtime.emit(text),
        on_final=lambda text, err, why:  bridge.final.emit(text, err, why),
        on_status=lambda msg:            bridge.status.emit(msg),
    )

    # 모델 로딩을 백그라운드 스레드에서 수행 (UI 블로킹 방지)
    def _start():
        bridge.status.emit("⏳ Cohere 모델 로딩 중... (최초 1회)")
        pipeline.start()
        bridge.status.emit("✅ 대기 중")

    threading.Thread(target=_start, daemon=True).start()

    # Ctrl+C → Qt 종료
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # Qt 이벤트 루프가 SIGINT를 처리하도록 200ms 타이머 유지
    heartbeat = QTimer()
    heartbeat.start(200)
    heartbeat.timeout.connect(lambda: None)

    ret = app.exec()
    pipeline.stop()
    sys.exit(ret)


if __name__ == "__main__":
    main()
