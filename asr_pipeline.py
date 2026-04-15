"""
2단계 STT 파이프라인
  - 실시간 자막: Qwen3-ASR-0.6B (vLLM 스트리밍)
  - 최종 확정:   Cohere Transcribe (오프라인 고정밀)
"""

import asyncio
import threading
import queue
import io
import os
import wave
import base64
import time
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundcard as sc
import torch
from openai import AsyncOpenAI
from transformers import pipeline as hf_pipeline

from config import Config


# ──────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────

def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """16-bit PCM → WAV bytes"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


# ──────────────────────────────────────────────────────────
# 1. 마이크 캡처
# ──────────────────────────────────────────────────────────

class AudioCapture:
    """
    AUDIO_SOURCE = "mic"      → 마이크 입력 (sounddevice)
    AUDIO_SOURCE = "loopback" → 시스템/브라우저 오디오 (soundcard WASAPI loopback)
    """

    CHUNK_SAMPLES = 512  # Silero VAD 고정 요구사항

    def __init__(self, cfg: Config, audio_queue: queue.Queue):
        self.cfg = cfg
        self.queue = audio_queue
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

    def start(self):
        self._stop_event.clear()
        if self.cfg.AUDIO_SOURCE == "loopback":
            self._thread = threading.Thread(target=self._loopback_worker, daemon=True)
            self._thread.start()
        else:
            self._start_mic()

    def stop(self):
        self._stop_event.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()

    # ── 마이크 모드 ────────────────────────────────────────

    def _start_mic(self):
        def _cb(indata, frames, t, status):
            pcm = (indata[:, 0] * 32768).astype(np.int16).tobytes()
            self.queue.put(pcm)

        self._stream = sd.InputStream(
            samplerate=self.cfg.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.CHUNK_SAMPLES,
            callback=_cb,
        )
        self._stream.start()

    # ── 루프백(시스템 오디오) 모드 ─────────────────────────

    def _loopback_worker(self):
        """WASAPI loopback — 브라우저/유튜브 오디오 캡처"""
        default_speaker = sc.default_speaker()
        loopback_mic = sc.get_microphone(
            default_speaker.id, include_loopback=True
        )
        with loopback_mic.recorder(
            samplerate=self.cfg.SAMPLE_RATE,
            channels=1,
            blocksize=self.CHUNK_SAMPLES,
        ) as recorder:
            while not self._stop_event.is_set():
                data = recorder.record(numframes=self.CHUNK_SAMPLES)
                # soundcard → float32 (N, ch), 16-bit PCM으로 변환
                pcm = (data[:, 0] * 32768).astype(np.int16).tobytes()
                self.queue.put(pcm)


# ──────────────────────────────────────────────────────────
# 2. VAD — Silero VAD (딥러닝 기반)
# ──────────────────────────────────────────────────────────

class VADProcessor:
    """
    Silero VAD 기반 발화 구간 감지.

    - 청크 크기: 512 샘플 고정 (16kHz = 32ms)
    - 확률값 THRESHOLD 이상이면 말소리로 판단
    - SILENCE_DURATION 초 이상 침묵이면 발화 종료
    """

    CHUNK_SAMPLES = 512          # Silero VAD 16kHz 요구 사항
    CHUNK_MS      = 32           # 512 / 16000 * 1000

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        self._model.eval()

        self._threshold      = cfg.VAD_THRESHOLD
        self._silence_limit  = int(cfg.SILENCE_DURATION * 1000 / self.CHUNK_MS)
        self._min_frames     = int(cfg.MIN_SPEECH_SEC   * 1000 / self.CHUNK_MS)

        self._buf: list[bytes] = []
        self._silence  = 0
        self._speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def get_buffer(self) -> bytes:
        return b"".join(self._buf)

    def _speech_prob(self, chunk: bytes) -> float:
        """PCM bytes → Silero VAD 확률값 (0.0 ~ 1.0)"""
        audio = torch.from_numpy(
            np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        )
        with torch.no_grad():
            prob = self._model(audio, self.cfg.SAMPLE_RATE).item()
        return prob

    def process(self, chunk: bytes) -> tuple[bool, Optional[bytes]]:
        """
        Returns:
          (is_speaking, completed_utterance_pcm | None)
        """
        prob     = self._speech_prob(chunk)
        is_speech = prob >= self._threshold

        if is_speech:
            self._buf.append(chunk)
            self._silence  = 0
            self._speaking = True
            return True, None

        if self._speaking:
            self._buf.append(chunk)   # 후행 묵음 포함 (자연스러운 끝처리)
            self._silence += 1
            if self._silence >= self._silence_limit:
                utterance      = b"".join(self._buf)
                self._buf      = []
                self._silence  = 0
                self._speaking = False

                total_frames = len(utterance) // (2 * self.CHUNK_SAMPLES)
                if total_frames < self._min_frames:
                    return False, None  # 너무 짧으면 버림
                return False, utterance

        return False, None


# ──────────────────────────────────────────────────────────
# 3. 실시간 ASR — Qwen3-ASR (vLLM)
# ──────────────────────────────────────────────────────────

class Qwen3RealtimeASR:
    def __init__(self, cfg: Config):
        self.client = AsyncOpenAI(base_url=cfg.QWEN3_BASE_URL, api_key="EMPTY")
        self.model = cfg.QWEN3_MODEL
        self.language = cfg.LANGUAGE
        self.hotwords = cfg.HOTWORDS

    async def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""

        wav = pcm_to_wav(pcm, 16000)
        audio_b64 = base64.b64encode(wav).decode()

        prompt = "Transcribe the audio."
        if self.hotwords:
            prompt += f" Key terms: {self.hotwords}"
        if self.language:
            prompt += f" Language: {self.language}."

        try:
            result = ""
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                stream=True,
                max_tokens=512,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    result += delta
            return result.strip()
        except Exception as e:
            return f"[Qwen3 오류: {e}]"


# ──────────────────────────────────────────────────────────
# 4. 오프라인 ASR — Cohere Transcribe
# ──────────────────────────────────────────────────────────

class CohereOfflineASR:
    def __init__(self, cfg: Config):
        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=cfg.COHERE_MODEL,
            device=cfg.COHERE_DEVICE,
        )
        self.language = cfg.LANGUAGE

    def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        audio = pcm_to_float32(pcm)
        try:
            result = self._pipe(
                audio,
                generate_kwargs={"language": self.language},
            )
            return result["text"].strip()
        except Exception as e:
            return f"[Cohere 오류: {e}]"


# ──────────────────────────────────────────────────────────
# 5. 파일 저장 및 오류 로깅
# ──────────────────────────────────────────────────────────

class TranscriptWriter:
    """
    확정 텍스트를 OUTPUT_FILE 에 append 저장.
    오류/미인식은 OUTPUT_FILE.err.txt 에 기록.

    오류 판단 기준:
      - 텍스트가 비어 있거나 너무 짧음 (< 2자)
      - 모델 오류 메시지 ([Qwen3 오류] / [Cohere 오류])
      - 반복 hallucination (동일 단어가 전체의 70% 이상)
    """

    MIN_LENGTH = 2

    def __init__(self, cfg: Config):
        base, ext = os.path.splitext(cfg.OUTPUT_FILE)
        ext = ext or ".txt"
        self.output_file = cfg.OUTPUT_FILE
        self.error_file  = f"{base}.err{ext}"

    def write(self, text: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        is_err, reason = self._check_error(text)

        if is_err:
            self._append(self.error_file, f"[{ts}] [{reason}] {text}\n")
        else:
            self._append(self.output_file, f"[{ts}] {text}\n")

        return is_err, reason

    # ── 내부 ──────────────────────────────────────────────

    def _check_error(self, text: str) -> tuple[bool, str]:
        stripped = text.strip()

        if len(stripped) < self.MIN_LENGTH:
            return True, "EMPTY"

        if stripped.startswith("[") and "오류" in stripped:
            return True, "MODEL_ERROR"

        # Hallucination: 단어 중복 비율 70% 초과
        words = stripped.split()
        if len(words) >= 6:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return True, "HALLUCINATION"

        return False, ""

    @staticmethod
    def _append(path: str, line: str):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


# ──────────────────────────────────────────────────────────
# 6. 메인 파이프라인 오케스트레이터
# ──────────────────────────────────────────────────────────

class HybridASRPipeline:
    """
    실시간(Qwen3) + 오프라인 확정(Cohere)을 동시에 운용하는 파이프라인.

    콜백:
      on_realtime(text)        — 말하는 중 주기적으로 호출
      on_final(text, err, why) — 발화 종료 후 Cohere 확정 결과 호출
                                 err=True 이면 오류 발화 (why: 사유)
      on_status(msg)           — 상태 메시지 (말하는 중 / 처리 중 등)
    """

    def __init__(
        self,
        cfg: Config,
        on_realtime: Callable[[str], None],
        on_final: Callable[[str, bool, str], None],
        on_status: Callable[[str], None],
    ):
        self.cfg = cfg
        self.on_realtime = on_realtime
        self.on_final = on_final
        self.on_status = on_status

        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self._utterance_q: queue.Queue[bytes] = queue.Queue()

        self._capture = AudioCapture(cfg, self._audio_q)
        self._vad = VADProcessor(cfg)
        self._qwen3 = Qwen3RealtimeASR(cfg)
        self._cohere = CohereOfflineASR(cfg)
        self._writer = TranscriptWriter(cfg)

        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── 시작 / 종료 ────────────────────────────────────

    def start(self):
        self._running = True

        # 오프라인 Cohere 처리용 스레드 (별도)
        self._offline_thread = threading.Thread(
            target=self._offline_worker, daemon=True
        )
        self._offline_thread.start()

        # asyncio 루프 — VAD + Qwen3 스트리밍
        self._loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._run_loop, daemon=True
        )
        self._async_thread.start()

        # 마이크 캡처 시작
        self._capture.start()

    def stop(self):
        self._running = False
        self._capture.stop()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._vad_and_stream_loop())

    # ── VAD + 실시간 스트리밍 루프 ────────────────────

    async def _vad_and_stream_loop(self):
        last_stream_time = 0.0

        while self._running:
            try:
                chunk = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            is_speaking, utterance = self._vad.process(chunk)

            now = time.monotonic()

            # 말하는 중: 일정 주기로 Qwen3에 현재 버퍼 전송
            if is_speaking and (now - last_stream_time) >= self.cfg.STREAM_INTERVAL:
                buf = self._vad.get_buffer()
                if buf:
                    self.on_status("🎙 말하는 중...")
                    text = await self._qwen3.transcribe(buf)
                    if text:
                        self.on_realtime(text)
                    last_stream_time = now

            # 발화 종료: Cohere 오프라인 큐에 넣기
            if utterance is not None:
                self.on_status("⏳ Cohere 분석 중...")
                self._utterance_q.put(utterance)
                last_stream_time = 0.0  # 다음 발화 리셋

    # ── 오프라인 Cohere 워커 ───────────────────────────

    def _offline_worker(self):
        while self._running:
            try:
                utterance = self._utterance_q.get(timeout=0.5)
            except queue.Empty:
                continue

            text = self._cohere.transcribe(utterance)
            is_err, reason = self._writer.write(text)
            self.on_final(text, is_err, reason)
            self.on_status("✅ 대기 중")
