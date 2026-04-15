class Config:
    # ── 오디오 입력 소스 ─────────────────────────────
    SAMPLE_RATE  = 16000
    AUDIO_SOURCE = "loopback"  # "loopback" = 브라우저/시스템 오디오
                               # "mic"      = 마이크 직접 입력

    # ── VAD (Silero VAD) ─────────────────────────────
    VAD_THRESHOLD    = 0.5  # 말소리 판단 확률 임계값 (0.0~1.0, 높을수록 엄격)
    SILENCE_DURATION = 1.2  # 이 시간(초) 침묵이면 발화 종료로 판단
    MIN_SPEECH_SEC   = 0.4  # 이것보다 짧은 발화는 무시

    # ── 실시간: Qwen3-ASR via vLLM ──────────────────
    QWEN3_BASE_URL  = "http://localhost:8000/v1"
    QWEN3_MODEL     = "Qwen/Qwen3-ASR-0.6B"
    STREAM_INTERVAL = 2.0   # 실시간 갱신 주기 (초)

    # ── 오프라인: Cohere Transcribe ──────────────────
    COHERE_MODEL  = "CohereLabs/cohere-transcribe-03-2026"
    COHERE_DEVICE = "cuda"  # GPU 없으면 "cpu"

    # ── 언어 / 핫워드 ────────────────────────────────
    LANGUAGE = "ko"
    HOTWORDS = ""           # 예: "삼성전자, 갤럭시 S25, HBM3E"

    # ── 출력 파일 ────────────────────────────────────
    OUTPUT_FILE = "transcript.txt"
    # 오류 로그는 자동 생성: transcript.err.txt
