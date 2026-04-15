# CLAUDE.md — Korean Hybrid STT Pipeline

## 프로젝트 개요

한국어 특화 2단계 STT 파이프라인.
- **오디오 소스**: 브라우저/YouTube (WASAPI 루프백) 또는 마이크
- **실시간 자막**: Qwen3-ASR-0.6B (vLLM 스트리밍)
- **최종 확정**: Cohere Transcribe (오프라인 고정밀)
- **VAD**: Silero VAD (딥러닝 기반 노이즈 필터링)
- **파일 저장**: 확정 텍스트 → `transcript.txt` / 오류 → `transcript.err.txt`

자세한 구조는 `architecture.md` 참고.

---

## 파일 구조

```
stt/
├── config.py          # 모든 설정값 (여기만 수정)
├── asr_pipeline.py    # 핵심 로직 — 컴포넌트 6개
│   ├── AudioCapture       마이크 or 루프백 → PCM 청크
│   ├── VADProcessor       Silero VAD 발화 감지
│   ├── Qwen3RealtimeASR   vLLM 스트리밍 전사
│   ├── CohereOfflineASR   HuggingFace 오프라인 전사
│   ├── TranscriptWriter   파일 저장 + 오류 로깅
│   └── HybridASRPipeline  오케스트레이터
├── main.py            # 진입점 + Rich 터미널 UI
├── requirements.txt   # 의존성
└── architecture.md    # 아키텍처 문서
```

---

## 설정 (config.py)

모든 튜닝은 `config.py` 에서 수행. 코드 수정 불필요.

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `AUDIO_SOURCE` | `"loopback"` | `"loopback"` = 브라우저 오디오, `"mic"` = 마이크 |
| `VAD_THRESHOLD` | `0.5` | Silero VAD 민감도. 소음 심하면 `0.7` |
| `SILENCE_DURATION` | `1.2` | 발화 종료 판단 침묵(초). 빠른 말은 `0.8` |
| `STREAM_INTERVAL` | `2.0` | 실시간 자막 갱신 주기(초) |
| `HOTWORDS` | `""` | Qwen3 핫워드. 예: `"삼성전자, HBM3E"` |
| `LANGUAGE` | `"ko"` | 인식 언어 코드 |
| `COHERE_DEVICE` | `"cuda"` | GPU 없으면 `"cpu"` |
| `OUTPUT_FILE` | `"transcript.txt"` | 확정 텍스트 저장 파일 (append) |

오류 로그 파일은 `OUTPUT_FILE` 에서 자동 생성: `transcript.err.txt`

---

## 의존성 및 실행 환경

```bash
pip install -r requirements.txt

# Qwen3 vLLM 서버 필수 (별도 터미널)
vllm serve Qwen/Qwen3-ASR-0.6B --port 8000

# 실행
python main.py
```

- Python 3.10+
- CUDA GPU 권장 (VRAM ~7GB: Qwen3 2GB + Cohere 5GB)
- Silero VAD 는 torch.hub 에서 자동 다운로드 (최초 1회)
- Cohere 모델은 HuggingFace 에서 자동 다운로드 (최초 1회, ~4GB)
- **Windows 전용**: `soundcard` WASAPI 루프백은 Windows에서만 동작

---

## 코드 수정 가이드

### 오디오 소스 전환
- `config.py` → `AUDIO_SOURCE = "loopback"` or `"mic"`
- 루프백: `asr_pipeline.py` → `AudioCapture._loopback_worker()` (soundcard WASAPI)
- 마이크: `asr_pipeline.py` → `AudioCapture._start_mic()` (sounddevice)

### VAD 교체 / 튜닝
- `asr_pipeline.py` → `VADProcessor` 클래스
- Silero VAD 청크 크기는 **512 샘플 고정** (변경 불가)
- `AudioCapture.CHUNK_SAMPLES = 512` 과 반드시 일치해야 함

### 실시간 ASR 교체
- `asr_pipeline.py` → `Qwen3RealtimeASR` 클래스
- vLLM OpenAI 호환 API 사용 (`/v1/chat/completions`)
- 오디오는 WAV bytes → base64 인코딩 후 `input_audio` 타입으로 전달

### 오프라인 ASR 교체
- `asr_pipeline.py` → `CohereOfflineASR` 클래스
- HuggingFace `pipeline("automatic-speech-recognition")` 인터페이스
- 입력: float32 numpy array ([-1.0, 1.0] 정규화)

### 파일 저장 / 오류 기준 수정
- `asr_pipeline.py` → `TranscriptWriter._check_error()` 메서드
- 현재 오류 판단: `EMPTY`(2자 미만) / `MODEL_ERROR` / `HALLUCINATION`(단어 반복 70%)
- `MIN_LENGTH` 클래스 변수로 최소 길이 조정 가능

### UI 수정
- `main.py` → `make_display()` 함수
- Rich `Layout` + `Panel` 구조
- 오류 발화는 빨간색으로 표시
- `on_realtime` / `on_final(text, is_err, reason)` / `on_status` 콜백으로 상태 수신

---

## 알려진 제약사항

1. **Cohere Word Boosting 미지원**
   - Qwen3 실시간 단계에서 핫워드 적용 → 최종 확정 단계에서 누락 가능
   - 해결책: `CohereOfflineASR.transcribe()` 이후 LLM 후처리 추가

2. **Cohere 타임스탬프 미지원**
   - 타임스탬프 필요 시 Qwen3-ForcedAligner-0.6B 별도 연동

3. **스트리밍 모드에서 Qwen3 타임스탬프 불가**
   - 오프라인 모드(`ForcedAligner`)에서만 타임스탬프 지원

4. **Cohere 언어 자동 감지 없음**
   - `LANGUAGE` 코드 반드시 명시 필요

5. **루프백 캡처 Windows 전용**
   - macOS/Linux는 `AUDIO_SOURCE = "mic"` 사용 또는 BlackHole(mac) 등 가상 오디오 드라이버 필요

---

## 화자 분리 추가 시 대안

현재 파이프라인은 화자 분리를 지원하지 않음. 추가 필요 시 아래 중 선택:

| 라이브러리 | 방식 | 특징 |
|-----------|------|------|
| **pyannote.audio** | `HybridASRPipeline` 내 후처리로 연동 | HuggingFace 모델, 높은 정확도, 상용 라이선스 확인 필요 |
| **VibeVoice-ASR** | Cohere 대신 전체 교체 | ASR + 화자분리 통합, WER 7.77% |
| **NeMo MSDD** | 별도 파이프라인으로 연동 | NVIDIA NeMo, 대화 시나리오 특화 |
