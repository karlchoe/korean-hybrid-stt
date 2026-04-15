# Korean Hybrid STT Pipeline — Architecture

## 개요

말하는 중에는 Qwen3-ASR로 실시간 자막을 표시하고,
발화가 끝나면 Cohere Transcribe로 고정밀 텍스트를 확정하는 2단계 파이프라인.

```
[마이크]
   │ 512샘플(32ms) 청크
   ▼
[Silero VAD]  ── 노이즈 차단, 발화 구간만 통과
   │
   ├── 발화 중 (2초마다) ──► [Qwen3-ASR-0.6B / vLLM]  ──► 실시간 자막 갱신
   │
   └── 발화 종료 ──────────► [Cohere Transcribe 2B]    ──► 최종 텍스트 확정
```

---

## 컴포넌트

### 1. AudioCapture
- 라이브러리: `sounddevice`
- 청크 크기: **512 샘플 고정** (Silero VAD 요구사항)
- 출력: 16-bit PCM bytes → `audio_queue`

### 2. VADProcessor — Silero VAD
- 모델: `snakers4/silero-vad` (torch.hub, ~2MB)
- 입력: 512샘플 PCM → 확률값 0.0~1.0
- `VAD_THRESHOLD(0.5)` 이상이면 말소리 판단
- `SILENCE_DURATION(1.2초)` 침묵 감지 시 발화 종료 신호 발생
- webrtcvad 대비 장점: 노이즈 환경 강함, 짧은 발화 오탐 적음

### 3. Qwen3RealtimeASR — 실시간
- 모델: `Qwen/Qwen3-ASR-0.6B` (Alibaba, Apache 2.0)
- 백엔드: vLLM (OpenAI 호환 API, port 8000)
- 방식: 누적 버퍼를 `STREAM_INTERVAL(2초)` 마다 전송 → 스트리밍 토큰 수신
- Word Boosting: `HOTWORDS` 를 프롬프트에 주입
- RTFx ~2000x

### 4. CohereOfflineASR — 오프라인 확정
- 모델: `CohereLabs/cohere-transcribe-03-2026` (Apache 2.0)
- 백엔드: HuggingFace `transformers` pipeline
- 발화 종료 후 전체 버퍼를 한 번에 전송
- WER 5.42% (Open ASR 리더보드 1위, 2026-03)
- RTFx 524x
- ⚠️ Word Boosting 미지원 → 필요 시 LLM 후처리로 보완

### 5. HybridASRPipeline — 오케스트레이터
- `AudioCapture` → `audio_queue` → VAD 루프 (asyncio)
- 발화 종료 신호 → `utterance_queue` → Cohere 워커 스레드 (별도)
- 콜백 인터페이스:
  - `on_realtime(text)` — 실시간 자막 갱신
  - `on_final(text)` — 최종 텍스트 확정
  - `on_status(msg)` — 상태 메시지

---

## 스레드 모델

```
Main Thread
  └── Rich Live UI (0.5초 갱신)

AsyncIO Thread
  └── _vad_and_stream_loop()
        ├── audio_queue 소비
        ├── Silero VAD 처리
        └── Qwen3 await (스트리밍)

Offline Thread
  └── _offline_worker()
        ├── utterance_queue 소비
        └── Cohere transcribe (동기)

sounddevice Callback Thread
  └── 마이크 → audio_queue 생산
```

---

## 모델별 성능 비교

| 항목 | Qwen3-ASR-0.6B | Cohere Transcribe 2B | VibeVoice-ASR |
|------|:-:|:-:|:-:|
| WER (평균) | ~6% | **5.42%** | 7.77% |
| RTFx | ~2000x | **524x** | 51x |
| 실시간 스트리밍 | **✅** | ❌ | ❌ |
| Word Boosting | **✅** | ❌ | **✅** |
| 타임스탬프 | **✅** | ❌ | **✅** |
| 화자 분리 | ❌ | ❌ | **✅** |
| 한국어 | **✅** | **✅** | **✅** |
| VRAM | ~2GB | ~5GB | ~8GB |

---

## VRAM 요구사항

| 컴포넌트 | VRAM |
|---------|------|
| Silero VAD | 무시 가능 (~10MB) |
| Qwen3-ASR-0.6B (vLLM) | ~2 GB |
| Cohere Transcribe 2B | ~5 GB |
| **합계** | **~7 GB** |

GPU 부족 시 `config.py` → `COHERE_DEVICE = "cpu"` (처리 속도 저하)

---

## 향후 확장 포인트

### 단기
- **Cohere 단 LLM 후처리**: 발화 확정 후 Claude/GPT에 도메인 사전 전달해 전문용어 교정
- **한국어 파인튜닝**: AIHub 데이터셋으로 Cohere 또는 Qwen3 LoRA 파인튜닝

### 중기
- **화자 분리**: pyannote-audio 또는 VibeVoice-ASR 연동
- **타임스탬프 정밀화**: Qwen3-ForcedAligner-0.6B 연동 (word/sentence/paragraph 단위)
- **Logit Boosting**: 디코딩 시 특정 토큰 logit 직접 가중 (Cohere Word Boosting 대안)

### 장기
- **Apple Silicon 지원**: mlx-qwen3-asr (moona3k/mlx-qwen3-asr)
- **화자 적응**: 사용자별 음성 프로파일 등록
- **Shallow Fusion**: 도메인 n-gram LM과 결합

---

## 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. Qwen3 vLLM 서버 (별도 터미널)
vllm serve Qwen/Qwen3-ASR-0.6B --port 8000

# 3. 파이프라인 실행
python main.py
```
