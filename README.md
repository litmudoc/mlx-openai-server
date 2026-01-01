# mlx-openai-server

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

> 이 프로젝트는 [cubist38/mlx-openai-server](https://github.com/cubist38/mlx-openai-server) 저장소를 fork하여, 언어 모델 기능의 개선을 진행합니다. 실험적이며 안정적이지 않을 수 있습니다. 원본은 링크의 저장소를 사용하세요.

## 설명 (Description)
이 저장소는 MLX 모델을 위한 OpenAI 호환 엔드포인트를 제공하는 고성능 API 서버를 호스팅합니다. Python으로 개발되고 FastAPI 프레임워크를 기반으로 하는 이 서버는 OpenAI 호환 인터페이스를 통해 로컬에서 MLX 기반 멀티모달 모델을 실행할 수 있는 효율적이고 확장 가능하며 사용자 친화적인 솔루션을 제공합니다. 이 서버는 향상된 Flux 시리즈 모델 지원과 함께 텍스트, 비전, 오디오 처리 및 이미지 생성 기능을 지원합니다.

> **참고:** 이 프로젝트는 Apple Silicon에 최적화된 Apple의 프레임워크인 MLX를 활용하므로 현재 **M 시리즈 칩이 탑재된 MacOS**만 지원합니다.

## 목차 (Table of Contents)
- [주요 기능 (Key Features)](#주요-기능-key-features)
- [OpenAI 호환성 (OpenAI Compatibility)](#openai-호환성-openai-compatibility)
- [지원되는 모델 유형 (Supported Model Types)](#지원되는-모델-유형-supported-model-types)
- [설치 (Installation)](#설치-installation)
- [사용법 (Usage)](#사용법-usage)
  - [서버 시작 (Starting the Server)](#서버-시작-starting-the-server)
  - [CLI 사용법 (CLI Usage)](#cli-사용법-cli-usage)
  - [로깅 구성 (Logging Configuration)](#로깅-구성-logging-configuration)
  - [API 사용 (Using the API)](#api-사용-using-the-api)
  - [구조화된 출력 (Structured Outputs)](#json-스키마를-이용한-구조화된-출력-structured-outputs-with-json-schema)
- [요청 대기열 시스템 (Request Queue System)](#요청-대기열-시스템-request-queue-system)
- [API 응답 스키마 (API Response Schemas)](#api-응답-스키마-api-response-schemas)
- [예제 노트북 (Example Notebooks)](#예제-노트북-example-notebooks)
- [대규모 모델 (Large Models)](#대규모-모델-large-models)
- [라이선스 (License)](#라이선스-license)

---

## 주요 기능 (Key Features)
- 🚀 MLX 모델을 위한 **빠르고 로컬에서 실행되는 OpenAI 호환 API**
- 🖼️ 비전, 오디오 및 텍스트를 포함한 **멀티모달 모델 지원**
- 🎨 MLX Flux 시리즈 모델(schnell, dev, Krea-dev, kontext)을 사용한 **고급 이미지 생성 및 편집**
- 🔌 앱에서 OpenAI API를 대체할 수 있는 **드롭인(Drop-in) 교체**
- 📈 **성능 및 대기열 모니터링 엔드포인트**
- 🧑‍💻 **쉬운 Python 및 CLI 사용**
- 🛡️ **견고한 오류 처리 및 요청 관리**
- 🎛️ 미세 조정된 이미지 생성 및 편집을 위한 **LoRA 어댑터 지원**
- ⚡ 최적의 성능을 위한 **구성 가능한 양자화** (4-bit, 8-bit, 16-bit)
- 🧠 메모리 최적화 및 성능 조정을 위한 **사용자 정의 가능한 컨텍스트 길이**

## OpenAI 호환성 (OpenAI Compatibility)

이 서버는 OpenAI API 인터페이스를 구현하여 애플리케이션에서 OpenAI 서비스의 드롭인 대체품으로 사용할 수 있습니다. 다음을 지원합니다:
- 채팅 완성 (스트리밍 및 비 스트리밍 모두)
- 멀티모달 상호 작용 (텍스트, 이미지 및 오디오)
- Flux 시리즈 모델을 사용한 고급 이미지 생성 및 편집
- 임베딩 생성
- 함수 호출(Function calling) 및 도구 사용(Tool use)
- 표준 OpenAI 요청/응답 형식
- 일반적인 OpenAI 매개변수 (temperature, top_p 등)

## 지원되는 모델 유형 (Supported Model Types)

서버는 6가지 유형의 MLX 모델을 지원합니다:

1. **텍스트 전용 모델** (`--model-type lm`) - 순수 언어 모델을 위해 `mlx-lm` 라이브러리를 사용합니다.
2. **멀티모달 모델** (`--model-type multimodal`) - 텍스트, 이미지 및 오디오를 처리할 수 있는 멀티모달 모델을 위해 `mlx-vlm` 라이브러리를 사용합니다.
3. **이미지 생성 모델** (`--model-type image-generation`) - 향상된 구성의 Flux 시리즈 이미지 생성 모델을 위해 `mflux` 라이브러리를 사용합니다.
4. **이미지 편집 모델** (`--model-type image-edit`) - Flux 시리즈 이미지 편집 모델을 위해 `mflux` 라이브러리를 사용합니다.
5. **임베딩 모델** (`--model-type embeddings`) - 최적화된 메모리 관리를 통한 텍스트 임베딩 생성을 위해 `mlx-embeddings` 라이브러리를 사용합니다.
6. **Whisper 모델** (`--model-type whisper`) - 오디오 전사 및 음성 인식을 위해 `mlx-whisper` 라이브러리를 사용합니다. ⚠️ *ffmpeg 설치가 필요합니다.*

### Whisper 모델

> **⚠️ 참고:** Whisper 모델은 오디오 처리를 위해 ffmpeg가 설치되어 있어야 합니다: `brew install ffmpeg`

### Flux 시리즈 이미지 모델

서버는 고급 이미지 생성 및 편집을 위해 여러 Flux 및 Qwen 모델 구성을 지원합니다:

#### 이미지 생성 모델
- **`flux-schnell`** - 4개의 기본 단계로 빠른 생성, 가이던스 없음 (빠른 반복 작업에 최적)
- **`flux-dev`** - 25개의 기본 단계, 3.5 가이던스로 고품질 생성 (품질/속도 균형)
- **`flux-krea-dev`** - 28개의 기본 단계, 4.5 가이던스로 프리미엄 품질 (최고 품질)
- **`qwen-image`** - 50개의 기본 단계, 4.0 가이던스의 Qwen 이미지 생성 모델 (고품질 Qwen 기반 생성)
- **`z-image-turbo`** - 빠른 이미지 생성을 위한 Z-Image Turbo 모델
- **`fibo`** - 이미지 생성을 위한 Fibo 모델

#### 이미지 편집 모델
- **`flux-kontext-dev`** - 28개의 기본 단계, 2.5 가이던스로 문맥 인식 편집 (문맥 이미지 편집에 특화)
- **`qwen-image-edit`** - 50개의 기본 단계, 4.0 가이던스의 Qwen 이미지 편집 모델 (고품질 Qwen 기반 편집)

각 구성은 다음을 지원합니다:
- **양자화 수준**: 메모리/성능 최적화를 위한 4-bit, 8-bit 또는 16-bit
- **LoRA 어댑터**: 미세 조정된 생성 및 편집을 위한 사용자 정의 스케일링이 포함된 다중 LoRA 경로 (모든 Flux 및 Qwen 이미지 모델 지원).
- **사용자 정의 매개변수**: 단계(Steps), 가이던스(Guidance), 네거티브 프롬프트(Negative prompts) 등

### 컨텍스트 길이 구성 (Context Length Configuration)

서버는 메모리 사용량과 성능을 최적화하기 위해 언어 모델에 대한 사용자 정의 가능한 컨텍스트 길이를 지원합니다:

- **기본 동작**: `--context-length`가 지정되지 않으면 서버는 모델의 기본 컨텍스트 길이를 사용합니다.
- **메모리 최적화**: 더 작은 컨텍스트 길이를 설정하면 특히 대형 모델의 경우 메모리 사용량을 크게 줄일 수 있습니다.
- **성능 조정**: 특정 사용 사례 및 사용 가능한 시스템 리소스에 따라 컨텍스트 길이를 조정합니다.
- **지원되는 모델**: 컨텍스트 길이 구성은 텍스트 전용(`lm`) 및 멀티모달(`multimodal`) 모델 유형 모두에서 작동합니다.
- **프롬프트 캐싱**: 서버는 컨텍스트 길이가 지정될 때 메모리 사용을 최적화하기 위해 프롬프트 캐싱을 사용합니다.

**사용 사례 예:**
- **짧은 대화**: 채팅 애플리케이션에는 더 작은 컨텍스트 길이(예: 2048, 4096)를 사용하세요.
- **문서 처리**: 긴 문서 분석에는 더 큰 컨텍스트 길이(예: 8192, 16384)를 사용하세요.
- **메모리가 제한된 시스템**: 제한된 RAM에 더 큰 모델을 맞추려면 컨텍스트 길이를 줄이세요.

### 사용자 정의 채팅 템플릿 (Custom Chat Templates)

서버는 언어 모델(`lm`) 및 멀티모달 모델(`multimodal`)을 위한 사용자 정의 채팅 템플릿을 지원합니다. 채팅 템플릿은 대화 메시지가 모델로 전송되기 전에 형식이 지정되는 방식을 정의합니다.

**기능:**
- **사용자 정의 서식**: 모델의 기본 채팅 템플릿을 사용자 정의 Jinja2 템플릿으로 덮어씁니다.
- **모델 호환성**: 텍스트 전용 및 멀티모달 모델 모두에서 작동합니다.
- **파일 기반 구성**: 서버를 시작할 때 `.jinja` 템플릿 파일의 경로를 지정합니다.

**사용법:**
```bash
# 사용자 정의 채팅 템플릿으로 서버 시작
python -m app.main \
  --model-path <path-to-model> \
  --model-type lm \
  --chat-template-file /path/to/custom_template.jinja
```

**템플릿 파일 형식:**
채팅 템플릿은 Jinja2 구문을 사용하며 토크나이저/프로세서가 예상하는 표준 형식을 따라야 합니다. 템플릿은 대화 기록이 포함된 `messages` 변수를 받습니다.

**사용 사례 예:**
- **사용자 정의 프롬프트 서식**: 시스템 프롬프트, 사용자 메시지 및 어시스턴트 응답 형식을 수정합니다.
- **모델별 요구 사항**: 특정 서식이 필요한 모델에 맞게 템플릿을 조정합니다.
- **파인 튜닝 호환성**: 파인 튜닝 데이터 형식과 일치하는 템플릿을 사용합니다.

> **참고:** 채팅 템플릿 파일이 존재하지 않으면 서버 시작 중에 오류가 발생합니다. 파일 경로가 올바르고 파일에 액세스할 수 있는지 확인하세요.

## 설치 (Installation)

MLX 기반 서버를 설정하려면 다음 단계를 따르세요:

### 전제 조건
- Apple Silicon (M 시리즈) 칩이 탑재된 MacOS
- Python 3.11 (기본 ARM 버전)
- pip 패키지 관리자

### 설정 단계
1. 프로젝트를 위한 가상 환경을 생성합니다:
    ```bash
    python3.11 -m venv mlx-openai-server
    ```

2. 가상 환경을 활성화합니다:
    ```bash
    source mlx-openai-server/bin/activate
    ```

3. 패키지를 설치합니다:
    ```bash
    # 옵션 1: GitHub에서 직접 설치
    pip install git+https://github.com/akirose/mlx-openai-server.git
    
    # 옵션 2: 복제 후 개발 모드로 설치
    git clone https://github.com/akirose/mlx-openai-server.git
    cd mlx-openai-server
    pip install -e .
    ```

### Conda 사용 (권장)

더 나은 환경 관리와 아키텍처 문제를 피하기 위해 conda 사용을 권장합니다:

1. **Conda 설치** (아직 설치되지 않은 경우):
    ```bash
    mkdir -p ~/miniconda3
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda init --all
    ```

2. Python 3.11로 **새 conda 환경 생성**:
    ```bash
    conda create -n mlx-server python=3.11
    conda activate mlx-server
    ```

3. **패키지 설치**:
    ```bash
    # 옵션 1: GitHub에서 직접 설치
    pip install git+https://github.com/akirose/mlx-openai-server.git
    
    # 옵션 2: 복제 후 개발 모드로 설치
    git clone https://github.com/akirose/mlx-openai-server.git
    cd mlx-openai-server
    pip install -e .
    ```

### 선택적 종속성 (Optional Dependencies)

서버는 향상된 기능을 위해 선택적 종속성을 지원합니다:

#### 기본 설치
```bash
pip install git+https://github.com/akirose/mlx-openai-server.git
```
**포함 내용:**
- 텍스트 전용 언어 모델 (`--model-type lm`)
- 멀티모달 모델 (`--model-type multimodal`) 
- 임베딩 모델 (`--model-type embeddings`)
- 모든 핵심 API 엔드포인트 및 기능

#### 이미지 생성 및 편집 지원
서버는 이미지 생성 및 편집 기능을 지원합니다:

**추가 기능:**
- 이미지 생성 모델 (`--model-type image-generation`)
- 이미지 편집 모델 (`--model-type image-edit`)
- MLX Flux 시리즈 모델 지원
- Qwen Image 모델 지원
- 미세 조정된 생성 및 편집을 위한 LoRA 어댑터 지원

#### Whisper 모델 지원
Whisper 모델이 제대로 작동하려면 ffmpeg를 설치해야 합니다:

```bash
# Homebrew를 사용하여 ffmpeg 설치
brew install ffmpeg
```

**ffmpeg 포함 기능:**
- 오디오 전사 모델 (`--model-type whisper`)
- 음성 인식 기능
- 다양한 오디오 형식 지원 (WAV, MP3, M4A 등)

> **참고:** Whisper 모델은 오디오 처리를 위해 ffmpeg가 필요합니다. whisper 모델 유형을 사용하기 전에 설치했는지 확인하세요.

### 문제 해결
**문제:** OS 및 Python 버전이 요구 사항을 충족하지만 `pip`가 일치하는 배포판을 찾을 수 없습니다.

**원인:** 네이티브가 아닌 Python 버전을 사용하고 있을 수 있습니다. 다음 명령을 실행하여 확인하세요:
```bash
python -c "import platform; print(platform.processor())"
```
출력이 `i386` (M 시리즈 머신에서)이면 네이티브가 아닌 Python을 사용하고 있는 것입니다. 네이티브 Python 버전으로 전환하세요. 좋은 방법은 [Conda](https://stackoverflow.com/questions/65415996/how-to-specify-the-architecture-or-platform-for-a-new-conda-environment-apple)를 사용하는 것입니다.

## 사용법 (Usage)

### 서버 시작 (Starting the Server)

Python 모듈 또는 CLI 명령을 사용하여 MLX 서버를 시작할 수 있습니다. 두 방법 모두 로깅 구성 옵션을 포함한 동일한 매개변수를 지원합니다.

#### 방법 1: Python 모듈
```bash
# 텍스트 전용 또는 멀티모달 모델의 경우
python -m app.main \
  --model-path <path-to-mlx-model> \
  --model-type <lm|multimodal> \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100

# 이미지 생성 모델의 경우 (Flux 시리즈, Qwen, Z-Image Turbo 또는 Fibo)
python -m app.main \
  --model-type image-generation \
  --model-path <path-to-local-model> \
  --config-name <flux-schnell|flux-dev|flux-krea-dev|qwen-image|z-image-turbo|fibo> \
  --quantize <4|8|16> \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100

# 이미지 편집 모델의 경우 (Flux 시리즈 또는 Qwen)
python -m app.main \
  --model-type image-edit \
  --model-path <path-to-local-model> \
  --config-name <flux-kontext-dev|qwen-image-edit> \
  --quantize <4|8|16> \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100

# 임베딩 모델의 경우
python -m app.main \
  --model-type embeddings \
  --model-path <embeddings-model-path> \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100

# Whisper 모델의 경우
python -m app.main \
  --model-type whisper \
  --model-path <whisper-model-path> \
  --max-concurrency 1 \
  --queue-timeout 600 \
  --queue-size 50

# 로깅 구성 옵션 포함
python -m app.main \
  --model-path <path-to-mlx-model> \
  --model-type lm \
  --no-log-file \
  --log-level INFO

python -m app.main \
  --model-path <path-to-mlx-model> \
  --model-type lm \
  --log-file /tmp/custom.log \
  --log-level DEBUG
```

#### 서버 매개변수
- `--model-path`: MLX 모델 디렉터리 경로(로컬 경로 또는 Hugging Face 모델 저장소). `lm`, `multimodal`, `embeddings`, `image-generation`, `image-edit`, `whisper` 모델 유형에 필요합니다.
- `--model-type`: 실행할 모델 유형:
  - 텍스트 전용 모델의 경우 `lm`
  - 멀티모달 모델(텍스트, 비전, 오디오)의 경우 `multimodal`
  - 이미지 생성 모델의 경우 `image-generation`
  - 이미지 편집 모델의 경우 `image-edit`
  - 임베딩 모델의 경우 `embeddings`
  - Whisper 모델(오디오 전사)의 경우 `whisper`
  - 기본값: `lm`
- `--context-length`: 언어 모델의 컨텍스트 길이. 텍스트 처리 및 메모리 사용 최적화를 위한 최대 시퀀스 길이를 제어합니다. 기본값: `None` (모델의 기본 컨텍스트 길이 사용).
- `--config-name`: 사용할 모델 구성. `image-generation` 및 `image-edit` 모델 유형에만 사용됩니다:
  - `image-generation`의 경우: `flux-schnell`, `flux-dev`, `flux-krea-dev`, `qwen-image`, `z-image-turbo`, `fibo`
  - `image-edit`의 경우: `flux-kontext-dev`, `qwen-image-edit`
  - 기본값: image-generation의 경우 `flux-schnell`, image-edit의 경우 `flux-kontext-dev`
- `--quantize`: Flux 모델의 양자화 수준. 사용 가능한 옵션: `4`, `8`, `16`. 기본값: `8`
- `--lora-paths`: 쉼표로 구분된 LoRA 어댑터 파일 경로.
- `--lora-scales`: 쉼표로 구분된 LoRA 어댑터 스케일 팩터. LoRA 경로 수와 일치해야 합니다.
- `--max-concurrency`: 최대 동시 요청 수 (기본값: 1)
- `--queue-timeout`: 요청 시간 초과(초) (기본값: 300)
- `--queue-size`: 대기 중인 요청의 최대 대기열 크기 (기본값: 100)
- `--port`: 서버를 실행할 포트 (기본값: 8000)
- `--host`: 서버를 실행할 호스트 (기본값: 0.0.0.0)
- `--disable-auto-resize`: 자동 모델 크기 조정을 비활성화합니다. Vision Language Models에서만 작동합니다.
- `--enable-auto-tool-choice`: 자동 도구 선택(Auto tool choice)을 활성화합니다. 언어 모델(`lm` 또는 `multimodal` 모델 유형)에서만 작동합니다.
- `--tool-call-parser`: 자동 감지 대신 사용할 도구 호출 파서(Tool call parser)를 지정합니다. 언어 모델(`lm` 또는 `multimodal` 모델 유형)에서만 작동합니다. 사용 가능한 옵션: `qwen3`, `glm4_moe`, `qwen3_moe`, `qwen3_next`, `qwen3_vl`, `harmony`, `minimax`.
- `--reasoning-parser`: 자동 감지 대신 사용할 추론 파서(Reasoning parser)를 지정합니다. 언어 모델(`lm` 또는 `multimodal` 모델 유형)에서만 작동합니다. 사용 가능한 옵션: `qwen3`, `glm4_moe`, `qwen3_moe`, `qwen3_next`, `qwen3_vl`, `harmony`, `minimax`.
- `--trust-remote-code`: 모델을 로드할 때 `trust_remote_code`를 활성화합니다. 이를 통해 모델 저장소에서 사용자 정의 코드를 로드할 수 있습니다. 기본값: `False` (비활성화됨). `lm` 또는 `multimodal` 모델 유형에서만 작동합니다.
- `--chat-template-file`: 사용자 정의 채팅 템플릿 파일의 경로입니다. 언어 모델(`lm`) 및 멀티모달 모델(`multimodal`)에서만 작동합니다. 기본값: `None` (모델의 기본 채팅 템플릿 사용).
- `--log-file`: 로그 파일 경로입니다. 지정하지 않으면 기본적으로 'logs/app.log'에 로그가 기록됩니다.
- `--no-log-file`: 파일 로깅을 완전히 비활성화합니다. 콘솔 출력만 표시됩니다.
- `--log-level`: 로깅 수준을 설정합니다. 선택 사항: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. 기본값: `INFO`.

### 파서 구성 (Parser Configuration)

서버는 도구 호출 및 추론/생각(thinking) 내용 추출을 위한 파서의 수동 구성을 지원합니다. 파서가 명시적으로 지정되지 않으면 기본적으로 `None`이 되며, 이는 파싱이 수행되지 않음을 의미합니다.

#### 사용 가능한 파서

다음 파서는 도구 호출 및 추론 파싱 모두에 사용할 수 있습니다:

- **`qwen3`**: Qwen3 모델 형식용 파서
- **`glm4_moe`**: GLM4 MoE 모델 형식용 파서
- **`qwen3_moe`**: Qwen3 MoE 모델 형식용 파서
- **`qwen3_next`**: Qwen3 Next 모델 형식용 파서
- **`qwen3_vl`**: Qwen3 Vision-Language 모델 형식용 파서
- **`harmony`**: Harmony/GPT-OSS 모델용 통합 파서 (thinking과 tools 모두 처리)
- **`minimax`**: MiniMax 모델 형식용 파서

#### 파서 매개변수

- **`--tool-call-parser`**: 모델 응답에서 도구 호출을 추출하는 데 사용할 파서를 지정합니다.
- **`--reasoning-parser`**: 모델 응답에서 추론/생각 내용을 추출하는 데 사용할 파서를 지정합니다.
- **`--enable-auto-tool-choice`**: 도구 호출 사용 시 자동 도구 선택을 활성화합니다.

#### 사용 예

**파서가 없는 기본 사용 (기본값):**
```bash
python -m app.main launch \
  --model-path /path/to/model \
  --model-type lm
```

**도구 호출 파서만 사용:**
```bash
python -m app.main launch \
  --model-path /path/to/model \
  --model-type lm \
  --tool-call-parser qwen3
```

**두 파서 모두 사용:**
```bash
python -m app.main launch \
  --model-path /path/to/model \
  --model-type lm \
  --tool-call-parser qwen3 \
  --reasoning-parser qwen3
```

**자동 도구 선택 활성화:**
```bash
python -m app.main launch \
  --model-path /path/to/model \
  --model-type lm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3 \
  --reasoning-parser qwen3
```

**Harmony 파서 사용 (통합 파서):**
```bash
python -m app.main launch \
  --model-path /path/to/model \
  --model-type lm \
  --reasoning-parser harmony \
  --tool-call-parser harmony
```

> **참고:** 파서 구성은 언어 모델(`lm` 또는 `multimodal` 모델 유형)에만 적용됩니다. 파서를 지정하지 않으면 서버는 파싱을 수행하지 않으며 원시 모델 응답이 반환됩니다.

#### 구성 예

**텍스트 전용 모델:**
```bash
python -m app.main \
  --model-path mlx-community/gemma-3-4b-it-4bit \
  --model-type lm \
  --context-length 8192 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

**멀티모달 모델:**
```bash
python -m app.main \
  --model-path mlx-community/llava-phi-3-vision-4bit \
  --model-type multimodal \
  --context-length 4096 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

**trust_remote_code가 활성화된 모델:**
```bash
python -m app.main \
  --model-path <path-to-model-requiring-custom-code> \
  --model-type lm \
  --trust-remote-code \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

**사용자 정의 채팅 템플릿이 있는 모델:**
```bash
python -m app.main \
  --model-path <path-to-model> \
  --model-type lm \
  --chat-template-file /path/to/custom_template.jinja \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

**이미지 생성 모델:**

*Schnell로 빠른 생성:*
```bash
python -m app.main \
  --model-type image-generation \
  --model-path <path-to-local-flux-model> \
  --config-name flux-schnell \
  --quantize 8 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*Dev로 고품질 생성:*
```bash
python -m app.main \
  --model-type image-generation \
  --model-path <path-to-local-flux-model> \
  --config-name flux-dev \
  --quantize 8 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*Krea-Dev로 프리미엄 품질 생성:*
```bash
python -m app.main \
  --model-type image-generation \
  --model-path <path-to-local-flux-model> \
  --config-name flux-krea-dev \
  --quantize 8 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*Qwen Image로 고품질 생성:*
```bash
python -m app.main \
  --model-type image-generation \
  --model-path <path-to-local-qwen-model> \
  --config-name qwen-image \
  --quantize 8 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*Z-Image Turbo로 빠른 생성:*
```bash
python -m app.main launch --model-path z-image-turbo --model-type image-generation --config-name z-image-turbo
```

*Fibo로 생성:*
```bash
python -m app.main launch --model-path fibo --model-type image-generation --config-name fibo
```

*Kontext로 이미지 편집:*
```bash
python -m app.main \
  --model-type image-edit \
  --model-path <path-to-local-flux-model> \
  --config-name flux-kontext-dev \
  --quantize 8 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*Qwen Image Edit으로 이미지 편집:*
```bash
python -m app.main \
  --model-type image-edit \
  --model-path <path-to-local-qwen-model> \
  --config-name qwen-image-edit \
  --quantize 8 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*LoRA 어댑터 포함 (이미지 생성):*
```bash
python -m app.main \
  --model-type image-generation \
  --model-path <path-to-local-flux-model> \
  --config-name flux-dev \
  --quantize 8 \
  --lora-paths "/path/to/lora1.safetensors,/path/to/lora2.safetensors" \
  --lora-scales "0.8,0.6" \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*LoRA 어댑터 포함 (이미지 편집):*
```bash
python -m app.main \
  --model-type image-edit \
  --model-path <path-to-local-flux-model> \
  --config-name flux-kontext-dev \
  --quantize 8 \
  --lora-paths "/path/to/lora1.safetensors,/path/to/lora2.safetensors" \
  --lora-scales "0.8,0.6" \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

**Whisper 모델:**

*Whisper로 오디오 전사:*
```bash
python -m app.main \
  --model-type whisper \
  --model-path mlx-community/whisper-large-v3-mlx \
  --max-concurrency 1 \
  --queue-timeout 600 \
  --queue-size 50
```

### CLI 사용법 (CLI Usage)

서버는 쉬운 시작 및 관리를 위한 편리한 CLI 인터페이스를 제공합니다:

**버전 및 도움말 확인:**
```bash
python -m app.main --version
python -m app.main --help
python -m app.main launch --help
```

**서버 시작:**
```bash
# 텍스트 전용 또는 멀티모달 모델의 경우
python -m app.main launch --model-path <path-to-mlx-model> --model-type <lm|multimodal> --context-length 8192

# 이미지 생성 모델의 경우 (Flux 시리즈, Qwen, Z-Image Turbo 또는 Fibo)
python -m app.main launch --model-type image-generation --model-path <path-to-local-model> --config-name <flux-schnell|flux-dev|flux-krea-dev|qwen-image|z-image-turbo|fibo>

# 이미지 편집 모델의 경우 (Flux 시리즈 또는 Qwen)
python -m app.main launch --model-type image-edit --model-path <path-to-local-model> --config-name <flux-kontext-dev|qwen-image-edit>

# Whisper 모델의 경우
python -m app.main launch --model-path mlx-community/whisper-large-v3-mlx --model-type whisper

# LoRA 어댑터 포함 (이미지 생성)
python -m app.main launch --model-type image-generation --model-path <path-to-local-flux-model> --config-name flux-dev --lora-paths "/path/to/lora1.safetensors,/path/to/lora2.safetensors" --lora-scales "0.8,0.6"

# LoRA 어댑터 포함 (이미지 편집)
python -m app.main launch --model-type image-edit --model-path <path-to-local-flux-model> --config-name flux-kontext-dev --lora-paths "/path/to/lora1.safetensors,/path/to/lora2.safetensors" --lora-scales "0.8,0.6"

# 사용자 정의 로깅 구성 포함
python -m app.main launch --model-path <path-to-mlx-model> --model-type lm --log-file /tmp/server.log --log-level DEBUG

# 파일 로깅 비활성화 (콘솔만)
python -m app.main launch --model-path <path-to-mlx-model> --model-type lm --no-log-file

# 기본 로깅 사용 (logs/app.log, INFO 레벨)
python -m app.main launch --model-path <path-to-mlx-model> --model-type lm

# 도구 호출 및 추론을 위한 파서 구성 포함
python -m app.main launch \
  --model-path <path-to-mlx-model> \
  --model-type lm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3 \
  --reasoning-parser qwen3

# trust_remote_code 활성화 (사용자 정의 코드가 필요한 모델용)
python -m app.main launch \
  --model-path <path-to-mlx-model> \
  --model-type lm \
  --trust-remote-code

# 사용자 정의 채팅 템플릿 파일 사용
python -m app.main launch \
  --model-path <path-to-mlx-model> \
  --model-type lm \
  --chat-template-file /path/to/custom_template.jinja

# python -m app.main 사용 (대체 방법)
python -m app.main --model-path <path-to-mlx-model> --model-type lm --no-log-file
python -m app.main --model-path <path-to-mlx-model> --model-type lm --log-file /tmp/custom.log
python -m app.main --model-path <path-to-mlx-model> --model-type lm --trust-remote-code
python -m app.main --model-path <path-to-mlx-model> --model-type lm --chat-template-file /path/to/custom_template.jinja
```

> **참고:** 이제 텍스트 전용 모델(`--model-type lm`)과 멀티모달 모델(`--model-type multimodal`) 모두에서 `/v1/embeddings` 엔드포인트를 통한 텍스트 임베딩을 사용할 수 있습니다.

### 로깅 구성 (Logging Configuration)

서버는 MLX 서버를 모니터링하고 디버깅하는 데 도움이 되는 유연한 로깅 옵션을 제공합니다:

#### 로깅 옵션

- **`--log-file`**: 로그 파일의 사용자 정의 경로를 지정합니다.
  - 기본값: `logs/app.log`
  - 예: `--log-file /tmp/my-server.log`

- **`--no-log-file`**: 파일 로깅을 완전히 비활성화합니다.
  - 콘솔 출력만 표시됩니다.
  - 개발 중이거나 영구 로그가 필요하지 않을 때 유용합니다.

- **`--log-level`**: 로깅의 상세도를 제어합니다.
  - 선택 사항: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
  - 기본값: `INFO`
  - `DEBUG`: 가장 상세하며 상세한 디버깅 정보를 포함합니다.
  - `INFO`: 표준 작동 메시지 (기본값)
  - `WARNING`: 잠재적인 문제에 대한 중요 알림
  - `ERROR`: 오류 메시지만 표시
  - `CRITICAL`: 치명적인 시스템 오류만 표시

#### 로깅 예시

```bash
# 기본 로깅 사용 (logs/app.log, INFO 레벨)
python -m app.main launch --model-path <path-to-model> --model-type lm

# 디버그 레벨로 사용자 정의 로그 파일 사용
python -m app.main launch --model-path <path-to-model> --model-type lm --log-file /tmp/debug.log --log-level DEBUG

# 콘솔 전용 로깅 (파일 출력 없음)
python -m app.main launch --model-path <path-to-model> --model-type lm --no-log-file

# 고수준 로깅 (오류만)
python -m app.main launch --model-path <path-to-model> --model-type lm --log-level ERROR

# 로깅 옵션과 함께 python -m app.main 사용
python -m app.main --model-path <path-to-model> --model-type lm --no-log-file
python -m app.main --model-path <path-to-model> --model-type lm --log-file /tmp/custom.log --log-level DEBUG
```

#### 로그 파일 기능

- **자동 회전**: 로그 파일은 500MB에 도달하면 자동으로 회전됩니다.
- **보존**: 로그 파일은 기본적으로 10일 동안 보관됩니다.
- **서식 있는 출력**: 콘솔 및 파일 로그 모두 타임스탬프, 로그 수준 및 구조화된 형식을 포함합니다.
- **색상화된 콘솔**: 콘솔 출력에는 가독성을 높이기 위해 색상 코딩이 포함됩니다.

### API 사용 (Using the API)

서버는 표준 OpenAI 클라이언트 라이브러리와 함께 사용할 수 있는 OpenAI 호환 엔드포인트를 제공합니다. 다음은 몇 가지 예입니다:

#### 텍스트 완성 (Text Completion)
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # 로컬 서버에는 API 키가 필요하지 않습니다
)

response = client.chat.completions.create(
    model="local-model",  # 로컬 서버의 경우 모델 이름은 중요하지 않습니다
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7
)
print(response.choices[0].message.content)
```

#### 멀티모달 모델 (비전 + 오디오)
```python
import openai
import base64

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 이미지 로드 및 인코딩
with open("image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

response = client.chat.completions.create(
    model="local-multimodal",  # 로컬 서버의 경우 모델 이름은 중요하지 않습니다
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)
print(response.choices[0].message.content)
```

#### 오디오 입력 지원
```python
import openai
import base64

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 오디오 파일 로드 및 인코딩
with open("audio.wav", "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

response = client.chat.completions.create(
    model="local-multimodal",  # 로컬 서버의 경우 모델 이름은 중요하지 않습니다
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this audio?"
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav"
                    },
                },
            ],
        }
    ],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

#### Flux 시리즈 모델을 사용한 고급 이미지 생성

```python
import openai
import base64
from io import BytesIO
from PIL import Image

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 기본 이미지 생성
response = client.images.generate(
    prompt="A serene landscape with mountains and a lake at sunset",
    model="local-image-generation-model",
    size="1024x1024",
    n=1
)

# 생성된 이미지 표시
image_data = base64.b64decode(response.data[0].b64_json)
image = Image.open(BytesIO(image_data))
image.show()
```

#### 사용자 정의 매개변수를 사용한 고급 이미지 생성
```python
import requests

# 더 많은 제어를 위해 직접 API 호출 사용
payload = {
    "prompt": "A beautiful cyberpunk city at night with neon lights",
    "model": "local-image-generation-model",
    "size": "1024x1024",
    "negative_prompt": "blurry, low quality, distorted",
    "steps": 8,
    "seed": 42,
    "priority": "normal"
}

response = requests.post(
    "http://localhost:8000/v1/images/generations",
    json=payload,
    headers={"Authorization": "Bearer fake-api-key"}
)

if response.status_code == 200:
    result = response.json()
    # base64 이미지 데이터 처리
    image_data = base64.b64decode(result['data'][0]['b64_json'])
    image = Image.open(BytesIO(image_data))
    image.show()
```

**이미지 생성 매개변수:**
- `prompt`: 원하는 이미지에 대한 텍스트 설명 (필수, 최대 1000자)
- `model`: 모델 식별자 (기본값 "local-image-generation-model")
- `size`: 이미지 크기 - "256x256", "512x512" 또는 "1024x1024" (기본값: "1024x1024")
- `negative_prompt`: 생성된 이미지에서 피해야 할 것 (선택 사항)
- `steps`: 추론 단계 수, 1-50 (구성에 따라 기본값 다름: Schnell의 경우 4, Dev의 경우 25, Krea-Dev의 경우 28, Qwen Image의 경우 50)
- `seed`: 재현 가능한 생성을 위한 랜덤 시드 (선택 사항)
- `priority`: 작업 우선순위 - "low", "normal", "high" (기본값: "normal")
- `async_mode`: 비동기적으로 처리할지 여부 (기본값: false)

> **참고:** 이미지 생성은 `--model-type image-generation`으로 서버를 실행해야 합니다. 서버는 MLX Flux 시리즈 모델(flux-schnell, flux-dev, flux-krea-dev), Qwen Image 모델(qwen-image), Z-Image Turbo(z-image-turbo), Fibo(fibo) 모델을 지원하여 구성 가능한 품질/속도 균형을 갖춘 고품질 이미지 생성을 제공합니다.

#### Flux 시리즈 모델을 사용한 이미지 편집

```python
import openai
import base64
from io import BytesIO
from PIL import Image

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 기존 이미지 편집
with open("images/china.png", "rb") as image_file:
    result = client.images.edit(
        image=image_file,
        prompt="make it like a photo in 1800s",
        model="flux-kontext-dev"
    )

# 편집된 이미지 표시
image_data = base64.b64decode(result.data[0].b64_json)
image = Image.open(BytesIO(image_data))
image.show()
```

#### 사용자 정의 매개변수를 사용한 고급 이미지 편집
```python
import requests

# 더 많은 제어를 위해 양식 데이터와 함께 직접 API 호출 사용
with open("images/china.png", "rb") as image_file:
    files = {"image": image_file}
    data = {
        "prompt": "make it like a photo in 1800s",
        "model": "flux-kontext-dev",
        "negative_prompt": "modern, digital, high contrast",
        "guidance_scale": 2.5,
        "steps": 4,
        "seed": 42,
        "size": "1024x1024",
        "response_format": "b64_json"
    }
    
    response = requests.post(
        "http://localhost:8000/v1/images/edits",
        files=files,
        data=data,
        headers={"Authorization": "Bearer fake-api-key"}
    )

if response.status_code == 200:
    result = response.json()
    # base64 이미지 데이터 처리
    image_data = base64.b64decode(result['data'][0]['b64_json'])
    image = Image.open(BytesIO(image_data))
    image.show()
```

**이미지 편집 매개변수:**
- `image`: 편집할 이미지 파일 (필수, PNG, JPEG 또는 JPG 형식, 최대 10MB)
- `prompt`: 원하는 편집에 대한 텍스트 설명 (필수, 최대 1000자)
- `model`: 모델 식별자 (기본값 "flux-kontext-dev")
- `negative_prompt`: 편집된 이미지에서 피해야 할 것 (선택 사항)
- `guidance_scale`: 모델이 프롬프트를 얼마나 가깝게 따를지 제어 (기본값: flux-kontext-dev의 경우 2.5, qwen-image-edit의 경우 4.0)
- `steps`: 추론 단계 수, 1-50 (기본값: flux-kontext-dev의 경우 4, qwen-image-edit의 경우 50)
- `seed`: 재현 가능한 편집을 위한 랜덤 시드 (기본값: 42)
- `size`: 출력 이미지 크기 - "256x256", "512x512" 또는 "1024x1024" (선택 사항)
- `response_format`: 응답 형식 - "b64_json" (기본값: "b64_json")

> **참고:** 이미지 편집은 `--model-type image-edit`으로 서버를 실행해야 합니다. 서버는 MLX Flux 시리즈 모델(flux-kontext-dev) 및 Qwen Image Edit 모델(qwen-image-edit)을 지원하여 구성 가능한 품질/속도 균형을 갖춘 고품질 이미지 편집을 제공합니다.

#### 함수 호출 (Function Calling)
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 메시지 및 도구 정의
messages = [
    {
        "role": "user",
        "content": "What is the weather in Tokyo?"
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city to get the weather for"}
                }
            }
        }
    }
]

# API 호출 수행
completion = client.chat.completions.create(
    model="local-model",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# 도구 호출 응답 처리
if completion.choices[0].message.tool_calls:
    tool_call = completion.choices[0].message.tool_calls[0]
    print(f"Function called: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
    
    # 도구 호출 처리 - 일반적으로 여기에서 실제 함수를 호출합니다
    # 이 예제에서는 날씨 응답을 하드코딩합니다
    weather_info = {"temperature": "22°C", "conditions": "Sunny", "humidity": "65%"}
    
    # 대화에 도구 호출 및 함수 응답 추가
    messages.append(completion.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": str(weather_info)
    })
    
    # 함수 결과를 사용하여 대화 계속
    final_response = client.chat.completions.create(
        model="local-model",
        messages=messages
    )
    print("\nFinal response:")
    print(final_response.choices[0].message.content)
```

#### JSON 스키마를 이용한 구조화된 출력 (Structured Outputs with JSON Schema)

서버는 JSON 스키마를 사용한 구조화된 출력을 지원하므로 특정 JSON 형식의 응답을 얻을 수 있습니다:

```python
import openai
import json

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 메시지 및 응답 형식 정의
messages = [
    {
        "role": "system",
        "content": "Extract the address from the user input into the specified JSON format."
    },
    {
        "role": "user",
        "content": "Please format this address: 1 Hacker Wy Menlo Park CA 94025"
    }
]

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "Address",
        "schema": {
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {
                            "type": "string", 
                            "description": "2 letter abbreviation of the state"
                        },
                        "zip": {
                            "type": "string", 
                            "description": "5 digit zip code"
                        }
                    },
                    "required": ["street", "city", "state", "zip"]
                }
            },
            "required": ["address"],
            "type": "object"
        }
    }
}

# 구조화된 출력으로 API 호출 수행
completion = client.chat.completions.create(
    model="local-model",
    messages=messages,
    response_format=response_format
)

# 구조화된 응답 파싱
response_content = completion.choices[0].message.content
parsed_address = json.loads(response_content)
print("Structured Address:")
print(json.dumps(parsed_address, indent=2))
```

**응답 형식 매개변수:**
- `type`: 구조화된 출력의 경우 `"json_schema"`로 설정해야 합니다.
- `json_schema`: 예상되는 응답 구조를 정의하는 JSON 스키마 객체
  - `name`: 스키마의 선택적 이름
  - `schema`: 속성, 유형 및 요구 사항이 포함된 실제 JSON 스키마 정의

**응답 예:**
```json
{
  "address": {
    "street": "1 Hacker Wy",
    "city": "Menlo Park",
    "state": "CA",
    "zip": "94025"
  }
}
```

> **참고:** 구조화된 출력은 텍스트 전용 모델(`--model-type lm`)에서 작동합니다. 모델은 제공된 JSON 스키마에 따라 응답 형식을 지정하려고 시도합니다.

#### 임베딩 (Embeddings)

1. 텍스트 전용 모델 임베딩:
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 단일 텍스트에 대한 임베딩 생성
embedding_response = client.embeddings.create(
    model="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-MLX-Q8",
    input=["The quick brown fox jumps over the lazy dog"]
)
print(f"Embedding dimension: {len(embedding_response.data[0].embedding)}")

# 여러 텍스트에 대한 임베딩 생성
batch_response = client.embeddings.create(
    model="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-MLX-Q8",
    input=[
        "Machine learning algorithms improve with more data",
        "Natural language processing helps computers understand human language",
        "Computer vision allows machines to interpret visual information"
    ]
)
print(f"Number of embeddings: {len(batch_response.data)}")
```

2. 멀티모달 모델 임베딩:
```python
import openai
import base64
from PIL import Image
from io import BytesIO

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 이미지를 base64로 인코딩하는 헬퍼 함수
def image_to_base64(image_path):
    image = Image.open(image_path)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_data = buffer.getvalue()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

# 이미지 인코딩
image_uri = image_to_base64("images/attention.png")

# 텍스트+이미지에 대한 임베딩 생성
multimodal_embedding = client.embeddings.create(
    model="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    input=["Describe the image in detail"],
    extra_body={"image_url": image_uri}
)
print(f"Multimodal embedding dimension: {len(multimodal_embedding.data[0].embedding)}")
```

> **참고:** 필요에 따라 모델 이름과 이미지 경로를 바꾸세요. `extra_body` 매개변수는 이미지 데이터 URI를 API에 전달하는 데 사용됩니다.

> **경고:** 멀티모달 요청(이미지 또는 오디오 포함)을 할 때는 서버가 `--model-type multimodal`로 실행되고 있는지 확인하세요. `--model-type lm`(텍스트 전용 모델)으로 실행되는 서버에 멀티모달 요청을 보내면 멀티모달 요청이 지원되지 않는다는 400 오류 메시지를 받게 됩니다.

## 요청 대기열 시스템 (Request Queue System)

서버는 MLX 모델 추론 요청을 관리하고 최적화하기 위해 강력한 요청 대기열 시스템을 구현합니다. 이 시스템은 효율적인 리소스 활용과 공정한 요청 처리를 보장합니다.

### 주요 기능

- **동시성 제어**: 리소스 고갈을 방지하기 위해 동시 모델 추론 수를 제한합니다.
- **요청 대기열**: 보류 중인 요청에 대해 공정한 선착순 대기열을 구현합니다.
- **시간 초과 관리**: 구성된 시간 초과를 초과하는 요청을 자동으로 처리합니다.
- **실시간 모니터링**: 대기열 상태 및 성능 메트릭을 모니터링하는 엔드포인트를 제공합니다.

### 아키텍처

대기열 시스템은 두 가지 주요 구성 요소로 구성됩니다:

1. **RequestQueue**: 다음을 수행하는 비동기 대기열 구현체입니다:
   - 구성 가능한 대기열 크기로 보류 중인 요청 관리
   - 세마포어를 사용하여 동시 실행 제어
   - 시간 초과 및 오류를 우아하게 처리
   - 실시간 대기열 통계 제공

2. **모델 핸들러**: 다양한 모델 유형에 대한 특수 핸들러:
   - `MLXLMHandler`: 텍스트 전용 모델 요청 관리
   - `MLXVLMHandler`: 멀티모달 모델 요청 관리
   - `MLXFluxHandler`: Flux 시리즈 이미지 생성 요청 관리

### 대기열 모니터링

`/v1/queue/stats` 엔드포인트를 사용하여 대기열 통계를 모니터링합니다:

```bash
curl http://localhost:8000/v1/queue/stats
```

응답 예:
```json
{
  "status": "ok",
  "queue_stats": {
    "running": true,
    "queue_size": 3,
    "max_queue_size": 100,
    "active_requests": 5,
    "max_concurrency": 2
  }
}
```

### 오류 처리

대기열 시스템은 다양한 오류 조건을 처리합니다:

1. **대기열 가득 참 (429)**: 대기열이 최대 크기에 도달했을 때
```json
{
  "detail": "Too many requests. Service is at capacity."
}
```

2. **요청 시간 초과**: 요청이 구성된 시간 초과를 초과했을 때
```json
{
  "detail": "Request processing timed out after 300 seconds"
}
```

3. **모델 오류**: 모델이 추론 중에 오류를 발생시켰을 때
```json
{
  "detail": "Failed to generate response: <error message>"
}
```

### 스트리밍 응답

서버는 적절한 청크 형식으로 스트리밍 응답을 지원합니다:
```python
{
    "id": "chatcmpl-1234567890",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "local-model",
    "choices": [{
        "index": 0,
        "delta": {"content": "chunk of text"},
        "finish_reason": null
    }]
}
```

## API 응답 스키마 (API Response Schemas)

서버는 기존 애플리케이션과의 원활한 통합을 보장하기 위해 OpenAI 호환 API 응답 스키마를 구현합니다. 다음은 주요 응답 형식입니다:

### 채팅 완성 응답 (Chat Completions Response)

```json
{
  "id": "chatcmpl-123456789",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "local-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "This is the response content from the model."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### 임베딩 응답 (Embeddings Response)

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.001, 0.002, ..., 0.999],
      "index": 0
    }
  ],
  "model": "local-model",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

### 함수/도구 호출 응답 (Function/Tool Calling Response)

```json
{
  "id": "chatcmpl-123456789",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "local-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Tokyo\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 25,
    "total_tokens": 40
  }
}
```

### 이미지 생성 응답 (Image Generation Response)

```json
{
  "created": 1677858242,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAA...",
      "url": null
    }
  ]
}
```

### 오류 응답 (Error Response)

```json
{
  "error": {
    "message": "Error message describing what went wrong",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

## 예제 노트북 (Example Notebooks)

이 저장소에는 API의 다양한 측면을 시작하는 데 도움이 되는 예제 노트북이 포함되어 있습니다:

- **function_calling_examples.ipynb**: 로컬 모델에서 함수 호출을 구현하고 사용하는 방법에 대한 실용 가이드:
  - 함수 정의 설정
  - 함수 호출 요청 수행
  - 함수 호출 응답 처리
  - 스트리밍 함수 호출 작업
  - 도구 사용을 통한 다중 턴 대화 구축

- **structured_outputs_examples.ipynb**: JSON 스키마를 사용하여 구조화된 출력을 사용하는 방법에 대한 포괄적인 가이드:
  - JSON 스키마 정의 설정
  - 응답 형식 사양으로 요청 수행
  - 구조화된 응답 파싱
  - 복잡한 중첩 스키마 작업
  - 구조화된 출력을 사용한 데이터 추출 파이프라인 구축

- **vision_examples.ipynb**: API의 비전 기능을 사용하는 방법에 대한 포괄적인 가이드:
  - 다양한 형식의 이미지 입력 처리
  - 비전 분석 및 객체 감지
  - 이미지가 포함된 다중 턴 대화
  - 상세한 이미지 설명 및 분석을 위한 비전 모델 사용

- **lm_embeddings_examples.ipynb**: 텍스트 전용 모델을 위한 임베딩 API 사용에 대한 포괄적인 가이드:
  - 단일 및 배치 입력에 대한 임베딩 생성
  - 텍스트 간 의미적 유사성 계산
  - 간단한 벡터 기반 검색 시스템 구축
  - 개념 간의 의미적 관계 비교

- **vlm_embeddings_examples.ipynb**: Vision-Language Model 임베딩 작업에 대한 상세 가이드:
  - 텍스트 프롬프트로 이미지에 대한 임베딩 생성
  - VLM으로 텍스트 전용 임베딩 생성
  - 텍스트와 이미지 표현 간의 유사성 계산
  - 멀티모달 모델의 공유 임베딩 공간 이해
  - VLM 임베딩의 실제 적용

- **simple_rag_demo.ipynb**: 로컬 MLX 서버를 사용하여 PDF 문서에 대한 경량 RAG(Retrieval-Augmented Generation) 파이프라인을 구축하는 실용 가이드:
  - PDF 문서 읽기 및 청킹
  - MLX 서버를 통한 텍스트 임베딩 생성
  - 검색을 위한 간단한 벡터 저장소 생성
  - 관련 청크를 기반으로 한 질문 답변 수행
  - Qwen3 로컬 모델을 사용한 문서 QA의 종단 간 데모
  <p align="center">
    <a href="https://youtu.be/ANUEZkmR-0s">
      <img src="https://img.youtube.com/vi/ANUEZkmR-0s/0.jpg" alt="RAG Demo" width="600">
    </a>
  </p>

- **audio_examples.ipynb**: MLX 서버의 오디오 처리 기능에 대한 포괄적인 가이드:
  - 오디오 처리를 위한 MLX 서버 연결 설정
  - API 전송을 위한 오디오 파일 로드 및 인코딩
  - 분석을 위해 멀티모달 모델에 오디오 입력 전송
  - 풍부하고 문맥을 인식하는 응답을 위해 오디오와 텍스트 프롬프트 결합
  - 다양한 유형의 오디오 분석 프롬프트 탐색
  - 오디오 전사 및 콘텐츠 분석 기능 이해

- **image_generations.ipynb**: MLX Flux 시리즈 및 Qwen Image 모델을 사용한 이미지 생성에 대한 포괄적인 가이드:
  - 이미지 생성을 위한 MLX 서버 연결 설정
  - 기본 매개변수를 사용한 기본 이미지 생성
  - 사용자 정의 매개변수(네거티브 프롬프트, 단계, 시드)를 사용한 고급 이미지 생성
  - 다양한 Flux 구성(schnell, dev, Krea-dev) 및 Qwen Image 모델 작업
  - 미세 조정된 생성을 위한 LoRA 어댑터 사용
  - 양자화 설정으로 성능 최적화

- **image_edit.ipynb**: MLX Flux 시리즈 및 Qwen Image Edit 모델을 사용한 이미지 편집에 대한 포괄적인 가이드:
  - 이미지 편집을 위한 MLX 서버 연결 설정
  - 기본 매개변수를 사용한 기본 이미지 편집
  - 사용자 정의 매개변수(가이던스 스케일, 단계, 시드)를 사용한 고급 이미지 편집
  - 문맥 편집을 위한 flux-kontext-dev 및 qwen-image-edit 구성 작업
  - 미세 조정된 편집을 위한 LoRA 어댑터 사용
  - 생성 및 편집 워크플로우 간의 차이점 이해
  - 효과적인 이미지 편집 프롬프트를 위한 모범 사례

## 대규모 모델 (Large Models)
시스템의 사용 가능한 RAM에 비해 큰 모델을 사용하는 경우 성능이 저하될 수 있습니다. mlx-lm은 모델과 캐시에서 사용하는 메모리를 와이어링(wiring)하여 속도를 향상시키려고 시도합니다. 이 최적화는 macOS 15.0 이상에서만 사용할 수 있습니다.
만약 다음 경고 메시지가 표시된다면:
> [WARNING] Generating with a model that requires ...
모델이 컴퓨터에서 느리게 실행될 수 있음을 의미합니다. 모델이 RAM에 맞는 경우 시스템의 와이어드 메모리 제한을 높여 성능을 향상시킬 수 있습니다. 이렇게 하려면 다음을 실행하세요:
```bash
bash configure_mlx.sh
```
## 라이선스 (License)
이 프로젝트는 [MIT 라이선스](LICENSE)에 따라 라이선스가 부여됩니다. 라이선스 조건에 따라 자유롭게 사용, 수정 및 배포할 수 있습니다.

## 🏗️ 핵심 기술
- **[MLX 팀](https://github.com/ml-explore/mlx)** - Apple Silicon에서 효율적인 기계 학습을 위한 기반을 제공하는 획기적인 MLX 프레임워크 개발
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)** - 효율적인 대규모 언어 모델 지원 및 최적화
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm/tree/main)** - MLX 생태계 내에서 멀티모달 모델 지원 개척
- **[mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)** - 최적화된 메모리 관리를 통한 텍스트 임베딩 생성
- **[mflux](https://github.com/filipstrand/mflux)** - 고급 구성을 갖춘 Flux 시리즈 이미지 생성 모델
- **[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)** - Apple Silicon에서 최적화된 추론을 통한 오디오 전사 및 음성 인식
- **[mlx-community](https://huggingface.co/mlx-community)** - 다양한 고품질 MLX 모델 컬렉션 큐레이팅 및 유지 관리

