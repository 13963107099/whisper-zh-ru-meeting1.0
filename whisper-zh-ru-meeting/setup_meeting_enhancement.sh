#!/bin/bash
# Independent environment: never pip-install into the user's main Python or pyannote venv.
set -euo pipefail
BASE="$HOME/whisper-ru"
ENV_DIR="${ENHANCEMENT_ENV:-$BASE/enhancement-venv}"
PYTHON="${ENHANCEMENT_PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in "$(command -v python3.11 2>/dev/null || true)" /opt/local/bin/python3.11 /usr/local/bin/python3.11 /opt/homebrew/bin/python3.11; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then PYTHON="$candidate"; break; fi
  done
fi
if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
  echo '需要独立 Python 3.11，不使用 Python 3.14 安装这些模型依赖。' >&2
  echo '已有 Homebrew：brew install python@3.11 ffmpeg' >&2
  echo '已有 MacPorts：sudo port install python311 ffmpeg' >&2
  echo '两种包管理器选已有的一种；安装后可设置 ENHANCEMENT_PYTHON=/实际路径/python3.11' >&2
  exit 2
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "请使用 Python 3.11"'
if [ -d "$ENV_DIR" ]; then
  "$ENV_DIR/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,11), "现有增强环境不是 Python 3.11；请指定新的 ENHANCEMENT_ENV"'
else
  "$PYTHON" -m venv "$ENV_DIR"
fi
PIP="$ENV_DIR/bin/python"
"$PIP" -m pip install --upgrade 'pip<26' setuptools wheel
# Avoid sentencepiece C++ compilation on Intel Mac: use a compatible pre-built wheel.
"$PIP" -m pip install --only-binary=:all: 'sentencepiece==0.2.2'
# Fixed compatibility baseline, not an assertion that these are the newest packages.
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = x86_64 ]; then
  TORCH='2.2.2'
else
  TORCH='2.5.1'
fi
"$PIP" -m pip install \
  'numpy==1.26.4' "torch==$TORCH" "torchaudio==$TORCH" \
  'speechbrain==1.0.3' 'transformers==4.46.3' 'huggingface-hub==0.27.1' \
  'soundfile==0.12.1' 'sacremoses==0.1.1'
if [ "${INSTALL_DEEPFILTER:-0}" = 1 ]; then
  # Separate switch: some architectures may need Rust/build dependencies for libdf.
  "$PIP" -m pip install 'deepfilternet==0.5.6' 'numpy==1.26.4' \
    "torch==$TORCH" "torchaudio==$TORCH" 'huggingface-hub==0.27.1'
fi
"$PIP" -m pip check
"$PIP" -c 'import torch, torchaudio, soundfile, speechbrain; from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration; print("增强环境导入检查通过，torch", torch.__version__)'
if [ "${INSTALL_DEEPFILTER:-0}" = 1 ]; then
  "$PIP" -c 'from df.enhance import init_df, enhance; print("DeepFilter依赖导入通过；首次使用仍需下载权重")'
fi
"$PIP" -m pip freeze > "$ENV_DIR/installed_versions.txt"
echo "环境：$ENV_DIR"
echo '本脚本只安装依赖；不训练模型、不读取或上传会议录音。'
echo 'SepFormer 与 M2M100 权重在第一次使用时下载。V3.3.5 不运行 ECAPA 声纹识别。'
echo '既有 Whisper / pyannote 环境不改变；若缺少 ffmpeg，请使用已有包管理器安装。'
