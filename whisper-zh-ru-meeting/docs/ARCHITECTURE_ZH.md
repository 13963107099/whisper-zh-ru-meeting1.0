# 架构说明（V3.3.5）

```text
私人音频（仅本地）
  -> whisper.cpp + Silero VAD（独立安装；生成 V3.3 原始 ASR 缓存）
  -> meeting_v335.py render（去磕绊、过滤疑似幻觉、S 段顺序编号）
     +-> translate：M2M100 中俄双向辅助译文，源文与译文分开保存
     +-> detect：pyannote segmentation 独立重叠语音检测（不做声纹识别）
     +-> separate：SepFormer 两路分离 -> whisper.cpp 分别转写，候选附于文末
  -> all_clean.txt / 结构化 JSON（均位于本地 output 目录，不上传 Github）
```

S01、S02 表示转写片段顺序，不代表人物。自动重叠区间是候选，不能假定有且仅有两人；SepFormer 只能产生两条估计音轨。完整 pyannote 全局声纹聚类已从 V3.3.5 的主流程移除。Whisper 仅调用 `localhost` 服务。Hugging Face 与 pyannote 遥测被强制关闭，模型 Hub 默认离线；只有在运行前明确设置 `WHISPER_MEETING_ALLOW_MODEL_DOWNLOADS=1` 才允许首次下载模型，会议音频和文字不作为下载请求输入。

## 可移植性与验证缺口

- `meeting_v335.py render` 需要 `render_meta.json`, `chunk_plan.json`, `asr_cache/chunk_*.json` 和对应 WAV，当前不提供真实会议示例数据。
- 旧版 V3.3 脚本的 CLI 提供从新录音开始的路线，要求用户安装本地外部二进制并通过参数指定路径。
- `test_v335.py` 使用临时与模拟输入，**不是**完整模型性能评测。
- 真实中俄口译、噪声、同时发言条件下的准确率与 CPU 速度尚待独立测评。
