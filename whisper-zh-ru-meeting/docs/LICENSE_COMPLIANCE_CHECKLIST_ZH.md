# 第三方模型来源与开源许可核验清单

版本范围：V3.3.5 / `v0.1.0-alpha`；核对日期：2026-11-13。

**结论：**仓库可以公开你有权授权的原创 Python 编排代码；第三方模型、框架和音频工具应分别署名并遵守各自许可证。**“写了出处”不等于“获得了任意再分发权”，也不能替代代码权属审查。**

## 1. 每个功能究竟是谁开发的？

| 功能 | 真正的上游 / 模型或库 | 许可信息及入口 | 对外应如何表述 |
|---|---|---|---|
| 中俄语音转写 | OpenAI [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)；[whisper.cpp](https://github.com/ggml-org/whisper.cpp) 推理实现；[ggerganov 量化 GGML 模型](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-large-v3-turbo-q5_0.bin) | 各上游仓库标示 MIT | 基于开源模型的本地会议转录流程；**不是自行训练 Whisper** |
| 人声活动检测 | [Silero Team / Silero VAD](https://github.com/snakers4/silero-vad) | [MIT](https://github.com/snakers4/silero-vad/blob/master/LICENSE) | 调用 Silero 模型检测语音边界 |
| 中俄翻译辅助 | [Meta AI / M2M100 418M](https://huggingface.co/facebook/m2m100_418M) | MIT | 使用 Meta 预训练模型辅助翻译；非真实口译原话 |
| 重叠语音检测 | [pyannote / segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) | MIT + 访问条件 | 识别疑似同时说话区间，不等于人物身份识别 |
| 两路语音分离 | [SpeechBrain / SepFormer WHAMR16k](https://huggingface.co/speechbrain/sepformer-whamr16k) | Apache-2.0 | 调用预训练分离模型，生成两路候选音轨 |
| 可选神经降噪 | [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) | MIT 或 Apache-2.0 双许可 | 可选使用上游预训练模型进行降噪 |
| 可选/遗留声纹匹配 | [SpeechBrain / ECAPA-TDNN VoxCeleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | Apache-2.0 | 源码存在条件调用，不是默认运行功能 |
| 遗留完整说话人聚类 | [pyannote / speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | MIT + 访问条件 | 旧代码保留入口；V3.3.5 默认不用 |

英文版完整清单以及框架级别的官方出处见 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。机器可读索引见 [`third_party_models.json`](../third_party_models.json)。

## 2. 公开之前逐项完成

- [ ] 确认即将上传的每一份 Python、Shell、测试和文档内容均由你有权公开的人员编写，或有适用的复用许可。尤其检查是否复制了雇主、客户、合作方、第三方项目的代码。
- [ ] 顶层 `LICENSE` 中的版权权利主体与实际权利情况一致。该 MIT 许可证只能授予你拥有授权权利的内容，不能覆盖上游模型。
- [ ] 在仓库首页显著链接 `THIRD_PARTY_NOTICES.md`、本清单及上游官方地址。
- [ ] 所有上游模型权重单独由使用者根据许可证下载；不上传 `.bin`, `.pt`, `.ckpt`, `.gguf`, `.safetensors` 等。
- [ ] 不上传真实录音、会议逐字稿、翻译数据、企业名称和谈判金额、会务人员身份、任何 HF_TOKEN / API 密钥。
- [ ] 如把 Apache-2.0 授权的组件（例如 SepFormer 权重或 SpeechBrain 源码）打包再分发，补全该组件许可证与适用 NOTICE、注明修改；仅写链接不足以履行所有复制分发义务。
- [ ] 如果以后打包 FFmpeg，明确具体二进制是 LGPL 还是 GPL 构建，落实相应分发义务；当前代码只调用外部 FFmpeg，不附带其二进制。
- [ ] 若以后修改依赖版本、替换模型仓库、下载微调权重，**重新核对该具体发布版本的许可证**，不要沿用旧声明。
- [ ] 如未完成代码权属或敏感信息核查，维持 GitHub 仓库为 Private，暂不切换 Public。

## 3. 常见误区

**“我已经在 README 贴了模型链接，可以把模型文件也上传 GitHub 吗？”** 不能这样推理。MIT/Apache-2.0 允许特定条件下复制分发，但必须满足版权与许可证附带要求；受限访问模型还应核对其访问条款。最简单的做法是不上传模型文件，仅提供官方链接。

**“我们的项目用 MIT 许可证，所以 FFmpeg、PyTorch、SpeechBrain 全都变成 MIT 了吗？”** 不是。顶层 MIT 许可证仅覆盖发布者有权许可的项目代码；依赖的许可证保持独立。

**“别人有权用这段代码，我可以公开之前处理的真实会议吗？”** 不可以由此得出。会议录音和转写内容涉及独立的隐私、保密和潜在合同义务，与模型许可证不同。

**“只要扫描脚本通过，就保证不侵权？”** 不能。扫描主要检查常见敏感文件与字符串，无法自动确认代码著作权、许可证冲突、员工协议、第三方模型再分发权，或音频中被谈话人的权利。

## 4. 公开仓库前实际操作

1. 把新的开源审核包解压到与会议录音完全分开的文件夹。
2. 本地执行 `python3 check_publication.py` 和 `python3 check_third_party_attributions.py`；再运行 `python3 -m unittest -v test_v335`。
3. 使用 Finder 检查根目录是否有 `THIRD_PARTY_NOTICES.md`、`third_party_models.json`、`LICENSE`、`README.md`。
4. 在 GitHub 的 **Private** 仓库上传这些代码文件；审核提交历史。尚未检查完毕时不要改为 Public。
5. 公开后每次变更模型 ID、安装器和依赖版本都更新这份清单与上游链接。

## 5. 关于学术引用

如需发表论文或技术报告，除了许可声明，可以参考上游论文：

- Whisper: https://arxiv.org/abs/2212.04356
- M2M100: https://arxiv.org/abs/2010.11125
- SepFormer: https://arxiv.org/abs/2010.13154

这些引用用于学术致谢，不代替相应模型的许可证与版权声明。
