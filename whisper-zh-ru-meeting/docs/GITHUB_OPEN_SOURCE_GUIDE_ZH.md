# GitHub 开源发布教程｜Whisper ZH/RU Meeting（从零开始）

本教程面向 macOS / zsh 初学者。公开前一定确认你拥有代码发布权，且仓库不含会议音频、转写记录、个人姓名、商业机密、Hugging Face Token、已下载权重。**公开发布是将数据提供给任何可访问 GitHub 的人。**

## 一、先理解五个基础概念

- Git 是本地版本工具，追踪文件每次改动；GitHub 是远程托管平台。
- repo（repository）是仓库；`main` 是主分支；commit 是带说明的一次保存；push 是把本地 commit 发送到 GitHub。
- `README.md` 为项目首页；`.gitignore` 指定新文件哪些不纳入 Git；`LICENSE` 给别人代码使用权限。
- 建议名称 `whisper-zh-ru-meeting`；建议首个 release 为 `v0.1.0-alpha`，不要将内部 V3.3.5 与对外语义版本混淆。

## 二、在 Mac 准备工作目录（不要直接将私有缓存目录作为仓库）

1. 将 `whisper-zh-ru-meeting-oss-starter.zip` 解压到 Downloads，得到项目文件夹。建议改名为 `whisper-zh-ru-meeting`。
2. 检查项目仅包含 `.py` / `.sh` / `.md` / 许可证 / GitHub 工作流等安全文件；模型和音频仍放在原有 `~/whisper-ru` 目录。
3. 仔细阅读 `README.md`、`.gitignore` 和 `LICENSE`，确认许可及项目描述准确。移除你没有权利再授权的第三方代码片段。
4. 执行不下载模型的本地测试：

```bash
cd "$HOME/Downloads/whisper-zh-ru-meeting"
python3 -m compileall -q .
python3 -m unittest -v test_v335
```

测试通过代表基本程序逻辑符合测试，不代表多语言准确率或完整新机器安装成功。

## 三、注册 GitHub 并设置 Git

1. 在 https://github.com/signup 注册账号，登录并启用双因素验证。
2. 打开 macOS 终端：

```bash
git --version
```

如果提示 Git 缺失，按照系统提示安装 Apple Command Line Tools；安装后再次执行 `git --version`。

3. 配置提交身份（用自己的 GitHub 用户名；邮箱可使用 GitHub 设置中的 noreply 隐私邮箱）：

```bash
git config --global user.name "YOUR_GITHUB_USERNAME"
git config --global user.email "YOUR_GITHUB_NOREPLY_EMAIL"
git config --global --list
```

这里填写的姓名和邮箱会进入 Git 提交元数据。不要误用不希望公开的私人邮箱。

## 四、建立本地 Git 仓库，确保没有私人文件

```bash
cd "$HOME/Downloads/whisper-zh-ru-meeting"
git init -b main
git status --short
```

第一次提交前，先用 `git status --short` 看文件列表，确认没有任何 `.mp3`、`.wav`、`all_clean.txt`、`HF_TOKEN` 文件、真实会议资料。可选进一步扫描：

```bash
find . -type f \( -iname '*.mp3' -o -iname '*.wav' -o -iname '*.safetensors' -o -iname '*.bin' \) -print
grep -RInE 'hf_[A-Za-z0-9]{12,}|-----BEGIN .*PRIVATE KEY-----' --include='*.py' --include='*.md' . || true
```

扫描只是防漏提醒，**不能代替人工核对**。

把经过检查的项目文件加入暂存区：

```bash
git add README.md LICENSE .gitignore .env.example SECURITY.md CONTRIBUTING.md CHANGELOG.md VERSION.txt
git add -- '*.py' '*.sh' '.github' 'docs'
git diff --cached --name-only
git diff --cached --stat
```

确定暂存列表不含敏感文件后执行：

```bash
git commit -m "chore: publish initial alpha source"
git log --oneline -1
```

注：单独的 `git add .` 并非一定错误，但第一次公开前建议明确指定文件类型并核对暂存列表。

## 五、在网站创建一个空的远程仓库

1. 访问 https://github.com/new 。
2. Owner：自己的 GitHub 账号；Repository name：`whisper-zh-ru-meeting`。
3. Description：`Local Chinese/Russian meeting transcription with overlap detection and bilingual translation aids`。
4. Visibility：希望开源就选 **Public**。
5. **不要**再勾选自动生成 README、`.gitignore` 或 LICENSE：这些文件在本地已经准备好，避免首次推送合并冲突。
6. 点击 **Create repository**。网页会显示形如 `https://github.com/USERNAME/whisper-zh-ru-meeting.git` 的地址。

来源：https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github

## 六、认证并上传

**最少依赖的 HTTPS 方案：** 安装 GitHub CLI（gh）或使用系统/网页提供的凭证管理方式完成 GitHub 认证，**不能用 GitHub 账号密码直接作为 Git HTTPS 密码**。如果已经安装 `gh`，推荐：

```bash
gh --version
gh auth login
```

在交互菜单中选择 `GitHub.com` → `HTTPS` → `Login with a web browser`，按显示的一次性代码在浏览器完成授权。如果没有 `gh`，可以使用 GitHub Desktop 登录并发布，或配置 SSH 密钥；不用为了本教程强行安装新的 CLI 工具。

认证后，返回本地项目目录，将 **USERNAME 替换为自己的用户名**：

```bash
cd "$HOME/Downloads/whisper-zh-ru-meeting"
git remote add origin https://github.com/USERNAME/whisper-zh-ru-meeting.git
git remote -v
git push -u origin main
```

刷新仓库网页，看到 README 与源码文件，才代表已经发布成功。

**完全不想用终端做远程认证？** 使用 https://desktop.github.com/ 下载 GitHub Desktop，登录账号，在 File → Add Local Repository 中选择这个仓库，再用 Publish repository（选择 Public）完成发布。初始仓库需要先 `git init` 和提交，或者直接用 Desktop 将文件夹创建成仓库；二选一，避免重复初始化。

## 七、制作首个 Release

仓库页 → `Releases` → `Draft a new release` → 选择/创建标签 `v0.1.0-alpha` → Title `v0.1.0-alpha — experimental source preview` → 勾选 `Set as a pre-release` → 填写简要变更说明 → 发布。

建议变更说明：**源码开放、S 顺序编号、中俄翻译候选、独立抢话检测、两音轨分离、缓存复用；模型与真实录音不随源码分发；尚未完成真实世界精度基准。**

参考：https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository

## 八、以后修改代码的标准循环

```bash
cd "$HOME/Downloads/whisper-zh-ru-meeting"
git status
# 修改 Python / README 文件
python3 -m unittest -v test_v335
git add meeting_v335.py README.md
git diff --cached
git commit -m "fix: improve separation cache handling"
git push
```

多人协作时新建分支（例如 `fix/overlap-cache`），推送后在 GitHub 打开 Pull Request；审查后合并到 `main`。项目问题通过 Issues 记录，绝不附私人会议音频。

## 九、常见错误处理

- `fatal: not a git repository`：没有 `cd` 到包含 `.git` 的项目目录。
- `remote origin already exists`：先运行 `git remote -v`，必要时用 `git remote set-url origin NEW_URL`，不要盲目重复添加。
- `Permission denied` / `Authentication failed`：GitHub 认证未配置，重新运行 `gh auth login` 或检查 SSH/凭证；不要输入账号登录密码代替 Token。
- `src refspec main does not match any`：还没有有效 commit，或者分支不叫 main；检查 `git branch`、`git log -1`。
- `failed to push some refs`：远程可能已经自动创建 README 或被他人更新；不要直接 `git push --force`，先检查远程分支和冲突。
- 大文件被拒绝：先找到来源，删除暂存、清理本地未推送历史再提交，不能仅把文件加入 `.gitignore` 就以为历史中已经消失。
- 意外上传了 Token：**先去 Hugging Face 立即撤销并新建 Token**，再按 GitHub 官方敏感数据移除指南处理历史。公开后删除文件并不等于从历史和其他副本中完全消失。

## 十、发布前的最终验收

1. 项目作者有权公开全部自己准备授权的代码；外部模型和依赖的版权要求分别遵守。
2. 所有示例数据为合成或已获许可的数据；无真实会议文字、录音、访问 Token。
3. 本地单元测试通过，README 中列出未验证项，不把测试计数当成模型效果。
4. `.gitignore`、LICENSE、SECURITY、CONTRIBUTING 已认真检查。
5. `git diff --cached --name-only` 仅有预期公开的文件。
6. 你明确选择 Public 后再推送；公开后别人可以克隆仓库。
7. 发布后用浏览器以未登录视角查看 README、文件和 Release，确保没有任何不希望公开的内容。

**重要区别：**GitHub 仓库公开的是代码，不是转写服务。用户还需要自行安装外部 whisper.cpp、模型权重和相关环境；V3.3.5 的 `render` 等命令要求先前 ASR 缓存。后续要实现“一键从新 MP3 开始”，应另做可验证的安装与集成测试，再发布正式 1.0 版本。
