# 贡献指南 (Contributing Guide)

感谢你关注并愿意为本项目贡献力量！为了保持高效协作与高质量的代码维护，请在提交代码前阅读以下规范。

---

## 1. 协作与分支模型

本项目遵循标准的 **GitHub Flow** 工作流：

1. **Fork 本仓库** 到你个人的 GitHub 账号；
2. **基于 `main` 分支拉取新的特性分支**：
   ```bash
   git checkout -b feat/your-feature-name
   # 或者缺陷修复分支
   git checkout -b fix/issue-description
   ```
3. 在本地完成修改，确保自测通过并补充相应测试用例；
4. 提交更改并推送到你的远程分支：
   ```bash
   git push origin feat/your-feature-name
   ```
5. 在 GitHub 上向本仓库的 `main` 分支发起 **Pull Request**。

---

## 2. Commit 提交信息规范

本项目遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/) 规范，统一采用以下格式：

```text
<type>(<scope>): <subject>
```

### 常用类型说明

| 类型 | 说明 | 示例 |
| :--- | :--- | :--- |
| `feat` | 新增功能或特性 | `feat(auth): 支持 OAuth2 登录授权` |
| `fix` | 缺陷与 Bug 修复 | `fix(parser): 修复空字符串解析导致空指针的问题` |
| `docs` | 仅文档更新或修改 | `docs: 完善快速开始与环境配置文档` |
| `style` | 代码格式调整（空格、分号等，不影响逻辑） | `style: 优化代码排版与换行` |
| `refactor` | 代码重构（既非新增特性也非修复缺陷） | `refactor(core): 抽取公共工具类` |
| `perf` | 性能优化 | `perf(cache): 引入本地二级缓存提升吞吐量` |
| `test` | 增加或重构单元测试与集成测试 | `test: 补充用户服务边界条件测试用例` |
| `build` | 构建系统、外部依赖或脚手架调整 | `build: 升级依赖版本至最新稳定版` |
| `ci` | CI/CD 流水线与 GitHub Actions 脚本修改 | `ci: 优化 release 自动化发版流程` |
| `chore` | 其他琐碎杂项（不改动源码与测试） | `chore: 更新 .gitignore 忽略规则` |
| `revert` | 恢复或回滚此前的某次历史提交 | `revert: feat(auth): 回退登录授权变动` |

### 🛡️ 本地 Git 提交钩子守门 (Git Hooks)

本项目通过 Git 原生钩子 [`.githooks/commit-msg`](./.githooks/commit-msg) 在本地提交阶段实时校验信息格式，零外部依赖（无需安装 Node.js、Husky 或 Python 环境）：
- **一键激活**：运行 `bash scripts/setup.sh` 会自动激活；或在仓库根目录下手动执行：
  ```bash
  git config core.hooksPath .githooks
  ```
- **智能放行**：自动放行分支合并 (`Merge branch...`)、代码回退 (`Revert "..."`) 以及变基临时提交 (`fixup!` / `squash!`)；
- **错误引导**：若提交信息不符合规范，钩子将中断提交、高亮输出格式诊断，并引导使用 `commit.sh` 快速组装。

### 💡 推荐：使用规范化提交助手

本项目内置了 POSIX Bash 编写的提交助手 [scripts/commit.sh](./scripts/commit.sh)，同时支持**人工交互式向导**与 **AI / 自动化非交互式调用**：

#### 方式 A：交互式向导模式（适合人工日常提交）
通过编号菜单引导选择类型、输入范围与描述，自动组装符合规范的提交信息并支持一键推送（支持输入 `0` 或 `q` 随时退出）：
```bash
# 在 Linux / macOS 或 Windows Git Bash 中直接运行
bash scripts/commit.sh
```

#### 方式 B：非交互式命令行模式（适合 AI Agent / 自动化脚本）
支持通过标准命令行参数一键组装 Conventional Commit 并推送到远端，无需任何键盘等待交互（推荐先使用 `git add <file>` 显式安全暂存目标文件）：
```bash
# 标准规范化提交并自动推送 (推荐先显式精准暂存)
git add <file-path>
bash scripts/commit.sh -t feat -s core -m "新增用户认证能力" -l "支持多工单关联" -c "12, 13" -p -y

# 自动变基推送 (若远程存在更新，自动受控变基同步后推送)
git add <file-path>
bash scripts/commit.sh -t feat -s core -m "实现新功能" -p -y -r

# 修复 Bug 且不自动推送
git add <file-path>
bash scripts/commit.sh -t fix -s parser -m "修复空指针异常" -c 14 -y

# 标记破坏性更新 (Breaking Changes)
git add <file-path>
bash scripts/commit.sh -t feat -s api -m "重构对外接口协议" -b -p -y

# 空跑演练模式 (预览生成的提交信息与 Git Trailers，不实际执行提交与推送)
bash scripts/commit.sh -t feat -s core -m "演练测试" -l "条目1" -c "12, 13" -n
```

| 参数选项 | 说明 |
| :--- | :--- |
| `-t, --type <type>` | 提交类型 (`feat\|fix\|docs\|style\|refactor\|perf\|test\|build\|ci\|chore\|revert`) |
| `-s, --scope <scope>` | 可选影响范围（推荐: `core`、`cli`、`api`、`workflow`、`deps`、`docs`） |
| `-m, --message <msg>` | 提交主体描述 / 标题（必填，简明扼要） |
| `-l, --item <text>` | 详细改动条目说明（可多次传入，自动拼接为 Markdown 列表） |
| `-d, --detail, --body <text>` | 详细正文描述段落（可与 `-l` 结合使用） |
| `-c, --close, --closes <issues>` | 关联并自动关闭 GitHub Issue（支持单/多工单，如 `-c 12 -c 13` 或 `-c "12, 13"`，按行独立生成 `Closes #ID` Git Trailer） |
| `-i, --issue, --ref, --refs <issues>` | 关联引用 GitHub Issue（格式同上，按行独立生成 `Refs #ID` Git Trailer） |
| `-b, --breaking` | 标记为破坏性更新（自动追加 `!` 标识） |
| `-n, --dry-run` | 空跑演练模式（仅打印提交信息组装预览与暂存状态，不执行提交与推送） |
| `-a, --all` | 自动暂存全部已修改及未跟踪文件（相当于 `git add -A`，请谨慎使用） |
| `-p, --push` | 提交成功后自动推送至当前分支远程仓库（内置 5 秒超时预检、全新分支 `-u` 上游自适应与并发竞态守卫） |
| `-r, --rebase` | 当远程分支存在更新时，允许在提交后自动执行受控变基同步 (`git pull --rebase origin <branch>`)；变基遇冲突自动执行 `git rebase --abort` 原子回滚 |
| `-y, --yes` | 跳过确认提示直接执行提交 |
| `-h, --help` | 查看详细帮助说明 |

> 💡 **Windows 终端提示**：无论使用 PowerShell 还是 CMD，只要系统安装了 Git，直接输入 `bash scripts/commit.sh` 即可调用 Git Bash 解释器顺畅执行。

---

## 3. Pull Request 与交付规范

### 📝 特性即文档与单工单闭环门禁 (Doc-as-Feature & Per-Ticket Sync)
为了杜绝“代码已合入但使用文档脱节、下游接入全凭猜”的现象，本项目推行**特性即文档**与**单工单文档准出闭环**准则：
- **同工单/同提交交付**：任何新增功能、对外接口、关键配置改造，必须在同一个 Pull Request 与提交中同步更新相关文档与使用示例；
- **三板斧质量标准**：
  1. **一句话场景痛点**：清晰说明为什么需要该特性、解决了什么业务/工程问题；
  2. **可直接运行的最小配置/代码用例**：提供复制即用的 YAML、Java、命令或示例代码；
  3. **预期控制台/SQL/契约呈现**：明确展示标准调用或成功运行后的控制台输出、API 响应或数据落库形态。
- **缺少文档与用例的代码改动，一律视为未达验收标准。**

### 🔄 PR 流程与合并规约
- **PR 标题规范**：PR 标题必须同样遵循 [Conventional Commits](#2-commit-提交信息规范) 格式（如 `feat: 新增能力` 或 `fix: 修复缺陷`），CI 会对其进行自动化合规校验；
- **模版填写**：发起 PR 时，请按模版完整填写变更背景、解决的问题以及关联的 Issue（如 `close #12`）；
- **CI 绿灯**：确保 CI 流水线测试全部处于通过状态；
- **审查与合并**：代码审查（Code Review）提出修改意见后，在原分支继续提交即可自动同步至 PR；合并后特性分支将被删除；
- **推荐合并方式**：仓库推荐采用 **Squash and merge（压缩合并）**，合入时自动采用合规的 PR 标题作为主干提交信息，保证 Release 变更日志 100% 精确。

---

## 4. 版本发版机制与发布说明

本项目通过 GitHub Actions 实现了现代化的**多通道自动化发版体系**：
- **通道一（推荐首选，零脚本零终端依赖）**：在 GitHub 仓库 Actions 界面通过 `workflow_dispatch` 手动输入版本号一键触发发版；
- **通道二（纯原生）**：在本地通过原生 `git tag` & `git push` 命令触发；
- **通道三（进阶可选工具箱）**：使用内置发版防呆脚本 `scripts/release.sh`，在本地完成 5 重前置自检（Pre-flight）、分支提交状态对齐与多语言版本文件递增。

---

### 🌐 方式一：GitHub Actions 网页端云端调度发版 (推荐首选)

若维护者未在本地配置终端发版环境，可直接在 GitHub 网页端一键触发：
1. 前往 GitHub 仓库页面，点击顶部 **Actions** 标签页；
2. 在左侧选择 **GitHub Release** 流水线；
3. 点击右侧蓝色的 **Run workflow** 下拉按钮；
4. 在 **发布版本号** 输入框中填入目标版本号（如 `v1.2.0`）；
5. 点击绿色 **Run workflow** 按钮启动流水线。
GitHub Actions 将自动执行版本格式与触发分支校验（自动拦截非默认主干分支上的误操作）、在当前主干提交显式创建并推送附注 Git Tag，随后自动提取日志完成 Release 发布。

---

### 🛠️ 方式二：原生 Git 命令行手动打标触发 (纯原生)

若偏好纯手动敲命令，可按照标准步骤在本地打标并推送：

```bash
# 步骤 A：确保本地最新代码已推送到 main 分支且工作区干净
git checkout main
git pull origin main
git push origin main

# 步骤 B：打附注版本标签并推送到 GitHub (支持 v1.0.0, v1.0.0-beta.1 等)
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

---

### 🚀 方式三 (进阶可选工具箱)：使用本地发版防呆脚本

面向重度终端用户，项目提供了具备 5 重前置防呆自检的 POSIX Bash 发版脚本 [scripts/release.sh](./scripts/release.sh)。它能够自动校验当前是否处于 `main` 主干分支、工作区是否干净无未提交代码、本地是否落后或超前远端最新提交（超前提交自动同步）、目标 Tag 是否重名冲突，推送失败时自动回滚本地标签避免脏 Tag 阻塞，并智能推导下一个语义化版本（Patch/Minor/Major）：

```bash
# 1. 安全演练模式（强烈推荐在正式发版前演练，只检查并打印拟执行动作，零污染 Git 历史）
bash scripts/release.sh --dry-run
# 演练指定版本发版
bash scripts/release.sh v1.0.0 --dry-run

# 2. 交互式发版（自动提取最新 Tag，交互引导选择 Patch / Minor / Major，输入 0 可随时退出）
bash scripts/release.sh

# 3. 指定版本号快速发版（可追加 -y 跳过交互确认，适合 AI Agent / CI 流水线静默执行）
bash scripts/release.sh v1.0.0
bash scripts/release.sh v1.0.0 -y
```

> 💡 **Windows 终端提示**：无论使用 PowerShell 还是 CMD，只要系统安装了 Git，直接输入 `bash scripts/release.sh` 即可调用 Git Bash 解释器顺畅执行。
>
> 🧩 **多语言工程版本文件递增（扩展插槽）**：
> 若你的具体项目需要在打 Tag 时同步更新工程元数据文件中的版本号（如 `pom.xml`、`package.json`、`Cargo.toml` 等），只需打开 [scripts/release.sh](./scripts/release.sh)，在顶部的 `custom_bump_version()` 函数插槽中解除对应语言的一行命令注释即可。脚本在自检通过后会自动调用该命令、创建 `chore(release): bump version to ...` 提交并自动推送到主干分支。

---

### ⚙️ 自动化流水线运行
标签推送后，GitHub Actions 将会自动执行 [`.github/workflows/release.yml`](./.github/workflows/release.yml)：
- 自动提取自上一版本以来的全部提交与 PR，由 `git-cliff` 关联 PR 链接与新贡献者致谢，格式化为发布日志；
- 自动创建 GitHub Release 并挂载发布内容；
- 将打包产物与 `checksums.txt` 挂载至附件（若配置了构建步骤）；
- **生态分发与镜像解耦**：若需自动分发至 Maven Central / npm / PyPI / Crates.io 等包仓库或构建多架构 Docker 镜像，可直接选用模版库 [`.github/workflow-templates/`](./.github/workflow-templates/) 中对应的 `publish.yml` 与 `docker.yml` 独立流水线。

> 💡 **安全校验和 (SHA-256 Checksums) 验证指引**：
> 下游用户或测试者下载产物与 `checksums.txt` 后，可在终端通过原生命令一键验证文件完整性：
> - **Linux**：`sha256sum -c checksums.txt --ignore-missing`
> - **macOS**：`shasum -a 256 -c checksums.txt`
> - **Windows (PowerShell)**：`Get-FileHash .\your-asset-file.tar.gz -Algorithm SHA256`

### 中心仓库发布凭据 (Secrets) 参考

如需发布至官方包管理仓库，请在仓库的 **Settings -> Secrets and variables -> Actions** 中配置对应凭证：

- **npm**: 推荐配置 [npm Trusted Publishing (OIDC)](https://docs.npmjs.com/trusted-publishers)，或配置 `NPM_TOKEN`；
- **Maven Central**: 推荐使用现代化 Sonatype Central Portal，配置 `CENTRAL_USERNAME` 与 `CENTRAL_PASSWORD`（或传统 OSSRH 凭据），以及 `MAVEN_GPG_PRIVATE_KEY` 与 `MAVEN_GPG_PASSPHRASE`；
- **PyPI**: 推荐使用官方 [PyPI Trusted Publisher (OIDC)](https://docs.pypi.org/trusted-publishers/)，免配静态密钥；
- **Docker 镜像**:
  - **GHCR (推荐)**: 默认直接使用系统内置 `GITHUB_TOKEN`，零 Secret 配置；
  - **Docker Hub**: 配置 `DOCKERHUB_USERNAME` 与 `DOCKERHUB_TOKEN`；
- **Go 二进制工具 (GoReleaser)**: 默认直接使用系统内置 `GITHUB_TOKEN` 上传附件至 Release，零 Secret 配置；
- **GitHub Pages (前端/文档)**: 默认使用官方工作流部署，仅需在仓库 **Settings -> Pages -> Build and deployment -> Source** 切换为 **GitHub Actions** 即可。

