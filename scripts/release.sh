#!/usr/bin/env bash
# ==============================================================================
# 全生命周期发版防呆脚本 (Release Automation Script)
#
# 提供严谨的前置自检 (Pre-flight)、语义化版本推荐、演练模式与生态扩展插槽。
#
# @author Ateng
# @since 2026-10-08
# ==============================================================================

set -eo pipefail

# ==============================================================================
# 语言生态与版本号递增扩展插槽 (Version Bump Hook Slot)
#
# 若 downstream 具体项目需要自动更新工程文件版本号，解除下方对应语言的一行配置，
# 或在 custom_bump_version 函数中编写自定义命令。传入参数 $1 为不带 v 前缀的版本号（如 1.2.0）。
# ==============================================================================
# shellcheck disable=SC2034
custom_bump_version() {
  local RAW_VERSION="$1"   # 纯版本号，如 "1.2.0"
  local FULL_VERSION="v$1" # 带 v 前缀的完整标签，如 "v1.2.0"
  : "${RAW_VERSION}" "${FULL_VERSION}"

  # --- [选项 1] Java (Maven) ---
  # mvn versions:set -DnewVersion="$RAW_VERSION" -DgenerateBackupPoms=false

  # --- [选项 2] Node.js (npm / pnpm) ---
  # npm version "$RAW_VERSION" --no-git-tag-version

  # --- [选项 3] Rust (Cargo) ---
  # cargo set-version "$RAW_VERSION"

  # --- [选项 4] Python (Poetry) ---
  # poetry version "$RAW_VERSION"

  # 若本项目无需更新任何文件（如 Go 项目或纯通用底座），保持此处为空即可。
  return 0
}

# 终端色彩定义（在支持的终端下提供视觉区分，兼容非 TTY 降级）
if [ -t 1 ]; then
  COLOR_RESET="\033[0m"
  COLOR_BOLD="\033[1m"
  COLOR_GREEN="\033[32m"
  COLOR_BLUE="\033[34m"
  COLOR_YELLOW="\033[33m"
  COLOR_RED="\033[31m"
  COLOR_CYAN="\033[36m"
else
  COLOR_RESET=""
  COLOR_BOLD=""
  COLOR_GREEN=""
  COLOR_BLUE=""
  COLOR_YELLOW=""
  COLOR_RED=""
  COLOR_CYAN=""
fi

log_info() {
  printf "%b[INFO]%b %s\n" "${COLOR_BLUE}" "${COLOR_RESET}" "$1"
}

log_success() {
  printf "%b[SUCCESS]%b %s\n" "${COLOR_GREEN}" "${COLOR_RESET}" "$1"
}

log_warn() {
  printf "%b[WARN]%b %s\n" "${COLOR_YELLOW}" "${COLOR_RESET}" "$1" >&2
}

log_error() {
  printf "%b[ERROR]%b %s\n" "${COLOR_RED}" "${COLOR_RESET}" "$1" >&2
}

# 参数解析
DRY_RUN=false
TARGET_VERSION=""
AUTO_CONFIRM=false

for arg in "$@"; do
  case "$arg" in
    -n|--dry-run)
      DRY_RUN=true
      ;;
    -y|--yes)
      AUTO_CONFIRM=true
      ;;
    -h|--help)
      printf "%b\n" "用法: bash scripts/release.sh [版本号] [选项]"
      printf "%b\n" ""
      printf "%b\n" "参数说明:"
      printf "%b\n" "  [版本号]      可选。符合 SemVer 格式的标签名（如 v1.0.0）。若未提供且处于交互终端则启动向导。"
      printf "%b\n" "  -y, --yes     可选。跳过最终确认提示，直接执行发版（适合 CI 或 AI 自动化调用）。"
      printf "%b\n" "  -n, --dry-run 可选。演练模式，仅执行检查并预览拟执行命令，不产生任何真实 Git 变更。"
      printf "%b\n" "  -h, --help    显示帮助信息。"
      printf "%b\n" ""
      printf "%b\n" "无参数运行示例 (交互向导模式):"
      printf "%b\n" "  bash scripts/release.sh"
      printf "%b\n" ""
      printf "%b\n" "非交互式调用示例 (AI / 自动化执行):"
      printf "%b\n" "  bash scripts/release.sh v1.4.0 -y"
      printf "%b\n" "  bash scripts/release.sh v1.4.0 --dry-run"
      exit 0
      ;;
    v*)
      TARGET_VERSION="$arg"
      ;;
    *)
      if [[ "$arg" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]]; then
        TARGET_VERSION="v$arg"
      else
        log_error "无法识别的参数: $arg (版本号必须以 v 开头或为数字版本，如 v1.0.0；选项支持 -y/--yes, -n/--dry-run, -h/--help)"
        exit 1
      fi
      ;;
  esac
done

printf "\n%b\n" "${COLOR_BOLD}${COLOR_CYAN}================================================================${COLOR_RESET}"
printf "%b\n" "${COLOR_BOLD}${COLOR_CYAN}         OSS 自动化发版防呆自检程序 (Release Guard)              ${COLOR_RESET}"
printf "%b\n" "${COLOR_BOLD}${COLOR_CYAN}================================================================${COLOR_RESET}"

if [ "$DRY_RUN" = true ]; then
  printf "%b\n\n" "${COLOR_BOLD}${COLOR_YELLOW}>>> 【演练模式 DRY-RUN 已激活】以下操作仅做安全预览，不修改任何 Git 状态 <<<\n${COLOR_RESET}"
fi

# ==============================================================================
# 1. 前置防呆自检 (Pre-flight Checks)
# ==============================================================================
log_info "正在执行发版前置环境自检 (Pre-flight Checks)..."

# 1.1 检查 Git 工具
if ! command -v git >/dev/null 2>&1; then
  log_error "未检测到 git 命令，请先安装 Git。"
  exit 1
fi

# 1.2 检查是否在 Git 工作区并自动定位至仓库根目录
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  log_error "当前目录不在 Git 仓库内，请在 Git 仓库内执行。"
  exit 1
fi
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# 1.3 校验当前分支是否为默认主干分支 (main 或 master)
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ] && [ "$CURRENT_BRANCH" != "master" ]; then
  log_error "当前处于分支 [$CURRENT_BRANCH]，严禁在非主干分支发版！请先切换至 main 分支: git checkout main"
  exit 1
fi
log_success "当前处于合规主干分支: $CURRENT_BRANCH"

# 1.4 校验工作区是否干净 (脏代码拦截)
DIRTY_FILES=$(git status --porcelain)
if [ -n "$DIRTY_FILES" ]; then
  log_error "检测到本地工作区存在未提交的变更或未跟踪文件！"
  printf "%b\n" "脏文件列表:"
  printf "%s\n" "$DIRTY_FILES"
  log_error "为防止未经验证的代码随版本发版，发版前工作区必须绝对干净！"
  log_info "提示: 你可以先运行 'bash scripts/commit.sh' 规范化提交变更，或使用 'git stash' 暂存工作区。"
  exit 1
fi
log_success "工作区状态检查通过: 干净无未提交代码"

# 1.5 校验与远端仓库同步状态 (防漏拉远端代码与超前提交检测，双重 5 秒超时守卫)
log_info "正在拉取远端最新状态 (git fetch origin)..."
AHEAD_COUNT=0
if GIT_SSH_COMMAND="ssh -o ConnectTimeout=5 -o BatchMode=yes" git -c http.timeout=5 fetch origin >/dev/null 2>&1; then
  BEHIND_COUNT=$(git rev-list "HEAD..origin/$CURRENT_BRANCH" --count 2>/dev/null || echo "0")
  if [ "$BEHIND_COUNT" -gt 0 ]; then
    log_error "本地分支落后于远端 origin/$CURRENT_BRANCH ($BEHIND_COUNT 个提交未合并)！"
    log_error "发版前必须合并远端最新代码，请先执行: git pull origin $CURRENT_BRANCH"
    exit 1
  fi
  AHEAD_COUNT=$(git rev-list "origin/$CURRENT_BRANCH..HEAD" --count 2>/dev/null || echo "0")
  if [ "$AHEAD_COUNT" -gt 0 ]; then
    log_warn "检测到本地主干分支超前于远端 ($AHEAD_COUNT 个提交未推送)。发版流程将在打标签前自动推送主干分支。"
  else
    log_success "与远程分支状态同步: 本地主干与远端完全一致"
  fi
else
  log_warn "无法连接远程仓库进行同步校验（可能处于离线环境或网络受阻），跳过远端拉取校验。"
fi

# ==============================================================================
# 2. 版本号解析与语义化推导
# ==============================================================================
# 获取最近一次历史 Git Tag (优先匹配标准 v* 标签；若存量项目未带 v 则兼容匹配纯数字标签)
LATEST_TAG=$(git describe --tags --abbrev=0 --match "v[0-9]*" 2>/dev/null || echo "")
if [ -z "$LATEST_TAG" ]; then
  LATEST_TAG=$(git describe --tags --abbrev=0 --match "[0-9]*" 2>/dev/null || echo "")
fi

if [ -z "$LATEST_TAG" ]; then
  log_info "未检测到历史版本 Tag，将基于初始版本 v0.0.0 推导。"
  BASE_VERSION="v0.0.0"
  V_MAJOR=0
  V_MINOR=0
  V_PATCH=0
else
  if [[ "$LATEST_TAG" =~ ^[0-9] ]]; then
    log_info "检测到存量项目历史版本 Tag（未带 v 前缀）: $LATEST_TAG，本次发版将自动对齐为标准 SemVer (v$LATEST_TAG)"
    BASE_VERSION="v$LATEST_TAG"
  else
    log_info "检测到当前最新版本 Tag: $LATEST_TAG"
    BASE_VERSION="$LATEST_TAG"
  fi
  # 提取版本号数字部分 (去掉 v 前缀以及可能的预发布后缀与构建元数据)
  CLEAN_VER=$(echo "${BASE_VERSION#v}" | cut -d'-' -f1 | cut -d'+' -f1)
  V_MAJOR=$(echo "$CLEAN_VER" | cut -d'.' -f1)
  V_MINOR=$(echo "$CLEAN_VER" | cut -d'.' -f2)
  V_PATCH=$(echo "$CLEAN_VER" | cut -d'.' -f3)

  # 数字容错保底
  V_MAJOR=${V_MAJOR:-0}
  V_MINOR=${V_MINOR:-0}
  V_PATCH=${V_PATCH:-0}
fi

NEXT_PATCH="v${V_MAJOR}.${V_MINOR}.$((V_PATCH + 1))"
NEXT_MINOR="v${V_MAJOR}.$((V_MINOR + 1)).0"
NEXT_MAJOR="v$((V_MAJOR + 1)).0.0"

# 若未通过命令行传入版本号，启动交互式选择向导
if [ -z "$TARGET_VERSION" ]; then
  if [ ! -t 0 ]; then
    log_error "非交互式环境下必须显式提供版本号参数！例如: bash scripts/release.sh v1.4.0 -y"
    exit 1
  fi

  printf "\n%b\n" "${COLOR_BOLD}请选择本次发布的目标版本号 (Semantic Versioning):${COLOR_RESET}"
  printf "  %b0)%b 退出发版流程 (Quit / Exit)\n" "${COLOR_YELLOW}" "${COLOR_RESET}"
  printf "  %b1)%b Patch  (🐛 补丁修复 / 缺陷修复)   : %b%s%b\n" "${COLOR_GREEN}" "${COLOR_RESET}" "${COLOR_CYAN}" "$NEXT_PATCH" "${COLOR_RESET}"
  printf "  %b2)%b Minor  (🚀 新功能引入 / 向下兼容) : %b%s%b\n" "${COLOR_GREEN}" "${COLOR_RESET}" "${COLOR_CYAN}" "$NEXT_MINOR" "${COLOR_RESET}"
  printf "  %b3)%b Major  (💥 破坏性更新 / 架构大改) : %b%s%b\n" "${COLOR_GREEN}" "${COLOR_RESET}" "${COLOR_CYAN}" "$NEXT_MAJOR" "${COLOR_RESET}"
  printf "  %b4)%b Custom (✍️  自定义输入版本号，如 pre-release)\n\n" "${COLOR_GREEN}" "${COLOR_RESET}"

  CHOICE=""
  while [ -z "$TARGET_VERSION" ]; do
    printf "%b请输入选项 [0-4]: %b" "${COLOR_BOLD}" "${COLOR_RESET}"
    read -r CHOICE
    CHOICE=$(echo "$CHOICE" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    case "$CHOICE" in
      0|[qQ]|[eE][xX][iI][tT])
        log_info "已取消发版。"
        exit 0
        ;;
      1) TARGET_VERSION="$NEXT_PATCH" ;;
      2) TARGET_VERSION="$NEXT_MINOR" ;;
      3) TARGET_VERSION="$NEXT_MAJOR" ;;
      4)
        printf "%b请输入自定义版本号 (需符合 SemVer，如 v1.2.3 或 v1.2.3-beta.1): %b" "${COLOR_BOLD}" "${COLOR_RESET}"
        read -r CUSTOM_VER
        CUSTOM_VER=$(echo "$CUSTOM_VER" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        if [[ "$CUSTOM_VER" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]]; then
          CUSTOM_VER="v$CUSTOM_VER"
        fi
        TARGET_VERSION="$CUSTOM_VER"
        ;;
      *)
        log_warn "无效选项，请输入 0 到 4 之间的有效编号。"
        ;;
    esac
  done
fi

# 严格校验最终版本号格式
SEMVER_REGEX="^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$"
if ! echo "$TARGET_VERSION" | grep -Eq "$SEMVER_REGEX"; then
  log_error "目标版本号 [$TARGET_VERSION] 不符合语义化版本规范！必须形如 v1.0.0 或 v1.0.0-rc.1"
  exit 1
fi
log_success "目标发版版本号校验合规: $TARGET_VERSION"

# 校验本地及远端是否已存在同名 Tag
if git rev-parse -q --verify "refs/tags/$TARGET_VERSION" >/dev/null 2>&1; then
  log_error "本地已存在相同标签 [$TARGET_VERSION]，禁止重复发版！"
  exit 1
fi

if GIT_SSH_COMMAND="ssh -o ConnectTimeout=5 -o BatchMode=yes" git -c http.timeout=5 ls-remote --tags origin "refs/tags/$TARGET_VERSION" 2>/dev/null | grep -q "$TARGET_VERSION"; then
  log_error "远程仓库已存在相同标签 [$TARGET_VERSION]，禁止重复发版！"
  exit 1
fi
log_success "标签唯一性检查通过: $TARGET_VERSION 未曾被使用"

# 提取不带 v 的纯版本号供插槽使用
RAW_VERSION="${TARGET_VERSION#v}"

# ==============================================================================
# 3. 拟执行动作汇总与最终确认
# ==============================================================================
printf "\n%b\n" "${COLOR_BOLD}${COLOR_CYAN}----------------------- 发版计划摘要 -----------------------${COLOR_RESET}"
printf "  基准历史版本 : %s\n" "$BASE_VERSION"
printf "  本次目标版本 : %b%s%b\n" "${COLOR_BOLD}${COLOR_GREEN}" "$TARGET_VERSION" "${COLOR_RESET}"
printf "  目标 Git 分支: %s\n" "$CURRENT_BRANCH"
if [ "$AHEAD_COUNT" -gt 0 ]; then
  printf "  主干前置同步 : 推送本地超前提交至 origin/%s (%s 个提交)\n" "$CURRENT_BRANCH" "$AHEAD_COUNT"
fi
printf "  拟执行动作   : 1. 执行工程版本更新钩子 (custom_bump_version)\n"
printf "                 2. 创建附注标签: git tag -a %s -m \"Release %s\"\n" "$TARGET_VERSION" "$TARGET_VERSION"
printf "                 3. 原子同步分支与标签: git push origin %s %s\n" "$CURRENT_BRANCH" "$TARGET_VERSION"
printf "                 4. 触发 GitHub Actions 自动发版与产物挂载流水线\n"
printf "%b\n\n" "${COLOR_BOLD}${COLOR_CYAN}------------------------------------------------------------${COLOR_RESET}"

if [ "$DRY_RUN" = true ]; then
  log_info "[DRY-RUN] 模拟执行 custom_bump_version $RAW_VERSION..."
  log_info "[DRY-RUN] 模拟执行 git tag -a $TARGET_VERSION -m 'Release $TARGET_VERSION'..."
  log_info "[DRY-RUN] 模拟执行 git push origin $CURRENT_BRANCH $TARGET_VERSION (原子合并推送)..."
  log_success "[DRY-RUN] 演练完成！所有发版防呆检查均已通过，实际发版将按上述规划执行。"
  exit 0
fi

if [ "$AUTO_CONFIRM" = true ] || [ ! -t 0 ]; then
  log_info "已指定 -y 或处于非交互环境，跳过人工确认，直接执行发版流程。"
else
  printf "%b确认以此计划立即发版并推送到 GitHub? [Y/n]: %b" "${COLOR_BOLD}" "${COLOR_RESET}"
  read -r CONFIRM_RELEASE
  CONFIRM_RELEASE=$(echo "$CONFIRM_RELEASE" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  case "$CONFIRM_RELEASE" in
    [nN][oO]|[nN])
      log_warn "发版操作已由用户取消，未执行任何改动。"
      exit 0
      ;;
    *)
      ;;
  esac
fi

# ==============================================================================
# 4. 执行版本号更新钩子与变更提交
# ==============================================================================
log_info "正在执行工程版本号更新钩子 (custom_bump_version)..."
custom_bump_version "$RAW_VERSION"

# 检查钩子是否修改了版本文件
HOOK_CHANGES=$(git status --porcelain)
NEED_PUSH_BRANCH=false
if [ -n "$HOOK_CHANGES" ]; then
  log_info "检测到扩展钩子更新了工程文件，正在全量暂存并创建版本提交..."
  git add -A
  git commit -m "chore(release): bump version to $TARGET_VERSION"
  NEED_PUSH_BRANCH=true
elif [ "$AHEAD_COUNT" -gt 0 ]; then
  log_info "检测到本地有 $AHEAD_COUNT 个超前提交，将随发版标签一同推送到远端..."
  NEED_PUSH_BRANCH=true
else
  log_info "未检测到工程文件变动（当前工程保持无版本文件模式）。"
fi

# ==============================================================================
# 5. 打标签与原子推送 (含推送失败自动回滚)
# ==============================================================================
log_info "正在创建附注 Git Tag: $TARGET_VERSION..."
git tag -a "$TARGET_VERSION" -m "Release $TARGET_VERSION"

if [ "$NEED_PUSH_BRANCH" = true ]; then
  log_info "正在原子推送主干变更与版本标签 (git push origin $CURRENT_BRANCH $TARGET_VERSION)..."
  if ! git push origin "$CURRENT_BRANCH" "$TARGET_VERSION"; then
    log_error "原子推送分支与标签到远程仓库失败！"
    log_warn "正在自动回滚本地标签，以防本地脏 Tag 阻塞后续发版..."
    git tag -d "$TARGET_VERSION" >/dev/null 2>&1 || true
    log_error "发版中断：推送失败，本地标签已安全清理。"
    exit 1
  fi
  log_success "主干变更与版本标签已原子同步至远端。"
else
  log_info "正在推送标签到远程仓库 (git push origin $TARGET_VERSION)..."
  if ! git push origin "$TARGET_VERSION"; then
    log_error "推送标签 $TARGET_VERSION 到远程仓库失败！"
    log_warn "正在自动回滚本地标签，以防本地脏 Tag 阻塞后续发版..."
    git tag -d "$TARGET_VERSION" >/dev/null 2>&1 || true
    log_error "发版中断：标签推送失败，本地标签已安全清理。"
    exit 1
  fi
  log_success "版本标签推送成功！"
fi

# 获取远端仓库地址生成 GitHub Release 网页直达链接
REMOTE_URL=$(git config --get remote.origin.url || echo "")
REPO_SLUG=""
if [[ "$REMOTE_URL" =~ github\.com[:/]([^/]+)/([^/.]+)(\.git)? ]]; then
  REPO_SLUG="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
fi

printf "\n%b\n" "${COLOR_BOLD}${COLOR_GREEN}================================================================${COLOR_RESET}"
printf "%b\n" "${COLOR_BOLD}${COLOR_GREEN}               🎉 版本 $TARGET_VERSION 发版指令已成功触发！      ${COLOR_RESET}"
printf "%b\n" "${COLOR_BOLD}${COLOR_GREEN}================================================================${COLOR_RESET}"
if [ -n "$REPO_SLUG" ]; then
  printf "GitHub Actions 发版流水线已自动开始运行，稍后可访问 Release 页面查看产物:\n"
  printf "👉 %bhttps://github.com/%s/releases/tag/%s%b\n\n" "${COLOR_CYAN}" "$REPO_SLUG" "$TARGET_VERSION" "${COLOR_RESET}"
fi
