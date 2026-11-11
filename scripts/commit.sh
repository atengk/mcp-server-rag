#!/usr/bin/env bash
# ==============================================================================
# Conventional Commits 规范化提交助手 (支持交互式与非交互式/AI自动化调用)
#
# 引导开发者或自动化脚本按规范组装提交信息，提供变更预览、类型校验与一键推送。
#
# @author Ateng
# @since 2026-10-08
# ==============================================================================

set -eo pipefail

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

# ==============================================================================
# 1. 命令行参数解析 (支持非交互式调用)
# ==============================================================================
CLI_TYPE=""
CLI_SCOPE=""
CLI_MSG=""
CLI_BREAKING=false
CLI_STAGE_ALL=false
CLI_PUSH=false
CLI_REBASE=false
AUTO_CONFIRM=false
IS_NON_INTERACTIVE=false
DRY_RUN=false
CLI_BODY=""
CLI_ITEMS=()
CLI_CLOSES=()
CLI_REFS=()

show_help() {
  printf "%b\n" "用法: bash scripts/commit.sh [选项]"
  printf "%b\n" ""
  printf "%b\n" "选项说明 (支持交互向导与非交互式/AI自动化调用):"
  printf "%b\n" "  -t, --type <type>       提交类型 (feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
  printf "%b\n" "  -s, --scope <scope>     可选影响范围 (推荐: core, cli, api, workflow, deps, docs)"
  printf "%b\n" "  -m, --message <msg>     提交描述 / 标题 (必填，简明扼要)"
  printf "%b\n" "  -l, --item <text>       详细改动条目说明 (可多次传入，自动拼接为 Markdown 列表)"
  printf "%b\n" "  -d, --detail, --body <text> 详细正文描述 (可与 -l 结合使用)"
  printf "%b\n" "  -c, --close, --closes <issues> 关联并自动关闭的 GitHub Issue (支持单个如 12、多个如 -c 12 -c 13 或逗号分隔 \"12, 13\")"
  printf "%b\n" "  -i, --issue, --ref, --refs <issues> 仅关联引用的 GitHub Issue (支持格式同上)"
  printf "%b\n" "  -b, --breaking          标记为破坏性更新 (Breaking Changes，自动追加 !)"
  printf "%b\n" "  -n, --dry-run           演练模式 (预览完整提交信息与暂存状态，不实际提交与推送)"
  printf "%b\n" "  -a, --all               自动暂存全部变动 (注意: 相当于 git add -A，请谨慎使用)"
  printf "%b\n" "  -p, --push              提交成功后自动推送到当前分支远程仓库"
  printf "%b\n" "  -r, --rebase            当远程分支有更新时，允许在提交后自动执行受控变基同步 (git pull --rebase)"
  printf "%b\n" "  -y, --yes               跳过确认提示，直接执行提交"
  printf "%b\n" "  -h, --help              显示帮助信息"
  printf "%b\n" ""
  printf "%b\n" "无参数运行示例 (交互向导模式):"
  printf "%b\n" "  bash scripts/commit.sh"
  printf "%b\n" ""
  printf "%b\n" "非交互式调用示例 (推荐先精准 git add，再调用提交助手):"
  printf "%b\n" "  git add <file-path>"
  printf "%b\n" "  bash scripts/commit.sh -t feat -s core -m \"实现新功能\" -l \"支持多工单关联\" -c \"12, 13\" -p -y"
  printf "%b\n" "  bash scripts/commit.sh -t feat -s core -m \"实现新功能\" -p -y -r"
  printf "%b\n" ""
  printf "%b\n" "空跑演练预览示例 (不实际执行提交与推送):"
  printf "%b\n" "  bash scripts/commit.sh -t fix -s parser -m \"修复解析异常\" -c 15 -n"
}

while [ $# -gt 0 ]; do
  case "$1" in
    -t|--type)
      CLI_TYPE="$2"
      IS_NON_INTERACTIVE=true
      shift 2
      ;;
    -s|--scope)
      CLI_SCOPE="$2"
      shift 2
      ;;
    -m|--message)
      CLI_MSG="$2"
      IS_NON_INTERACTIVE=true
      shift 2
      ;;
    -l|--item)
      CLI_ITEMS+=("$2")
      shift 2
      ;;
    -d|--detail|--body)
      if [ -n "$CLI_BODY" ]; then
        CLI_BODY="${CLI_BODY}"$'\n\n'"$2"
      else
        CLI_BODY="$2"
      fi
      shift 2
      ;;
    -c|--close|--closes)
      for issue in $(echo "$2" | tr ',' ' '); do
        clean_id=$(echo "$issue" | tr -d '#' | tr -d '[:space:]')
        if [ -n "$clean_id" ]; then
          CLI_CLOSES+=("$clean_id")
        fi
      done
      shift 2
      ;;
    -i|--issue|--ref|--refs)
      for issue in $(echo "$2" | tr ',' ' '); do
        clean_id=$(echo "$issue" | tr -d '#' | tr -d '[:space:]')
        if [ -n "$clean_id" ]; then
          CLI_REFS+=("$clean_id")
        fi
      done
      shift 2
      ;;
    -b|--breaking)
      CLI_BREAKING=true
      shift
      ;;
    -n|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -a|--all)
      CLI_STAGE_ALL=true
      shift
      ;;
    -p|--push)
      CLI_PUSH=true
      shift
      ;;
    -r|--rebase)
      CLI_REBASE=true
      shift
      ;;
    -y|--yes)
      AUTO_CONFIRM=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      log_error "未知参数: $1 (使用 -h 或 --help 查看支持的选项)"
      exit 1
      ;;
  esac
done

# ==============================================================================
# 2. 基础环境与工作区自检
# ==============================================================================
if ! command -v git >/dev/null 2>&1; then
  log_error "未检测到 git 命令，请先安装 Git。"
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  log_error "当前目录不在 Git 仓库内，请在 Git 仓库内执行。"
  exit 1
fi

# 自动定位并切换至仓库根目录，支持在任意子目录下执行
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# 自动检测并注册本地 Git 提交规范钩子 (自愈机制)
if [ -d ".githooks" ] && [ "$(git config core.hooksPath 2>/dev/null || echo '')" != ".githooks" ]; then
  chmod +x .githooks/* 2>/dev/null || true
  git config core.hooksPath .githooks 2>/dev/null || true
fi

# 检查 Git 提交者配置
GIT_USER=$(git config user.name 2>/dev/null || echo "")
GIT_EMAIL=$(git config user.email 2>/dev/null || echo "")
if [ -z "$GIT_USER" ] || [ -z "$GIT_EMAIL" ]; then
  log_error "未检测到 Git 用户身份配置！请先设置:"
  printf "  git config user.name \"Your Name\"\n"
  printf "  git config user.email \"you@example.com\"\n"
  exit 1
fi

STATUS_OUTPUT=$(git status --porcelain)
if [ -z "$STATUS_OUTPUT" ]; then
  if [ "$DRY_RUN" = true ]; then
    log_info "【演练模式】当前工作区无修改，继续执行提交信息组装预览..."
  else
    log_warn "当前工作区没有任何修改或未跟踪的文件，无需提交。"
    exit 0
  fi
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# ==============================================================================
# 远端状态探测辅助函数 (按需调用，双重超时，防假死守卫)
# ==============================================================================
REMOTE_CHECKED=false
REMOTE_BRANCH_EXISTS=false
BEHIND_COUNT=0

detect_remote_status() {
  local target_branch="$1"
  if [ "$REMOTE_CHECKED" = true ]; then
    return 0
  fi

  log_info "正在预检远程分支状态 (origin/${target_branch})..."

  local fetch_output
  # 统一 5 秒超时守卫 (兼容 HTTPS 与 SSH 协议，禁止 SSH 交互按键挂起)
  if fetch_output=$(GIT_SSH_COMMAND="ssh -o ConnectTimeout=5 -o BatchMode=yes" \
      git -c http.timeout=5 fetch origin "$target_branch" 2>&1); then
    REMOTE_BRANCH_EXISTS=true
  else
    # 区分“远程分支尚不存在 (新分支)”与“网络超时/离线/无法连接”
    if echo "$fetch_output" | grep -Eq "(couldn't find remote ref|fatal: couldn't find remote ref|Remote branch .* not found)"; then
      REMOTE_BRANCH_EXISTS=false
      log_info "检测到远程仓库尚无对应分支 [origin/${target_branch}]，判定为全新分支。"
      REMOTE_CHECKED=true
      return 0
    else
      REMOTE_BRANCH_EXISTS=false
      log_warn "无法连接远程仓库进行同步校验（可能处于离线环境或网络受阻）: $(echo "$fetch_output" | head -n 1)"
      REMOTE_CHECKED=true
      return 1
    fi
  fi

  REMOTE_CHECKED=true
  # 精准计算落后提交数
  BEHIND_COUNT=$(git rev-list "HEAD..origin/${target_branch}" --count 2>/dev/null || echo "0")
  return 0
}

# ==============================================================================
# 3. 暂存区检查与智能暂存 (先确认变动再撰写信息)
# ==============================================================================
if [ "$CLI_STAGE_ALL" = true ]; then
  if [ "$DRY_RUN" = true ]; then
    log_info "【演练模式】模拟自动暂存全部变动 (git add -A)，跳过真实暂存..."
  else
    log_info "正在自动暂存当前所有改动 (git add -A)..."
    git add -A
  fi
elif [ "$IS_NON_INTERACTIVE" = false ] && [ "$DRY_RUN" = false ]; then
  printf "\n%b\n" "${COLOR_BOLD}${COLOR_CYAN}================================================================${COLOR_RESET}"
  printf "%b\n" "${COLOR_BOLD}${COLOR_CYAN}         Conventional Commits 规范化提交助手                     ${COLOR_RESET}"
  printf "%b\n" "${COLOR_BOLD}${COLOR_CYAN}================================================================${COLOR_RESET}"
  printf "当前工作分支: %b\n\n" "${COLOR_GREEN}${CURRENT_BRANCH}${COLOR_RESET}"

  printf "%b当前文件变更概览 (git status):%b\n" "${COLOR_BOLD}" "${COLOR_RESET}"
  git status --short
  printf "\n"

  if git diff --cached --quiet; then
    # 暂存区无内容，但工作区有改动
    printf "%b当前暂存区 (Staged) 无内容。是否暂存当前所有改动 (git add -A)? [Y/n]: %b" "${COLOR_BOLD}" "${COLOR_RESET}"
    read -r STAGE_ALL_CHOICE
    STAGE_ALL_CHOICE=$(echo "$STAGE_ALL_CHOICE" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    case "$STAGE_ALL_CHOICE" in
      [nN][oO]|[nN])
        log_warn "暂存区为空且未暂存改动，终止提交。"
        exit 0
        ;;
      *)
        log_info "正在暂存全部变动 (git add -A)..."
        git add -A
        ;;
    esac
  else
    # 暂存区已有内容
    log_info "检测到暂存区已有文件已暂存。"
    printf "%b是否仅提交已暂存的变动? (选 n 则追加暂存全部变动) [Y/n]: %b" "${COLOR_BOLD}" "${COLOR_RESET}"
    read -r ONLY_STAGED_CHOICE
    ONLY_STAGED_CHOICE=$(echo "$ONLY_STAGED_CHOICE" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    case "$ONLY_STAGED_CHOICE" in
      [nN][oO]|[nN])
        log_info "正在追加暂存全部变动 (git add -A)..."
        git add -A
        ;;
      *)
        log_info "将仅提交当前已暂存的改动。"
        ;;
    esac
  fi
fi

# 二次确认暂存区是否有内容
if git diff --cached --quiet; then
  if [ "$DRY_RUN" = true ]; then
    log_info "【演练模式】暂存区目前为空，跳过暂存强制拦截。"
  else
    log_error "暂存区无任何可提交的内容，请先使用 'git add' 暂存改动或传入 '-a' 参数。"
    exit 1
  fi
fi

# 提交前远端状态预检与防卫守卫 (Pre-commit Fetch Guard)
if [ "$CLI_PUSH" = true ] && [ "$DRY_RUN" = false ]; then
  if ! detect_remote_status "$CURRENT_BRANCH"; then
    # 网络不可达或超时
    if [ "$AUTO_CONFIRM" = true ] || [ ! -t 0 ]; then
      log_error "由于显式指定了远程推送 (-p)，但当前无法连接远程仓库进行状态校验！"
      log_error "为防止盲目提交后导致本地分叉或推送失败，流程安全中止 (Fail-Fast)。"
      log_info "若仅需在本地完成提交，请去除 '-p' 参数后重试。"
      exit 1
    else
      log_warn "当前无法连接远程仓库进行状态校验（可能处于离线环境）。"
      printf "%b是否取消远程推送计划，仅执行本地提交? [Y/n]: %b" "${COLOR_BOLD}" "${COLOR_RESET}"
      read -r OFFLINE_PUSH_CHOICE
      OFFLINE_PUSH_CHOICE=$(echo "$OFFLINE_PUSH_CHOICE" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
      case "$OFFLINE_PUSH_CHOICE" in
        [nN][oO]|[nN])
          log_warn "用户取消操作，未执行任何提交。"
          exit 0
          ;;
        *)
          log_info "已取消远程推送计划，将仅执行本地提交。"
          CLI_PUSH=false
          ;;
      esac
    fi
  elif [ "$REMOTE_BRANCH_EXISTS" = true ] && [ "$BEHIND_COUNT" -gt 0 ]; then
    # 守卫 1: 脏工作区防卫 (Dirty Workspace Defense)
    # 检查是否存在未暂存的修改 (包含已跟踪文件的未暂存改动或未跟踪的新文件)
    if git status --porcelain | grep -Eq "^(.[MDAR]|\?\?)"; then
      log_error "检测到本地分支落后于远程 origin/${CURRENT_BRANCH} (${BEHIND_COUNT} 个提交)，且当前工作区存在未暂存的修改！"
      log_error "为防止变基同步污染或破坏未暂存代码，已触发安全熔断。"
      log_info "解决建议: 请先暂存所有改动 (git add) 或使用 'git stash' 暂存工作区后再执行提交。"
      exit 1
    fi

    # 守卫 2: 双模变基受控契约 (Dual-mode Rebase Contract)
    if [ "$AUTO_CONFIRM" = true ] || [ ! -t 0 ]; then
      if [ "$CLI_REBASE" != true ]; then
        log_error "检测到本地分支落后于远程 origin/${CURRENT_BRANCH} (${BEHIND_COUNT} 个提交)！"
        log_error "自动化模式下默认保持严格快速失败 (Fail-Fast 防卫)，禁止隐式自动变基。"
        log_info "若授权自动化变基同步并推送，请显式追加 '-r' 或 '--rebase' 参数。"
        log_info "手动处理方案: git pull --rebase origin ${CURRENT_BRANCH} && git push origin ${CURRENT_BRANCH}"
        exit 1
      else
        log_info "已授权 -r/--rebase，将在本地提交完成后自动执行受控变基同步。"
      fi
    else
      if [ "$CLI_REBASE" != true ]; then
        log_warn "检测到本地分支落后于远程 origin/${CURRENT_BRANCH} (${BEHIND_COUNT} 个提交)。"
        printf "%b是否授权在本地提交后自动执行 git pull --rebase 同步并推送? [Y/n]: %b" "${COLOR_BOLD}" "${COLOR_RESET}"
        read -r AUTO_REBASE_CHOICE
        AUTO_REBASE_CHOICE=$(echo "$AUTO_REBASE_CHOICE" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        case "$AUTO_REBASE_CHOICE" in
          [nN][oO]|[nN])
            log_warn "用户未授权变基同步，终止提交流程。请先手动拉取合并远端代码后再试。"
            exit 0
            ;;
          *)
            log_info "已授权自动变基，将在本地提交后自动同步远端提交。"
            CLI_REBASE=true
            ;;
        esac
      fi
    fi
  fi
fi

# ==============================================================================
# 4. 收集与校验提交信息 (Type, Scope, Breaking, Subject)
# ==============================================================================
VALID_TYPES="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)$"

if [ "$IS_NON_INTERACTIVE" = true ]; then
  # 非交互式模式：校验传入参数
  TYPE="$CLI_TYPE"
  if ! echo "$TYPE" | grep -Eq "$VALID_TYPES"; then
    log_error "无效的提交类型: [$TYPE]。必须为下列之一:"
    log_error "feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert"
    exit 1
  fi

  SUBJECT=$(echo "$CLI_MSG" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  if [ -z "$SUBJECT" ]; then
    log_error "提交描述 (-m, --message) 不能为空！"
    exit 1
  fi

  SCOPE=$(echo "$CLI_SCOPE" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  if [ "$CLI_BREAKING" = true ]; then
    BREAKING_FLAG="!"
  else
    BREAKING_FLAG=""
  fi
else
  # 交互式向导模式
  printf "\n%b请选择本次提交类型 (Commit Type):%b\n" "${COLOR_BOLD}" "${COLOR_RESET}"
  printf "  %b1)%b  feat     : 🚀 新特性 / 新功能 (Features)\n" "${COLOR_GREEN}" "${COLOR_RESET}"
  printf "  %b2)%b  fix      : 🐛 缺陷与 Bug 修复 (Bug Fixes)\n" "${COLOR_GREEN}" "${COLOR_RESET}"
  printf "  %b3)%b  docs     : 📝 仅文档变动 (Documentation)\n" "${COLOR_GREEN}" "${COLOR_RESET}"
  printf "  %b4)%b  style    : 🎨 代码格式与排版调整（不影响逻辑，Style）\n" "${COLOR_GREEN}" "${COLOR_RESET}"
  printf "  %b5)%b  refactor : 🚜 代码重构（既非新特性也非缺陷修复，Refactor）\n" "${COLOR_GREEN}" "${COLOR_RESET}"
  printf "  %b6)%b  perf     : ⚡ 性能优化 (Performance)\n" "${COLOR_GREEN}" "${COLOR_RESET}"
  printf "  %b7)%b  test     : 🧪 测试相关（新增用例或重构测试，Tests）\n" "${COLOR_GREEN}" "${COLOR_RESET}"
  printf "  %b8)%b  build    : 📦 构建系统、外部依赖或脚手架调整 (Build)\n" "${COLOR_GREEN}" "${COLOR_RESET}"
  printf "  %b9)%b  ci       : 👷 CI/CD 流水线与 GitHub Actions 修改 (CI)\n" "${COLOR_GREEN}" "${COLOR_RESET}"
  printf "  %b10)%b chore    : 🧹 杂项维护（不改动源码与测试，Chore）\n" "${COLOR_GREEN}" "${COLOR_RESET}"
  printf "  %b11)%b revert   : ⏪ 代码回滚 (Revert)\n" "${COLOR_GREEN}" "${COLOR_RESET}"
  printf "  %b0)%b  q / exit : 🚪 取消并退出\n\n" "${COLOR_YELLOW}" "${COLOR_RESET}"

  TYPE=""
  while [ -z "$TYPE" ]; do
    printf "%b请输入选项编号 [0-11]: %b" "${COLOR_BOLD}" "${COLOR_RESET}"
    read -r TYPE_CHOICE
    TYPE_CHOICE=$(echo "$TYPE_CHOICE" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    case "$TYPE_CHOICE" in
      1) TYPE="feat" ;;
      2) TYPE="fix" ;;
      3) TYPE="docs" ;;
      4) TYPE="style" ;;
      5) TYPE="refactor" ;;
      6) TYPE="perf" ;;
      7) TYPE="test" ;;
      8) TYPE="build" ;;
      9) TYPE="ci" ;;
      10) TYPE="chore" ;;
      11) TYPE="revert" ;;
      0|[qQ]|[eE][xX][iI][tT])
        log_warn "用户取消操作，退出提交向导。"
        exit 0
        ;;
      *)
        log_warn "无效选项，请输入 0 到 11 之间的编号。"
        ;;
    esac
  done

  # 收集可选影响范围
  printf "\n%b请输入影响范围 Scope (可选，如 core / cli / api，留空按回车): %b" "${COLOR_BOLD}" "${COLOR_RESET}"
  read -r SCOPE
  SCOPE=$(echo "$SCOPE" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

  # 收集破坏性更新标记
  printf "%b是否包含破坏性更新 (Breaking Changes)? [y/N]: %b" "${COLOR_BOLD}" "${COLOR_RESET}"
  read -r IS_BREAKING
  IS_BREAKING=$(echo "$IS_BREAKING" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  case "$IS_BREAKING" in
    [yY][eE][sS]|[yY])
      BREAKING_FLAG="!"
      ;;
    *)
      BREAKING_FLAG=""
      ;;
  esac

  # 收集提交描述
  SUBJECT=""
  while [ -z "$SUBJECT" ]; do
    printf "%b请输入提交描述 Subject (必填，简要概括本次变动): %b" "${COLOR_BOLD}" "${COLOR_RESET}"
    read -r SUBJECT
    SUBJECT=$(echo "$SUBJECT" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [ -z "$SUBJECT" ]; then
      log_warn "提交描述不能为空，请重新输入。"
    fi
  done

  # 收集详细改动明细条目列表
  printf "\n%b请输入详细改动明细条目 (可选，每行一条，直接按回车结束):%b\n" "${COLOR_BOLD}" "${COLOR_RESET}"
  ITEM_INDEX=1
  while true; do
    printf "  %b[%d]%b " "${COLOR_CYAN}" "$ITEM_INDEX" "${COLOR_RESET}"
    read -r LINE_ITEM
    LINE_ITEM=$(echo "$LINE_ITEM" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [ -z "$LINE_ITEM" ]; then
      break
    fi
    CLI_ITEMS+=("$LINE_ITEM")
    ITEM_INDEX=$((ITEM_INDEX + 1))
  done

  # 收集关联或关闭的 Issue
  printf "\n%b请输入待关联或关闭的 GitHub Issue 编号 (可选，如 12 或 '12, 13'，直接回车跳过): %b" "${COLOR_BOLD}" "${COLOR_RESET}"
  read -r ISSUE_INPUT
  ISSUE_INPUT=$(echo "$ISSUE_INPUT" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  if [ -n "$ISSUE_INPUT" ]; then
    printf "%b该提交是否旨在解决并自动闭环所选 Issue? [Y/n]: %b" "${COLOR_BOLD}" "${COLOR_RESET}"
    read -r IS_CLOSE_CHOICE
    IS_CLOSE_CHOICE=$(echo "$IS_CLOSE_CHOICE" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    for raw_id in $(echo "$ISSUE_INPUT" | tr ',' ' '); do
      clean_id=$(echo "$raw_id" | tr -d '#' | tr -d '[:space:]')
      if [ -n "$clean_id" ]; then
        case "$IS_CLOSE_CHOICE" in
          [nN][oO]|[nN])
            CLI_REFS+=("$clean_id")
            ;;
          *)
            CLI_CLOSES+=("$clean_id")
            ;;
        esac
      fi
    done
  fi
fi

# 拼装提交消息 Header
if [ -n "$SCOPE" ]; then
  COMMIT_HEADER="${TYPE}(${SCOPE})${BREAKING_FLAG}: ${SUBJECT}"
else
  COMMIT_HEADER="${TYPE}${BREAKING_FLAG}: ${SUBJECT}"
fi

# 拼装提交消息 Body 与 Git Trailers
COMMIT_BODY=""
if [ -n "$CLI_BODY" ]; then
  COMMIT_BODY="$CLI_BODY"
fi

if [ ${#CLI_ITEMS[@]} -gt 0 ]; then
  ITEMS_BLOCK=""
  for item in "${CLI_ITEMS[@]}"; do
    if [ -n "$ITEMS_BLOCK" ]; then
      ITEMS_BLOCK="${ITEMS_BLOCK}"$'\n'"- ${item}"
    else
      ITEMS_BLOCK="- ${item}"
    fi
  done
  if [ -n "$COMMIT_BODY" ]; then
    COMMIT_BODY="${COMMIT_BODY}"$'\n\n'"${ITEMS_BLOCK}"
  else
    COMMIT_BODY="${ITEMS_BLOCK}"
  fi
fi

# 拼装 Git Trailers (Closes 与 Refs，逐行独立，自动去重)
TRAILERS_BLOCK=""

UNIQUE_CLOSES=()
for id in "${CLI_CLOSES[@]}"; do
  already=false
  for exist in "${UNIQUE_CLOSES[@]}"; do
    if [ "$exist" = "$id" ]; then already=true; break; fi
  done
  if [ "$already" = false ]; then
    UNIQUE_CLOSES+=("$id")
  fi
done

for id in "${UNIQUE_CLOSES[@]}"; do
  trailer="Closes #${id}"
  if [ -n "$TRAILERS_BLOCK" ]; then
    TRAILERS_BLOCK="${TRAILERS_BLOCK}"$'\n'"${trailer}"
  else
    TRAILERS_BLOCK="${trailer}"
  fi
done

UNIQUE_REFS=()
for id in "${CLI_REFS[@]}"; do
  already=false
  for exist in "${UNIQUE_CLOSES[@]}"; do
    if [ "$exist" = "$id" ]; then already=true; break; fi
  done
  for exist in "${UNIQUE_REFS[@]}"; do
    if [ "$exist" = "$id" ]; then already=true; break; fi
  done
  if [ "$already" = false ]; then
    UNIQUE_REFS+=("$id")
  fi
done

for id in "${UNIQUE_REFS[@]}"; do
  trailer="Refs #${id}"
  if [ -n "$TRAILERS_BLOCK" ]; then
    TRAILERS_BLOCK="${TRAILERS_BLOCK}"$'\n'"${trailer}"
  else
    TRAILERS_BLOCK="${trailer}"
  fi
done

if [ -n "$TRAILERS_BLOCK" ]; then
  if [ -n "$COMMIT_BODY" ]; then
    COMMIT_BODY="${COMMIT_BODY}"$'\n\n'"${TRAILERS_BLOCK}"
  else
    COMMIT_BODY="${TRAILERS_BLOCK}"
  fi
fi

# ==============================================================================
# 5. 预览与确认
# ==============================================================================
printf "\n%b\n" "${COLOR_BOLD}${COLOR_CYAN}------------------- 拟生成的提交信息预览 -------------------${COLOR_RESET}"
printf "%b\n" "${COLOR_BOLD}${COLOR_GREEN}${COMMIT_HEADER}${COLOR_RESET}"
if [ -n "$COMMIT_BODY" ]; then
  printf "\n%b%s%b\n" "${COLOR_CYAN}" "$COMMIT_BODY" "${COLOR_RESET}"
fi
printf "%b\n\n" "${COLOR_BOLD}${COLOR_CYAN}------------------------------------------------------------${COLOR_RESET}"

if [ "$DRY_RUN" = true ]; then
  log_success "【演练模式 DRY-RUN】提交信息组装校验完成，未执行任何实际提交与推送。"
  exit 0
fi

if [ "$AUTO_CONFIRM" = true ] || [ ! -t 0 ]; then
  log_info "已指定 -y 或处于非交互环境，跳过人工确认，直接执行提交。"
else
  printf "%b是否确认以此信息执行提交? [Y/n]: %b" "${COLOR_BOLD}" "${COLOR_RESET}"
  read -r CONFIRM_COMMIT
  CONFIRM_COMMIT=$(echo "$CONFIRM_COMMIT" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  case "$CONFIRM_COMMIT" in
    [nN][oO]|[nN])
      log_warn "操作已由用户取消，未执行任何提交。"
      exit 0
      ;;
    *)
      ;;
  esac
fi

# ==============================================================================
# 6. 执行 Git Commit
# ==============================================================================
if [ -n "$COMMIT_BODY" ]; then
  git commit -m "$COMMIT_HEADER" -m "$COMMIT_BODY"
else
  git commit -m "$COMMIT_HEADER"
fi
log_success "代码提交成功！"

# ==============================================================================
# 7. 远程推送与受控变基 (Push & Controlled Rebase Guard)
# ==============================================================================
if [ "$CLI_PUSH" = false ] && [ -t 0 ] && [ "$AUTO_CONFIRM" = false ]; then
  printf "\n%b是否立即推送到远程分支 (origin/%s)? [y/N]: %b" "${COLOR_BOLD}" "$CURRENT_BRANCH" "${COLOR_RESET}"
  read -r CONFIRM_PUSH
  CONFIRM_PUSH=$(echo "$CONFIRM_PUSH" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  case "$CONFIRM_PUSH" in
    [yY][eE][sS]|[yY])
      CLI_PUSH=true
      # 交互模式补做探测
      if ! detect_remote_status "$CURRENT_BRANCH"; then
        log_warn "无法连接远程仓库，跳过远程推送。你可以在联网后随时手动推送。"
        CLI_PUSH=false
      elif [ "$REMOTE_BRANCH_EXISTS" = true ] && [ "$BEHIND_COUNT" -gt 0 ]; then
        if git status --porcelain | grep -Eq "^(.[MDAR]|\?\?)"; then
          log_error "本地分支落后于远端 (${BEHIND_COUNT} 个提交)，且工作区存在未暂存修改，无法自动变基推送。"
          log_info "请先暂存修改并手动执行: git pull --rebase origin ${CURRENT_BRANCH} && git push origin ${CURRENT_BRANCH}"
          CLI_PUSH=false
        else
          printf "%b检测到远程分支有 %s 个更新，是否自动执行 git pull --rebase 同步后推送? [Y/n]: %b" "${COLOR_BOLD}" "$BEHIND_COUNT" "${COLOR_RESET}"
          read -r INTERACTIVE_REBASE_CHOICE
          INTERACTIVE_REBASE_CHOICE=$(echo "$INTERACTIVE_REBASE_CHOICE" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
          case "$INTERACTIVE_REBASE_CHOICE" in
            [nN][oO]|[nN])
              log_info "已取消自动变基与推送。本地提交已安全保留。"
              CLI_PUSH=false
              ;;
            *)
              CLI_REBASE=true
              ;;
          esac
        fi
      fi
      ;;
    *)
      log_info "已跳过远程推送。本地提交已安全落盘，你可以在就绪后随时运行 'git push origin ${CURRENT_BRANCH}'。"
      ;;
  esac
fi

if [ "$CLI_PUSH" = true ]; then
  # 若此前未探测过（例如某些边缘分支），兜底探测
  detect_remote_status "$CURRENT_BRANCH" || true

  # 1. 变基同步流程 (仅在分支存在、有落后且获授权时执行)
  if [ "$REMOTE_BRANCH_EXISTS" = true ] && [ "$BEHIND_COUNT" -gt 0 ]; then
    if [ "$CLI_REBASE" = true ]; then
      # 再次确认工作区无未暂存脏改动
      if git status --porcelain | grep -Eq "^(.[MDAR]|\?\?)"; then
        log_error "变基前检测到工作区存在未暂存改动！为防污染未暂存代码，终止变基。"
        log_warn "本地提交已安全保留。请手动排查并执行: git pull --rebase origin ${CURRENT_BRANCH}"
        exit 1
      fi

      log_info "正在执行变基同步 (git pull --rebase origin ${CURRENT_BRANCH})..."
      if ! git pull --rebase origin "$CURRENT_BRANCH"; then
        log_error "变基同步遇到冲突或失败！"
        log_warn "正在自动执行 git rebase --abort 彻底回滚变基，杜绝中间态挂死工作区..."
        git rebase --abort >/dev/null 2>&1 || true
        log_error "变基冲突已原子回滚！本地 Commit 已安全落盘，但未推送到远端。"
        log_info "请手动执行以下命令排查解决冲突后重新推送:"
        log_info "  git pull --rebase origin ${CURRENT_BRANCH}"
        log_info "  git push origin ${CURRENT_BRANCH}"
        exit 1
      fi
      log_success "受控变基同步成功！本地提交已平移至远端最新基线之上。"
    else
      log_error "本地分支落后于远程分支 (${BEHIND_COUNT} 个提交)，且未授权自动变基，终止推送。"
      log_info "本地提交已安全保留。请手动同步: git pull --rebase origin ${CURRENT_BRANCH} && git push origin ${CURRENT_BRANCH}"
      exit 1
    fi
  fi

  # 2. 最终推送执行 (新分支自动追加 -u，竞态报错保护)
  if [ "$REMOTE_BRANCH_EXISTS" = false ]; then
    log_info "正在推送到全新远程分支并建立上游追踪 (git push -u origin ${CURRENT_BRANCH})..."
    if ! git push -u origin "$CURRENT_BRANCH"; then
      log_error "新分支推送到远程仓库失败！"
      log_warn "注意：本地 Commit 已安全落盘，请勿重复提交。"
      log_info "你可以在检查网络或权限后随时运行 'git push -u origin ${CURRENT_BRANCH}'。"
      exit 1
    fi
    log_success "新分支推送成功并已关联上游！"
  else
    log_info "正在自动推送到 origin/${CURRENT_BRANCH}..."
    if ! git push origin "$CURRENT_BRANCH"; then
      log_error "推送到远程仓库失败！"
      log_warn "注意：本地 Commit 已安全落盘，请勿重复提交。"
      log_info "若发生并发竞态推送冲突，请执行 'git pull --rebase origin ${CURRENT_BRANCH} && git push origin ${CURRENT_BRANCH}'。"
      exit 1
    fi
    log_success "推送成功！"
  fi
fi
