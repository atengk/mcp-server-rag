# 安全策略与漏洞披露指南 (Security Policy)

本项目十分重视代码与系统安全。如果你在本项目中发现了疑似安全漏洞或隐患，请**切勿在公开的 GitHub Issue 或讨论区直接披露**，以防在漏洞修复前造成安全风险。

---

## 1. 支持的版本 (Supported Versions)

我们通常仅对最新的主干分支 (`main`) 及最新发布的正式版本提供安全补丁支持：

| 版本系列 | 是否支持安全补丁 | 说明 |
| :--- | :--- | :--- |
| `main` / 最新 Release | :white_check_mark: 支持 | 优先接收安全补丁与热修复 |
| 历史旧版本 | :x: 不再支持 | 建议尽快升级至最新版本 |

---

## 2. 漏洞报告渠道 (Reporting a Vulnerability)

推荐通过以下方式安全、私密地向维护者报告漏洞：

### 方式一：GitHub 私密安全报告 (推荐首选)
1. 前往本仓库的 **[Security -> Advisories](https://github.com/atengk/mcp-server-rag/security/advisories/new)** 页面；
2. 点击 **「Report a vulnerability」**；
3. 详细填写漏洞描述、复现步骤及潜在影响后提交。该报告仅仓库维护者可见。

### 方式二：安全联络邮箱
若无法访问 GitHub Advisories，请直接发送加密邮件至安全联系人：
- **安全联系邮箱**：`2385569970@qq.com`（*请在邮件主题中注明 `[SECURITY VULNERABILITY]`*）

---

## 3. 漏洞响应流程 (Response Process)

- **初步确认**：我们将在 **48 小时内** 确认收到你的安全报告并启动技术研判；
- **补丁研发**：确认漏洞存在后，维护团队将在私有分支中编写修复补丁并验证防御效果；
- **安全发布**：修复补丁合并后，我们将发布全新的补丁版本（Patch Release），并在 GitHub Security Advisories 中公开致谢发现者（若你希望保持匿名，亦可提前说明）；
- **负责任披露**：在补丁正式发布前，维护者与报告人双方承诺遵守协同披露原则，避免提前公开 PoC 细节。
