# Codex 首次交接｜B001：Mac 仓库起步与官方数据检查

**编写日期：2026-09-28。范围：项目说明 + 首批实施任务。**

这是交给用户 Mac 上 Codex 的实施说明。它不是运行报告，也不表示任何代码已经实现或推送。

历史说明（截至本文初始提交）：准备本文的 ChatGPT 尚未读取 `yehu77/mimo_rl` 的文件、分支或 commit，GitHub 写入连接也尚未建立。B001 已完成真实仓库检查、实现、数据检查并推送；不要把这段历史状态当作当前状态。Notion 记录的 0/36 是此前验收快照，不是对当前工作区的检查结论。

## 当前批次：B002（GitHub Issue #4）

评审基线：`ba30dbf422c81139653f00bd0f05f76da1f88955`，工作分支：`codex/b002-task-contract`，目标 PR base：`codex/b001-r1`。

Issue #4 授权本批次推进：先修复 unmatched image 详情绑定，再只读固定官方 recipe、mimoagent/uni-agent 子模块和 slime 源码，建立 `docs/UPSTREAM_CONTRACT.md`，实现轻量 TaskBundle、solver/verifier 隔离和 `prepare_code_tasks.py`，复用严格 provenance 检查完成固定真实数据合同检查。保留 R1 回归；本轮不启动 Docker、模型、任务容器、官方 loader runtime、agent loop 或训练。B001 原始结果和 R1 记录保留在 `reports/handoffs/B001.md`。

本段取代旧的 B001-R1 当前批次限制；旧批次段落仍作为历史交接保留，不再约束本批次。

## 1. 给 Codex 的项目背景

用户要做一个真正经历数据接入、agent rollout、verifier、RL 更新、调试、评测的项目，获得训练工程经验和机制判断能力，不只是把论文名写进 README。

项目将优先使用：

| 项目 | 当前约定 | 验证边界 |
|---|---|---|
| 仓库 | `yehu77/mimo_rl` | 用户已创建；你需检查本地 checkout 与远端 |
| 数据 | `XiaomiMiMo/MiMo-V2.6-RL-oss` 的 code 子集 | 下载实际固定版本，不能用网页预览代替 schema |
| 初始模型 | 优先验证 `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` | 尚未验收其与选定 slime 后端的兼容性 |
| 主训练框架 | slime | 具体 commit、后端、并行与优化配置待固定 |
| 执行层 | 优先复用官方 mimoagent | 根据固定源码确认 Docker、bash-only、code adapter 和 verifier |
| 本地开发 | Mac、CPU 可测试核心 | Mac 型号/内存/Docker 状态由你实际采集 |
| 真实训练 | 后续 Linux/NVIDIA GPU 机器 | 用户曾提供十多张约 80GB 卡的资源计划，实际分配待确认 |

**第一条真实训练链路：**官方任务 → 独立任务容器 → 当前 policy 的多轮交互 → 提交代码 → 可信 verifier → 同题多条轨迹 → 优势与 loss → 参数更新 → checkpoint → 新权重再次交互。

正式实验是后续工作，不是本批次立即执行的任务：
- A0：固定任务、H1 和起点的 outcome-RL。
- A1：与 A0 对照，只引入组相对长度惩罚，比较成功率和成本，不能只报变短。
- A2：与 A0/A1 对照，引入可信质量评分与 GAR；评分器校准、质量系数映射及成本必须说明。
- Multi-harness：在同一数据划分上比较 H1-only、H2-only、mixed，再在未训练接口 H3 上评测；同组使用同一个 harness。

以上是项目设计，不是报告保证在 9B/小预算下复现的结果。原报告的 GAR Figure 8 使用 MiMo-V2.6-Flash；不要把它称为已经有 9B 同配置复现。

## 2. 本批次只做什么

**B001 目标：在 Mac 得到一个可独立安装和测试的最小仓库，并实际读取一份固定版本的官方 code 数据。**

对应 Notion：T01、T03 的本地开发/版本记录部分、T04。T03 的 GPU 环境验收不能随本批次一起勾完。

先做只读检查，再实现下面的小步。低风险且在范围内的选择由你执行并记录；重大框架替换、额外开销、数据语义变化先向用户说明。

### B001.1｜检查真实仓库，保护已有工作

在正确本地目录执行并核对：

```bash
pwd
git rev-parse --show-toplevel
git status --short
git branch --show-current
git log -3 --oneline
```

本地检查 origin 是否对应 `yehu77/mimo_rl`，但不要把可能包含凭据的 remote URL 原样发布。读取已有 README、AGENTS、pyproject、测试和源码。明确哪些已存在、哪些缺失。

如果还未 clone，在用户指定目录 clone 该仓库；不要猜用户想放在哪里。如果远端确实无初始 commit，按真实状态建立初始分支；如果已有代码，保留其历史和约定。

如果工作区不干净，指出相关文件并保护它们，不自动 stash/reset/clean。无冲突时在实际基线创建 `codex/b001-mac-bootstrap` 或不冲突的等价分支，记录 base commit。

**验收：**远端身份、分支、base commit 和本批次可修改范围明确；没有丢失现有修改。

### B001.2｜记录本地能力，建立可重建开发环境

只采集必要信息：macOS、架构、内存、可用 Python、Docker 客户端及 daemon 是否可用。不需要用户名、机器序列号、完整环境变量或密钥。

Python 3.12 是当前项目起点；如已存在合适 Python，建立仓库级虚拟环境。没有时先报告并让用户确认安装方式，不擅自修改系统 Python。沿用仓库已采用的包管理器；空仓库可选择一种简单方案，记录版本和重建步骤，不同时建立多套锁文件体系。

开发依赖只服务于本批次：配置/JSON、Parquet、HF 文件访问、pytest、lint 等所需最小集合。没有代码依赖的库不必提前装。不要安装 slime/SGLang/Megatron/CUDA，更不要下载 9B 模型权重。

**验收：**新建本地开发环境可复现；核心包可导入；`import mimo_rl` 不要求 GPU、不访问网络、不启动引擎。Docker 不可用如实记录，不阻塞本批次数据工作。

### B001.3｜完成最小仓库包装，而非搭一套大框架

尊重现有目录；空仓库的建议增量如下，按实际需要创建：

```text
mimo_rl/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── .gitattributes
├── .env.example
├── docs/CODEX_HANDOFF.md
├── src/mimo_rl/__init__.py
├── scripts/
│   ├── fetch_code_data.py
│   └── inspect_dataset.py
├── tests/
│   ├── unit/
│   └── fixtures/
├── configs/runtime/
│   ├── mac.yaml
│   └── linux_gpu.example.yaml
├── manifests/versions.yaml
├── artifacts/
└── runs/
```

README 说明目的、当前里程碑、实测安装/测试/数据检查命令和 Mac→Linux 的重建原则。不宣称后续训练功能已经可用。

`.gitignore` 排除 `.venv`、`.env`、凭据、权重、原始下载、任务镜像、运行轨迹、编译缓存和 `.DS_Store`；保留 `.env.example`。不要忽略锁文件、测试源码和脱敏的轻量验收报告。使用 UTF-8、LF，注意执行脚本的文件权限。

运行配置用于路径和服务地址；未知 GPU 目录/设备/模型地址留空或明确未配置。不要用 localhost 假服务填上“可运行”配置。共享算法与实验参数本批次不冻结，不制造空 train、GAR、grader 或 harness 插件实现。

### B001.4｜记录来源版本，仅下载官方 code 必要文件

数据源：`XiaomiMiMo/MiMo-V2.6-RL-oss`。

1. 查询真实 dataset revision，并把具体 commit SHA 写入 `manifests/versions.yaml`。不能仅写 `main`。
2. 实现数据下载入口，显式接收或读取该 revision，按固定版本定位 `code.parquet` 和 `image-mapping.jsonl`。
3. 只下载这两份数据；许可/README 可查阅。不抓取其他领域文件、模型权重或任务镜像。
4. 保存每份文件的相对位置、大小、SHA-256 和来源 revision。重复执行可重用已验证缓存，但不得使用另一 revision 的同名文件而不报错。
5. 记录 slime、mimoagent、官方 recipe 的准备采用 commit（如网络可用）；这只表示版本选择，不能写成兼容性测试通过。未经访问无法核对的项为 null，并附原因。训练镜像/CUDA/并行仍待目标机。

建议命令行接口：

```text
python scripts/fetch_code_data.py --revision <真实固定SHA> --output-dir <本地数据目录>
```

实际交付以 README 的实测命令为准。

### B001.5｜检查真实 schema，不提前臆造 TaskSpec/verifier

实现 `scripts/inspect_dataset.py`，接收本地两份文件和输出目录；内部可复用小型解析函数以便测试。

检查并记录：
- Parquet 实际行数、列名、字段类型、嵌套结构。
- `prompt`、`reward_model`、`extra_info` 是否存在、实际结构是什么。
- `extra_info.instance_json` 若为字符串才 JSON 解码；若已是结构体则直接检查；缺失、类型错误、畸形 JSON 带原始记录索引报告。
- task/instance 标识和空题面，重复 ID、缺失 ID、解析失败。不可假设 prompt 一定是普通字符串。
- 镜像映射文件的真实 schema、重复键、可解析对应关系和未匹配项。对于需要官方 loader 才能解析的映射，明确列为“待 loader 核对”，不自己拼 registry/tag。
- 分清数据已提供的信息与需要镜像/loader 才知道的信息，保留原始索引；不默默删除异常任务、不修改官方题目或 verifier。

预期本地产物：

```text
artifacts/data/schema.json
artifacts/data/sample_records.json
artifacts/data/validation_summary.json
artifacts/data/provenance.json
```

`sample_records.json` 保存少量完整真实记录用于本地核对，不默认提交整个题面、参考补丁、隐藏测试或原始样本。可提交脱敏后的字段名/类型、汇总统计、hash 和可重现命令。

### B001.6｜测试与干净重建

单元测试至少覆盖：可移植包导入、字符串/结构体两类嵌套记录、缺失/错误类型、畸形 JSON、空题面、重复标识、映射无法解析、输出目录和序列化。

人工畸形 fixture 仅用于测试，命名或元数据明确 `synthetic`；生产脚本不能在下载失败时切到 fixture。测试不需要 Docker/CUDA/收费 API。

另外执行一次真实数据检查，并记录真实命令及退出码。纯单元测试 PASS 与真实数据运行 PASS 分开。

在干净 checkout 或临时导出目录中建立新 venv 复跑安装与测试，避免依赖当前 shell 的 PYTHONPATH 或用户 site-packages；不要通过清理用户当前仓库来制造“干净环境”。

### B001.7｜交付可由远端 reviewer 检查的证据

代码在本批次分支提交。新增或更新 `reports/handoffs/B001.md`，至少记录：

```text
批次：B001
Notion 关联：T01 / T03-local / T04
本批次结论：PASS / PARTIAL / BLOCKED
环境：macOS、架构、Python（脱敏）
base_commit：
tested_source_commit：
分支与 PR：
实际变更文件：
实际命令及退出码：
单元测试：passed / failed / skipped
真实数据检查：PASS / FAIL / NOT_RUN
数据 revision、文件大小与 SHA-256：
真实行数、解析错误/重复/缺失统计：
Mac Docker：可用 / 不可用 / 未检查
GPU 模型加载 / RL 更新 / 权重同步：NOT_RUN
参数是否真实更新：false
发现的问题及未证实的假设：
原始日志位置（本地，脱敏）：
建议核对的文件与关键函数：
下一步建议：仅一项
```

确认没有敏感内容后，可使用已有 GitHub 权限将本批次分支 push 并发起 PR；不得 force-push 或自动合并主分支。

**到此停止。不要自动开始 B002、GAR、环境镜像拉取或训练。**

## 3. 本批次明确不做

不接收费模型 API；不下载全部模型权重；不启动任务容器；不运行真实模型生成的命令；不安装整套 CUDA 栈；不实现 GAR/长度惩罚/多 harness；不写新的 agent/训练框架；不把 36 项一次性做完；不进行训练吞吐或能力提升声明。

## 4. 全局阶段地图

| 阶段 | 对应任务 | 核心结果 |
|---|---|---|
| 资源、数据、版本 | T01–T05 | 真实仓库、固定来源、schema、目标模型兼容性 |
| 可信环境 | T06–T11 | 官方任务适配、隔离、正负控、patch 往返、重置、失败分类 |
| Agent 轨迹 | T12–T17 | H1、完整轨迹、真实 token/prob、loss mask、slime 接入 |
| M0 训练闭环 | T18–T23 | loss 合同、两端概率、真实更新、权重同步、恢复 |
| A0 基线 | T24–T26 | 冻结数据划分、pilot、可信 outcome-RL |
| 效率与质量 | T27–T32 | A1 长度、grader 校准、GAR 旁路和 A2 |
| 交互泛化 | T33–T36 | H2/H3、组级 mixed、跨接口矩阵、种子与总结 |

GPU 目标机一旦可用，尽早验证 T05，不要等所有研究功能都写完才暴露模型适配问题。

## 5. 两端如何沟通

ChatGPT 与 Mac Codex 没有自动直连会话。GitHub 是共享代码/说明/证据的载体，不代表自动唤醒或持续监听。

每轮流程：reviewer 给出明确批次 → Codex 读取当前仓库与批次 → 在分支实现和测试 → push 代码与回报 → 用户提供 commit/PR → reviewer 读取实际变更、反馈并核对 Notion。

本交接只安排 B001。下一轮任务修改 `docs/CODEX_HANDOFF.md` 的当前批次段，旧批次的输入与回报保留在 Git 历史/PR/轻量报告中。

## 6. 未决事项

Mac 架构/内存/Python/Docker；训练 GPU 型号/节点/时限；训练后端与并行；模型兼容性；学习率和 loss/KL；长度惩罚系数；grader 版本/预算/校准；GAR 排序到质量系数的映射；未见 harness 设计。

这些未决项不妨碍 B001。不能为清单看起来完整而猜填。

## 7. 导入与执行规则

用户启动本批次时的一句话：

> 读取 AGENTS.md 和 docs/CODEX_HANDOFF.md，检查现有仓库后执行 B001。不要只给计划；做真实本地实现与测试，按约定报告证据。本批次结束即停止，不自动进入下一阶段。

## 8. 来源与原文边界

本任务拆分、Mac 开发方式、目录、验收报告和比较设计来自用户与 ChatGPT 已确认的项目计划，不是论文原配置。

官方入口：
- 数据：https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss
- 模型：https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
- 执行层：https://github.com/XiaomiMiMo/mimoagent
- 官方训练 recipe：https://github.com/XiaomiMiMo/verl
- slime：https://github.com/THUDM/slime

论文依据为用户提供的 MiMo-V2.6 技术报告：§7.1–7.2 支持发布 9B 起点和开放环境做 GRPO；§4.2.5 支持 mini-harness；§4.3.2 支持 GAR；§4.3.3 支持组相对长度惩罚；§5.1 支持使用 rollout 时生成概率；§6.1 支持只对模型生成 segment 计算 loss；§6.2 支持同一组使用共同 harness 配置。

实现论文算法的批次必须先查原文和固定源码。原文未披露的超参数/映射应明确是本项目选择；原文不支持的效果结论不得编造。
