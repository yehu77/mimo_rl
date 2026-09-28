# mimo_rl — coding-agent collaboration contract

本文面向在 `yehu77/mimo_rl` 中工作的 Codex 和其他编码 agent。这里的 Codex 是项目开发者，不是将被 RL 训练的 policy。说明使用中文；代码、标识符和工具协议使用适当的英文。

## 先读什么

1. 先遵守当前会话及仓库已经存在的指令，检查现有文件和未提交修改。本文若是导入的交接材料，不得盲目覆盖已有 `AGENTS.md`。
2. 阅读 `README.md`（存在时）和 `docs/CODEX_HANDOFF.md`，只执行其中明确授权的当前批次。
3. 详细任务与项目状态维护在用户的 Notion 工作台：
   https://app.notion.com/p/3e7e839bd96180a78669f8b97a7b6a3f
   不能访问 Notion 时使用当前交接任务；不要臆造其更新或声称已同步。
4. 仓库承载代码、review 和可追溯的执行报告，不复制第二份 T01–T36 总看板。

## 项目要做什么

基于用户提供的《MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement》，实现一个以真实训练经验与机制对照为目标的 Code Agent RL 项目。

已确认方向：
- 先在 Mac 开发可移植代码，再到 Linux/NVIDIA GPU 机器做真实训练验收。
- 采用 `XiaomiMiMo/MiMo-V2.6-RL-oss` 的 code 子集，不先合成新任务，不混入其他领域。
- 主训练框架为 slime；优先验证模型 `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`，不是已经证明它在所选后端兼容。
- 优先复用官方 mimoagent 的任务适配、Docker、bash-only agent 和 verifier，只补必要接口。
- H1 起步：单 agent、单上下文、单 bash 工具、同步采样。官方工具实际格式以固定源码为准，不按聊天中的示例猜测。
- 首先得到可信的 outcome-RL 基线；之后分别研究组相对长度惩罚、Groupwise Advantage Redistribution（GAR）和 multi-harness。

不是要缩小复刻整套 MiMo，不是开发新的训练框架，也不是制作只有 mock 能通过的演示平台。

## 里程碑不可混淆

- **M-dev**：Mac 上实际安装、真实数据解析、纯逻辑/契约测试及部署说明通过。
- **M0**：真实目标模型在官方任务环境中生成轨迹，独立 verifier 判分，真实 optimizer 更新，保存/恢复 checkpoint，新权重重新 rollout。
- **后续实验**：同一起点、冻结的数据划分和预算下进行 A0 outcome、A1 长度惩罚、A2 GAR，以及单/多 harness 对照。

M-dev 不能替代 M0。能聊天、能启动进程、打印 loss、测试全部 skip 都不是训练成功证据。

## 工程边界

- 数据合同、轨迹结构、配置校验和优势纯函数应能在无 GPU 环境独立测试。默认导入 `mimo_rl` 不加载 slime、SGLang、Megatron 或 CUDA 扩展。
- 可移植开发依赖与 Linux/GPU 训练镜像分开管理；不在 Mac 硬装 CUDA 栈，不用通用依赖同步覆盖训练镜像的 GPU 库。
- Python 3.12 是开发起点候选，按实际固定上游版本核对。不要未经确认修改系统 Python 或全局安装环境。
- 机器路径、模型目录、服务地址、资源分配由运行配置注入。共享实验配置不写死 `/Users/...`、`/mnt/...` 或凭据。
- 不把 Mac 的 `.venv`、动态库、wheel、编译缓存迁移到 Linux；在目标机重建依赖。
- 先查看固定上游源码，再写薄适配。不要同时维护 verl 和 slime 两套长期训练工程，不复制官方仓库源码来冒充个人实现。
- 不提前创建几十个空模块、通用插件系统、空 `train()` 或总是成功的占位后端。未实现的生产入口明确失败；测试替身仅放在测试路径中。

## 数据与 RL 正确性

- 数据使用真实官方记录；只有单元测试的畸形输入/边界样例可以人工构造，必须标为 synthetic。
- 真实训练必须记录生成时原始 token IDs、对应 log-probabilities、policy version、task/group/rollout/harness 身份。
- 工具结果和 system/user 输入不参与 loss；多轮历史不能重复计算旧生成 token 的 loss。
- 禁止将缺失 log-prob 填零，禁止事后重编码对话并称其为原始采样 token；禁止把外部教师 API 的输出冒充当前训练 policy 的 on-policy 轨迹。
- 同一个 GRPO 组的 task、policy version、harness 配置保持一致。奖励全同的组如实记账，不伪造有梯度的结果。
- 原始测试结果、有效训练奖励和基础设施失败分别记录。INFRA_ERROR 不直接当模型负奖励；组级排除/重试策略须可追溯。
- 使用固定初始环境和可信 verifier，检查重置、补丁往返、新文件提交、隐藏资产隔离。官方 grader 存在不等于 GAR 质量评分已验证。
- 不根据测试集上的模型表现挑题或调参；正式划分按仓库/同源关系隔离。调试反复使用的题不能继续充当独立测试证据。

## 执行安全与 Git 协作

- 用户已创建远端仓库 `https://github.com/yehu77/mimo_rl.git`。先验证本地路径、origin、默认分支和工作区；不要再次盲目初始化仓库。
- 不删除、覆盖、stash 或 reset 用户的未提交修改；发生冲突先说明。禁止 force-push，不自动合并主分支。
- 在本批次功能分支提交；已获用户授权且有正常凭据时可 push 并开 PR，保留默认分支供 review。没有网络/权限则交付本地 commit 和阻塞原因，不能声称已经推送。
- 不发布 API key、token、SSH 文件、环境变量全集、用户目录、私有 Notion 内容或未脱敏日志。不要为了提交而索取用户粘贴凭据。
- 真实模型命令只能在已检查的隔离环境执行。禁止为图方便切换到直接执行 Mac 宿主机命令的 local 后端。
- 第一阶段不调用收费模型 API、不租机器、不下载整个模型权重或全库任务镜像、不执行未审查的上游安装脚本。
- 对外发布保留原有许可和 attribution；不要擅自给用户仓库选择许可证。

## 如何回报

每批次提供：实际 base/tested/implementation commit、分支或 PR、修改文件、可重跑命令、退出码、通过/失败/跳过数量、真实数据 revision、结果摘要和下一步。

`unit`、`contract`、`docker_integration`、`gpu_integration` 分开记录。未执行写 NOT_RUN；有命令证据才能写 PASS。mock/replay 联调必须说明未发生真实参数更新。

当前交接文件规定本批次报告位置；不要每轮另外创建 roadmap_v2、status_final 等重复进度入口。报告是执行证据，不等于 Notion 任务自动完成。GitHub 也不是自动唤醒另一端 agent 的会话通道。
