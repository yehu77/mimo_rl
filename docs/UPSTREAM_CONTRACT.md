# B002 upstream code contract

本文件只记录 B002 读取到的 code 路径合同。它是源码核对结果，不是官方
loader、容器或训练栈的运行报告。所有版本都来自 `manifests/versions.yaml`，
读取日期为 2026-09-29，`validation_scope=source_read`，本机没有安装或运行
这些上游依赖。

## 固定源码

| 用途 | 仓库与 commit | 关键路径 |
|---|---|---|
| 官方 recipe | [XiaomiMiMo/verl @ `a2ad9f6160b03ff2d47e59832bfb6b289f37c917`](https://github.com/XiaomiMiMo/verl/tree/a2ad9f6160b03ff2d47e59832bfb6b289f37c917) | `recipes/code/dataset.py`, `recipes/code/mimoagent_runner.py`, `scripts/code/env.example` |
| recipe gitlink | [verl-project/verl-recipe @ `e7f889574b8301cc0f0fc1d57c6d67f31ffeb689`](https://github.com/verl-project/verl-recipe/tree/e7f889574b8301cc0f0fc1d57c6d67f31ffeb689) | `XiaomiMiMo/verl/.gitmodules` 的 `recipe` gitlink |
| environment/agent | [XiaomiMiMo/MiMo-Agent @ `467f0a19016f0ac4d63b8d17a1f0da9ba07f232c`](https://github.com/XiaomiMiMo/MiMo-Agent/tree/467f0a19016f0ac4d63b8d17a1f0da9ba07f232c) | `src/mimoagent/environments/datasets/opensource_code.py`, `base.py`, `environments/utils.py` |
| Uni-Agent 子模块 | [XiaomiMiMo/uni-agent @ `c63e0b01c375ebede95e01fe92bc367df24e5bf3`](https://github.com/XiaomiMiMo/uni-agent/tree/c63e0b01c375ebede95e01fe92bc367df24e5bf3) | `uni_agent/tasks/base.py`, `gateway/session/types.py`, `gateway/session/session.py` |
| rollout/Sample | [THUDM/slime @ `89bfada990a00663846e0ac804de1454685ceed3`](https://github.com/THUDM/slime/tree/89bfada990a00663846e0ac804de1454685ceed3) | `examples/coding_agent_rl/generate.py`, `slime/rollout/{data_source,base_types,sample_hooks}.py`, `slime/ray/rollout.py` |

verl 在固定 commit 的 `.gitmodules` 中把 `third_party/mimoagent-osr` 固定到
`467f0a19016f0ac4d63b8d17a1f0da9ba07f232c`，把 `third_party/uni_agent` 固定到
`c63e0b01c375ebede95e01fe92bc367df24e5bf3`，所以本批次没有分别选择其他
仓库的最新 HEAD。slime 的默认分支实际是 `main`；verl、MiMo-Agent 和
Uni-Agent 的默认分支实际是 `mimo-oss`。

## 数据进入官方 recipe 的位置

在固定 verl recipe 的
[`recipes/code/dataset.py`](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/recipes/code/dataset.py#L14-L51)，
`MimoAgentSWEDataset.__getitem__` 先调用 `RLHFDataset.__getitem__`，再读取
`row["extra_info"]`。它从 `extra_info.instance_json` 解码字符串（对象则直接
使用），要求得到带 `docker_image` 的 dict，将副本放进
`tools_kwargs["instance"]`，把 Parquet item 放进 `tools_kwargs["dataset_index"]`，
并把 `tools_kwargs` 同时回写到 `row` 和 `row["extra_info"]`。固定源码在此处
没有读取本项目的 `image-mapping.jsonl`；映射必须在进入该 adapter 前完成。

固定 runner 的
[`_extract_task`](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/recipes/code/mimoagent_runner.py#L140-L149)
优先使用非空字符串 `raw_prompt`，其次使用 prompt 列表中第一个非空 user
消息，最后才回退到 `instance["problem_statement"]`。在
[`_run_sync`](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/recipes/code/mimoagent_runner.py#L244-L325)，
`tools_kwargs["instance"]` 传给 `make_dataset_env`，`setup_environment()`
先于 agent 创建，`agent.run(task)` 接收上面的任务文本，agent 返回后才调用
`environment.calculate_reward()`。

runner 的环境配置来自 `config["environment"]`，并在
[`make_dataset_env`](https://github.com/XiaomiMiMo/MiMo-Agent/blob/467f0a19016f0ac4d63b8d17a1f0da9ba07f232c/src/mimoagent/environments/utils.py#L54-L124)
中解析 dataset type、选择 `docker_image`、把 `instance["cwd"]` 传入 base
environment，最后调用 `create_from_registry`。该函数本身只构造对象；真正
的底层环境启动在后续的 `DatasetEnvironment.setup_environment()`。

## 官方 instance 与判分语义

MiMo-Agent 固定 commit 的
[`opensource_code.py` 顶部合同](https://github.com/XiaomiMiMo/MiMo-Agent/blob/467f0a19016f0ac4d63b8d17a1f0da9ba07f232c/src/mimoagent/environments/datasets/opensource_code.py#L7-L36)
要求以下八个字段：

```text
dataset_type, docker_image, cwd, instance_id,
problem_statement, test_patch, test_command, verifier_timeout_sec
```

其中 `cwd` 是容器内绝对路径，`instance_id` 用于日志和导出，
`problem_statement` 是任务题面。`test_patch` 是隐藏测试和 verifier 脚本的
补丁，不能解释为参考修复答案；固定源码明确说明它只在 reward 阶段出现。

[`OpenSourceCodeEnvironment.__init__`](https://github.com/XiaomiMiMo/MiMo-Agent/blob/467f0a19016f0ac4d63b8d17a1f0da9ba07f232c/src/mimoagent/environments/datasets/opensource_code.py#L63-L112)
校验字段存在性并保存 `test_patch`、`test_command`、timeout；在
[`_setup_dataset_specific`](https://github.com/XiaomiMiMo/MiMo-Agent/blob/467f0a19016f0ac4d63b8d17a1f0da9ba07f232c/src/mimoagent/environments/datasets/opensource_code.py#L118-L124)
中启动后捕获容器工作树的真实 HEAD 到 `_base_ref`。这个 `_base_ref` 来自任务
容器，不是 HF dataset revision、verl commit、task id 或本项目分支，因此
本批次 `RuntimeSpec.base_ref` 保持 `null`/`PENDING_RUNTIME`。

[`_do_calculate_reward`](https://github.com/XiaomiMiMo/MiMo-Agent/blob/467f0a19016f0ac4d63b8d17a1f0da9ba07f232c/src/mimoagent/environments/datasets/opensource_code.py#L190-L251)
的顺序固定为：

1. 用捕获的 `_base_ref` 重置 test patch 触碰的路径；
2. 将 `test_patch` 复制到容器并执行 `git apply`；
3. 在 `cwd` 执行 `test_command`，退出码为 0 才得到 reward 1；
4. 在 `finally` 再次重置测试路径。

reset 或 apply 失败带 `error_category=reward/testbed_corrupted`，不能简单当成
模型错误的负奖励。B002 只保存这些字段并逐字段生成 controller instance，
不会在 Mac 上执行 command、patch 或 verifier。

## Uni-Agent 与 bash-only 入口

固定 Uni-Agent 的 [`TaskConfig`](https://github.com/XiaomiMiMo/uni-agent/blob/c63e0b01c375ebede95e01fe92bc367df24e5bf3/uni_agent/tasks/base.py#L29-L87)
将 `sandbox`、`agent`、`prompt` 和 `metadata` 分开，且 `extra="forbid"`；
[`Task.build_sandbox/build_agent`](https://github.com/XiaomiMiMo/uni-agent/blob/c63e0b01c375ebede95e01fe92bc367df24e5bf3/uni_agent/tasks/base.py#L100-L134)
才是后续运行层创建 sandbox/agent 的入口。固定 session 的
[`SessionHandle`](https://github.com/XiaomiMiMo/uni-agent/blob/c63e0b01c375ebede95e01fe92bc367df24e5bf3/uni_agent/gateway/session/types.py#L26-L42)
只提供 session id、模型 gateway base URL 和 reward-info URL；
`Trajectory` 还会携带 prompt/response ids、mask、generation spans、logprobs
和 reward metadata。它们是后续 controller 接口，B002 不导入其运行依赖。

固定 verl runner 通过 `agents.factory.get_agent_class` 选择 agent（包括
`bashonly-agent`），然后用 `agent_cls(model, environment.env, ...)` 构造并调用
`agent.run(task)`。这条路径在
[`recipes/code/mimoagent_runner.py#L252-L283`](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/recipes/code/mimoagent_runner.py#L252-L283)
中可核对；本轮只记录该入口，不调用它。

## slime 的后续接入边界

固定 slime 的 `RolloutDataSource.get_samples` 在
[`slime/rollout/data_source.py#L50-L118`](https://github.com/THUDM/slime/blob/89bfada990a00663846e0ac804de1454685ceed3/slime/rollout/data_source.py#L50-L118)
按 `n_samples_per_prompt` 复制 `Sample` 并分配 group/sample index。
`RolloutFnTrainOutput`、`call_rollout_fn` 和 `finalize_rollout_groups` 位于
[`slime/rollout/base_types.py#L12-L100`](https://github.com/THUDM/slime/blob/89bfada990a00663846e0ac804de1454685ceed3/slime/rollout/base_types.py#L12-L100)，
自定义 rollout 的结果必须仍是 `Sample` 或受支持的 rollout reference。
`slime/ray/rollout.py#L276-L310` 负责调用 generate、加载/接受 rollout 数据、
校验 rollout id 并交给 batch builder；`rollout_sample_hooks.py#L37-L64` 是未来
对 Sample 做过滤/扩展的 hook 边界。slime 本批次只做源码定位，未安装、导入、
启动 Ray/SGLang、生成 Sample 或计算 reward。

## 本项目的转换边界

`src/mimo_rl/task.py` 的 `TaskBundle` 复用上述字段语义但不依赖上游包：

* `SolverTask.to_solver_payload()` 只返回显式白名单：`task_id`、原始
  `messages`、`problem_statement` 和容器 `cwd`。它不带 `extra_info`、
  `test_patch`、`test_command`、私有嵌套字段或未知字段。
* `RuntimeSpec` 保留原始 `dataset_image`、精确映射得到的
  `dockerhub_image`、`cwd`、数据 revision、Parquet 0-based
  `source_row_index`、原始 `extra_info.index` 和 JSONL mapping 行号。
  不自动添加 registry/tag/digest，也不创建或解析 Mac 路径。
* `VerifierSpec` 只供未来 controller，保留 `test_patch`、`test_command`、
  `verifier_timeout_sec`，默认 repr 和公开摘要只给长度/hash。`to_mimoagent_instance()`
  才恢复官方八字段；它不等于已经调用官方 loader。
* `repo_identity`、任务仓库的真实 `base_ref` 和 `image_digest` 没有来源时
  保持 `null`，状态写为 `PENDING_RUNTIME`；不从 task id、dataset revision 或
  upstream commit 推断。

`prepare_code_tasks.py` 先用 `inspect_dataset(..., --strict-provenance)` 验证
manifest、真实文件 hash 和全部 2698 条输入，再逐条构造 `TaskBundle`。默认只
写 `catalog_summary.json`、`task_catalog.jsonl` 和字段摘要；只有同时给出明确
`--task-id --write-private` 才写入忽略目录的完整 controller bundle。任何
`test_patch` 或 `test_command` 都只被保存/哈希，绝不通过 `eval`、`exec` 或
`subprocess` 执行。

## 结论与未执行范围

题目进入模型的位置是 runner 的 `_extract_task` → `agent.run(task)`；隐藏测试
进入环境的位置是 reward 阶段的 reset → apply `test_patch` → `test_command`；
后续应复用 `MimoAgentSWEDataset`、`make_dataset_env`、
`OpenSourceCodeEnvironment`、Uni-Agent session 和 slime `Sample`/rollout
接口，而不是在本项目重写 verifier 或训练引擎。

本文件的源码合同核对不代表 official loader runtime、Docker、模型、agent loop、
容器 verifier、slime/SGLang、rollout、参数更新或训练已通过；这些在
`reports/handoffs/B002.md` 中保持 `NOT_RUN`。

## B003 轨迹导出合同（source_read）

B003 只实现一个不依赖上游包的原始调用记录和离线编译器。下面的文件是在
固定 slime `89bfada990a00663846e0ac804de1454685ceed3`、Uni-Agent
`c63e0b01c375ebede95e01fe92bc367df24e5bf3` 上读取的源码；链接和行号用于
复核实际符号，不表示本机加载了这些模块或运行了官方引擎。

* slime 的 [`Sample.append_response_tokens`](https://github.com/THUDM/slime/blob/89bfada990a00663846e0ac804de1454685ceed3/slime/utils/types.py#L268-L391)
  把追加的 token 计入 `response_length`。可训练 token 必须有同长度的原始
  `rollout_log_probs`；非生成/不可训练段使用 `loss_mask=0` 和 `0.0` 占位。
  `effective_response_length` 是 `sum(loss_mask)`，而
  [`_validate_response_metadata_lengths`](https://github.com/THUDM/slime/blob/89bfada990a00663846e0ac804de1454685ceed3/slime/utils/types.py#L542-L553)
  要求 `loss_mask` 与 `rollout_log_probs` 都精确等于 `response_length`。
* 固定的 [`sglang_rollout`](https://github.com/THUDM/slime/blob/89bfada990a00663846e0ac804de1454685ceed3/slime/rollout/sglang_rollout.py#L172-L253)
  请求 `return_logprob=True`，从 `output_token_logprobs` 读取每个真实生成
  token 的 `(logprob, token_id)`，再交给 `append_response_tokens`。B003 因此
  不会用零填充来掩盖模型生成位置的概率缺失。
* Uni-Agent 固定的 [`GatewaySession.run_generation`](https://github.com/XiaomiMiMo/uni-agent/blob/c63e0b01c375ebede95e01fe92bc367df24e5bf3/uni_agent/gateway/session/session.py#L224-L300)
  返回 backend 的 token ids 与概率；[`_prepare_generation_inputs`](https://github.com/XiaomiMiMo/uni-agent/blob/c63e0b01c375ebede95e01fe92bc367df24e5bf3/uni_agent/gateway/session/session.py#L443-L565)
  会把上下文续接为输入，并对输入/工具位置使用 mask 0 和 0.0。其
  [`_assert_response_logprob_alignment`](https://github.com/XiaomiMiMo/uni-agent/blob/c63e0b01c375ebede95e01fe92bc367df24e5bf3/uni_agent/gateway/session/session.py#L725-L729)
  只允许空值或与响应长度完全一致的概率数组。

本项目的 `src/mimo_rl/trajectory.py` 对这些语义做了轻量映射：每个
`GenerationEvent` 保存一次调用的完整输入前缀、输出 token、原始概率和
policy 版本；`compile_single_context_trace` 只接受单 context 的 append-only
前缀，新增输入/工具 token 固定为 `mask=0, logprob=0.0`，生成 token 固定为
`mask=1` 并要求真实概率。B003 仅支持 `temperature=1.0`、`top_p=1.0`、
`top_k=0`、`repetition_penalty=1.0` 的 full-vocabulary 概率；其它采样合同
会被离线检查拒绝。`inspect_trajectory.py` 只读取本地 JSON/JSONL 并输出
脱敏摘要，`--export-training` 仍会拒绝 synthetic、缺失 engine 原始捕获证据、
概率/长度不一致、身份混用和不完整 group，且始终 `parameter_update=false`。

这些是固定源码和结构测试得出的合同（`validation_scope=source_read`）。本轮
没有启动 SGLang、Uni-Agent、slime、官方 loader、模型或训练，因此不存在真实
engine capture、真实 rollout 或参数更新的 runtime 通过证据；带
`capture_evidence=engine_raw` 的单元 fixture 只验证拒绝规则的形状，不能冒充
官方运行结果。
