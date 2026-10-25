# EEG-Agent Core v0.4.2 — 前五项问题修复说明

**基线**：用户上传的 Core v0.4.1（`fix/versioned-replanning-evidence`）。
本补丁只针对 Core；不包含 UI 或 vLLM 权重。

## 修复内容

1. **ReAct 失败状态**：工具异常、参数绑定失败、QC 门禁拒绝与工具返回 `failed` / `error` / `partial_failure` 均写入 `trace`、`error`、`tool_failures`，终态不会误标 `completed`。成功工具与失败工具混合的其他运行可使用 `partial_success`。不会再把 LLM 提供了文本答复当作工具执行成功。
2. **NPZ 单位**：连续二维 NPZ 的 `data_unit`（`V` / `mV` / `uV`）自动识别并转换为 MNE 内部伏特；显式 `input_unit` 与元数据冲突时抛错。`PreprocessArgs`、`InspectArgs`、`QcArgs` 支持 `input_unit`。没有 `data_unit` 的旧 NPY/NPZ/MAT 为兼容旧数据仍默认为 `uV`，请在未知单位时显式核实；CLI `--mat-unit` 默认改为自动识别。**更新了预处理缓存标识为 `preprocess-v0.4.2`，防止旧版单位错误产生的缓存继续被命中**。
3. **统一 QC 门禁**：三种 Agent 策略共享 `qc_policy.evaluate_tool()`。初始 QC 为 `requires_review` / `unavailable` 等时，预处理、批预处理和科学分析/可视化工具必须获得独立人审授权；无法审批时直接拒绝/触发安全重规划。无损检查、完整性校验、QC 报告/预处理建议仍可执行。科学参数明显不合理时拒绝，批准时仍支持修改参数后重新审批。**纯 Python 无审批运行不会静默视作批准**。
4. **完整 DAG 就绪队列**：`ready_batch` 不再按 cursor 的连续片段筛选，而是遍历全计划找出依赖证据已完成的节点；即使计划顺序不拓扑排序也能执行。`Plan` 验证所有依赖存在且无环。只并发执行 `parallel_safe=True` 的工具；需要审批或依赖 `$latest_npz` 的工具单独运行。每步状态为 `step_status`，每次重规划重新初始化；已完成旧计划的证据仍保留完整版本键。`cursor` 兼容旧界面，但现在表示当前计划完成的步骤数，不再表示下一项索引。
5. **失败日志和前端事件**：`tool_progress.phase` 区分 `started` / `completed` / `failed` / `rejected`；添加 `step_id`、`attempt_id`、错误信息、耗时。`trace` 记录开始时间、结束时间、耗时和错误；运行清单记录 `step_status`、`qc_summary`、`tool_failures`。原有 `tool_progress.data` 保留兼容。

## 行为/接口说明

- 新的运行状态 `partial_success` 表示先前工具出错，后续安全计划完成但本次运行有失败历史；请在 UI 中将其区别于 `completed`。
- 审批机制依赖 `checkpoint=True` 和 `approval_tools`。默认 session 在检查点开启时审批 `preprocess_eeg` / `batch_preprocess`；若还想允许在异常 QC 下执行分析工具，须将所需工具显式加入 `approval_tools` 并由用户批准。`checkpoint=False` 时异常 QC 的高风险工具将拒绝。
- 在计划中优先引用 `$E2.output_npz`（重规划后自动转换为 `$P2:E2.output_npz`）并填写 `depends_on`；`$latest_npz` 是共享可变引用，故工具单独调度，但不会推断未声明的隐式依赖。
- 不具备明确 `data_unit` 元数据的旧 NPZ **并不能**从信号幅度可靠推断其物理单位；仍默认 `uV`。必要时在工具调用中显式设置 `input_unit`。
- 对已存在的正在执行的 LangGraph checkpoint，建议使用原 Core 版本完成或重新创建任务，再切换 v0.4.2；改变节点选择语义可能改变旧检查点重放的调度方式。

## 本地运行/验证

在 Core 根目录、依赖已经安装好的虚拟环境中：

```bash
python -m pytest -q
python -m pytest -q tests/agent/test_p0_p1_fixes.py
```

使用 uv 安装完整测试/Agent 环境（WSL/Linux）：

```bash
uv pip install -e '.[agent,dev]' --python /path/to/venv/bin/python
/path/to/venv/bin/python -m pytest -q
```

环境未安装 LangGraph 时，基于离线模拟器的节点/编排测试仍可运行；针对 LangGraph 的真实集成测试将自动跳过。**本次审查环境对 PyPI 的 DNS 无法连接，因此没有声称真实 vLLM/Qwen 或 LangGraph Checkpoint 的集成运行已通过。**
