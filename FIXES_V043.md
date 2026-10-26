# EEG-Agent Core v0.4.3 — Three P0 correctness fixes

修复范围仅 Core，**不包含独立的 EEG-Agent-UI、Milvus RAG 或 vLLM 部署**。

## P0-1：DAG `$latest_npz` 引用安全

- `schema.Plan` 在执行前静态分析 DAG 的 `depends_on` 图。
- 有 `$latest_npz` 的步骤，必须通过依赖链可达 **恰好一个** `preprocess_eeg` 步骤；否则拒绝计划。
- 对可证明唯一的生产者自动重写为 `$E2.output_npz`（重规划时为 `$P2:E2.output_npz`），不再运行时读取可能被并行任务覆盖的全局指针。
- 当前计划中的 `$E*.field` 必须通过 `depends_on` 链可达；原本的 `activate_plan` 仍会为合法的显式引用补充直接依赖。
- `ready_batch` 即使直接被调用，也会在选取可执行步骤前重新验证整个 DAG。
- 跨计划显式引用 `$P1:E1.field` 由 `activate_plan` 校验历史证据；不会把最新 NPZ 隐式继承到新计划。

**示例**（推荐显式引用）：

```json
{"steps": [
  {"id":"E1","tool":"preprocess_eeg","args":{"source":"$input"},"depends_on":[]},
  {"id":"E2","tool":"analyze_spectrum","args":{"source":"$E1.output_npz"},"depends_on":["E1"]}
]}
```

有两个并列上游 `preprocess_eeg` 的场景不可使用 `$latest_npz`，必须指定具体步骤。

## P0-2：批处理每个 EEG 文件独立 QC

- `batch.preprocess_dataset` 对每个被发现的文件调用 `assess_eeg_quality`，随后按照 `evaluate_tool('preprocess_eeg', ...)` 校验。
- `status=pass` 的文件才会被批量预处理；`requires_review`、`unavailable` 或科学参数无效的文件 **不调用 `preprocess_eeg`**，在清单中逐项记录。
- 单个文件 QC 无法加载时采用 **fail-closed**，不因其他文件通过 QC 而被忽略。
- 新返回字段 `requires_review` 是文件数量；原有 `completed`、`failed` 保持语义。包含待审核或失败文件的批次返回 `status=partial_failure`。
- Manifest / batch metadata 包含逐文件简化的 `qc` 信息和原因。
- **重要审批边界：批准执行 batch_preprocess ≠ 允许对所有有 QC 警告的文件自动进行预处理。** 有问题的文件需科研人员检查后按单文件 `preprocess_eeg` 审批流程执行；不能通过 batch 参数绕过逐文件 QC。
- 每个文件的 QC 默认检查最多 16 个窗口或原始信号的前 20 秒；这是当前 QC 的抽样检查，不代表完整临床级评估。

## P0-3：ReAct 完成条件必须有证据

- 默认 Agent 执行模式 `task_mode='analysis'`：至少存在一个 `trace.status=ok` 且带合法 `result` 字典的工具调用，否则标记 `failed`，不能仅凭模型说“分析完成”判定成功。
- 可选的宿主侧合同 `required_tools=['preprocess_eeg','analyze_spectrum']`：每种工具至少有一次成功调用。
- 可选 `required_artifacts=['output_npz']`：至少有一项成功工具结果包含对应的非空字段；如果是 `*_npz`、`*_json`、`*_csv` 或 `*_html`，还要求文件真实存在。
- `task_mode='chat'` 允许纯文字回复，无需 EEG 工具；也跳过原始 EEG QC 预检。UI 的自由聊天也可继续使用 `EEGSession.chat()`。
- 这些是宿主应用设置的执行要求，**不能根据模型自身的声明信任其已完成的步骤**。缺失的要求会写入 `error`，`status='failed'` 和 `final_answer`。
- **边界**：默认规则只要求至少一项成功工具证据；严格的多步任务请由调用方提供 `required_tools` / `required_artifacts`。本版本不尝试通过自然语言关键词猜测任务的全部科研目标。

## 测试

新增 `tests/agent/test_p0_v043_regression.py`：

1. 无上游生产者的 `$latest_npz` 在执行前拒绝（包括直接调用调度器）；
2. 唯一上游生产者自动绑定为明确 evidence 引用，覆盖重规划命名空间；
3. 多生产者歧义拒绝；
4. 显式 evidence 引用缺少依赖及不存在的跨计划证据拒绝；
5. 混合 QC 批次只处理合格文件，manifest 有逐文件记录，重复执行仍然阻止坏文件；
6. 无法读取的原始文件被阻止；
7. ReAct 无工具的分析性文字回复标记失败；
8. Chat 模式正常无工具完成；
9. 缺少指定工具或工件时拒绝完成；
10. 有指定工具和证据时正常完成；
11. 工具报告了不存在的输出文件时仍然失败。

```bash
python -m pytest -q
```

本地测试依赖已提供的科学包和 pytest，使用轻量的图节点模拟器；没有安装 LangGraph / LangChain 的环境中，真实 LangGraph 集成 smoke test 会被跳过。集成环境仍需要对照 vLLM、SQLite checkpoint、HITL 和 EEG-Agent-UI 联调。
