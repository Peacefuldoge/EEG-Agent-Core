# 相对 EEG_Preprocess_n 的修改记录

- 保留旧版四数据集读取与标签处理逻辑，不直接改动原始标签数据。
- 新增 `eeg_universal.py`：通用 EEG 文件/数值数据处理，支持 NxT 可变输入通道、按名/按数目标通道、任意样本可表达的窗口、步长、事件限制、NPZ/FC。
- 新增 `channel_layout.py`：共享标准 26、64 名称、空间均匀选点、HydroCel-129 对应关系、输入参数验证。
- 原 `*_FINAL.py` 改用 `EEG_TARGET_CHANNELS`、`EEG_TARGET_COUNT`、`EEG_WINDOW_SECONDS`、`EEG_TARGET_SFREQ`、`EEG_DATASET_ROOT`、`EEG_OUTPUT_DIR`。
- 动态修改硬编码的 `64x64`、`26x26`、`(N,64,250)` 断言及报告字段；修复 MODMA/TDBRAIN 非整数采样窗口的时间戳累计偏差。
- 允许 HydroCel 已有电极总数范围内的可选通道（最多 129），源传感器严格一对一映射。
- 单通道跳过 CAR、FastICA；FC 的 1×1 对角固定为零。
- 旧报告/Notebook 的固定结果没有重生成；仅作为历史版本参考。
- 增加无真实 EEG 输入即可执行的合成数据测试和示例配置。

## 已执行的验证

- `python -m compileall -q .`：通过。
- `python -m pytest -q tests/test_universal.py`：15 tests passed。
- 使用模拟 `.npz` 数据执行完整 CLI 流程：8→4 通道、1.3 秒窗口、0.7 秒步长；输出 `(13,4,325)`、FC `(13,4,4)`，元数据可读取。
- 使用模拟 HydroCel 数据测试 `E22/E9/E36/E104 → Fp1/Fp2/C3/C4` 命名映射 CLI：通过。
- HBN/MODMA 输出布局的 1、16、26、32、64、65、80、129 和 native 模式导入检查：全部通道映射唯一。

## 未做的验证

本包不包含大型实际 EEG 记录，仅包含原项目代码及标签表。因此未进行真实 PRED+CT、HBN、MODMA、TDBRAIN 全集重新预处理。上线前建议用每个数据集 1～2 个真实被试检查电极坐标、原始单位、参考方式和事件准确性。

## Visualization integration (2026-10-23)

- Added `eeg_visualization.py` (17 figure types, unit-safe old/new NPZ ingestion, FC fallback, dynamic channel/time window handling, file-level QC).
- Replaced hardcoded-path `FC_graph.py` / `FC_Matrix.py` with compatible CLI shims. Originals archived in `legacy/scripts/`.
- Restored old notebooks to `legacy/notebooks/`, replaced root notebook with fully runnable new analysis, added compact transparent and 17-chart gallery notebooks.
- Added interactive navigation via local `index.html`, descriptive dataset comparison, CSV validation/overlap/JS reports, optional vector SVG export.
- `eeg_universal.py --visualize` now executes the same report generator after saving NPZ, and exports `channel_positions_xyz` for montage-informed plots.
- Demo generator and precomputed **synthetic** gallery in `examples/`. No patient raw EEG bundled.
- Added visualizer and end-to-end integration tests; 28 tests passed in development environment.
