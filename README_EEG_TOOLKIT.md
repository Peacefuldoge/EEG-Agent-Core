# EEG 通用预处理（基于 EEG_Preprocess_n 改造）

保留旧项目中 **PRED+CT、HBN、MODMA、TDBRAIN** 的数据读取、事件标签与原有预处理流程，并把通道布局和窗口参数改成可配置。另提供独立的 `eeg_universal.py`，可以处理**任意数量的实际 EEG 输入电极**以及自定义目标通道、采样率、窗口长度和滑窗步长。

> **重要：**“任意输出通道数”不等于可以无中生有生成真实测量电极。默认仅允许从实际输入电极中挑选目标通道；目标电极缺失时默认报错。通用入口支持通过 `--interpolate-missing` **显式**基于有坐标的电极插值，但插值电极是估计信号，不应当作为真实测量通道报告。跨数据集比较请指定**同一份目标通道名称与顺序**，不能只指定相同数量。

## 文件说明

| 文件 | 用途 |
|---|---|
| `eeg_universal.py` | **推荐通用入口**，从 MNE Raw/文件或数值 NxT 处理为统一 NPZ；支持命令行和 Python API |
| `channel_layout.py` | 四个旧版数据集脚本公用的目标通道解析和 HydroCel 映射 |
| `PREDCT_FINAL.py` | 原 PRED+CT BIDS/EEGLAB SET+FDT 读取、EO/EC 事件分段 |
| `HBN_FINAL.py` | 原 HBN EEGLAB SET，HydroCel-129 电极映射及 EO/EC 分段 |
| `MODMA_FINAL.py` | 原 MODMA EEGLAB MAT 结构解析、在线 Cz 参考处理 |
| `TDBRAIN_FINAL.py` | 原 TDBRAIN BDF、MDD/HC 标签表和 EC 分段 |
| `configs/` | 26/64 标准通道顺序及 HydroCel 映射 JSON |
| `tests/test_universal.py` | 无需 EEG 数据集的合成测试 |
| `README_LEGACY.md` | 旧版说明和原始实验设置，仅供参考 |

## 环境

推荐 Python 3.10+。在代码文件夹执行：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests/
```

## 方案 A：通用入口（推荐新数据集）

**例 1：任选输入通道，输出 26 个实际存在的通道、1 秒窗口。**

```powershell
python eeg_universal.py `
  --input-file "D:\EEG\subject01.edf" `
  --output-dir "D:\EEG\processed_26_1s" `
  --target-channels "@configs/standard_26.json" `
  --sfreq 250 --window-seconds 1.0 --default-label "Eyes Closed"
```

若缺少 `configs/standard_26.json` 中任意目标通道，将明确报错；请调整通道清单或使用有依据的重映射。

**例 2：32 通道输入，自动选 16 个空间分布较均匀的现有 EEG 通道，切 0.75 秒窗口，0.25 秒滑动。**

```powershell
python eeg_universal.py `
  --input-file "D:\EEG\recording.bdf" --output-dir "D:\EEG\processed" `
  --target-count 16 --window-seconds 0.75 --stride-seconds 0.25
```

**例 3：任意 NxT 连续数值矩阵**（`.npy` / `.npz` / `.mat`）：

```powershell
python eeg_universal.py `
  --input-file "D:\EEG\sample.npy" --input-sfreq 512 `
  --input-channels "Fp1,Fp2,F3,F4,C3,C4,P3,P4" --mat-unit uV `
  --target-channels "Fp1,F3,C3,P3" --sfreq 250 `
  --window-seconds 1.6 --output-dir "D:\EEG\processed"
```

`.npy` 默认输入形状 `(C,T)`，必须提供采样率；`.npz` 可包含 `data (C,T)`、`sfreq`、`channel_names`。`.mat` 支持简单 `data/sfreq` 或 EEGLAB `EEG.data/EEG.srate` 结构。MAT/NPY/NPZ 输入默认为 **uV**，用 `--mat-unit V` / `mV` / `uV` 设置实际单位。原始 EDF/BDF/SET/FIF/VHDR/CNT 由 MNE 读取为伏特，不再手动缩放。**通用 MAT 读取不执行 MODMA 原版对缺失 Cz 在线参考的自动重建；MODMA 原数据请优先使用其专用适配器。**

**例 4：按事件边界切窗（不允许跨 EO/EC 段）**：

```powershell
python eeg_universal.py `
  --input-file "D:\EEG\recording.set" --events-tsv "D:\EEG\events.tsv" `
  --target-count 16 --window-seconds 1.25 --output-dir "D:\EEG\processed"
```

TSV 至少包含 `onset`、`duration`，推荐带 `trial_type`；单位秒。未提供 TSV 时优先使用 MNE annotations 中正时长标注，没有可用标注时才把全文件看成单一段（标签默认 `Unknown`）。

**例 5：HydroCel E1...E128 → 标准 26 电极名称（使用旧版固定映射）**：

```powershell
python eeg_universal.py `
  --input-file "D:\HBN\sub01.set" --output-dir "D:\HBN\preprocessed" `
  --source-map "configs/hydrocel_to_26.json" `
  --target-channels "@configs/standard_26.json" `
  --window-seconds 1.0
```

`--source-map` 的 JSON 格式为 `{ "目标电极名": "源电极名" }`。对不同系统的电极对齐是一种固定位置近似，并不是任意传感器均可无损重命名。

**例 6：指定缺失电极的坐标插值（默认禁用）**：

```powershell
python eeg_universal.py --input-file "D:\EEG\sample.edf" `
  --output-dir "D:\EEG\estimated" --target-channels "Fp1,Fp2,C3,C4,Cz" `
  --montage standard_1020 --interpolate-missing
```

只有在源电极和目标电极的坐标可用时才能执行。输出 metadata 包含 `virtual_interpolated_channels` 和插值来源，且这是基于逆距离加权的**估计信号**。

命令行帮助：`python eeg_universal.py --help`。目录批量处理：`--input-dir D:\EEG --glob "**/*.edf"`，输出保持相对目录结构，避免不同被试的同名记录互相覆盖。默认不会覆盖已存在 NPZ，覆盖需 `--overwrite`；每次运行生成 `manifest.json`。

## 方案 B：继续运行原来的四个数据集脚本

旧版逐数据集实现保留原来的 ICA/EOG 处理及事件识别、标签表处理、QC、输出结构；将以下环境变量应用于**当前 PowerShell 会话**：

```powershell
$env:EEG_TARGET_CHANNELS = "26"       # 26、32、64、native、Fp1,F3,C3... 或 @list.json
$env:EEG_WINDOW_SECONDS = "1.0"      # 任意正数；最终使用整数采样点
$env:EEG_TARGET_SFREQ = "250"        # 目标采样率（Hz）
$env:EEG_DATASET_ROOT = "D:\EEG\raw"
$env:EEG_OUTPUT_DIR = "D:\EEG\preprocessed_26_1s"
# TDBRAIN 可另外设置 $env:EEG_LABEL_TABLE = "D:\EEG\TDBRAIN_MDD_HC.csv"
python .\HBN_FINAL.py
```

`EEG_TARGET_CHANNELS` 也可以替换为 `$env:EEG_TARGET_COUNT = "32"`（二者不可同时设置）。`26` 使用共享 26 通道清单，`64` 使用共享 64 通道清单，其他数字按电极坐标选择相对均匀的子集。`native` 对 HydroCel 选择 E1…E128 + Cz；对于 PRED+CT/TDBRAIN 保留各自已知的源电极。`HBN/MODMA` 支持在 64 以上混用已映射的标准电极名称和其他未占用的原生 E# 电极名称，最多可达其实际 129 个传感器。

**注意：** PRED+CT 的既有 EEG 列表为 64 个，TDBRAIN 为 26 个。要求 64 通道的 TDBRAIN 在原始数据缺少电极时会**报错**；不会默默创造 38 个通道。通用 `eeg_universal.py` 支持符合物理约束的显示插值。多个数据集均需统一 26 通道时使用相同配置；并确保输出目录区分 `通道数+窗口长度`，避免旧脚本中 `OVERWRITE=True` 覆盖别的实验设置。

四个原脚本默认参数（未设置环境变量时）尽量保留旧版：PRED+CT/HBN/MODMA 64 通道，TDBRAIN 26 通道；窗口均为 2 秒，采样率 250 Hz。输出分别保留原项目原有文件命名、标签字段和目录组织。`README_LEGACY.md` 中的固定形状描述仅适用于旧实验。

## 通用处理步骤

读取数据 → EEG/EOG 类型区分 → 通道选择/映射 →（可选）缺失通道显式插值 → 50/60 Hz 陷波 → 0.5–45 Hz Butterworth 滤波 → 目标采样率 → 坏道检测（稳健 z 值）与基于坐标的近邻修复 → CAR（单通道跳过）→ 有真实 EOG 时的 FastICA → 事件段内完整滑窗 → 逐窗基线校正与幅度/RMS QC → Pearson FC → NPZ。

- 滑窗长度为 `round(window_seconds * sfreq)` 个样点，故秒数有约 `1/sfreq` 精度。**Pearson FC 要求窗口至少 2 个采样点。**
- 输出 `data` 为 `(N,C,T)`，单位 **uV**；`fc` 为 `(N,C,C)`，对角置零；还有 `channel_names`、`sfreq`、`labels`、`window_times_sec`、`metadata_json`。
- QC 被剔除的窗口不写入文件；所有窗口剔除时直接报错。不完整尾窗丢弃，事件窗口严格处于对应事件段内。
- 通用入口不会自动做训练/测试被试划分。进行 EEG SSL、抑郁检测时，应先确保下游被试**不出现在任何自监督预训练集合**中，且所有划分按被试进行。
- 单通道、无电极坐标、无 EOG、无有效事件等边界输入均有明确的退化/跳过行为，具体请检查 `metadata_json`。

## 测试与限制

测试：`python -m pytest -q tests/test_universal.py`。已包含通道选择、可变窗口、事件边界、缺失电极保护、插值和旧数据集适配器动态配置；还应在真实数据上检查滤波、单位、坐标映射和标签一致性。

仅用合成波形进行功能测试，**不代表在四个真实数据集上完成了重新预处理与结果复现**。旧的分析 Notebook、CSV、PNG（若另行保留）属于历史输出，不应当用来代表此次新配置。官方实验请固定依赖版本、通道清单、随机种子，并保存 NPZ 元数据与 subject-level split 清单。

---

## 新增：可视化与数据一致性分析（本项目集成）

**新版统一入口：`eeg_visualization.py`**。支持前述 `eeg_universal.py` 和四个旧数据集适配器导出的 NPZ，无需改动文件内部结构。输入只要求包含：

- `data`: `(N, C, T)`，N 为窗口数，C 为**任意正整数通道数**，T 为可变采样点数；`sfreq` 必填且正数；
- `channel_names` 可选（缺省自动生成 Ch1...）；`fc` 可选（缺省现算 Pearson）；
- `data_unit` 推荐设置 `uV`，也支持 `mV` 与 `V`，转换到 µV 后出图；没有单位标记的旧文件默认视为 µV，可用 `--assume-unit` 改写；
- `window_times_sec`、`labels`、`subject_ids`、`dataset_names`、`metadata_json`、`channel_positions_xyz` 都是可选增强字段。

### 图表类型：17 种

| 参数名称 | 图表 / 使用场景 |
|---|---|
| `traces` | 多通道原始波形重叠对比 |
| `stacked_traces` | EEG 堆叠波形及电极标签 |
| `psd` | 平均 Welch 功率谱密度 (dB µV²/Hz) |
| `fft` | 单边 FFT 幅度谱 (µV) |
| `spectrogram` | 单通道窗口时频谱 |
| `bandpower` | Delta/Theta/Alpha/Beta/Gamma 相对频带功率 |
| `bandpower_map` | Alpha（或可用频带）头皮地形图 |
| `bandpower_time` | 随窗口变化的频带功率热图 |
| `fc_heatmap` | Pearson FC 热力矩阵 |
| `fc_circle` | FC 圆环曲线图（支持高亮指定电极对） |
| `fc_network` | 电极空间连接网络；无坐标时注明圆形回退 |
| `fc_strength` | 每通道平均绝对连接强度 |
| `fc_distribution` | 非对角 Pearson 连接分布 |
| `rms` | 窗口 RMS 分布 |
| `rms_time` | 随时间变化的 RMS |
| `channel_qc` | 每通道 RMS / 峰峰值箱线图 |
| `qc_summary` | 参数、窗口数、ICA/坏道/QC 元数据摘要 |

### 快速体验（已内附合成示例 NPZ 和 HTML 图册）

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests

# 自动生成四个不同输入通道数、不同窗口长度的合成示例
python examples/create_demo.py

# 全部 17 种可视化 + 跨文件比较 + HTML 图册
python eeg_visualization.py --input-dir examples/demo_npz `
  --output-dir reports/demo --plots all --compare
```

直接打开 `reports/demo/index.html`。压缩包也提供了预生成的 `examples/demo_report/index.html`，方便直接查看效果（**只包含合成数据**，不是真实研究结果）。

### 与通用预处理器一键对接

```powershell
python eeg_universal.py `
  --input-file "D:\EEG\recording.edf" `
  --output-dir "D:\EEG\processed" `
  --target-count 26 --window-seconds 1.3 --stride-seconds 0.65 `
  --sfreq 250 --visualize --viz-plots all --viz-max-windows 64
```

输出 `D:\EEG\processed\xxx_preprocessed.npz`，以及 `D:\EEG\processed\visualizations\index.html`。如果输入批量记录，可同时生成跨记录对比图。**该操作不会向已存 NPZ 再执行滤波或改变数据**。默认不生成可视化，只有 `--visualize` 打开后才运行，避免生产批次无意生成大量图片。

### 四个旧数据集适配器的结果也能直接分析

```powershell
# 先按原说明运行各 *_FINAL.py，得到 NPZ
python eeg_visualization.py --input-dir "D:\EEG\preprocessed" `
  --glob "**/*.npz" --output-dir "D:\EEG\reports" `
  --plots default --compare --max-windows 80 --max-files 200
```

`FC_graph.py` 和 `FC_Matrix.py` 已从旧版固定 Windows 路径重构为兼容性 CLI 入口，分别默认画 `fc_circle`、`fc_heatmap`：

```powershell
python FC_graph.py --input-file "D:\EEG\subject01.npz" `
  --output-dir "D:\EEG\fc_circle" --masked-pairs "Fp1:O2,F3:P4" `
  --top-edges 40 --transparent --formats png,svg --dpi 300

python FC_Matrix.py --input-file "D:\EEG\subject01.npz" `
  --output-dir "D:\EEG\fc_heatmap" --fc-mode single --window-index 12 `
  --absolute-fc --fc-diagonal-one
```

`--fc-mode` 接受 `mean`（默认，平均窗口 FC）、`single`（选择原始窗口索引）、`concatenate`（拼接已加载窗后重算 FC）。注意这三种方式的估计量**并不等价**，`concatenate` 还可能引入窗口边界效应，论文请明确报告方法。`--max-windows` 默认 64，等距抽取并强制保留指定窗口；大文件仍需先完整解压 NPZ 的 EEG 数组，不能视为内存映射。

### 报告与可复现输出

每次批量执行会生成：

- `index.html`：自包含本地 HTML 图片目录（离线浏览，不依赖网络）；
- `<编号>_<NPZ文件名>/`：各类 PNG / 可选 SVG 与 `report.json`；
- `dataset_summary.csv`：各文件通道数、窗口长度、有效采样率、RMS 等；
- `fc_validation.csv`：FC 形状、对称性、对角线、范围检查；
- `preprocessing_metadata.csv`：ICA、坏道、参考方式、坐标来源；
- 使用 `--compare` 时额外生成 `channel_overlap.csv`、`psd_js_divergence.csv`、`compare_*.png`、共同通道 FC 对比图；若跨文件完全无共享通道则写明跳过原因；
- `failed_files.json`：记录不能读取的文件及原因，不会偷偷忽略。

项目根目录的 `Dataset_consistency_analysis.ipynb`、`Dataset_consistency_analysis_compact_transparent.ipynb` 和 `EEG_visualization_gallery.ipynb` 都采用新接口；历史 Notebook 及原始 FC 代码存档在 `legacy/`，防止丢失原实验版本。

### 研究解释与注意事项

1. **没有输入前后 EEG 对照就不能画真实的前后去伪迹变化**；当前质量图描述输出数据，并通过 `metadata_json` 呈现处理记录，不会伪造“处理前”结果。
2. 头皮图优先使用 NPZ 中的 `channel_positions_xyz`；旧数据没有坐标则尝试 MNE 标准 montage，**只是近似位置**。若无法确认至少四个电极位置则清楚标记不可绘制；这些地形图不能替代电极坐标验证。
3. 不同数据集比较只比较重叠的**同名通道**，不是同样数量的任意通道。PSD 的 JS divergence 是描述性距离，不是显著性检验；采样率、窗口长度、分布差异都可能影响它。
4. 频带功率仅在该窗口的频率分辨率足以覆盖完整频段时计算；短窗可能不显示低频段。滑窗可能有重叠，**窗的数量不是独立被试数量**。报告统计按文件等权，不能替代 subject-level 评估或站点校正。
5. 图表默认只对最多 64 个窗进行等间隔抽样，保证速度；完整数据 QC 仍以预处理时的 `window_qc` 为准。若希望真实实验全样本统计可提高 `--max-windows`，内存占用也会增大。
6. **28 个合成 / CLI 功能测试通过**；尚未在真实 PRED+CT、HBN、MODMA、TDBRAIN 全量数据上跑通，也尚未核验各设备对应的真实电极位置。不要将示例图用作论文证据。
