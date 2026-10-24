# EEG NPZ 可视化指南

[返回中文项目说明](../README.zh-CN.md) | [查看源码](../eeg_visualization.py)

`eeg_visualization.py` 将已经切窗的 EEG NPZ 转换为波形、频谱、频段功率、Pearson 功能连接和描述性质控图，并支持跨记录、跨数据集报告。它兼容通用预处理以及 PRED+CT、HBN、MODMA、TDBRAIN 专用脚本的输出，不修改源 NPZ。

## 安装与快速开始

以下命令均在仓库根目录执行，参数写成单行，适用于 PowerShell 和常见 Linux shell。

```bash
python -m pip install -e ".[viz]"

# 使用仓库自带的合成数据，生成全部图表和比较报告
python eeg_visualization.py --input-dir examples/demo_npz --output-dir reports/demo --plots all --max-windows 12 --compare --dpi 150

# 只处理一个文件
python eeg_visualization.py --input-file examples/demo_npz/synthetic_TDBRAIN.npz --output-dir reports/single
```

打开输出目录的 `index.html` 查看报告。需要 PNG 图片时直接使用记录子目录；需要矢量图时设置 `--formats svg` 或 `--formats png,svg`。

## 输入约定

必须有以下字段：

| 字段 | 要求 |
| --- | --- |
| `data` | 非空三维数组 `(windows, channels, samples)`，每窗至少 2 点 |
| `sfreq` | 有效的正数采样率，单位 Hz |

推荐保留以下字段：

| 字段 | 行为 |
| --- | --- |
| `channel_names` | 长度须等于通道数且不能重复；缺失时使用 `Ch1` 等名称 |
| `data_unit` | 支持 V、mV、uV 及微伏符号，读取后统一为 µV |
| `fc` | 支持 `(N,C,C)` 或 `(C,C)`；缺失或不能使用时根据 EEG 重算 |
| `window_times_sec` | `(N,2)` 原记录时间区间 |
| `window_start_sec/window_end_sec` | 缺少 `window_times_sec` 时采用 |
| `labels` | 长度须等于窗口数；缺失时使用 `Unknown` |
| `dataset_names/subject_ids` | 用于分组和标题；缺失时从元数据或路径回退 |
| `metadata_json` | JSON 字符串，用于展示处理参数与质控记录 |
| `channel_positions_xyz` | `(C,3)` 有限坐标，支持头皮位置投影 |

缺少所有窗口时间字段时，代码按窗口顺序推算连续时间，不能据此还原原记录中的间隙。旧 NPZ 缺少 `data_unit` 时默认假设 µV，可通过 `--assume-unit V` 或 `mV` 改变假设；已有单位字段优先生效。

工具使用 `allow_pickle=False` 读取 NPZ。所选 EEG 窗口含 NaN 或 Inf 时会拒绝该记录。虽然图表默认最多选 64 个窗口，但 NPZ 的 `data` 数组会先完整加载，因此 `--max-windows` 不能消除大文件的读取内存开销。

## 命令行用法

三种输入方式互斥：重复指定 `--input-file`、指定一个 `--input-dir`，或指定通用预处理生成的 `--manifest`。

```bash
# 自选图表
python eeg_visualization.py --input-file examples/demo_npz/synthetic_TDBRAIN.npz --output-dir reports/basic --plots traces,psd,bandpower,fc_heatmap,channel_qc

# 多文件比较
python eeg_visualization.py --input-file examples/demo_npz/synthetic_HBN.npz --input-file examples/demo_npz/synthetic_MODMA.npz --output-dir reports/compare --compare

# 按目录递归匹配 NPZ，最多处理 20 个文件
python eeg_visualization.py --input-dir preprocessed --glob "**/*.npz" --max-files 20 --output-dir reports/batch

# 读取 eeg_universal.py 输出的 manifest.json
python eeg_visualization.py --manifest preprocessed/manifest.json --output-dir reports/from_manifest --compare

# 查看原始索引为 3 的窗口，并输出透明 PNG/SVG
python eeg_visualization.py --input-file examples/demo_npz/synthetic_TDBRAIN.npz --output-dir reports/window3 --plots stacked_traces,fc_heatmap --fc-mode single --window-index 3 --formats png,svg --transparent
```

`--manifest` 读取包含 `status: "OK"` 和 `file` 字段的条目，不是任意 Agent 批处理 manifest 的通用读取入口。混合存放特征 NPZ、连接矩阵 NPZ 和 EEG NPZ 时，应通过目录或 `--glob` 缩小范围；只有符合 EEG 输入约定的文件才能处理。

### 参数速查

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input-file` | 无 | NPZ 路径，可重复指定 |
| `--input-dir` | 无 | 扫描目录 |
| `--manifest` | 无 | 通用预处理清单 |
| `--glob` | `**/*.npz` | 目录匹配规则 |
| `--output-dir` | 必填 | 输出目录 |
| `--plots` | `default` | `default`、`all` 或逗号分隔的图名 |
| `--max-windows` | `64` | 每文件最多选择的窗口数 |
| `--max-files` | `100` | 最多处理文件数 |
| `--fc-mode` | `mean` | `mean/single/concatenate` |
| `--window-index` | `0` | 原 NPZ 中的窗口索引，从 0 开始 |
| `--top-edges` | `35` | 圆图和网络图最多展示的强连接边数 |
| `--max-traces` | `12` | 波形图最多展示的通道数 |
| `--dpi` | `180` | 单记录图片分辨率，以及比较图 PNG 分辨率 |
| `--formats` | `png` | `png`、`svg` 或 `png,svg`；比较图仍为 PNG |
| `--assume-unit` | `uV` | 仅当文件没有单位字段时使用 |
| `--compare` | 关闭 | 至少两条记录成功时生成比较报告 |
| `--masked-pairs` | 空 | 圆图指定高亮通道对，如 `Fp1:O2,F3:P4` |
| `--absolute-fc` | 关闭 | 仅将 FC 热图显示为相关系数绝对值 |
| `--fc-diagonal-one` | 关闭 | 仅将 FC 热图对角线显示为 1 |
| `--transparent` | 关闭 | 单记录图透明背景 |

窗口采用覆盖全记录的确定性均匀索引抽样，不是随机抽样；指定的 `window-index` 会被纳入选取。波形和时频图使用该窗口，PSD、频段功率和质控统计通常汇总所选窗口。

## 17 种图表

`--plots all` 生成全部 17 种；默认集合有 13 种，下表标注默认是否包含。

| 图名 | 默认 | 内容与计算范围 |
| --- | --- | --- |
| `traces` | 是 | 指定窗口中部分通道的叠加波形 |
| `stacked_traces` | 否 | 指定窗口中部分通道的偏移堆叠波形 |
| `psd` | 是 | 所选窗口和通道平均的 Welch 功率谱，显示为 dB |
| `fft` | 否 | 所选窗口和通道平均的单边 FFT 幅值谱 |
| `spectrogram` | 是 | 指定窗口第一个通道的时频图 |
| `bandpower` | 是 | 所选窗口和通道平均的相对频段功率 |
| `bandpower_map` | 是 | 优先 alpha 的相对功率头皮分布；不可用时选首个有效频段 |
| `bandpower_time` | 否 | 各窗口、各频段的平均相对功率热图 |
| `fc_heatmap` | 是 | Pearson 连接矩阵热图 |
| `fc_circle` | 是 | 按绝对连接强度选边的连接圆图 |
| `fc_network` | 是 | 头皮投影连接图；位置不全时回退为圆形布局 |
| `fc_strength` | 是 | 每通道非对角元素的平均绝对 Pearson 相关 |
| `fc_distribution` | 否 | 唯一通道对的 Pearson 相关分布 |
| `rms` | 是 | 所选窗口、通道的 RMS 分布 |
| `rms_time` | 是 | 平均通道 RMS 随窗口起始时间的变化 |
| `channel_qc` | 是 | 每通道 RMS 和峰峰值箱线图 |
| `qc_summary` | 是 | 记录参数、窗口数、坏导、ICA 和质控摘要 |

默认频段为 delta 1–4 Hz、theta 4–8 Hz、alpha 8–13 Hz、beta 13–30 Hz、gamma 30–45 Hz。相对功率在绘图中显示为百分比，分母是 1 Hz 到 `min(45 Hz, Nyquist)` 的 PSD 积分；超出 Nyquist 或频点不足的频段会被省略。短窗口频率分辨率有限，不能仅凭图形细节推断精确的低频结构。

注意与特征 API 区分：`extract_features` 的相对功率特征使用比例值，不是这里图上的百分数。

## 功能连接的三种模式

| 模式 | 计算方法 | 解释 |
| --- | --- | --- |
| `mean` | 对所选窗口的已有 FC 求均值；不能使用已有 FC 时从 EEG 重算 | 默认，用于描述各窗连接的平均水平 |
| `single` | 取 `--window-index` 指定窗口的 FC | 用于检查单窗状态 |
| `concatenate` | 将所选窗口按时间拼接后重新计算 Pearson | 与逐窗平均不是同一种估计，也不能视为原始连续 EEG |

若输入只存一个 `(C,C)` 矩阵，`mean` 和 `single` 都使用该矩阵，不能从它恢复真正的逐窗 FC。最终展示矩阵会做对称化、范围裁剪与对角线归零；`--fc-diagonal-one` 仅改变热图显示。图表不会写回这些变化。

连接边默认按 `abs(r)` 从大到小排序，颜色保留正负相关信息。`--masked-pairs` 用于圆图高亮，并不删除连接。绘图不是统计显著性检验，强相关也不等于因果关系。

此模块面向 EEG NPZ 中的 Pearson 连接。`analyze_connectivity` 输出的 `fc_alpha` 等矩阵文件不含必需的三维 `data`，不能直接作为这里的输入。

## 报告目录与错误处理

```text
reports/demo/
├── index.html                       # 总览入口
├── dataset_summary.csv              # 每文件描述统计
├── fc_validation.csv                # FC 对称性、对角线、范围检查
├── preprocessing_metadata.csv       # 处理与单位元数据摘要
├── failed_files.json                # 读取或处理失败的记录
├── 001_<文件名>/
│   ├── traces.png                   # 按请求生成的图片
│   ├── psd.png
│   └── report.json                  # 所选索引、图片列表和跳过原因
└── ...
```

使用 `--compare` 且至少两条记录成功时，还会输出 RMS、平均 PSD、相对频段功率及公共通道 FC 比较。相关表包括 `channel_overlap.csv` 与 `psd_js_divergence.csv`；有多个数据集组时还生成通道 Jaccard 相似度和 PSD Jensen-Shannon 散度热图。

跨记录 FC 仅比较所有记录中同名通道的交集，少于两个公共通道时生成 `compare_common_fc_SKIPPED.txt`。跨数据集摘要按文件等权汇总，没有把每个窗口当成独立受试者；比较结果仍是描述性的，不是组间显著性结论。跨记录 FC 比较采用默认逐窗平均模式，不继承单记录图的 `--fc-mode`。

单个图不支持时，代码会记录 `report.json` 的 `plot_errors` 并继续生成其他图；这类跳过不一定使程序返回非零。文件级失败记录在 `failed_files.json`，命令行完成其余记录后以退出码 1 结束。应同时检查两种记录，不能只看退出码判断所有图均已成功。

重复使用同一输出目录可能覆盖同名图片，旧图也不会自动清理。比较不同参数时建议使用新的输出目录。

## Python 与 Notebook 调用

```python
from pathlib import Path
from eeg_visualization import visualize_file, make_report_index

out = Path("reports/python_api")
result = visualize_file(
    "examples/demo_npz/synthetic_TDBRAIN.npz",
    out,
    plots=("traces", "psd", "fc_heatmap"),
    max_windows=12,
    mode="mean",
    formats=("png", "svg"),
    dpi=180,
)
make_report_index(out, [{
    "record": result["record"],
    "figures": [p.relative_to(out) for p in result["figures"]],
}])
print(out / "index.html")
print(result["errors"])
```

`visualize_file` 自身保存图和 `report.json`；HTML 索引由 `make_report_index` 创建。直接在 Notebook 中显示单图：

```python
import matplotlib.pyplot as plt
from eeg_visualization import read_npz, plot_record

record = read_npz("examples/demo_npz/synthetic_TDBRAIN.npz", max_windows=12)
figure = plot_record(record, "psd")
plt.show()
plt.close(figure)
```

模块作为命令行运行时使用无界面的 Agg 后端；Notebook 导入时保留当前后端。直接调用 `plot_record` 时，由调用者负责显示、保存和关闭 Figure。

使用 `EEGTools` 时，`visualize_eeg` 只暴露 `source`、`plots` 和 `max_windows`，默认最多 32 个窗口；若需要 SVG、单窗 FC 或边数设置，请使用本模块命令行或直接 Python API。

兼容入口 `FC_graph.py` 和 `FC_Matrix.py` 分别预设 `fc_circle` 和 `fc_heatmap`，其余命令行参数与本模块一致。可交互示例见 [EEG_visualization_gallery.ipynb](../EEG_visualization_gallery.ipynb)。
