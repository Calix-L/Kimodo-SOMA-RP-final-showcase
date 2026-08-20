# Kimodo 推理与评测链路

本目录把项目中使用过的推理、结果校验、可视化数据组织和量化评测收敛为一条可复现链路。数据集差异通过输入清单和配置表达，不再为 `SEED`、`7951-games`、`748-games`、`75-testsplit` 分别维护一次性脚本。

## 1. 链路总览

```text
CSV / JSON / JSONL 元数据
        │
        ▼
prepare_benchmark.py
        │  meta.json + source_meta.json + manifest.json
        ▼
Kimodo benchmark layout
        │
        ▼
generate.py ── RP 原始权重或微调 checkpoint
        │
        ▼
overview/<sample_id>/motion.npz
        │
        ├── build_manifest.py ── 统一样本 ID、Prompt、帧数和输出路径
        ├── validate_outputs.py ── 检查漏样本、重复 ID、数组形状和 NaN/Inf
        │
        ├── evaluate_tmr_physics.py
        │      R@1、R@5、T2M Sim、Contact、Foot Skate Ratio/Height
        │
        └── evaluate_rte_bpe.py（需要逐条配对的 Game GT）
               RTE、RTE-XZ、RTE-Y、BPE、BPE-Upper、BPE-Lower
```

`run_pipeline.sh` 串联上述阶段。所有结果都写入用户指定的 `RUN_DIR`，源数据、原始 Kimodo 仓库和 checkpoint 只读。

## 2. 文件说明

| 文件 | 作用 |
|---|---|
| `scripts/prepare_benchmark.py` | 将通用元数据转换成官方 benchmark/overview 输入结构 |
| `scripts/generate.py` | 调用官方 `benchmark/generate_eval.py`；可选覆盖 denoiser checkpoint |
| `scripts/build_manifest.py` | 将 overview 输出整理成统一 `manifest.csv`/`manifest.jsonl` |
| `scripts/validate_outputs.py` | 在评测前逐条验证 `motion.npz` 完整性 |
| `scripts/evaluate_tmr_physics.py` | 计算语义检索与物理质量六项指标 |
| `scripts/evaluate_rte_bpe.py` | 将预测与 Game GT 精确配对，计算 RTE/BPE 六个分项 |
| `scripts/summarize_runs.py` | 将多个标准运行目录汇总为 Markdown/JSON 对比表 |
| `config.example.env` | 端到端运行配置模板 |
| `run_pipeline.sh` | 单模型、单数据集端到端入口 |

## 3. 环境要求

脚本应在能够正常运行原始 Kimodo 的环境中执行，并需要：

- Python、PyTorch、NumPy；
- 可导入的 Kimodo 源码目录；
- `Kimodo-SOMA-RP-v1.1` 与 `TMR-SOMA-RP-v1` 所需 checkpoint；
- 本地 LLM2Vec/TMR text encoder 文件；
- GPU 推理时可用的 CUDA 环境。

本目录不复制官方模型和第三方权重，避免把大文件或内部路径提交到仓库。

## 4. 准备 benchmark 输入

如果已有 Kimodo benchmark 目录并且根目录包含 `manifest.json`，可以跳过本节。

通用输入支持 CSV、JSON 列表或 JSONL。每条至少需要唯一 ID、英文 Prompt 和帧数，例如：

```csv
motion_id,text,num_frames,category
attack_001,A character swings a sword forward.,90,attack
hover_001,A character rises and hovers in the air.,150,hover
```

转换命令：

```bash
python inference/scripts/prepare_benchmark.py \
  --input /data/eval.csv \
  --output-dir /work/benchmark_input \
  --id-field motion_id \
  --text-field text \
  --frames-field num_frames \
  --category-field category \
  --fps 30 \
  --max-frames 300 \
  --diffusion-steps 100
```

超过 `max-frames` 的动作会被裁剪，并在 `source_meta.json` 和 `summary.json` 中记录。脚本拒绝覆盖已有 `manifest.json`，防止误改现有数据集。

## 5. 端到端运行

复制模板并只修改路径和实验参数：

```bash
cp inference/config.example.env /work/my_run.env
bash inference/run_pipeline.sh /work/my_run.env
```

### 原始 RP

设置：

```bash
MODEL_LABEL=rp_base
CHECKPOINT=
```

### 微调 checkpoint

设置：

```bash
MODEL_LABEL=gold_step3000
CHECKPOINT=/path/to/checkpoint_step_3000.pt
```

已知微调 checkpoint 必须包含 `denoiser_state_dict`。脚本使用 `strict=True` 加载，结构不一致时会直接停止，避免在只加载部分权重的情况下产出不可比较结果。

### 缓存文本特征

若评测集已经按训练流程缓存 LLM2Vec 特征，同时设置：

```bash
FEATURE_INDEX=/path/to/index.jsonl
TEXT_FEATURE_DIR=/path/to/text_features
```

两项留空时，使用 Kimodo 配置的本地 text encoder 逐条编码 Prompt。

## 6. 公平对比约定

模型对比建议固定：

- 同一份 `manifest.json` 和 Prompt；
- 同一帧率、帧数裁剪规则和随机种子；
- `diffusion_steps=100`；
- 同一 TMR evaluator 和本地 text encoder；
- 同一后处理开关；
- 同一 GT 配对规则。

项目的模型能力表默认使用关闭后处理的原始输出。Foot Locking/IK 会改变 Contact 和脚滑指标，因此开启后处理的结果应单独成表，不能和关闭后处理结果直接混排。

## 7. 指标解释

| 指标 | 方向 | 含义 |
|---|---:|---|
| R@1 / R@5 | ↑ | 在 TMR 联合空间中，正确动作能否排进文本检索结果前 1/5 名 |
| T2M Sim | ↑ | 配对文本与动作的 embedding 相似度 |
| Contact | ↑ | 模型输出的脚接触标签与高度/速度启发式接触判断的一致率 |
| Foot Skate Ratio | ↓ | 脚接近地面时发生明显滑动的区间比例 |
| Foot Skate Height | ↓ | 脚接近地面时脚趾的平均速度，实际单位为 m/s，并非几何高度 |
| RTE | ↓ | 公共世界坐标中根节点轨迹误差，即角色整体是否移动到正确位置 |
| RTE-XZ / RTE-Y | ↓ | RTE 的水平轨迹误差与垂直高度误差 |
| BPE | ↓ | 分别消除预测和 GT 的根位置、水平朝向后，关节相对姿态误差 |
| BPE-Upper / Lower | ↓ | 上半身与下半身的 BPE 分项 |

TMR 在 RP 数据域上训练，适合衡量原始能力保持和 SEED 泛化，但不能充分判断攻击、浮空、击倒等游戏动作是否还原。因此有逐条 Game GT 时，应同时报告 RTE/BPE；没有配对 GT 时，不能凭空计算这两项。

## 8. 输出目录

一次完整运行会生成：

```text
RUN_DIR/
├── generated/                         # 官方 overview 结构及 motion.npz
├── eval_input/
│   ├── manifest.csv                   # 评测统一入口
│   ├── manifest.jsonl
│   └── validation.json
├── metrics/
│   ├── tmr_physical/
│   │   ├── metrics_summary.json       # 六项汇总指标
│   │   ├── physical_per_sample.csv
│   │   ├── tmr_per_sample.csv
│   │   └── progress.jsonl
│   └── rte_bpe/
│       ├── summary.json
│       ├── <model>.csv                 # 逐样本 RTE/BPE
│       └── rte_bpe_comparison.md
└── logs/
```

用于可视化界面的目录是 `generated/**/text2motion/overview`，每个样本目录中的核心文件是 `motion.npz`。如果界面依赖 `meta.json` 或额外渲染资源，应保留整个 overview 目录，而不只复制 NPZ。

多模型完成后可生成统一表格：

```bash
python inference/scripts/summarize_runs.py \
  --run rp=/work/rp_run \
  --run step3000=/work/step3000_run \
  --dataset 75-testsplit \
  --output /work/comparison.md
```

## 9. 多 GPU 运行

当前方式是任务级并行：不同模型、数据 split 或 benchmark shard 各启动一个独立进程，并通过 `CUDA_VISIBLE_DEVICES` 分配 GPU；它不是把单条动作拆成 DDP。推荐每个进程使用独立 `RUN_DIR`，完成后分别构建 manifest 和评测，避免多个进程同时写入同一 overview 目录。

批量大小由显存决定。项目中使用过 `batch_size=256`，但新 GPU 或新模型应先用少量样本探测，确认没有 OOM 后再跑全量。

## 10. 常见问题

- **输出数量不对**：先运行 `validate_outputs.py`，不要直接计算平均指标。
- **RTE/BPE 缺样本**：预测 manifest 的 `key` 必须与 GT `index.jsonl` 的 `motion_id` 精确一致。
- **TMR 结果异常低**：检查 Prompt、文本特征和 motion manifest 是否逐条对齐。
- **Foot Skate Height 看起来不像高度**：官方类名为 `FootSkateFromHeight`，但返回的是近地脚趾速度，报告时必须带 `m/s`。
- **长动作被裁剪**：TMR 默认最多编码 300 帧；裁剪数量会写入 `tmr_summary.json`。
- **结果无法复现**：同时记录 checkpoint、Prompt、seed、帧数、扩散步数和后处理开关，仅记录模型名称不够。
