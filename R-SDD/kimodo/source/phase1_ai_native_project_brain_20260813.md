# Kimodo-SOMA-RP Phase 1 AI Native 项目大脑实践说明

## 1. 项目背景与目标

本项目 Phase 1 的目标是：在没有 Kimodo 官方后训练代码、也没有商业 bones-rigplay 700h 光捕训练数据的前提下，基于开源 `Kimodo-SOMA-RP-v1.1` 权重和开源 SEED 数据集，探索并验证一条可复现的后训练方法。

需要验证的问题不是简单“跑一次训练”，而是：

1. 能否建立完整训练管线：数据制备、feature cache、replay 混合、finetune、checkpoint 产出。
2. 能否建立有效评估管线：目标域专项 text2motion 评估、Content/Repetition 双 split、TMR/FID/foot-skate 等指标。
3. 能否找到一个细分动作域，使微调 ckpt 相比原始 `Kimodo-SOMA-RP-v1.1` 在目标域上有可解释提升。
4. 能否进一步通过 SEED 全测试 text2motion 保持集验证“目标域增强但整体能力不明显下降”。

最终 Phase 1 产出面向后续 Phase 2：等游戏动作数据准备好后，可以复用同一套流程做游戏动作域微调。

## 2. 项目大脑的角色定位

本次 AI 不是只承担“查资料”或“写文档”的辅助角色，而是作为项目大脑参与完整项目推进：

| 职责 | 实际承担内容 |
| --- | --- |
| 目标拆解 | 将 Phase 1 拆成训练管线验证、目标域增强验证、全域保持验证三层问题 |
| 技术方案设计 | 选择 SEED 目标域、设计 target + replay 的 domain finetune 流程 |
| 工程实现 | 补齐 manifest、cache、combine、train、eval、report 等脚本 |
| 远端执行 | 连接 lintrain，使用 H100 跑训练、评估和监控 |
| 结果判断 | 对每个子集和每个超参实验判断是否形成稳定收益 |
| 复盘沉淀 | 生成实验报告、workflow 纪要、保持评估计划和可复用脚本 |

## 3. 项目认知恢复

项目大脑首先建立了可恢复的项目认知，而不是依赖临时上下文。

| 项目认知项 | 当前状态 |
| --- | --- |
| 项目目标 | Phase 1 验证开源权重 + 开源数据上的后训练方法，目标域增强且整体能力不掉 |
| 成功标准 | 至少一个 SEED 细分子集在目标域 text2motion 指标上有可解释提升，并补 SEED 全测试保持评估 |
| 边界 | Kimodo 官方后训练代码未开源；商业 bones-rigplay 数据不可用；游戏动作数据尚未准备 |
| 远端环境 | lintrain，2 x H100，训练根目录 `/home/share/user/zyc/kimodo_seed_repro_train`，Kimodo 根目录 `/home/share/user/zyc/kimodo` |
| 当前候选 | `abl_object_lr1e7_s3000_20260811` |
| 当前未闭环项 | SEED 全测试 text2motion 保持评估仍在跑 |

关键产物：

```text
work/codex_pipeline/phase1_current_summary_20260813.md
work/codex_pipeline/phase1_workflow_20260813.md
work/codex_pipeline/phase1_ai_native_project_brain_20260813.md
```

## 4. AI 参与组织运行

项目推进过程中，AI 参与了任务拆解、执行、监控和结果判断。

### 4.1 任务拆解

Phase 1 被拆成以下任务链：

1. 远端环境连通：SSH key、HostKeyAlias、端口、batch mode。
2. 代码适配：补齐 Kimodo 后训练所需脚本。
3. 数据制备：从 SEED metadata 中构造目标域训练 manifest。
4. 训练缓存：生成 target cache、replay cache、combined cache。
5. 主实验训练：G+C、Dancing、Object 三个子集跑 4000-step 主实验。
6. 专项评估：为每个目标域构造 Content/Repetition text2motion benchmark。
7. 消融实验：对 Dancing/Object 复用 cache，跑 `lr=1e-7` 的 2000/3000-step 消融。
8. 报告整理：输出论文式实验报告和 workflow 纪要。
9. 保持评估：启动 SEED 全测试 text2motion benchmark，比较 baseline 与最佳候选。

### 4.2 远端执行与监控

AI 直接操作 lintrain 执行训练和评估，并做周期性监控：

| 阶段 | 监控内容 |
| --- | --- |
| benchmark 构造 | 检查 `meta.json`、`seed_motion.json`、manifest、split 数量 |
| 训练 | 检查 `train.py` 进程、loss、NaN、OOM、checkpoint |
| 评估 | 检查 `generate_eval.py` / `generate_eval_checkpoint.py` / `embed_folder.py` / `evaluate_folder.py` / `parse_folder.py` |
| GPU | 检查 H100 显存、利用率、是否干扰 GPU1 |
| 日志 | 检查 traceback、OOM、No space、No such file、proxy、nan、failed |
| 产物 | 检查 `summary_rows.json`、metrics、motion、embedding |

## 5. 过程和决策可追溯

项目推进中有几次关键决策被记录并修正。

| 决策点 | 初始做法 | 后续修正 | 原因 |
| --- | --- | --- | --- |
| 评估口径 | 早期使用 constraint/no-postprocess 小规模 suite 做 smoke test | 正式结论改为目标域 text2motion 专项评估 | 小规模 constraint suite 不能回答“目标域 text2motion 是否变好” |
| 子集选择 | 先参考 Gestures + Communication | 扩展到 G+C、Dancing、Object 三个子集 | 需要找可证明增强的细分动作域 |
| 训练强度 | 主实验 `lr=2e-7, steps=4000` | 增加 `lr=1e-7, steps=2000/3000` 消融 | 主实验有 foot-skate 副作用，需验证是否训练过强 |
| 最佳候选 | 初始关注 G+C 参考对象 | 当前选择 Object 3000-step | G+C/Dancing 收益不稳定，Object 指标更像正例 |
| 保持评估 | 早期误把小 suite 作为 full 参考 | 改为 SEED 全测试 text2motion 保持集 | “其它能力不掉”需要足够大、代表性的 text2motion 保持集 |

## 6. 根据结果持续调整

实验不是一次性跑完，而是根据结果持续调整。

### 6.1 G+C 主实验

Run：

```text
gc_safe_fp32_20260810_2045
```

观察：

- Content 上 T2M Similarity、FID、Contact Consistency、Foot Skate MaxVel 有提升。
- Repetition 上 R@1/R@3/R@5、Similarity、FID 多数下降。
- 结论：G+C 不是稳定正例，不能作为最终候选，只能说明训练和评估管线跑通。

### 6.2 Dancing 主实验与消融

Runs：

```text
cat_dancing_fp32_20260811_0005
abl_dancing_lr1e7_s2000_20260811
abl_dancing_lr1e7_s3000_20260811
```

观察：

- 4000-step 在部分 Repetition 指标上有收益。
- 2000/3000-step 没有稳定优于 baseline。
- foot-skate ratio/height 副作用明显。
- 结论：Dancing 暂不作为 Phase 1 正例。

### 6.3 Object Manipulation 主实验与消融

Runs：

```text
cat_object_manip_fp32_20260811_0005
abl_object_lr1e7_s2000_20260811
abl_object_lr1e7_s3000_20260811
```

观察：

- Object 3000-step 在 Content 检索、Similarity、FID、Contact 上较稳定。
- Repetition 的 R@2/R@3/R@5 和 FID 提升明显。
- foot-skate 类指标仍有退化。
- 结论：当前最佳候选是 `abl_object_lr1e7_s3000_20260811`。

最佳超参：

| 参数 | 值 |
| --- | --- |
| target domain | Object Manipulation |
| base model | `Kimodo-SOMA-RP-v1.1` |
| dtype | fp32 |
| lr | 1e-7 |
| steps | 3000 |
| warmup | 200 |
| batch | 4 |
| grad_accum | 8 |
| target repeat | 2 |
| replay size | 4000 |
| replay repeat | 1 |
| text_dropout | 0.10 |
| max_grad_norm | 0.5 |

## 7. 形成工作闭环

当前已经形成了一个 Phase 1 的工作闭环：

```text
目标定义
  -> 数据筛选
  -> target/replay manifest
  -> feature cache
  -> cache merge
  -> finetune
  -> checkpoint
  -> 目标域 benchmark
  -> baseline/candidate 评估
  -> 指标汇总
  -> 消融决策
  -> 报告沉淀
  -> SEED 全测试保持评估
```

拿掉 AI 后会损失的部分：

1. 远端训练和评估的持续监控能力。
2. 多个 run、cache、benchmark、summary 的状态同步。
3. 根据指标持续调整实验设计的过程记录。
4. 可复用脚本和报告模板的沉淀。
5. 从目标域增强到全域保持评估的闭环判断。

仍需要人工接管的部分：

1. 最终是否接受 foot-skate 副作用。
2. Phase 2 游戏动作数据的准备和授权。
3. 业务上哪些动作类别最值得增强。
4. 是否将 GPU1 并行纳入长时间保持评估。

## 8. Skill 与可复用沉淀

本项目已经沉淀出可复用资产：

| 类型 | 路径 |
| --- | --- |
| 训练 manifest 构造 | `work/codex_pipeline/build_training_subset_manifest.py` |
| 目标域 benchmark 构造 | `work/codex_pipeline/build_domain_text_benchmark.py` |
| SEED 全测试 benchmark 构造 | `work/codex_pipeline/build_seed_text_test_benchmark.py` |
| 目标域训练流水线 | `work/codex_pipeline/run_category_domain_finetune.sh` |
| 目标域评估流水线 | `work/codex_pipeline/run_domain_text_eval.sh` |
| SEED 全测试保持评估 | `work/codex_pipeline/run_seed_full_text_holdout_eval.sh` |
| 指标汇总 | `work/codex_pipeline/summarize_domain_text_ablation.py` |
| 报告生成 | `work/codex_pipeline/generate_phase1_current_summary.py` |

这些沉淀可以直接迁移到 Phase 2：

```text
游戏动作数据
  -> 游戏动作 manifest
  -> target/replay cache
  -> Kimodo-SOMA-RP finetune
  -> 游戏动作目标域评估
  -> SEED/非目标保持评估
```

## 9. 当前项目状态

| 项目 | 状态 |
| --- | --- |
| 训练管线 | 已跑通 |
| 目标域专项评估 | 已完成 |
| 主实验 | G+C、Dancing、Object 已完成 |
| 消融实验 | Dancing/Object 的 2000/3000-step 已完成 |
| 当前最佳候选 | `abl_object_lr1e7_s3000_20260811` |
| 全域保持评估 | 正在跑 |
| 最终 Phase 1 结论 | 等 SEED 全测试保持评估完成后确认 |

当前正在执行的保持评估：

| 项目 | 设置 |
| --- | --- |
| benchmark | SEED test Content + Repetition text2motion |
| 当前规模 | Content 3510, Repetition 3428, Total 6938 |
| baseline | `Kimodo-SOMA-RP-v1.1` |
| candidate | `abl_object_lr1e7_s3000_20260811` |
| output | `/home/share/user/zyc/kimodo/eval_work/seed_text_test_full_eval_20260813` |

## 10. 对照评分项自评

| 评分项 | 分值 | 证据 | 自评 |
| --- | ---: | --- | ---: |
| 项目认知能够恢复 | 2 | 记录了目标、边界、远端环境、run、路径、当前候选和未闭环项 | 2 |
| AI 参与组织运行 | 3 | AI 参与任务拆解、远端执行、训练监控、评估推进、结果复盘，而不是只回答问题 | 3 |
| 过程和决定可追溯 | 2 | 记录了评估口径修正、子集选择、超参调整、候选选择和保持评估设计 | 2 |
| 根据结果持续调整 | 2 | 从 G+C/Dancing 转向 Object，基于结果增加 2000/3000-step 消融，并补 SEED 全测试保持评估 | 2 |
| 形成工作闭环 | 1 | 已形成从数据到训练、评估、报告、保持评估的闭环；当前仅等待保持评估最终结果 | 1 |
| 合计 | 10 | 仍需等待 SEED 全测试保持评估完成后形成最终验收结论 | 10 |

## 11. 限分风险说明

可能影响评分的风险点：

1. SEED 全测试保持评估尚未完成，因此“整体能力不掉”还不是最终结论。
2. 当前最佳 Object candidate 仍有 foot-skate 类指标副作用，需要业务判断是否可接受。
3. GPU1 暂未并行使用，长评估耗时较长；若需要更快闭环，应拆 baseline/candidate 或 Content/Repetition 到双卡并行。
4. 目前沉淀的是项目内脚本和报告，还没有整理成正式可复用的团队级 Skill 包。

## 12. 下一步

1. 等待 SEED 全测试保持评估完成。
2. 将全域保持结果补入 `phase1_current_summary_20260813.md`。
3. 若保持评估通过，冻结 `abl_object_lr1e7_s3000_20260811` 为 Phase 1 候选权重。
4. 若保持评估不通过，优先做 replay 比例、target repeat、foot-skate/contact regularization 的下一轮消融。
5. 将当前 pipeline 整理成 Phase 2 游戏动作数据可直接复用的标准流程。
