# 实验设置说明 (Experiment Specification)

本文件描述本仓库这一次实验的完整设置:研究问题、数据、目标格式、训练方法、
模型、推理与评估。所有数值均与 `configs/experiment_gnn4plan.yaml`(默认口径,GNN4Plan
对齐)及 `taskbench_sft/` 代码一致。

---

## 1. 研究问题 (Research question)

在**同一个基座模型、同一份数据、同一套划分、同一组超参数**下,比较两种 TaskBench
SFT **目标格式 (target format)** 对工具规划质量的影响:

- **Mode A — Full JSON**:完整 TaskBench 计划对象 `{task_steps, task_nodes, task_links}`
- **Mode B — Tool Trajectory**:执行顺序的工具 ID 列表 `["Tool_A", "Tool_B", ...]`

目标是**验证 "SFT 相比 Base 是否涨点"**(并横向比较两种格式),而**不是**调最优超参。
因此对所有模型使用**同一套稳定的 LoRA(r=16) recipe**。

---

## 2. 模型 (Models)

6 个 instruct 模型,均为 Llama 风格结构(故 LoRA `target_modules` 通用):

| # | 模型 | 规模 | gated |
|---|---|---|---|
| 1 | `Qwen/Qwen3-8B` | 8B | 否 |
| 2 | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 否 |
| 3 | `lmsys/vicuna-7b-v1.5` | 7B | 否 |
| 4 | `meta-llama/Llama-2-7b-chat-hf` | 7B | 是 |
| 5 | `meta-llama/Llama-3.2-3B-Instruct` | 3B | 是 |
| 6 | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | 是 |

gated 模型需 `HF_TOKEN` + 在 HuggingFace 接受许可,否则自动跳过。

---

## 3. 数据 (Data)

### 3.1 来源与忠实性
- 来源:官方 **TaskBench**(microsoft/JARVIS),3 个域:
  `data_huggingface`、`data_multimedia`、`data_dailylifeapis`。
- **复用官方数据结构与评估逻辑,不重写、不修改 gold schema**
  (`taskbench_sft/official/` 中的指标函数从官方 `evaluate.py` 逐行复制)。
- 解析不了的脏 gold **如实剔除**(如 link 不是 `{source,target}`、node 不是
  `{task,arguments}`),不强行修补。

### 3.2 拓扑范围:只用 Node + Chain(排除 DAG)
`taskbench_sft/data/topology.py` + `data/split.py` 强制:

| 类型 | 用 | 说明 |
|---|---|---|
| **Node** (`single`) | ✅ | 单工具 |
| **Chain** | ✅ | 必须是**简单连通路径** (simple connected path) |
| **DAG** | ❌ | 分叉/合并,`dag_excluded` |

另排除空链、断链 (`not_simple_connected_path`)、重名歧义 (`ambiguous_repeated_names`)
等无法忠实恢复 trajectory 的样本。

### 3.3 Off-catalog 剔除
`data/prepare.py` 剔除 gold 中用到"工具表之外"工具的样本
(`require_catalog_faithful_gold: true`),保证评估忠实。

### 3.4 划分 (Split)
- `data/split.py`,**80 / 10 / 10** train/val/test,`seed=42`。
- **分层 (stratified)**:按 `domain × topology × chain_length_bucket`
  (bucket = `node` / `chain_2` / `chain_3` / `chain_4_plus`)分桶,保证不同 plan
  长度在各 split 均匀分布。
- **跨模型共享样本 ID**(`artifacts/splits/`):所有模型/格式用**完全相同**的
  train/val/test,保证公平可比。
- **每个域独立**训练/测试(domain-major),不跨域混合。
- 测试集分开存:`test_node.jsonl`、`test_chain.jsonl`、`test_all.jsonl`。

### 3.5 Tokenization
- `max_seq_length: 3072`(每模型出 token 长度报告确认覆盖率);
- `drop_truncated_targets: true`(目标被截断的样本丢弃,避免学坏)。

---

## 4. 两种 SFT 目标格式 (`taskbench_sft/targets.py`)

两种 mode 用**相同的 canonical JSON 格式化**,prompt 里的 one-shot 示例与 assistant
目标完全一致。

**Mode A — Full JSON**(verbatim 取自 gold):
```json
{"task_steps": [...], "task_nodes": [{"task": "...", "arguments": [...]}, ...], "task_links": [{"source": "...", "target": "..."}, ...]}
```

**Mode B — Trajectory**(执行顺序工具 ID 列表):
```json
["Tool_A", "Tool_B"]
```

---

## 5. 训练方法 (Training — `configs/experiment_gnn4plan.yaml`)

一套**稳定优先**的 QLoRA recipe,对所有模型通用:

| 项 | 值 | 说明 |
|---|---|---|
| method | **qlora** (4-bit) | 单张 24GB+ GPU 即可 |
| epochs | **5**(上限) | 配 early stopping,通常更早停 |
| eval_strategy | **epoch** | 每 epoch 评估一次 |
| early_stopping_patience | **2** | val loss 连续 2 epoch 不降则停 |
| learning_rate | **2e-4** | LoRA 常用稳定值 |
| scheduler | cosine, warmup_ratio 0.03 | |
| optim | **paged_adamw_8bit** | 省显存且稳 |
| per_device_train_batch_size | 1 | |
| gradient_accumulation_steps | 16 | 有效 batch = 16 |
| gradient_checkpointing | true | |
| 精度 | **bf16**(非 fp16) | 避免溢出/NaN |
| max_grad_norm | 1.0 | 梯度裁剪,防爆炸 |
| weight_decay | 0.0 | LoRA 参数不衰减 |
| seed | 42 | |
| **LoRA** | r=**16**, alpha=**32**, dropout 0.05 | alpha = 2r |
| target_modules | q/k/v/o/gate/up/down_proj | Llama 风格通用 |

### Checkpoint 选择 (`eval/score.py`)
按 val 上的加权 `common_score` 选最优 checkpoint:
```
0.4·node_f1 + 0.3·edge_f1 + 0.2·sequence_exact_match + 0.1·parse_valid_rate
```

---

## 6. 推理 (Inference)

- **确定性解码**:`do_sample=false`、`temperature=0`、`num_beams=1`
  (**无约束解码 / 无输出修复** — 直接评模型原始输出);
- `full_json_max_new_tokens: 1024`,`trajectory_max_new_tokens: 256`。

---

## 7. 评估 (Evaluation — `taskbench_sft/eval/`)

复用官方指标函数(`official/evaluate_lib.py`,逐行复制官方 `evaluate.py`)。
按 `domain × topology × chain_length` 以及 `overall` **分组**报告。

主要指标:
- **节点 (工具集合)**:`node_f1`(官方 micro,no-matching)、`node_macro_f1`、
  `multiset_node_f1/precision/recall`(区分重复工具);
- **边 (依赖)**:`edge_f1`、`adjacent_edge_f1`;
- **序列**:`trajectory_exact_match`、`sequence_exact_match`、`ned`(归一化编辑距离);
- **参数**(Full JSON):`parameter_name_f1`、`parameter_value_f1`、`rougeL`、
  `task_step_rougeL`;
- **健康度**:`parse_valid_rate`、`schema_valid_rate`、`hallucinated_tool_rate`、
  `over_selection_rate` / `under_selection_rate`、`samples_with_hallucination_rate`。

---

## 8. 实验矩阵 (Experiment matrix)

**6 模型 × 3 域 × 4 设置 = 72 个 unit**,每个域单独训练/测试。

4 设置 = **{Base, SFT} × {Full-JSON, Trajectory}**:
- **Base**:不训练,直接用基座模型 + 同样 prompt 推理(对照);
- **SFT**:用对应格式 LoRA 微调后推理。

涨点结论 = 同 (模型, 域, 格式) 下 **SFT vs Base** 指标对比。

---

## 9. 复现性 (Reproducibility)

- 固定 `seed=42`(split + 训练);
- 所有模型/格式共享同一份 split(相同样本 ID);
- 同一套超参,只换 `model.name` 与目标格式;
- 离线运行(预先缓存模型/数据,`HF_HUB_OFFLINE=1`)结果不依赖网络。

---

## 10. 产出报告 (Outputs)

每个 unit 产出:
- 训练 checkpoint(选中的最优)与 token 长度报告;
- 推理预测 JSONL;
- 分组 metrics 报告(domain / topology / chain_length / overall);
- 汇总后可对比 Base vs SFT、Full-JSON vs Trajectory、不同 chain 长度的表现。

---

## 附:运行方式

- **美国正式(UNITES,Slurm)**:`run.sh` → PART 1 联网 `prep_env.sh` + `prestage_all.sh`
  (下全部模型到 `/playpen-shared`);PART 2 `sbatch scripts/job_unites.sbatch`
  (离线 `job_env.sh` + `run_grid.sh`)。
- **中国烟测(单节点)**:`run_ch_test.sh`,同样 PART1 联网 / PART2 离线,
  小模型 + `MAX_STEPS` / `INFER_LIMIT` 限速,~10 分钟验证管路。
