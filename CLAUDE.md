# CLAUDE.md — 项目须知 & 踩过的坑（务必先读）

这些是反复踩出来的坑,**每次动手前先看,别再重复浪费时间**。详细运行步骤见
[RUNBOOK.md](RUNBOOK.md),实验设计见 [EXPERIMENT.md](EXPERIMENT.md)。

---

## 🔴 头号坑:torch / CUDA / venv（所有混乱的根源）

1. **venv 默认隔离,自己装 torch,别碰 base。**
   `prestage_all.sh` 默认建**隔离 venv**(无 `--system-site-packages`)。
   不要为了"复用 base torch"去开 `--system-site-packages` —— base torch 通常是旧的
   (2.1),会让 `pip install -r requirements.txt` 把 torch **升级成 cu13 wheel**。
   想复用 base 才设 `VENV_SYSTEM_SITE=1`(一般别设)。

2. **cu13 torch 在 CUDA 12.x 驱动上跑不了**(`torch.cuda.is_available()` 返回 False)。
   装完**永远要验证**:`python -c "import torch;print(torch.__version__, torch.cuda.is_available())"`
   必须是 `...+cu121 True`。

3. **China 的 torch 走 SJTU cu121 镜像**(`mirror.sjtu.edu.cn/pytorch-wheels/cu121`):
   - ❌ 别用清华默认 PyPI 的 `torch`(给最新版 = cu13);
   - ❌ 别用官方 `download.pytorch.org`(中国 ~3MB/s);
   - ✅ `setup_china.sh` 里 `TORCH_INDEX_URL` 已指向 SJTU cu121。

4. **装 requirements 时把 torch 钉死**(`-c` constraint),否则不固定版本的
   transformers/trl 会把 torch 偷偷升级成 cu13。`prestage_all.sh` 已经这么做了。

5. **装 CUDA wheel 不需要 GPU**,只有运行才需要。PART1(准备节点)可能无 GPU —— 它只装
   wheel + 下模型/数据,GPU 能不能用留到 PART2 验证。`prestage` 里用 `torch.version.cuda`
   (是不是 CUDA build)判断,**不要**用 `torch.cuda.is_available()`(无 GPU 会误判)。

6. venv 路径 = **`$WORK_DIR/taskbench_venv`**(不是 `venv`)。坏了就 `rm -rf` 重建。

---

## 🟠 两段式流程:联网准备 → 离线运行

- **PART 1(联网)**:`prep_env*.sh` + `prestage_all.sh` → 把 **env + 数据 + 模型**下到
  共享/数据盘。
- **PART 2(离线)**:`job_env.sh` + `run_grid.sh` → 只读缓存跑。
- **离线 ≠ 自动跳过联网。** 代码会真的尝试连网然后失败/卡住。HF 默认连用缓存前都要先联网
  检查更新,所以离线节点**必须设** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`,否则即使模型
  已缓存也会 `model not accessible` / FATAL。
- W&B 离线:`WANDB_MODE=offline`,事后 `wandb sync`。
- **新开 shell 跑 PART2 一定要重设 `WORK_DIR`/`HF_HOME`**(否则退回默认路径,找不到缓存)。
  命令别两行黏成一行(`export HF_HOME=...source ...` 会把路径拼坏)。

---

## 🟠 China 节点（AutoDL，`/root/autodl-tmp`）

- **政策:能走 AutoDL 学术加速的一律默认走。** `source scripts/setup_china.sh` 自动选:
  - **有 `/etc/network_turbo`** → source 它,用**官方 huggingface.co + git 走代理**;
  - **没有** → 退回 `HF_ENDPOINT=hf-mirror.com`。
- 🔴 **无论哪条,都必须 `HF_HUB_DISABLE_XET=1`(关 Xet)。** Xet 的下载器(hf-xet)**不走
  `http_proxy`**,Xet 一开,大权重就**绕过代理**直连美国 → ~3 MB/s(反复踩的坑)。关掉
  Xet,权重才走普通 HTTP 穿过 turbo 代理 → 快;配 `HF_HUB_ENABLE_HF_TRANSFER=1` 并行。
- `pip`(清华)、`torch`(SJTU cu121)镜像**始终用**(学术加速不覆盖 PyPI/pytorch.org)。
- `WORK_DIR=/root/autodl-tmp/tb_work`。`hf_transfer is deprecated` 警告无害。

---

## 🟠 US Slurm 集群（UNITES）

- **登录节点 unites1 有网可 SSH;计算节点 unites2-9 不能 SSH、离线。**
- **一切(repo、venv、数据、HF 缓存)必须在共享 NFS `/playpen-shared/`**,
  **不能放 `$HOME`** —— `/home` 是登录节点本地盘,计算节点看不到,`cd $HOME/...` 直接失败。
- ⚠️ **共享目录名可能 ≠ `$USER`**(实测:目录 `xinyu` 但 `$USER=xinyuzh`)→ 别盲目用
  `$USER` / `%u`,核对真实目录,改 `WORK_DIR`、sbatch 的 `cd` 与 `--output/--error`。
- **用 `sbatch` 提交,不是 `bash`**(`#SBATCH` 只有 sbatch 读;bash 会在登录节点本地跑)。
- 分区选 **`a100`**(Ampere,cu121 OK);**别用 `blackwell`**(新架构,cu121 torch
  "no kernel image")。
- 整节点内存用 **`--mem=0`**;**别写 `--mem=480G`**(超 a100/ada 物理上限 ~472GiB,会被拒)。
- 每人 **8 GPU 上限**。

---

## 🟡 密钥

- `HF_TOKEN` / `WANDB_API_KEY` 只放 **gitignored** 文件(`run.sh`、`run_ch_test*.sh`、
  `~/env.sh`),**绝不写进会提交的脚本**。离线 job **不需要**密钥(不认证)。
- gated 模型(Llama-2/3.2、Mistral)下载才需要 `HF_TOKEN` + 网页接受许可。

## 🟡 入口命令文件（gitignored，复制到节点用）

- `run.sh`(US 正式)、`run_ch_test.sh`(中国 node+chain 烟测)、`run_ch_test_dag.sh`
  (中国 DAG 烟测)是**给人复制到节点跑的命令文件**,含内联密钥,gitignored。
- **改了 `scripts/` 下的 setup/命令(setup_china、prep_env、job_env、prestage、
  submit_unites…),若入口命令需要跟着变,必须同步更新这三个文件。** 它们不走 git,
  得手动保持最新。
- 每个文件**绑定一个分支**(`git reset --hard origin/<branch>`):run.sh/run_ch_test.sh →
  `main`;run_ch_test_dag.sh → `exp-dag-fulljson`。**改某分支的脚本时,确认对应入口文件
  指向的分支上也有这些改动**(否则会拉到旧脚本,重演 cu13 等坑)。

---

## 🟡 分支 & 实验

- **`main`** = node+chain 实验,4 设置(Base/SFT × Full-JSON/Trajectory),6 模型 × 3 域。
- **`exp-dag-fulljson`** = 加入 DAG 的实验,**只 Full-JSON**(DAG 无线性顺序,Mode B 不适用),
  3 模型(vicuna-7b / Qwen3-8B / Mistral-7B)。跑法:

  ```bash
  CONFIG=configs/experiment_dag_fulljson.yaml MODES=full_json MODELS="..." bash scripts/run_grid.sh
  ```

  - DAG 通过 `include_topologies: [single, chain, dag]` 开启(`annotate_sample(include_dag=...)`)。
  - DAG 单独分桶(`dag`)、单独写 `test_dag.jsonl`;评估按 topology 分组天然给出
    node/chain/dag + overall。
- `run_ch_test_dag.sh`(gitignored)= DAG 的中国烟测,会自动 checkout `exp-dag-fulljson`。

---

## 🟡 run_grid.sh 的环境开关

`CONFIG` `MODES` `MODELS` `DOMAINS` `GPUS` `DELETE_MODELS` `TEST_SPLIT` `MAX_STEPS`
`INFER_LIMIT` `MAX_CACHED`。

- **烟测专用**:`MAX_STEPS`(训几步)、`INFER_LIMIT`(推理几条)—— **正式跑不要设**。
- 它会把 `--config` 透传给 split/train/infer,但**覆盖** `split.out_dir` 为
  `artifacts/splits/$domain`(每域隔离)。

---

## 🟡 CLI 用法

- `--config` 是**全局参数,放在子命令前**:`python -m taskbench_sft.cli --config X split`
  (不是 `split --config X`)。生成划分的命令是 **`split`**(不是 `make-split`)。
- `--set key.path=value` 可临时覆盖任意 config 项。

---

## 🟢 无害警告（别浪费时间排查）

- `OMP_NUM_THREADS Invalid value`(容器设了非法值)→ `export OMP_NUM_THREADS=1` 消除。
- `torch_dtype is deprecated`、`torchvision image.so / libjpeg` → 第三方库提示,文本任务无关。
- `Not uninstalling ... outside environment`(仅 `VENV_SYSTEM_SITE=1` 时)→ 影子,无害。

---

## 🟢 数据 & 评估要点

- **复用官方 TaskBench schema 与评估逻辑**,不重写、不改 gold;解析不了的脏样本如实剔除。
- main:只 node+chain(DAG 排除);branch:加 DAG。off-catalog gold 剔除。
- 80/10/10 分层(domain × topology × chain_length),`seed=42`,跨模型共享样本 ID。
- 推理确定性解码(temperature=0,无约束/无修复)。
