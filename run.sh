#!/usr/bin/env bash
# ============================================================================
# UNITES (US) 启动脚本。COMMITTED（已入库，无密钥，可直接 git pull/clone）。
# 一套代码两个实验，用 EXP 选：  EXP=dag（默认） | EXP=node-chain
#
# 师兄用法：
#   1) clone 到【共享 NFS】（不是 $HOME，计算节点看不到 /home）：
#        git clone -b run_exp https://github.com/LynnixZ/taskbench-sft-comparison.git \
#          /playpen-shared/<你的共享目录>/taskbench-sft-comparison
#        cd /playpen-shared/<你的共享目录>/taskbench-sft-comparison
#   2) 在你自己的 shell export 密钥（别写进这个文件）：
#        export HF_TOKEN=hf_xxx          # gated 模型用；token 须在 HF 接受过对应许可
#        export WANDB_API_KEY=wandb_xxx  # 可选
#   3) 改下面【要改①】SHARED、【要改②】GPU 数；选实验（默认 dag）：
#        bash run.sh                       # DAG 实验
#        EXP=node-chain bash run.sh        # node+chain 实验
#   以后更新只需 `git pull`。
# ============================================================================
set -e

# ======================【 要改①：共享目录名 SHARED 】======================
#   填你 /playpen-shared/<这个> 的真实目录名（可能 ≠ 登录名，如目录 xinyu vs 登录 xinyuzh）。
export SHARED="${SHARED:-xinyu}"                                   # <-- 改这里
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # 本 repo 位置（自动，须在 /playpen-shared）
export WORK_DIR="${WORK_DIR:-/playpen-shared/$SHARED/tb_work}"     # venv + 数据 + HF 缓存（共享 NFS）
export HF_HOME="$WORK_DIR/hf_home"
# ======================【 要改②：申请几张 GPU 】======================
#   分区自动投 a100,ada,a6000（cu121 能跑的，谁空用谁；不会用 blackwell）。
export GPUS_PER_JOB="${GPUS_PER_JOB:-4}"                           # 几张卡（≤8）；也可临时覆盖：GPUS_PER_JOB=2 bash run.sh
case "$REPO_DIR" in /playpen-shared/*) : ;; *) echo "⚠️ REPO_DIR=$REPO_DIR 不在 /playpen-shared，计算节点看不到 -> 请 clone 到共享盘";; esac

# ---- 选实验：EXP=dag（默认）| node-chain ----
EXP="${EXP:-dag}"
case "$EXP" in
  dag)
    EXP_CONFIG=configs/experiment_dag_fulljson.yaml
    EXP_MODES="full_json"                                          # DAG 无线性顺序，只 full_json
    EXP_MODELS="lmsys/vicuna-7b-v1.5 Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3" ;;
  node-chain|nodechain|nc)
    EXP_CONFIG=configs/experiment_models.yaml
    EXP_MODES="full_json trajectory"
    EXP_MODELS="Qwen/Qwen3-8B Qwen/Qwen2.5-1.5B-Instruct lmsys/vicuna-7b-v1.5 meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.2-3B-Instruct mistralai/Mistral-7B-Instruct-v0.3" ;;
  *) echo "EXP 必须是 dag 或 node-chain（当前: $EXP）"; exit 1 ;;
esac
echo "[run] EXP=$EXP  CONFIG=$EXP_CONFIG  MODES=$EXP_MODES"
[ -n "${HF_TOKEN:-}" ] || echo "WARN: HF_TOKEN 没设 -> gated 模型会被跳过；export HF_TOKEN=... 再跑可补上。"

# ---- PART 1: 登录节点联网准备（下环境 + 数据 + 模型）。tmux 里跑，别 Ctrl-C ----
source scripts/prep_env.sh
MODELS="$EXP_MODELS" bash scripts/prestage_all.sh
cat "$WORK_DIR/prestage_models_summary.txt"                        # 期望各模型 OK（权重已验证）

# ---- PART 2: 提交离线 Slurm 作业 ----
export CONFIG="$EXP_CONFIG" MODES="$EXP_MODES" MODELS="$EXP_MODELS"

bash scripts/submit_unites.sh                                     # 卡数见上面 GPUS_PER_JOB

squeue -u "$USER"
# tail -f "$WORK_DIR"/logs/grid-*.out
