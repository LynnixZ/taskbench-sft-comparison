#!/usr/bin/env bash
# ============================================================================
# UNITES (US) 启动脚本 — DAG 实验。COMMITTED（已入库，无密钥，可直接 git pull/clone）。
#
# 师兄用法：
#   1) clone 到【共享 NFS】（不是 $HOME，计算节点看不到 /home）：
#        git clone -b run_exp https://github.com/LynnixZ/taskbench-sft-comparison.git \
#          /playpen-shared/<你的共享目录>/taskbench-sft-comparison
#        cd /playpen-shared/<你的共享目录>/taskbench-sft-comparison
#   2) 在你自己的 shell export 密钥（别写进这个文件）：
#        export HF_TOKEN=hf_xxx          # 下 gated 的 Mistral 用；token 须在 HF 接受过 Mistral 许可
#        export WANDB_API_KEY=wandb_xxx  # 可选
#   3) 改下面【要改①】SHARED、【要改②】GPU 数，然后：  bash run.sh
#   以后更新只需 `git pull`，不用再同步代码。
# ============================================================================
set -e

# ======================【 要改①：共享目录名 SHARED 】======================
#   填你 /playpen-shared/<这个> 的真实目录名（可能 ≠ 登录名，如目录 xinyu vs 登录 xinyuzh）。
export SHARED="${SHARED:-xinyu}"                                   # <-- 改这里
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # 本 repo 位置（自动，须在 /playpen-shared）
export WORK_DIR="${WORK_DIR:-/playpen-shared/$SHARED/tb_work}"     # venv + 数据 + HF 缓存（共享 NFS）
export HF_HOME="$WORK_DIR/hf_home"
case "$REPO_DIR" in /playpen-shared/*) : ;; *) echo "⚠️ REPO_DIR=$REPO_DIR 不在 /playpen-shared，计算节点看不到 -> 请 clone 到共享盘";; esac

DAG_MODELS="lmsys/vicuna-7b-v1.5 Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3"
[ -n "${HF_TOKEN:-}" ] || echo "WARN: HF_TOKEN 没设 -> Mistral(gated) 会被跳过；export HF_TOKEN=... 再跑可补上。"

# ---- PART 1: 登录节点联网准备（下环境 + 数据 + 模型）。tmux 里跑，别 Ctrl-C ----
source scripts/prep_env.sh
MODELS="$DAG_MODELS" bash scripts/prestage_all.sh
cat "$WORK_DIR/prestage_models_summary.txt"                        # 期望 3 个都 OK（权重已验证）

# ---- PART 2: 提交离线 Slurm 作业 ----
export CONFIG=configs/experiment_dag_fulljson.yaml MODES=full_json MODELS="$DAG_MODELS"
# ======================【 要改②：申请几张 GPU 】======================
#   分区自动投 a100,ada,a6000（cu121 能跑的，谁空用谁；不会用 blackwell）。
GPUS_PER_JOB=4 bash scripts/submit_unites.sh                       # <-- 改卡数（≤8）

squeue -u "$USER"
# tail -f "$WORK_DIR"/logs/grid-*.out
