#!/usr/bin/env bash
# ============================================================================
# UNITES (US) 启动脚本。COMMITTED（已入库，无密钥，可直接 git pull/clone）。
# 一套代码多个实验，用 EXP 选（默认 gnn4plan，对齐 GRAFT/GTool）：
#   gnn4plan（默认）| gnn4plan-dag | dag | node-chain
#
# 用法：
#   1) clone 到【共享 NFS】（不是 $HOME，计算节点看不到 /home）：
#        git clone -b run_exp https://github.com/LynnixZ/taskbench-sft-comparison.git \
#          /playpen-shared/<你的共享目录>/taskbench-sft-comparison
#        cd /playpen-shared/<你的共享目录>/taskbench-sft-comparison
#   2) 在你自己的 shell export 密钥（别写进这个文件）：
#        export HF_TOKEN=hf_xxx          # gated 模型用；token 须在 HF 接受过对应许可
#        export WANDB_API_KEY=wandb_xxx  # 可选
#   3) 改下面【要改①】SHARED、【要改②】GPU 数；选实验（默认 gnn4plan）：
#        bash run.sh                          # GNN4Plan 对齐（只链，固定 test 500，比 GRAFT/GTool）
#        EXP=gnn4plan-dag bash run.sh         # GNN4Plan 数据 + DAG 增强
#        EXP=dag / EXP=node-chain bash run.sh # 旧的分层 DAG / node+chain
#   ★ US Slurm 必带 VENV_PYTHON（登录节点 conda 建的 venv 在计算节点会失效）：
#        VENV_PYTHON=/usr/bin/python3 bash run.sh   # China 不用设；透传给 prestage 建 venv
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

# ---- 选实验：EXP=gnn4plan（默认，严格对齐 GRAFT）| gnn4plan-dag | dag | node-chain ----
EXP="${EXP:-gnn4plan}"
GNN4PLAN=0
case "$EXP" in
  gnn4plan|gnn)
    # 严格对齐 GNN4Plan/GRAFT/GTool：同数据 + 同 test 集（split_ids.json，只链）。默认。
    EXP_CONFIG=configs/experiment_gnn4plan.yaml
    EXP_MODES="${MODES:-full_json trajectory}"
    EXP_MODELS="${MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3 meta-llama/Llama-3.2-3B-Instruct lmsys/vicuna-7b-v1.5}"
    GNN4PLAN=1 ;;
  gnn4plan-dag|gnndag)
    # GNN4Plan 数据 + DAG 增强（node+chain+DAG 混训；chain test 仍是那 500 条）。
    EXP_CONFIG=configs/experiment_gnn4plan_dag.yaml
    EXP_MODES="${MODES:-full_json trajectory}"                      # DAG 只在 full_json 生效；trajectory 自动只用单+链
    EXP_MODELS="${MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3 meta-llama/Llama-3.2-3B-Instruct lmsys/vicuna-7b-v1.5}"
    GNN4PLAN=1 ;;
  dag)
    EXP_CONFIG=configs/experiment_dag_fulljson.yaml
    EXP_MODES="${MODES:-full_json}"                                # DAG 无线性顺序，只 full_json
    EXP_MODELS="${MODELS:-lmsys/vicuna-7b-v1.5 Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3}" ;;
  node-chain|nodechain|nc)
    EXP_CONFIG=configs/experiment_models.yaml
    EXP_MODES="${MODES:-full_json trajectory}"
    EXP_MODELS="${MODELS:-Qwen/Qwen3-8B Qwen/Qwen2.5-1.5B-Instruct lmsys/vicuna-7b-v1.5 meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.2-3B-Instruct mistralai/Mistral-7B-Instruct-v0.3}" ;;
  rule-sweep|rule)
    # rule-aware label smoothing 扫参：Qwen3-8B 一个模型、只 trajectory、扫 alpha。
    # 用 GNN4Plan 对齐 split -> EM 可与 GRAFT/GTool 比。alpha=0 = baseline(平滑关)。
    EXP_CONFIG=configs/experiment_gnn4plan.yaml
    EXP_MODES="trajectory"
    EXP_MODELS="${MODELS:-Qwen/Qwen3-8B}"
    export RULE_ALPHAS="${RULE_ALPHAS:-0 0.05 0.1 0.2}"
    GNN4PLAN=1 ;;
  *) echo "EXP 必须是 gnn4plan | gnn4plan-dag | dag | node-chain | rule-sweep（当前: $EXP）"; exit 1 ;;
esac
echo "[run] EXP=$EXP  CONFIG=$EXP_CONFIG  MODES=$EXP_MODES${RULE_ALPHAS:+  RULE_ALPHAS=[$RULE_ALPHAS]}"
[ -n "${HF_TOKEN:-}" ] || echo "WARN: HF_TOKEN 没设 -> gated 模型会被跳过；export HF_TOKEN=... 再跑可补上。"

# ---- PART 1: 登录节点联网准备（下环境 + 数据 + 模型）。tmux 里跑，别 Ctrl-C ----
source scripts/prep_env.sh
[ "$GNN4PLAN" = 1 ] && bash scripts/download_gnn4plan.sh data/gnn4plan   # vendor GNN4Plan 数据 + split_ids
MODELS="$EXP_MODELS" bash scripts/prestage_all.sh
cat "$WORK_DIR/prestage_models_summary.txt"                        # 期望各模型 OK（权重已验证）

# ---- PART 2: 提交离线 Slurm 作业 ----
export CONFIG="$EXP_CONFIG" MODES="$EXP_MODES" MODELS="$EXP_MODELS"

bash scripts/submit_unites.sh                                     # 卡数见上面 GPUS_PER_JOB

squeue -u "$USER"
# tail -f "$WORK_DIR"/logs/grid-*.out
