# 可移植环境模版 (Portable env template)

把 `setup_env.sh` 拷到**任何 repo**,一条命令配好一个**干净、可复现、不踩 cu13 坑**的
Python 环境。它把这个项目反复踩出来的经验固化成一个自包含脚本。

## 它解决的坑（为什么这么写）

1. **隔离 venv,不碰 base** —— base/conda 的 torch 通常旧(2.1),`--system-site-packages`
   复用它会让 `pip install -r requirements.txt` 把 torch 升级成 **cu13** wheel,在
   CUDA 12.x 驱动上跑不了。模版默认建**隔离 venv**,自己装 cu121 torch。
2. **torch 钉版** —— 装 requirements 时用 `-c` 约束钉死 torch,不固定版本的
   transformers/trl 就不能偷偷升级它。
3. **网络自动选** —— AutoDL 学术加速 > 中国镜像 > 官方;HF/pip/torch 各走对的源。
4. **GPU 无关** —— 只装 wheel(无 GPU 也能跑 PART1);GPU 能不能用等真跑时验证。

## 用法

```bash
# 1) 拷贝
cp setup_env.sh /your/other/repo/

# 2) 配环境（PART 1，联网节点）
cd /your/other/repo
WORK_DIR=/data/yourproj REQUIREMENTS=requirements.txt bash setup_env.sh

# 3) 用它
source /data/yourproj/venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 想要 '...+cu121 True'
```

## 常用 knob

| 变量 | 默认 | 说明 |
|---|---|---|
| `WORK_DIR` | `./_env` | venv + 约束文件放哪 |
| `VENV_DIR` | `$WORK_DIR/venv` | venv 路径 |
| `REQUIREMENTS` | `requirements.txt` | 要装的依赖文件 |
| `REGION` | `auto` | `auto`/`china`/`china_turbo`/`us`,自动判断不准时手动指定 |
| `TORCH_INDEX_URL` | 按 region | 覆盖 torch wheel 源 |
| `VENV_SYSTEM_SITE` | 不设 | `=1` 才复用 base 包(不推荐) |

## 离线运行（PART 2，计算节点无网）

模版只管"配环境"。真正离线跑时,在激活 venv 后加:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
```

否则即使模型已缓存,HF 也会先联网检查更新而失败。

## 验收（环境配好的标志）

- `python -c "import torch; print(torch.cuda.is_available())"` 在 GPU 节点是 `True`;
- torch 版本带 `+cu121`(不是 `+cu13`);
- 没有 `Not uninstalling ... outside environment` 警告(那是 base 耦合的迹象)。
