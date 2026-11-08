# 操作文档：魔搭上跑通 8B LoRA 微调

对应方案 `docs/模型微调方案.md` V1.7。按执行顺序编号，每一步都有「命令 / 预期输出 / 失败怎么办」。

---

## 0. 当前进度（先看这里）

| 环节 | 状态 |
|---|---|
| 环境（torch/llamafactory/bnb） | ✅ 已在 A10 24GB 实例装好，`bnb-status.txt` = ok |
| 模型权重 | ✅ 已在 `/mnt/workspace/models/R1-0528-Qwen3-8B`（含 tokenizer_class 修正） |
| 基线推理 | ✅ 已出 `output/baseline_v2.jsonl`（思考段无法关闭，约 19s/题） |
| 训练样本 | ⛔ 81 条候选全部 `pending`，**卡在人工审核** |
| 正式训练 | ⛔ 未开始（需 ≥20 条 approved，理想 ≥50 条） |

**唯一阻塞项**：审核样本。审核通过后本文件第 1 节 → 第 5 节即可一路跑通。

**下一批要做的事（按顺序）**

| # | 位置 | 动作 |
|---|---|---|
| 1 | 控制面板 | 停止 GPU 实例（关终端不等于停止） |
| 2 | 本机 | 用 `scripts/review_cli.py` 逐条审核 81 条 |
| 3 | 本机 | 导出 `data/out/` 三个文件 |
| 4 | 本机 → 实例 | 重新上传 `train/` 与 `data/out/`（实例上的脚本是旧版） |
| 5 | 实例 | 开机体检：版本 grep + tokenizer 往返 + bnb 状态 + 数据齐全 |
| 6 | 实例 | 正式训练 → 对照 → 合并导出 |

---

## 0.1 开工前确认

| 项 | 要求 |
|---|---|
| 实例 | PAI-DSW GPU 8 核 / 32GB 内存 / 24GB 显存（CUDA），镜像 `ubuntu22.04-cuda13.0.3-py312-torch2.13` |
| 备用 | AMD 192GB（ROCm 7.2.3），仅在 CUDA 两条路都不通时启用 |
| 额度 | 自设累计 8 小时；单次会话不超过 2 小时 |
| 单次上限 | 平台 10 小时 / 空闲 1 小时自动关机 |
| 持久化 | 只有 `/mnt/workspace`（100GB），其余路径关机即清空 |
| 网络 | 实例内访问不了 HuggingFace，模型只走 ModelScope |

**时间预算**

| 步骤 | 预计 | 位置 |
|---|---|---|
| 1 造数据与转换 | 10 分钟 | 本地 |
| 2 tokenizer 自检 | 10-15 分钟 | CPU 实例（不耗 GPU 额度） |
| 3 环境 + 下载 + 基线 | 30-40 分钟 | GPU 第 1 次开机 |
| 4 人工审核样本 | 你自己定 | 本地 |
| 5 训练 | 20-40 分钟 | GPU 第 2 次开机 |
| 6 对照 / 合并 / 导出 | 15-25 分钟 | GPU 第 2 次开机 |

---

## 1. 本地：审核样本并导出训练数据

`data/` 下现有 81 条候选样本（`samples_seed` 2 / `samples_seed.add9` 9 / `batch2` 26 /
`batch3` 24 / `batch4` 20），**全部 `pending`**。内容取自 D005（功能声称规范）、
D006（心脑血管手册）、D007、D008（普通食品）、D009（异议）、D010/D011（仪器）、
D013（退换货）、D014（急救）的真实素材，覆盖正常问答、高危禁忌、资料不足、目录外功能、
反例纠正、停药红线、角色边界七类。

**先看模型实际会看到什么（只看不导出）**

```bash
cd train
python scripts/convert_to_alpaca.py \
  --input data/samples_seed.jsonl data/samples_seed.add9.jsonl data/samples_batch2.jsonl \
  data/samples_batch3.jsonl data/samples_batch4.jsonl data/samples_batch5.jsonl data/samples_batch6.jsonl data/samples_batch7.jsonl data/samples_batch8.jsonl data/samples_batch9.jsonl data/samples_batch10.jsonl data/samples_batch11.jsonl \
  --preview 3
```

默认排布 `question_first`：`instruction` 是员工的原话问法，`input` 是随问题给出的
事实与参考资料（没资料时为空），对应 LLaMA-Factory 官方 alpaca 的用法。

> ⚠️ 与线上的差异：线上 `_build_prompt` 目前是「证据在前、问题在后」合成一条消息。
> 训练改成问题在前，**P5 接入时必须把 `_build_prompt` 的分区顺序也改成问题在前**，
> 否则训练与线上输入分布不一致。不适合改线上时，导出加 `--style production` 切回旧排布。

### 1.1 审核工具

`scripts/review_cli.py` 负责逐条审核，**不要手改 JSONL**（手改容易破坏字段顺序或漏填 `reviewer_id`）。

```bash
# ① 有多少、分布如何
python scripts/review_cli.py stats \
  --input data/samples_seed.jsonl data/samples_seed.add9.jsonl data/samples_batch2.jsonl \
  data/samples_batch3.jsonl data/samples_batch4.jsonl data/samples_batch5.jsonl data/samples_batch6.jsonl data/samples_batch7.jsonl data/samples_batch8.jsonl data/samples_batch9.jsonl data/samples_batch10.jsonl data/samples_batch11.jsonl

# ② 按风险层拉清单（redline 排最前）
python scripts/review_cli.py list \
  --input data/samples_seed.jsonl data/samples_seed.add9.jsonl data/samples_batch2.jsonl \
  data/samples_batch3.jsonl data/samples_batch4.jsonl data/samples_batch5.jsonl data/samples_batch6.jsonl data/samples_batch7.jsonl data/samples_batch8.jsonl data/samples_batch9.jsonl data/samples_batch10.jsonl data/samples_batch11.jsonl \
  --tier redline --status pending

# ③ 看某几条的完整内容（问题 / 资料 / 目标答案）
python scripts/review_cli.py show \
  --input data/samples_seed.jsonl data/samples_seed.add9.jsonl data/samples_batch2.jsonl \
  data/samples_batch3.jsonl data/samples_batch4.jsonl data/samples_batch5.jsonl data/samples_batch6.jsonl data/samples_batch7.jsonl data/samples_batch8.jsonl data/samples_batch9.jsonl data/samples_batch10.jsonl data/samples_batch11.jsonl \
  --id sft-0004 sft-0005

# ④ 批量标结论（原地写回源文件，其余字段原样保留）
python scripts/review_cli.py set \
  --input data/samples_seed.jsonl data/samples_seed.add9.jsonl data/samples_batch2.jsonl \
  data/samples_batch3.jsonl data/samples_batch4.jsonl data/samples_batch5.jsonl data/samples_batch6.jsonl data/samples_batch7.jsonl data/samples_batch8.jsonl data/samples_batch9.jsonl data/samples_batch10.jsonl data/samples_batch11.jsonl \
  --id sft-0004 sft-0005 --status approved --reviewer <你的工号或姓名>

# ⑤ 导出前体检：重复 ID / 空 reviewer_id / 非法状态
python scripts/review_cli.py check \
  --input data/samples_seed.jsonl data/samples_seed.add9.jsonl data/samples_batch2.jsonl \
  data/samples_batch3.jsonl data/samples_batch4.jsonl data/samples_batch5.jsonl data/samples_batch6.jsonl data/samples_batch7.jsonl data/samples_batch8.jsonl data/samples_batch9.jsonl data/samples_batch10.jsonl data/samples_batch11.jsonl
```

风险层由 `source_ids` 自动判定：

| 层 | 覆盖 | 审错的后果 |
|---|---|---|
| `redline` | D005 声称规范、D008 普通食品铁律、D013 退换货、D014 急救 | 合规违规 / 人身风险 |
| `device` | D010 体脂仪、D011 骨密度仪 | 越界解读、误诊引导 |
| `product` | D006 / D007 产品与目录 | 事实错误 |
| `dialog` | D009 异议处理 | 话术失当 |
| `meta` | 角色边界 | 越权 |

**审核标准**：事实与来源一致、虚构信息有标记、错误话术没被肯定、权限正确、答复简洁、无无依据补充。
`redline` 层只要有疑问就标 `rejected` 并单独记下来讨论，不要「差不多就放」。

### 1.2 导出

审核完再导出（脚本只收 `approved`，一条都没有会直接拒绝）：

```bash
python scripts/convert_to_alpaca.py \
  --input data/samples_seed.jsonl data/samples_seed.add9.jsonl data/samples_batch2.jsonl \
  data/samples_batch3.jsonl data/samples_batch4.jsonl data/samples_batch5.jsonl data/samples_batch6.jsonl data/samples_batch7.jsonl data/samples_batch8.jsonl data/samples_batch9.jsonl data/samples_batch10.jsonl data/samples_batch11.jsonl \
  --out-dir data/out
```

**预期输出**

```json
{
  "source_status_counts": { "approved": 4, "pending": 6 },
  "exported": { "train": 4, "val": 0 },
  "dataset_dir": "data/out"
}
```

**产物**

| 文件 | 说明 |
|---|---|
| `train.jsonl` / `val.jsonl` | 只有 `system / instruction / input / output` 四个官方字段 |
| `dataset_info.json` | 数据集注册，必须与数据同目录（脚本已保证） |
| `index.jsonl` | `sample_id` / `group_id` / 来源 / 提示哈希，**不进训练** |
| `manifest.json` | 来源 sha256、导出计数、分组清单 |

**规则**：没有一条 approved 就拒绝导出；同一 `group_id` 不能跨训练/验证集；
验证集为 0 条时会有警告，训练脚本会自动去掉 `eval_dataset`（否则无法挑检查点）。

**常见失败**：脚本会用中文报出缺字段、`sample_id` 重复、分组串侧等原因并退出。
**不要手改产物绕过检查**，去改主数据。

---

## 2. CPU 实例：tokenizer 自检（不占 GPU 额度）

CPU 环境长期可用，用它把模板和长度问题挡在 GPU 之外。

**上传**：把本地 `train/` 目录整体上传到 `/mnt/workspace/train`（同名，不重命名）。

**命令**

```bash
cd /mnt/workspace
python train/scripts/10_download.py \
  --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --tokenizer-only
# 记下末尾打印的「模型目录」，默认是 /mnt/workspace/models/R1-0528-Qwen3-8B
ls /mnt/workspace/models/R1-0528-Qwen3-8B/tokenizer_config.json

python train/scripts/30_tokenize_check.py \
  --model-path /mnt/workspace/models/R1-0528-Qwen3-8B \
  --dataset /mnt/workspace/data/train.jsonl \
  --max-length 2048
```

> `--model-path` 必须指到**含 `tokenizer_config.json` 的那一层**。指向缓存父目录或不存在的
> 路径时，`transformers` 会把它当成 Hub 仓库名去校验，报出的
> `HFValidationError: Repo id must be in the form ...` 与真实原因无关。

**预期输出**

```
样本数: 3
总长度: 中位 620 / 最大 720 / 最小 540
答案段: 中位 45 / 最大 60
超过 max_length=2048: 0 条 []
提示前缀不匹配(模板边界不可信): 0 条 []
答案段为空的样本: 0 条 []
```

**失败怎么办**

| 现象 | 含义 | 动作 |
|---|---|---|
| `tokenizer 没有 chat_template` | tokenizer 与模型版本不匹配 | 确认 `--model-id` 和 `--revision` 与后续 GPU 侧完全一致 |
| `提示渲染不是完整渲染的前缀` | 模板对 system/user 的处理与预期不同 | 停下，把渲染结果带回讨论。此时训练的标签遮罩不可信 |
| 超过 max_length 的样本 > 0 | 样本太长 | 缩短参考资料，或调 `cutoff_len`。不要静默截断答案 |

> CPU 与 GPU 实例的 `/mnt/workspace` **不共享**，第 3 步还要再传一次。

---

## 3. GPU 第 1 次开机：环境 + 权重 + 基线（≤2 小时）

### 3.1 先记时间

```bash
date -Iseconds | tee -a /mnt/workspace/logs/session.log
```

每完成一个小步就再记一次。到 **1.5 小时**强制保存并关机，不管做到哪。

### 3.2 传文件

把本地 `train/` 传到 `/mnt/workspace/train`，把 `data/out/` 里的
`train.jsonl`、`val.jsonl`、`dataset_info.json` 放到 `/mnt/workspace/data/`。

> ⚠️ 上传时**整体覆盖** `/mnt/workspace/train/scripts/`，不要只传单个文件。上一轮实例上
> 出现过新旧脚本混装（`infer_lib.py` 旧版 + `50_compare.py` 新版）导致
> `TypeError: run_question() got an unexpected keyword argument 'enable_thinking'`。
> 传完用第 5.1 节那组 grep 核对，**对不上就不要开训**。

### 3.3 装环境

```bash
bash /mnt/workspace/train/scripts/00_env.sh
```

**三种可能结果，对应三条路**

| 输出 | 含义 | 下一步 |
|---|---|---|
| `bitsandbytes 状态：ok` | 4bit 可用 | 按原计划走 `sft_qlora_cuda.yaml` |
| `bitsandbytes 状态：fail` | 4bit 不可用 | 训练改用 `sft_lora_cuda_bf16.yaml`，不在这上面继续折腾 |
| `dry-run 计划安装/替换 torch` 后退出 | 装框架会顶掉镜像 torch | `TORCH_MODE=fresh bash /mnt/workspace/train/scripts/00_env.sh` 重跑一次 |

**成功时的关键输出**：`llamafactory-cli` 的版本号、`torch ... cuda ...`、`device available True`。

**失败怎么办**：`TORCH_MODE=fresh` 也失败 → 本次会话到此为止，把
`logs/pip-install.txt` 尾巴和 `logs/pip-dryrun.txt` 带回讨论，**不要在实例上现场拼依赖**。

### 3.4 下载权重（16.3GB）

```bash
source /mnt/workspace/venv/bin/activate
python /mnt/workspace/train/scripts/10_download.py \
  --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --local-dir /mnt/workspace/models/R1-0528-Qwen3-8B
```

**预期输出**：一段 JSON，`gib` 约 16.3，`files` 约 10 个，`config_sha256` 有值。
这个指纹就是模型版本记录，留着。

### 3.4b 分词器自检（必做，否则基线与训练都是废的）

该权重的 `tokenizer_config.json` 自报 `tokenizer_class: LlamaTokenizerFast`。用它编码中文会得到**空数组**，模型吐乱码，而长度统计看起来"正常"——实测踩过一次，整轮基线作废。

```bash
cd /mnt/workspace/models/R1-0528-Qwen3-8B
cp tokenizer_config.json tokenizer_config.json.orig
python - <<'PY'
import json, pathlib
p = pathlib.Path("tokenizer_config.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("原:", d.get("tokenizer_class"))
d["tokenizer_class"] = "Qwen2TokenizerFast"
p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
PY
```

验证（必须 `往返: True`）：

```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/mnt/workspace/models/R1-0528-Qwen3-8B', trust_remote_code=True)
s = '康源牌鱼油软胶囊'
ids = tok(s, add_special_tokens=False).input_ids
print(tok.__class__.__name__, len(ids), tok.decode(ids) == s)
"
```

并留痕（方案要求记录模型版本证据，这里改过模型目录）：

```bash
echo "$(date -Iseconds) 修正 tokenizer_class: LlamaTokenizerFast -> Qwen2TokenizerFast（原文件备份 .orig）" \
  >> /mnt/workspace/logs/model-notes.txt
```

### 3.5 基线推理

```bash
python /mnt/workspace/train/scripts/20_baseline.py \
  --model-path /mnt/workspace/models/R1-0528-Qwen3-8B \
  --questions /mnt/workspace/train/data/baseline_questions.demo.jsonl \
  --out /mnt/workspace/output/baseline.jsonl
```

**预期输出**：每题一行进度，最后一段 meta JSON（GPU 名、torch 版本、平均首 token
时延、平均总时延、峰值显存）。

**失败怎么办**

| 现象 | 动作 |
|---|---|
| 加载时 OOM | 加 `--max-new-tokens 256` 重试；仍 OOM 说明推理都放不下，转 AMD |
| 每题都 `EMPTY_ANSWER` | 思考段切分标记不对，把 `output_raw` 的前 200 字带回来 |
| 生成很慢（>60s/题） | 正常，24GB 卡上 8B bf16 推理本就慢；记录数字即可 |

### 3.6 关机

产物在 `/mnt/workspace/output/` 和 `/mnt/workspace/logs/`，`sync` 后关机。
**第一次开机不训练。**

---

## 4. 人工审核样本（不占 GPU）

这 81 条是起点：逐条审、把通过的标 approved；口径稳定后按方案 4.3 扩到 100–200 条。审核改的是主数据的
`review_status`，不是转换后的 `train.jsonl`。**操作命令见第 1.1 节（`review_cli.py`）。**

审核至少看六项：事实与来源一致、虚构信息有标记、错误话术没被肯定、权限正确、
答复简洁、无无依据补充。

**题库的 140 道标准答案不能进训练集**，只能拿来出测试题。

---

## 5. GPU 第 2 次开机：训练

### 5.1 预检

实例上的 `train/` 是上一轮的旧版（`30_tokenize_check.py` / `40_train.sh` / `50_compare.py`
混了新旧版本），**必须重新上传整个 `train/` 目录**，再跑这一组核对：

```bash
cd /mnt/workspace/train
grep -c "往返失败" scripts/30_tokenize_check.py      # 期望 ≥1（有往返守卫）
grep -c "enable_thinking" scripts/infer_lib.py         # 期望 ≥1
grep -c "enable_thinking" scripts/50_compare.py        # 期望 ≥1
grep -c "ALLOW_SMALL" scripts/40_train.sh              # 期望 ≥1
grep -c "build_prompt_ids" scripts/infer_lib.py        # 期望 ≥1（新版渲染入口）
ls scripts/review_cli.py scripts/25_template_probe.py   # 应存在
```

任何一项对不上就是上传没成功，**不要继续往下跑**。

### 5.1b 训练/推理格式一致性（已实测确认）

LLaMA-Factory 训练用**它自己代码里的 `deepseekr1` 模板**，推理脚本原本用
**tokenizer 自带的 `chat_template`**。两者 user 段相同，assistant 前缀不同。

**2026-10-30 实测结论**

| 项 | 实测值 |
|---|---|
| `<think>` / `</think>` | **在词表内，各为单 token**（151667 / 151668） |
| `<｜begin▁of▁thinking｜>` | **不在词表**（`convert_tokens_to_ids` 返回 None） |
| `deepseekr1` + `enable_thinking=False` | prompt 尾部 `<｜Assistant｜><think>\n\n</think>\n\n`（**空思考块**，Qwen3 no-thinking 标准格式） |
| `deepseekr1` + `enable_thinking=True` | 把 `<think>` 塞进响应，不可用 |
| `deepseek3` | 只输出 `<｜Assistant｜>`，无思考块 |
| tokenizer 自带模板 | 只输出 `<｜Assistant｜>`，**缺空思考块** |

**结论**：训练侧 `template: deepseekr1` + `enable_thinking: false` **正确，配置不动**。
缺口在推理侧——`20_baseline.py` / `50_compare.py` 必须加 `--template deepseekr1`。

重跑探针复核（`messages` 的最后一个必须是 **assistant** 消息，system 走独立参数）：

```bash
source /mnt/workspace/venv/bin/activate
python /mnt/workspace/train/scripts/25_template_probe.py \
  --model-path /mnt/workspace/models/R1-0528-Qwen3-8B \
  --template deepseekr1
```

> 两条已踩过的坑：system 塞进 `messages` 会报 `NotImplementedError: Unexpected role: system`；
> 只传 user 消息时 `response` 会打印成 BOS+system，看起来像模型坏了。
> 注册表里**没有** `deepseekr1_nothink`，启动时那条 reasoning 模板告警可忽略。

### 5.1c 环境与数据体检

```bash
date -Iseconds | tee -a /mnt/workspace/logs/session.log
source /mnt/workspace/venv/bin/activate
cat /mnt/workspace/logs/bnb-status.txt                   # 期望 ok
python -c "
import json; d=json.load(open('/mnt/workspace/models/R1-0528-Qwen3-8B/tokenizer_config.json',encoding='utf-8'))
print('tokenizer_class =', d.get('tokenizer_class'))"    # 期望 Qwen2TokenizerFast
wc -l /mnt/workspace/data/train.jsonl /mnt/workspace/data/val.jsonl
```

`train.jsonl` 行数 < 20 时，只有显式加 `ALLOW_SMALL=1` 才会训练——那种结果只能算链路验证，不是实验组。

### 5.2 启动

```bash
# 主路线：4bit QLoRA
CONFIG=/mnt/workspace/train/configs/sft_qlora_cuda.yaml \
RUN=r1-8b-qlora-r8-lr2e-5 \
bash /mnt/workspace/train/scripts/40_train.sh

# bnb 不可用时
CONFIG=/mnt/workspace/train/configs/sft_lora_cuda_bf16.yaml \
RUN=r1-8b-bf16-r8-lr2e-5 \
bash /mnt/workspace/train/scripts/40_train.sh
```

> 训练配置里的 `template:` 与 `enable_thinking:` 必须与 5.1b 的探针结论一致。
> 两者只改一个，训练与推理就对不上。

**预期输出**

```
训练样本 120 条
可用磁盘 80GB
训练已在后台启动，pid=12345，日志：/mnt/workspace/output/<RUN>/train.log
```

跟日志：`tail -f /mnt/workspace/output/<RUN>/train.log`

**日志里必须出现的三行**

1. `Using template: deepseekr1` —— 模板名不对，后面全白做
2. LoRA 可训练参数量（百万级，占比很小）
3. 每 `logging_steps` 一条 loss

### 5.3 训练中怎么判断

| 现象 | 含义 | 动作 |
|---|---|---|
| loss 前几步很高之后稳定下降 | 正常 | 继续 |
| loss 降到接近 0 且验证 loss 回升 | 过拟合 | 停，取中间检查点 |
| loss 不降（一直平） | 学习率太小或数据格式没进 loss | 先查标签遮罩，再谈调参 |
| 显存 OOM | 24GB 放不下 | 降 `cutoff_len` 到 1024（BF16 配置已默认） |
| 进程被杀 | 实例空闲关机或超额 | 查会话时间，恢复训练 |

### 5.4 收尾

训练结束后把 `output/<RUN>/` 与 `logs/` 打包下载到本机 `训练/output/<RUN>/`。
记录：数据清单 sha256（在 `data/out/manifest.json`）、基础模型 `config_sha256`、
配置、随机种子、耗时、峰值显存。

---

## 6. 对照、合并、导出（同一次会话内做完）

### 6.1 同题对照

```bash
python /mnt/workspace/train/scripts/50_compare.py \
  --model-path /mnt/workspace/models/R1-0528-Qwen3-8B \
  --adapter /mnt/workspace/output/<RUN> \
  --questions /mnt/workspace/train/data/baseline_questions.demo.jsonl \
  --baseline /mnt/workspace/output/baseline_v2.jsonl \
  --template deepseekr1 \
  --out /mnt/workspace/output/adapter_<RUN>.jsonl
```

> **基线必须用同一套渲染重跑过。** `baseline_v2.jsonl` 是旧版脚本用 tokenizer
> 模板生成的，没有 `render_source` 字段——脚本会打印警告，此时对照不成立。
> 正确做法：先按 5.1b 定下模板，再重跑基线，然后 Adapter 用**同样的 `--template`** 对照。
>
> ```bash
> python /mnt/workspace/train/scripts/20_baseline.py \
>   --model-path /mnt/workspace/models/R1-0528-Qwen3-8B \
>   --questions /mnt/workspace/train/data/baseline_questions.demo.jsonl \
>   --template deepseekr1 \
>   --tag base-lf-template \
>   --out /mnt/workspace/output/baseline_lf.jsonl
> ```
>
> `--out` 已存在时脚本会直接退出，避免覆盖上一轮结果。


**预期输出**：逐题打印「原模型 vs Adapter」两段答案与 token 数。

判读标准：Adapter 的回答**更短、更贴资料、缺资料时会说没有**，才算方向对。
只变长、爱重复、格式带出模板符号，都是模板或数据有问题。

### 6.2 合并（可选但推荐）

```bash
sed "s|__ADAPTER__|/mnt/workspace/output/<RUN>|" \
  /mnt/workspace/train/configs/merge_lora.yaml > /mnt/workspace/output/merge.yaml
llamafactory-cli export /mnt/workspace/output/merge.yaml
```

**注意**：合并必须用**未量化**的基础模型。24GB 卡上 `export_device: cpu` 已设好，
靠内存而不是显存做合并。合并失败不影响 Adapter 本身可用。

### 6.3 导出与归档

下载 `output/` 与 `logs/`，按方案 8.1 分别记录 sha256。**不推送到模型 Hub。**

---

## 7. 止损与降级阶梯

```
① 4bit QLoRA（sft_qlora_cuda.yaml）
      ↓ logs/bnb-status.txt = fail
② BF16 LoRA 挤 24GB（sft_lora_cuda_bf16.yaml，cutoff_len 1024）
      ↓ 仍 OOM 或框架根本装不上
③ AMD 192GB（sft_lora_amd.yaml，不量化）
```

| 触发条件 | 动作 |
|---|---|
| 30 分钟内没完成「装上框架 → 8B 加载 → 一次前向」 | 逐级降级，不硬耗 |
| 单次会话到 1.5 小时 | 保存、`sync`、关机 |
| 累计到 8 小时 | 停止一切 GPU 操作，先复盘再申请 |

**只在结束后才能结束实例**：关掉训练进程不等于实例停止计时，必须在控制台点停止。

---

## 8. 还没在实例上验证过的点

1. `template: deepseekr1` 的 assistant 段是否自动插入思考起始标记——该权重的
   `chat_template` **没有 thinking 分支**，`enable_thinking: false` 只在 LF 侧生效，
   推理端仍会输出思考段（实测约 500 token/题）。训练/推理分布是否因此不一致，待本轮对照验证。
2. `20_baseline.py` 用 tokenizer 自带 `chat_template`，LLaMA-Factory 用代码内 `deepseekr1`，
   两者应当等价但未逐字验证；输出风格明显不同时优先怀疑这里。
3. 24GB 上 `cutoff_len: 2048` + QLoRA 是否真放得下，还是必须降到 1024。
4. PAI-DSW 的 7860 端口映射入口在哪（仅影响可选的 `webchat` 观察界面）。

**已验证**：CUDA 13 + torch 2.13 上 `bitsandbytes` 可 import 且 4bit 前向成功
（`logs/bnb-status.txt` = ok）；`TORCH_MODE=fresh` 不需要触发；
`tokenizer_class` 修正后往返一致。

---

## 9. 目录速查

```
/mnt/workspace/
├── venv/                     虚拟环境（持久化，复用镜像 torch）
├── models/R1-0528-Qwen3-8B/  16.3GB 权重
├── data/                     train.jsonl / val.jsonl / dataset_info.json
├── train/                    本仓库 train/ 目录（脚本与配置）
├── output/<RUN>/             检查点、Adapter、train.log、run.yaml
└── logs/                     session.log、pip-*.txt、bnb-status.txt、指纹
```

| 记录文件 | 作用 |
|---|---|
| `logs/session.log` | 每次开机/关机的真实时间，算额度用 |
| `logs/bnb-status.txt` | 决定走 QLoRA 还是 BF16 |
| `logs/pip-dryrun.txt` | torch 是否被替换的证据 |
| `logs/model-notes.txt` | 改过模型目录的留痕 |
| `output/<RUN>/run.yaml` | 该次训练实际生效的完整配置 |
| `data/out/manifest.json` | 数据来源与 sha256，复现依据 |

本地 `train/scripts/` 一览：

| 脚本 | 作用 |
|---|---|
| `review_cli.py` | 主数据审核：stats / list / show / set / check |
| `convert_to_alpaca.py` | 主数据 → alpaca（train/val/dataset_info/index/manifest） |
| `prompt_template.py` | 与线上共用的提示渲染（system + 证据块） |
| `10_download.py` | 从 ModelScope 拉权重或仅 tokenizer |
| `20_baseline.py` | 原模型基线推理 |
| `21_probe.py` | 最小前向探针 |
| `30_tokenize_check.py` | 模板边界 / 长度 / 标签遮罩自检 |
| `40_train.sh` | LLaMA-Factory 训练封装（含样本数与磁盘门槛） |
| `50_compare.py` | Adapter 与基线逐题对照 |

---

## 10. 每次会话记录模板

往 `train/output/sessions.md` 追加：

| 日期 | 实例 | 用时 | 做了什么 | 产物 | 结论/失败原因 |
|---|---|---|---|---|---|
| | | | | | |
