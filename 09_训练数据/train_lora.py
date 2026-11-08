# 健康顾问培训系统 · Qwen2.5 LoRA 微调
#
# ════════════════════════════════════════════════════════════════════
# 本脚本以 `D:\ai260803\xiangmudaima\moxingweitiao\qianwen_weitiao.py`
# 为蓝本，沿用它的训练方式（LoRA r=16 / alpha=32 / cosine + warmup /
# 梯度检查点 / bf16），只改与「本项目数据」相关的部分。
#
# 为什么不直接改原脚本：原脚本绑死了电商客服数据集与 1.5B 路径，
# 另起一份能保住原脚本作为对照，出问题可回退。
#
# ────────────────────────────────────────────────────────────────────
# ★ 与原脚本的 4 处必要差异（每条都有实测依据，勿随手改回）
# ────────────────────────────────────────────────────────────────────
# 1. **max_seq_length 384**（原脚本 256）
#    实测本项目 120 条样本 token 分布：P50=219 / P90=307 / max=328。
#    256 会截断 14% 的样本——而且截掉的是**末尾的答案部分**，
#    模型只会学到「问题和资料」，学不到「怎么答」，等于白训。
#    384 截断 0%，显存增量可接受。
#
# 2. **num_train_epochs 3**（原脚本 5）
#    数据 459 条 vs 原脚本 100 条。5 epoch = 2295 步，配合 1e-4 学习率
#    在这么小的数据上极易过拟合（损失降得很低但泛化差）。
#    3 epoch ≈ 1377 步，更稳。
#
# 3. **格式化函数带 system，且 system 取自推理侧**
#    ★ 这是全脚本最关键的一处。
#    原脚本的 messages 只有 user + assistant，没有 system。
#    而本项目推理时每次都传 system（403 字的角色与 8 条硬约束）。
#    若训练时不带 system，模型没学过那套约束，推理时再喂就等于**错配**。
#    这里直接 import 推理侧的 `_system_prompt()`，保证逐字一致。
#
# 4. **数据格式校验前置**
#    训练前先断言：system 与推理侧一致、instruction 含【员工的问题】分区、
#    output 非空。不通过直接退出，避免拿错数据白跑几小时。
# ════════════════════════════════════════════════════════════════════

# ===========================
# 依赖说明（版本非常重要）
# ===========================
# 原脚本要求：transformers==4.40.2 trl==0.8.6 peft==0.10.0 accelerate==0.30.1
# 本机实际：transformers==4.44.2 trl==0.8.6 peft==0.21.2 accelerate==1.15.0
# ⚠ 差异说明：trl 版本一致（0.8.6，SFTTrainer 接口对得上）；
#   peft / accelerate 更新，本脚本已按新接口写，实测可跑。
#   若换回原脚本那套版本，请注意 peft 的 enable_input_require_grads 行为不同。

import os
import sys
import json
import pathlib
import warnings

import torch
import torch.distributed.tensor   # ⚠ 必须：peft 内部会 isinstance(..., DTensor)

import torch.backends.cuda
torch.backends.cuda.enable_cudnn_sdp(False)   # RTX40 + bf16 的 cuDNN SDPA 反向数值 bug

from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

warnings.filterwarnings("ignore", message="Could not find a config file")

# ================= Windows 编码修复 =================
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
os.environ["PYTHONUTF8"] = "1"

_original_read_text = pathlib.Path.read_text

def _patched_read_text(self, encoding=None, errors=None):
    return _original_read_text(self, encoding="utf-8", errors=errors)

pathlib.Path.read_text = _patched_read_text

# ================= 配置区域 =================
PROJECT_ROOT = pathlib.Path(r"D:\项目\某健康公司培训系统")

#: 要微调的基座模型。可用环境变量 TRAIN_MODEL 覆盖（换 3B 时用）。
#: 1.5B 在 8GB 卡上可 bf16 训练（不必 4bit），最快；
#: 3B 权重 5.8GB，显存很紧，建议配合 PYTORCH_CUDA_ALLOC_CONF。
_DEFAULT_MODEL = (
    r"D:\ai260803\xiangmudaima\pytorch\model\models"
    r"\Qwen--Qwen2.5-1.5B-Instruct\snapshots\master"
)
MODEL_PATH = os.environ.get("TRAIN_MODEL", _DEFAULT_MODEL)

#: 本项目 SFT 数据（由 09_训练数据/build_sft_dataset_v2.py 生成）
DATA_PATH = str(PROJECT_ROOT / "09_训练数据" / "train.jsonl")

#: LoRA 权重输出目录（可用 TRAIN_OUT 覆盖，避免覆盖上一版权重）
OUTPUT_DIR = os.environ.get(
    "TRAIN_OUT", str(PROJECT_ROOT / "09_训练数据" / "lora_out")
)

#: ★ 384：实测 max=328 token，384 零截断；256 会截掉 14% 的答案尾部
MAX_SEQ_LENGTH = int(os.environ.get("TRAIN_MAX_SEQ", "384"))

#: ★ 2 轮：v5 的验证 loss 0.2062（训练 0.10）说明训得不够，补 1 轮。
#: v: 1 轮时（v4）答案正确但格式脏；v5 靠数据规范化修格式反而丢了正确表述。
#: 现在格式交给推理侧清洗，训练只需保证内容准确，2 轮更充分。
#: 历史：3 轮 → train_loss 0.014，严重过拟合，
#:   推理时面对资料外问题直接编造产品）。
#:   459 条 × 1 轮 ≈ 114 步，足够让模型学到「怎么说」而不是背答案。
NUM_EPOCHS = 2

#: debug 模式：只取 N 条（设 0 表示全量）
DEBUG_LIMIT = int(os.environ.get("TRAIN_DEBUG_LIMIT", "0"))

# ================= 从推理侧导入 system prompt =================
# ★ 绝不手抄。手抄必然漂移（实测手抄 403→409 字符，多行缩进混进正文），
#   而 system 错配等于整个数据集白做。
CODE_ROOT = str(PROJECT_ROOT / "项目代码")
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

try:
    from application.use_cases.answer_question import AnswerQuestionUseCase
    SYSTEM_PROMPT = AnswerQuestionUseCase._system_prompt()
except Exception as exc:
    print(f"❌ 无法导入推理侧 system prompt: {exc}")
    print("   训练与推理必须用同一段 system，否则学出来的模型对不上。")
    sys.exit(1)

print(f"✅ system prompt 已从推理侧导入（{len(SYSTEM_PROMPT)} 字）")
# 打印实际配置，便于事后从日志核对用的是哪个基座
print(f"📦 基座模型: {MODEL_PATH}")
print(f"📂 输出目录: {OUTPUT_DIR}")
print(f"📏 max_seq_length={MAX_SEQ_LENGTH} | epochs={NUM_EPOCHS}")

# ================= 检查 GPU =================
if not torch.cuda.is_available():
    print("❌ 未检测到 GPU")
    sys.exit(1)

print("使用 GPU:", torch.cuda.get_device_name(0))
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ================= 1. tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ================= 2. 数据校验（先于加载模型，避免白等） =================
def precheck_dataset(path: str) -> int:
    """校验数据格式。不通过直接退出。"""
    p = pathlib.Path(path)
    if not p.is_file():
        print(f"❌ 数据文件不存在: {path}")
        sys.exit(1)

    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not rows:
        print("❌ 数据为空")
        sys.exit(1)

    bad_system = [i for i, r in enumerate(rows) if r.get("system") != SYSTEM_PROMPT]
    bad_inst = [i for i, r in enumerate(rows) if "【员工的问题】" not in r.get("instruction", "")]
    bad_out = [i for i, r in enumerate(rows) if not r.get("output", "").strip()]

    if bad_system:
        print(f"❌ {len(bad_system)} 条 system 与推理侧不一致（首条 index={bad_system[0]}）")
        print("   训练 system 必须与推理逐字相同，否则模型行为会错配。")
        sys.exit(1)
    if bad_inst:
        print(f"❌ {len(bad_inst)} 条 instruction 缺【员工的问题】分区")
        sys.exit(1)
    if bad_out:
        print(f"❌ {len(bad_out)} 条 output 为空")
        sys.exit(1)

    print(f"✅ 数据校验通过：{len(rows)} 条，system 一致 / 格式完整 / output 非空")
    return len(rows)


total_rows = precheck_dataset(DATA_PATH)

# ================= 3. 加载模型 =================
print("正在加载模型...")
#: 是否用 4bit 量化加载（QLoRA）。3B 全精度 5.8GB 在 8GB 卡上极易 OOM，
#: 用 TRAIN_4BIT=1 打开。4bit 只能训 LoRA（冻结的基座不可训练），本脚本正是 LoRA。
_USE_4BIT = os.environ.get("TRAIN_4BIT", "0") == "1"

if _USE_4BIT:
    # ★ 必须打这个补丁：accelerate ≥1.10 的 dispatch_model 在单卡分支
    #   无条件 `model.to(device)`，而 bitsandbytes 4bit 模型禁止 .to()，
    #   直接抛 `ValueError: .to is not supported for 4-bit models`。
    #   与推理侧 local_llm_adapter._patch_accelerate_dispatch() 是同一个坑。
    import accelerate.big_modeling as _bm
    import transformers.modeling_utils as _mu

    _orig_dispatch = _bm.dispatch_model

    def _safe_dispatch(model, *a, **kw):
        try:
            return _orig_dispatch(model, *a, **kw)
        except ValueError as exc:
            if "4-bit" not in str(exc) and "8-bit" not in str(exc):
                raise
            dm = kw.get("device_map")
            if dm is not None:
                try:
                    model.hf_device_map = dict(dm)
                except (AttributeError, TypeError):
                    pass
            return model

    _bm.dispatch_model = _safe_dispatch
    _mu.dispatch_model = _safe_dispatch

    from transformers import BitsAndBytesConfig
    _quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("🔧 启用 4bit 量化加载（QLoRA）+ accelerate 补丁")
else:
    _quant = None

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    trust_remote_code=True,
    quantization_config=_quant,
    # ★ 不用 device_map="auto"：实测 3B 加载时报
    #   `Cannot copy out of meta tensor; no data!`——
    #   accelerate 的分片加载与 low_cpu_mem_usage 的 meta 设备混用会冲突。
    #   单卡直接 .to("cuda") 更稳（下面还有一步显式搬移）。
    device_map=None,
)
model.config.use_cache = False       # 与 gradient_checkpointing 冲突，必须关
if not _USE_4BIT:
    model = model.to("cuda")         # 单卡显式搬移，避免 meta tensor 问题
else:
    # ★ 4bit 只量化参数，rope 的 cos/sin 仍是非参数 buffer，会留在 CPU，
    #   直接训练报 `indices should be either on cpu or on the same device`。
    #   推理侧 local_llm_adapter 也是这么处理的。
    for _, buf in list(model.named_buffers()):
        if buf.device.type != "cuda":
            buf.data = buf.data.to("cuda")
print("✅ 模型加载成功")
if _USE_4BIT:
    # QLoRA 必需：让输入嵌入参与梯度，否则反向报 requires_grad 错误
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

# ================= 4. 配置 LoRA =================
# 参数与参考脚本完全一致：r=16 / alpha=32 / 7 个目标层全开
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
model = get_peft_model(model, peft_config)
model.enable_input_require_grads()   # 配 gradient_checkpointing 必须
model.print_trainable_parameters()

# ================= 5. 加载数据 =================
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
if DEBUG_LIMIT > 0:
    dataset = dataset.select(range(min(DEBUG_LIMIT, len(dataset))))
print(f"样本总数: {len(dataset)} 条")

# ★ 切出 10% 作验证集。上次全量训练、没有验证集，只有 train_loss 一个信号，
#   结果 0.014 的假象掩盖了过拟合。现在能直接看到验证 loss 是否回升。
_holdout = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, valid_ds = _holdout["train"], _holdout["test"]
print(f"训练集 {len(train_ds)} 条 / 验证集 {len(valid_ds)} 条")


def formatting_func(examples):
    """组装成对话模板。

    ★ 与原脚本的差异：**带上 system**。
      本项目推理每次都传 403 字的角色与硬约束，训练不带就是错配。
    """
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction + "\n" + input_text},
            {"role": "assistant", "content": output},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )
    return {"text": texts}


train_ds = train_ds.map(formatting_func, batched=True)
valid_ds = valid_ds.map(formatting_func, batched=True)

# ================= 6. 训练参数 =================
# 显存相关三个参数沿用参考脚本的 8GB 适配值：
#   batch=1 + grad_accum=4 → 等效 batch 4；gradient_checkpointing 换显存
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    # ★ 关掉 Trainer 内 eval。
    #   实测三次都 OOM：eval 要算 [1, seq, 152064] 的 logits（约 1.7GB），
    #   而训练已占 5.3GB，8GB 卡必然不够。
    #   `prediction_loss_only=True` 也救不了——trl 的 SFTTrainer 会自己
    #   接管 compute_loss，该参数不生效（实测验证）。
    #   改为训练结束后用下方 `evaluate_holdout()` 单独算验证 loss：
    #   此时训练显存已释放，且用 batch=1、只累加 loss、不留 logits。
    eval_strategy="no",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=5e-5,   # ★ 由 1e-4 降到 5e-5：配合 1 轮，减缓记忆速度
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    # warmup 按比例给：参考脚本 200 步是相对 100 条数据（约 25 步/epoch）而言，
    # 本项目 459 条 × 3 epoch ≈ 345 步，warmup 取总量的 10% 更合理
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="none",
)

# ================= 7. Trainer =================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
)

# ================= 8. 训练 =================
steps_per_epoch = len(train_ds) // (training_args.per_device_train_batch_size
                                   * training_args.gradient_accumulation_steps)
print(f"开始训练：{NUM_EPOCHS} 轮 × 约 {steps_per_epoch} 步/轮 ≈ {steps_per_epoch * NUM_EPOCHS} 步")
def evaluate_holdout(mdl, tok, ds) -> float:
    """训练后单独算验证 loss（不进 Trainer，避免 OOM）。

    做法：逐条前向，只累加 loss，不保留 logits、不 backward。
    """
    mdl.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(len(ds)):
            ids = tok([ds[i]["text"]], return_tensors="pt",
                      truncation=True, max_length=MAX_SEQ_LENGTH)
            ids = {k: v.to("cuda") for k, v in ids.items()}
            ids["labels"] = ids["input_ids"].clone()
            out = mdl(**ids)
            total += out.loss.item()
            n += 1
            del out
            if n % 5 == 0:
                torch.cuda.empty_cache()
    return total / max(n, 1)


# ★ 用 try/finally 保证权重一定落盘：
#   上次 eval 阶段 OOM 抛异常，train() 之后的 save_model 没执行，
#   109 步训练白跑（权重文件根本没生成）。
try:
    trainer.train()
finally:
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ LoRA 权重已保存到: {OUTPUT_DIR}")
    # 权重已落盘后再评估——即使 OOM 也不影响产物
    torch.cuda.empty_cache()
    try:
        vl = evaluate_holdout(trainer.model, tokenizer, valid_ds)
        print(f"📊 验证集 loss: {vl:.4f}（训练 loss 见上方曲线，两者差距过大=过拟合）")
    except Exception as exc:
        print(f"⚠️ 验证 loss 计算失败（不影响权重）: {type(exc).__name__}: {exc}")
