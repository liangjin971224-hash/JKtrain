"""一次性探针：确认模板能否关闭思考，以及关掉后答案有多长。

不依赖 train/ 里的其它脚本，目的是在实例上快速拿到三个事实：
  1. chat_template 里有没有 thinking 相关分支
  2. apply_chat_template(enable_thinking=False) 是否被接受
  3. 关闭后生成 200 token 会得到什么
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/workspace/models/R1-0528-Qwen3-8B"
EVID = (
    "【结构化事实】（来自产品库，权威，可直接引用）\n"
    "产品：康源牌鱼油软胶囊（P001）\n"
    "不适宜人群：少年儿童、孕妇、乳母、出血倾向者和出血性疾病患者、肝功能不全者\n"
)
QUESTION = "客户吃着阿司匹林呢，还能吃鱼油不？"
SYSTEM = "你是连锁健康食品门店的培训助教，回答新员工的问题。"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
msgs = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": f"{EVID}\n【员工的问题】\n{QUESTION}"},
]

tpl = tok.chat_template or ""
print("1) 模板含 thinking 字样:", "thinking" in tpl.lower())
print("   模板长度:", len(tpl))

for flag in (False, True):
    try:
        p = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=flag
        )
        print(f"2) enable_thinking={flag} 被接受, 末尾={p[-70:]!r}")
    except Exception as exc:
        print(f"2) enable_thinking={flag} 被拒绝: {type(exc).__name__}: {exc}")

prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
)
model.eval()
enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
with torch.no_grad():
    out = model.generate(
        **enc, max_new_tokens=200, do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"3) 生成 {len(tok(text, add_special_tokens=False).input_ids)} token：")
print(text)
