"""MCP 五个能力的真实调用验证（不经过 MCP 协议，直接调函数）。

★ 为什么这样测
  走 MCP 协议需要起 stdio 子进程 + 构造 JSON-RPC，排障困难。
  先在函数层验证「业务逻辑正确」，再用独立脚本验证「协议能通」。
  两者分开，出错时能立刻定位是逻辑问题还是协议问题。
"""
import json
import logging
import sys

sys.path.insert(0, r"D:\项目\某健康公司培训系统\项目代码")
logging.basicConfig(level=logging.ERROR)

from config.assembly import build_container
from config.settings import get_settings
from domain.value_objects.role import Role
from interfaces.mcp.identity import McpIdentity, resolve_identity
from interfaces.mcp.prompts import expression_polish_template
from interfaces.mcp.resources import read_product
from interfaces.mcp.tools.drill import my_drill_reviews
from interfaces.mcp.tools.graph import graph_entity
from interfaces.mcp.tools.kb import kb_search

sys.stdout.reconfigure(encoding="utf-8")

c = build_container(get_settings())
mgr = resolve_identity(c, token="", debug_role="manager")
print(f"容器: {len(c.ready_ports())} 端口就绪 | pending={c.pending or '无'}")
print(f"身份: {mgr.employee_no} / {mgr.role.label}")
print()

# ---- 1. graph_entity（不依赖检索，先测这个最快） ----
print("=" * 70)
print("① graph_entity: 鱼油")
print("=" * 70)
out = graph_entity(c, mgr, name="鱼油")
d = json.loads(out)
if d.get("found"):
    p = d["product"]
    print(f"  产品: {p['name']} ({p['product_id']}) | 类别: {p['category']}")
    print(f"  批准功能原文: {p['claims_text']}")
    print(f"  禁用人群({len(p['contraindications'])}):")
    for x in p["contraindications"][:3]:
        print(f"     - {x['name']} severity={x['severity']} must_block={x['must_block']}")
else:
    print("  未找到:", d.get("note"))

# ---- 2. product resource ----
print()
print("=" * 70)
print("② product://P001")
print("=" * 70)
d = json.loads(read_product(c, mgr, "P001"))
print(f"  {d.get('name')} | 批准文号: {d.get('approval_no')}")
print(f"  claims: {[x['text'] for x in d.get('claims', [])]}")
print(f"  compliance_note: {str(d.get('compliance_note'))[:60]}")

# ---- 3. my_drill_reviews ----
print()
print("=" * 70)
print("③ my_drill_reviews（会话身份=manager/(dev)）")
print("=" * 70)
d = json.loads(my_drill_reviews(c, mgr, days=90))
print(f"  trainee_id={d['trainee_id']} 条数={d['count']}")
print(f"  note: {d.get('note', '')[:80]}")

# 用一个真实存在的学员工号测（验证身份绑定生效）
print()
print("  用真实学员身份 E001 再测：")
e001 = McpIdentity("E001", Role.TRAINEE)
d = json.loads(my_drill_reviews(c, e001, days=3650))
print(f"  trainee_id={d['trainee_id']} 条数={d['count']}")
for r in d["reviews"][:2]:
    print(f"     {r['case_id']} {r['case_type']} 风险={r['risk_count']} 要素={r['element_hit']}/{r['element_total']}")

# ---- 4. prompt ----
print()
print("=" * 70)
print("④ expression_polish")
print("=" * 70)
p = expression_polish_template("客户命中率 4/6，建议咨询医生。", "monthly_report")
print(f"  模板长度: {len(p)} 字")
print(f"  含「不改数字」: {'不改任何数字' in p}")
print(f"  含「不加强语气」: {'不加强语气' in p}")
print(f"  含场景约束: {'对外' in p}")

c.close()
