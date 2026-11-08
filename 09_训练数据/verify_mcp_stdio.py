"""MCP stdio 端到端验证：用官方 Client 真实走一遍协议。

★ 与前面测试的区别
  前面测的是「函数能跑」。这里测的是「**协议能通**」——
  外部客户端（WorkBuddy / Claude Desktop）看到的正是这一步的结果。

  走的是标准流程：
      启动子进程 → initialize → list_tools → call_tool → 收结果

★ 为什么这一步不能省
  MCP 的价值全在「外部能调用」。函数返回对了但协议序列化错了，
  客户端一样用不了。这类问题只有真走一遍才暴露。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"D:\项目\某健康公司培训系统\项目代码"
sys.path.insert(0, ROOT)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    env = dict(os.environ)
    # 开发模式：不要求 token，身份按 debug role 走
    env["AUTH_REQUIRED"] = "false"
    env["MCP_TOKEN"] = ""
    env["PYTHONPATH"] = ROOT
    env["PYTHONUNBUFFERED"] = "1"

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "interfaces.mcp.server"],
        env=env,
        cwd=ROOT,
    )

    print("启动 MCP Server（stdio 子进程）…")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"✅ 握手成功")
            # ★ 2.x 里是 server_info（snake_case），不是 serverInfo
            si = getattr(init, "server_info", None) or getattr(init, "serverInfo", None)
            print(f"   server: {si.name} v{si.version}")
            caps = init.capabilities
            print(f"   capabilities: tools={caps.tools is not None}, "
                  f"resources={caps.resources is not None}, "
                  f"prompts={caps.prompts is not None}")
            print()

            # ---- 列出能力 ----
            #
            # ★ 2.x 全面改用 snake_case：input_schema / uri_template / …
            #   照 1.x 教程写 camelCase 会在客户端侧全线 AttributeError。
            tools = await session.list_tools()
            print(f"📋 工具 {len(tools.tools)} 个：")
            for t in tools.tools:
                sch = getattr(t, "input_schema", None) or {}
                props = sch.get("properties", {})
                print(f"   - {t.name}({', '.join(props)})")
            print()

            templates = await session.list_resource_templates()
            tpl = getattr(templates, "resource_templates", None) or \
                getattr(templates, "resourceTemplates", [])
            print(f"📦 资源模板 {len(tpl)} 个："
                  f"{[str(getattr(x, 'uri_template', getattr(x, 'uriTemplate', '?'))) for x in tpl]}")

            prompts = await session.list_prompts()
            print(f"💬 提示 {len(prompts.prompts)} 个："
                  f"{[p.name for p in prompts.prompts]}")
            print()

            # ---- 调用 1：graph_entity（最快，不依赖 LLM） ----
            print("=" * 66)
            print("调用 graph_entity(name='鱼油')")
            print("=" * 66)
            r = await session.call_tool("graph_entity", {"name": "鱼油"})
            text = r.content[0].text if r.content else ""
            d = json.loads(text)
            print(f"found={d.get('found')} | "
                  f"产品={d.get('product', {}).get('name')}")
            print(f"批准功能={d.get('product', {}).get('claims_text')}")
            print()

            # ---- 调用 2：product resource ----
            print("=" * 66)
            print("读取 resource product://P001")
            print("=" * 66)
            rr = await session.read_resource("product://P001")
            body = rr.contents[0].text if rr.contents else ""
            d = json.loads(body)
            print(f"名称={d.get('name')} | 文号={d.get('approval_no')}")
            print()

            # ---- 调用 3：my_drill_reviews ----
            print("=" * 66)
            print("调用 my_drill_reviews(days=90)")
            print("=" * 66)
            r = await session.call_tool("my_drill_reviews", {"days": 90})
            d = json.loads(r.content[0].text)
            print(f"trainee={d.get('trainee_id')} | 条数={d.get('count')}")
            print(f"note={str(d.get('note'))[:70]}")
            print()

            # ---- 调用 4：prompt ----
            print("=" * 66)
            print("获取 prompt expression_polish")
            print("=" * 66)
            pr = await session.get_prompt(
                "expression_polish",
                {"text": "客户命中率 4/6，建议咨询医生。",
                 "scene": "monthly_report"},
            )
            msg = pr.messages[0].content
            body = getattr(msg, "text", str(msg))
            print(f"模板长度={len(body)} 字")
            print(f"含三条约束: "
                  f"{all(k in body for k in ['不改任何数字', '不添加任何事实', '不加强语气'])}")
            print()

            # ---- 调用 5：越权验证（未登记工具） ----
            print("=" * 66)
            print("越权验证：调用未登记的 build_paper")
            print("=" * 66)
            try:
                r = await session.call_tool("build_paper", {})
                print(f"结果: {r.content[0].text[:100] if r.content else '(空)'}")
            except Exception as exc:                    # noqa: BLE001
                print(f"客户端收到错误（预期）: {type(exc).__name__}: {str(exc)[:120]}")

    print()
    print("✅ stdio 端到端验证完成")


if __name__ == "__main__":
    asyncio.run(main())
