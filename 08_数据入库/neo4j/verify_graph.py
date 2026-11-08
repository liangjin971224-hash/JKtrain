"""
图谱数据校验
对 jiankangshujuku 执行 10 项校验，输出结论
"""
import sys
sys.path.insert(0, ".")
from run_cypher import run_cypher


def check(title, cypher, expect_empty=True, show_limit=10):
    ok, data = run_cypher(cypher)
    print(f"\n{'='*66}")
    print(f"{title}")
    print("-" * 66)
    if not ok:
        print(f"  ❌ 执行失败: {data}")
        return False

    if data is None or (isinstance(data, list) and len(data) == 0):
        print("  ✅ 通过（无记录 = 符合预期）" if expect_empty else "  ⚠️ 无记录")
        return True

    for row in data[:show_limit]:
        vals = " | ".join(str(v) for v in row["row"])
        print(f"  {vals}")
    if len(data) > show_limit:
        print(f"  ... 共 {len(data)} 条")

    if expect_empty:
        print(f"  ❌ 失败：预期为空，实际 {len(data)} 条")
        return False
    return True


def main():
    all_pass = True

    # ---------- 完整性校验 ----------
    print("\n" + "#" * 66)
    print("# 第一部分：数据完整性校验")
    print("#" * 66)

    all_pass &= check(
        "C1 普通食品不得有任何 CLAIMS 关系（应为空）",
        "MATCH (p:Product {category:'普通食品'})-[:CLAIMS]->(e:Efficacy) "
        "RETURN p.name AS 产品, e.name AS 错误功效",
    )

    all_pass &= check(
        "C2 所有 CONTRAINDICATED_FOR 都应有 reason（应为空）",
        "MATCH (p:Product)-[r:CONTRAINDICATED_FOR]->(c:Contraindication) "
        "WHERE r.reason IS NULL AND c.reason IS NULL "
        "RETURN p.name AS 产品, c.name AS 禁忌",
    )

    all_pass &= check(
        "C3 所有 Efficacy 都应在 24 项目录内或标注为营养素补充剂（应为空）",
        "MATCH (e:Efficacy) WHERE e.catalog_no IS NULL AND e.type IS NULL "
        "RETURN e.name AS 越界功效",
    )

    all_pass &= check(
        "C4 所有 Product 都应有 category（应为空）",
        "MATCH (p:Product) WHERE p.category IS NULL RETURN p.name AS 缺类别",
    )

    # ---------- 关键查询验证 ----------
    print("\n" + "#" * 66)
    print("# 第二部分：典型查询验证（图谱能否回答真实问题）")
    print("#" * 66)

    check(
        "Q1 客户血脂偏高，有哪些保健食品（应按血脂功效查）",
        "MATCH (p:Product)-[:CLAIMS]->(e:Efficacy {efficacy_id:'EFF_19'}) "
        "RETURN p.name AS 产品, p.category AS 类别, p.approval_no AS 批准文号",
        expect_empty=False,
    )

    check(
        "Q2 【核心能力】客户在服抗凝药，哪些产品要排除",
        "MATCH (p:Product)-[r:CONTRAINDICATED_FOR]->(c:Contraindication {contra_id:'CI_BLEED'}) "
        "RETURN p.name AS 产品, c.severity AS 严重度, r.reason AS 原因",
        expect_empty=False,
    )

    check(
        "Q3 【核心能力】客户肾不好，哪些蛋白类产品要排除",
        "MATCH (p:Product)-[r:CONTRAINDICATED_FOR]->(c:Contraindication {contra_id:'CI_RENAL'}) "
        "RETURN p.name AS 产品, r.reason AS 原因",
        expect_empty=False,
    )

    check(
        "Q4 【教学重点】外观相似但类别不同的产品对（普通食品 vs 保健食品）",
        "MATCH (a:Product)-[r:SIMILAR_TO]->(b:Product) "
        "RETURN a.name AS 产品A, b.name AS 产品B, r.lesson_point AS 教学点",
        expect_empty=False,
    )

    check(
        "Q5 中药配伍禁忌（人参/西洋参 vs 藜芦/五灵脂）",
        "MATCH (p:Product)-[r:INCOMPATIBLE_WITH]->(h:Ingredient) "
        "RETURN p.name AS 产品, h.name AS 禁忌中药, r.severity AS 严重度",
        expect_empty=False,
    )

    check(
        "Q6 某产品的完整画像（P001 鱼油）",
        "MATCH (p:Product {product_id:'P001'}) "
        "OPTIONAL MATCH (p)-[:CLAIMS]->(e:Efficacy) "
        "OPTIONAL MATCH (p)-[:CONTRAINDICATED_FOR]->(c:Contraindication) "
        "RETURN p.name AS 产品, collect(DISTINCT e.name) AS 批准功能, "
        "collect(DISTINCT c.name) AS 不适宜人群",
        expect_empty=False,
    )

    check(
        "Q7 从需求反查产品（血脂健康关注）",
        "MATCH (n:Need {need_id:'NEED_LIPID'})<-[:ADDRESSES]-(e:Efficacy)<-[:CLAIMS]-(p:Product) "
        "RETURN p.name AS 产品, e.name AS 依据功效",
        expect_empty=False,
    )

    # ---------- 统计 ----------
    print("\n" + "#" * 66)
    print("# 第三部分：入库结果统计")
    print("#" * 66)

    ok, data = run_cypher(
        "MATCH (n) RETURN labels(n)[0] AS 节点类型, count(n) AS 数量 ORDER BY 数量 DESC"
    )
    print("\n节点：")
    total_nodes = 0
    for row in data:
        print(f"  {row['row'][0]:20} {row['row'][1]:>4}")
        total_nodes += row["row"][1]
    print(f"  {'合计':20} {total_nodes:>4}")

    ok, data = run_cypher(
        "MATCH ()-[r]->() RETURN type(r) AS 关系类型, count(r) AS 数量 ORDER BY 数量 DESC"
    )
    print("\n关系：")
    total_rels = 0
    for row in data:
        print(f"  {row['row'][0]:22} {row['row'][1]:>4}")
        total_rels += row["row"][1]
    print(f"  {'合计':22} {total_rels:>4}")

    # 产品分类统计
    ok, data = run_cypher(
        "MATCH (p:Product) RETURN p.category AS 类别, count(p) AS 数量"
    )
    print("\n产品分类：")
    for row in data:
        print(f"  {row['row'][0]:12} {row['row'][1]:>4}")

    print("\n" + "=" * 66)
    print("校验结果：" + ("✅ 全部通过" if all_pass else "❌ 存在失败项，请检查上文"))
    print("=" * 66)


if __name__ == "__main__":
    main()
