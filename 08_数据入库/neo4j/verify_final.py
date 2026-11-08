"""
图谱最终校验
入库完成后执行，输出完整统计与关键查询验证
"""
import sys
sys.path.insert(0, ".")
from run_cypher import run_cypher


def q(cypher):
    """返回 (是否成功, 数据)。注意：查询成功但无结果时 data 为空列表，属正常"""
    ok, data = run_cypher(cypher)
    if not ok:
        return None, data
    return data or [], None


def section(title):
    print(f"\n{'='*68}")
    print(title)
    print("=" * 68)


def main():
    issues = []

    # ---------- 统计 ----------
    section("一、总体统计")

    data, _ = q("MATCH (n) RETURN labels(n)[0] AS t, count(n) AS c ORDER BY c DESC")
    print("\n节点：")
    nodes = 0
    for r in data:
        print(f"  {r['row'][0]:20} {r['row'][1]:>5}")
        nodes += r["row"][1]
    print(f"  {'合计':20} {nodes:>5}")

    data, _ = q("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC")
    print("\n关系：")
    rels = 0
    for r in data:
        print(f"  {r['row'][0]:22} {r['row'][1]:>5}")
        rels += r["row"][1]
    print(f"  {'合计':22} {rels:>5}")

    # ---------- 完整性 ----------
    section("二、完整性校验（预期全部为空）")

    checks = [
        ("C1 普通食品不应有 CLAIMS 关系",
         "MATCH (p:Product {category:'普通食品'})-[:CLAIMS]->(e) RETURN p.name, e.name"),
        ("C2 CONTRAINDICATED_FOR 必须有 reason",
         "MATCH (p)-[r:CONTRAINDICATED_FOR]->(c) WHERE r.reason IS NULL AND c.reason IS NULL RETURN p.name, c.name"),
        ("C3 Efficacy 必须在目录内或为营养素补充剂",
         "MATCH (e:Efficacy) WHERE e.catalog_no IS NULL AND e.type IS NULL RETURN e.name"),
        ("C4 每个产品必须有用法用量",
         "MATCH (p:Product) WHERE NOT EXISTS((p)-[:DOSED_AS]->()) RETURN p.product_id, p.name"),
        ("C5 每个保健食品必须有不适宜人群",
         "MATCH (p:Product {category:'保健食品'}) WHERE NOT EXISTS((p)-[:CONTRAINDICATED_FOR]->()) RETURN p.product_id, p.name"),
        ("C6 每个产品必须有适宜人群或为普通食品",
         "MATCH (p:Product {category:'保健食品'}) WHERE NOT EXISTS((p)-[:SUITABLE_FOR]->()) RETURN p.product_id, p.name"),
        ("C7 每个产品必须有别名",
         "MATCH (p:Product) WHERE NOT EXISTS((:Alias)-[:ALIAS_OF]->(p)) RETURN p.product_id, p.name"),
    ]

    for title, cypher in checks:
        data, err = q(cypher)
        if data is None:
            print(f"\n  {title}\n    ⚠️ 查询失败")
            issues.append(title)
        elif len(data) == 0:
            print(f"\n  {title}\n    ✅ 通过")
        else:
            print(f"\n  {title}\n    ❌ 有 {len(data)} 条不符合：")
            for r in data[:5]:
                print(f"       {' | '.join(str(v) for v in r['row'])}")
            issues.append(title)

    # ---------- 业务查询 ----------
    section("三、业务查询验证")

    queries = [
        ("Q1 客户血脂偏高有哪些产品",
         "MATCH (p:Product)-[:CLAIMS]->(e:Efficacy {efficacy_id:'EFF_19'}) "
         "RETURN p.name AS 产品, p.category AS 类别"),
        ("Q2 【核心】在服抗凝药需排除哪些产品",
         "MATCH (p:Product)-[r:CONTRAINDICATED_FOR]->(c:Contraindication {contra_id:'CI_BLEED'}) "
         "RETURN p.name AS 产品, c.severity AS 严重度"),
        ("Q3 【核心】肾功能不全需排除哪些",
         "MATCH (p:Product)-[r:CONTRAINDICATED_FOR]->(c:Contraindication {contra_id:'CI_RENAL'}) "
         "RETURN p.name AS 产品"),
        ("Q4 【核心】孕妇需排除哪些",
         "MATCH (p:Product)-[:CONTRAINDICATED_FOR]->(c:Contraindication {contra_id:'CI_PREG'}) "
         "RETURN p.name AS 产品"),
        ("Q5 口语别名能否映射到产品（鱼油/蛋白粉/钙片）",
         "MATCH (a:Alias)-[:ALIAS_OF]->(p:Product) "
         "WHERE a.name IN ['鱼油','蛋白粉','钙片','褪黑素','益生菌'] "
         "RETURN a.name AS 客户说法, p.name AS 标准产品"),
        ("Q6 合规规则清单",
         "MATCH (r:Rule) RETURN r.name AS 规则, r.level AS 级别"),
    ]

    for title, cypher in queries:
        data, _ = q(cypher)
        print(f"\n  {title}")
        if not data:
            print("    （无结果）")
            issues.append(title)
            continue
        for r in data[:8]:
            print(f"    {' | '.join(str(v) for v in r['row'])}")
        if len(data) > 8:
            print(f"    ... 共 {len(data)} 条")

    # ---------- 产品完整画像 ----------
    section("四、产品完整画像抽样（P001 鱼油）")
    data, _ = q(
        "MATCH (p:Product {product_id:'P001'}) "
        "OPTIONAL MATCH (p)-[:CLAIMS]->(e:Efficacy) "
        "OPTIONAL MATCH (p)-[:HAS_INGREDIENT]->(i:Ingredient) "
        "OPTIONAL MATCH (p)-[:DOSED_AS]->(u:Usage) "
        "OPTIONAL MATCH (p)-[:SUITABLE_FOR]->(t:TargetGroup) "
        "OPTIONAL MATCH (p)-[:CONTRAINDICATED_FOR]->(c:Contraindication) "
        "OPTIONAL MATCH (a:Alias)-[:ALIAS_OF]->(p) "
        "RETURN p.name AS 名称, p.category AS 类别, p.approval_no AS 文号, p.spec AS 规格, "
        "collect(DISTINCT e.name) AS 批准功能, "
        "collect(DISTINCT i.name) AS 成分, "
        "collect(DISTINCT u.dosage) AS 用法, "
        "collect(DISTINCT u.daily_limit) AS 上限, "
        "collect(DISTINCT t.name) AS 适宜, "
        "collect(DISTINCT c.name) AS 不适宜, "
        "collect(DISTINCT a.name) AS 别名"
    )
    if data:
        row = data[0]["row"]
        labels = ["名称", "类别", "文号", "规格", "批准功能", "成分", "用法", "上限", "适宜", "不适宜", "别名"]
        for k, v in zip(labels, row):
            print(f"  {k:8} {v}")

    # ---------- 结论 ----------
    section("五、结论")
    if issues:
        print(f"\n  ❌ 有 {len(issues)} 项需关注：")
        for i in issues:
            print(f"     - {i}")
    else:
        print("\n  ✅ 全部校验通过")
        print(f"     节点 {nodes} 个 / 关系 {rels} 条")


if __name__ == "__main__":
    main()
