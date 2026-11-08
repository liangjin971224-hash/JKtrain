"""P0 端到端验证：评阅结果真的落库了吗。

★ 为什么单独写脚本：
  单测用假仓储，只能证明「调用逻辑对」，证明不了
  「真 MySQL 里字段真的写进去了、定位真的准确」。

  这是本项目一贯的教训：**单测绿 ≠ 接真库可用**。

跑法::

    .venv/Scripts/python.exe scripts/verify_p0_persist.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient          # noqa: E402


def main() -> int:
    from config.settings import Settings
    from config.assembly import build_container
    import interfaces.http.app as appmod

    settings = Settings()
    container = build_container(settings)
    repo = container.repository
    if repo is None:
        print("仓储不可用，退出")
        return 1

    failures: list[str] = []
    app = appmod.app

    with TestClient(app) as client:
        r = client.post("/api/auth/login",
                        json={"employee_no": "E001",
                              "password": "trainee123"})
        if r.status_code != 200:
            print("登录失败:", r.status_code, r.text[:200])
            return 1
        tok = r.json().get("access_token") or r.json().get("token")
        h = {"Authorization": f"Bearer {tok}"}

        # ---- 用例 1：正常作答（尽量合规） ----
        cases = client.get("/api/cases", headers=h).json()
        target = next((c for c in cases if c["case_id"] == "C07"), cases[0])
        print(f"\n演练案例: {target['case_id']} {target['title']}")

        start = client.post("/api/drill/start",
                            json={"case_id": target["case_id"]},
                            headers=h)
        if start.status_code != 200:
            print("start 失败:", start.status_code, start.text[:200])
            return 1
        sid = start.json()["session_id"]

        # 答三轮（内容故意包含违规话术，确保有 risk_item）
        answers = [
            "这个肯定有效果，能降血脂的。",
            "您放心，吃了保证您血脂降下来。",
            "这个要一直吃，不能停。",
        ]
        for a in answers:
            resp = client.post("/api/drill/answer",
                               json={"session_id": sid, "answer": a},
                               headers=h)
            if resp.status_code != 200:
                print("answer 失败:", resp.status_code, resp.text[:200])
                break

        review = client.post("/api/drill/review",
                             json={"session_id": sid}, headers=h)
        print(f"评阅接口: {review.status_code}")
        if review.status_code != 200:
            print("评阅失败:", review.text[:300])
            return 1
        card = review.json()
        api_risks = card.get("risk_items", [])
        print(f"  接口返回违规数: {len(api_risks)}")
        for it in api_risks[:3]:
            print(f"    {it['location']} [{it['level']}] {it['rule_type']}")

        # ---- 验证 1：库里真的写进去了 ----
        row = repo.get_drill_review(sid)
        if row is None:
            failures.append("评阅结果没有落库（get_drill_review 返回 None）")
        else:
            print(f"\n【库中记录】drill_id={row.drill_id}")
            print(f"  case_id={row.case_id}")
            print(f"  case_type={row.case_type!r}")
            print(f"  trainee_id={row.trainee_id!r}")
            print(f"  要素 {row.element_hit}/{row.element_total} "
                  f"(含选答 {row.element_hit_all}/{row.element_total_all})")
            print(f"  违规 risk={row.risk_count} high={row.high_count} "
                  f"medium={row.medium_count} critical={row.critical_count}")
            print(f"  明细条数: {len(row.risk_items)}")

            # 字段校验
            if not row.case_type:
                failures.append("case_type 为空（能力项无法归并）")
            if not row.trainee_id:
                failures.append("trainee_id 为空")
            if row.element_total == 0:
                failures.append("element_total 为 0（必答要素没算出来）")
            # 一致性：库里的违规数与接口返回一致
            if row.risk_count != len(api_risks):
                failures.append(
                    f"违规数不一致: 库={row.risk_count} 接口={len(api_risks)}")
            if len(row.risk_items) != len(api_risks):
                failures.append(
                    f"明细条数不一致: 库={len(row.risk_items)} "
                    f"接口={len(api_risks)}")

            # 定位校验：库里的 turn/sentence 必须在接口返回的 location 里出现
            api_locs = {it["location"] for it in api_risks}
            for it in row.risk_items:
                loc = f"第{it.turn_index}轮第{it.sentence_index}句"
                if loc not in api_locs:
                    failures.append(f"定位对不上: 库={loc} 接口={sorted(api_locs)}")
                    break

        # ---- 验证 2：幂等（重复评阅不产生重复明细） ----
        before = len(row.risk_items) if row else 0
        client.post("/api/drill/review", json={"session_id": sid}, headers=h)
        row2 = repo.get_drill_review(sid)
        after = len(row2.risk_items) if row2 else 0
        print(f"\n【幂等验证】重复评阅前后明细条数: {before} → {after}")
        if after != before:
            failures.append(f"重复评阅导致明细累积: {before} → {after}")

        # ---- 验证 3：落库失败不影响接口返回 ----
        # 用一个会抛异常的假仓储替换，确认接口仍返回 200
        real_repo = container.repository

        class _BoomRepo:
            def __getattr__(self, name):
                raise RuntimeError("db down")

        container.repository = _BoomRepo()
        try:
            start2 = client.post("/api/drill/start",
                                 json={"case_id": target["case_id"]}, headers=h)
            sid2 = start2.json()["session_id"]
            client.post("/api/drill/answer",
                        json={"session_id": sid2, "answer": "保健食品不是药物。"},
                        headers=h)
            r3 = client.post("/api/drill/review",
                             json={"session_id": sid2}, headers=h)
            print(f"\n【降级验证】落库失败时评阅接口状态: {r3.status_code}")
            if r3.status_code != 200:
                failures.append(
                    f"落库失败把接口也带崩了: {r3.status_code}")
        finally:
            container.repository = real_repo

    container.close()

    print("\n" + "=" * 60)
    if failures:
        print("P0 验证失败:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("P0 验证全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
