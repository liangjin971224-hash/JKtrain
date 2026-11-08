"""噪声过滤策略对比：用评测集决定哪一档最优"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("HF_HUB_OFFLINE","1")
sys.path.insert(0,'.')

import re
from chunking_variants import (
    BASE_DIR_STR, TRAINING_PERMISSION, chunk_case_split, chunk_case_whole, chunk_heading,
)
from common import Embedder, get_client
from compare_chunking import create_collection, evaluate, EVAL_PATH

BASE = Path(BASE_DIR_STR)

STRATEGIES = {
    "F0_none":   [],
    "F1_min":    [r"^学习目标$", r"自测题"],
    "F2_mid":    [r"^学习目标$", r"自测题", r"^参考答案", r"参考答案$"],
}

def build(pats):
    units = []
    for f in sorted((BASE/"01_培训文档").glob("*.md")):
        doc_id = f.name[:4]
        perm = TRAINING_PERMISSION.get(doc_id,"internal")
        md = f.read_text(encoding="utf-8")
        title = md.splitlines()[0].lstrip("# ").strip() or f.stem
        for i,(sec,body) in enumerate(chunk_heading(md)):
            if len(body) < 30: continue
            if any(re.search(p, sec) for p in pats): continue
            units.append(dict(chunk_id=f"{doc_id}_c{i:03d}", text=body[:11000],
                doc_id=doc_id, title=title, section=sec, source_file=f.name,
                permission_level=perm, category="training", parent_text=""))
    for f in sorted((BASE/"02_情景案例").glob("*.md")):
        md = f.read_text(encoding="utf-8")
        for idx,(cid,ctitle,sec,body) in enumerate(chunk_case_split(md)):
            units.append(dict(chunk_id=f"{cid}_s{idx:02d}", text=body[:11000],
                doc_id=cid, title=ctitle, section=f"{ctitle} · {sec}", source_file=f.name,
                permission_level="restricted", category="case", parent_text=""))
    return units

def main():
    ev = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    client = get_client()
    emb = Embedder()
    print(f"评测 {len(ev['queries'])} 条\n")
    print(f"{'策略':<12}{'块数':>6}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}")
    print("-"*54)
    res = {}
    for name, pats in STRATEGIES.items():
        coll = f"eval_{name.lower()}"
        units = build(pats)
        vecs = emb.encode([u["text"] for u in units])
        rows = []
        for u,v in zip(units,vecs):
            r=dict(u); r["vector"]=v; rows.append(r)
        create_collection(client, coll)
        for i in range(0,len(rows),100):
            client.insert(collection_name=coll, data=rows[i:i+100])
        client.flush(coll)
        r = evaluate(client, emb, coll, ev)
        res[name]=r
        print(f"{name:<12}{len(units):>6}{r['recall@1']:>8.3f}{r['recall@3']:>8.3f}{r['recall@5']:>8.3f}{r['mrr']:>8.3f}")
    Path("filter_tune.json").write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
