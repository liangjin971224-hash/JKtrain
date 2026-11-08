"""在魔搭实例内下载模型或只下载 tokenizer，落到持久化目录并记录版本指纹。

    python scripts/10_download.py --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    python scripts/10_download.py --model-id ... --tokenizer-only   # CPU 实例够用

默认落盘 /mnt/workspace/models（唯一持久化目录）。写到 /root/.cache 的权重
在实例关闭后会被清除，所以这里显式指定 cache_dir。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

TOKENIZER_PATTERNS = ["*.json", "*.txt", "*.model", "*.tiktoken"]


def dir_fingerprint(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    config = path / "config.json"
    return {
        "path": str(path),
        "files": len(files),
        "bytes": total,
        "gib": round(total / 2**30, 2),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()
        if config.exists()
        else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--cache-dir", default="/mnt/workspace/models")
    ap.add_argument(
        "--local-dir",
        default=None,
        help="不指定时落到 <cache-dir>/<模型名>，保证 CPU 与 GPU 两侧路径一致",
    )
    ap.add_argument("--revision", default="master")
    ap.add_argument("--tokenizer-only", action="store_true")
    args = ap.parse_args()

    from modelscope import snapshot_download

    os.environ.setdefault("MODELSCOPE_CACHE", args.cache_dir)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    local_dir = args.local_dir or str(
        Path(args.cache_dir) / args.model_id.split("/")[-1]
    )

    started = datetime.now(timezone.utc)
    path = snapshot_download(
        args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=local_dir,
        allow_patterns=TOKENIZER_PATTERNS if args.tokenizer_only else None,
    )
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    fp = dir_fingerprint(Path(path))
    record = {
        "model_id": args.model_id,
        "revision_requested": args.revision,
        "tokenizer_only": args.tokenizer_only,
        "downloaded_at": started.isoformat(),
        "download_seconds": round(elapsed, 1),
        **fp,
    }
    log_dir = Path("/mnt/workspace/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    (log_dir / f"download-{stamp}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\n模型目录（后续 --model-path 用这个）：{path}")


if __name__ == "__main__":
    main()
