"""Load, validate and split the golden set."""

import hashlib
import json
from pathlib import Path

from evals.schemas import GoldenItem


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = GoldenItem(**json.loads(line))
        except Exception as e:  # noqa: BLE001 - re-raise with the line number, the thing you need to fix it
            raise ValueError(f"{path}:{lineno}: {e}") from e
        if item.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {item.id}")
        seen.add(item.id)
        items.append(item)
    if not items:
        raise ValueError(f"{path}: no questions found")
    return items


def split_of(item_id: str, test_fraction: float = 0.3) -> str:
    """Deterministic dev/test split from the id alone, so adding questions never reshuffles old ones."""
    bucket = int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 1000
    return "test" if bucket < int(test_fraction * 1000) else "dev"


def select(items: list[GoldenItem], split: str = "all") -> list[GoldenItem]:
    if split not in {"all", "dev", "test"}:
        raise ValueError("split must be all, dev or test")
    return items if split == "all" else [i for i in items if split_of(i.id) == split]
