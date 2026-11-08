"""配置层统一出口。"""
from __future__ import annotations

from config.assembly import Container, build_container, build_test_container
from config.settings import BASE_DIR, Settings, get_settings

__all__ = [
    "BASE_DIR",
    "Container",
    "Settings",
    "build_container",
    "build_test_container",
    "get_settings",
]
