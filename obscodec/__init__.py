"""ObsCodec — learned observation compression for multi-agent systems.

Pre-study for SemCom-MARL.
5 codec paradigms, 263 benchmark configurations across 7 MPE scenarios.
"""
__version__ = "2.1.0"
__author__ = "MacswareX"
__license__ = "MIT"

from collections.abc import Iterable
from typing import Any


def dedupe_by_name(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last record for each named configuration.

    Records without a ``"name"`` key are passed through as-is.
    """
    deduped: dict[str, dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for record in records:
        name = record.get("name")
        if name is None:
            passthrough.append(record)
        else:
            deduped[name] = record
    return passthrough + list(deduped.values())
