from __future__ import annotations

from typing import Dict, List


def scan_term_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    term_rows = payload.get("term_records")
    if isinstance(term_rows, list) and term_rows:
        return [dict(row) for row in term_rows]
    noun_rows = payload.get("noun_records")
    if isinstance(noun_rows, list):
        return [dict(row) for row in noun_rows]
    return []


def row_term(row: Dict[str, object]) -> str:
    if "term" in row:
        return str(row.get("term", ""))
    return str(row.get("noun", ""))
