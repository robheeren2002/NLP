import json
import re
from dataclasses import dataclass, asdict
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# =========================
# Regex patterns (EVDB pages)
# =========================

TITLE_RE = re.compile(r"^(.*) price and specifications - EV Database$")
MONTH_YEAR_RE = re.compile(
    r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$"
)

USABLE_BATT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*kWhUseable Battery")
REAL_RANGE_RE = re.compile(r"(\d+)\s*kmReal Range")


# =========================
# Data model (small output)
# =========================

@dataclass(frozen=True)
class SlimCarRecord:
    """
    Minimal record that still answers:
      - "Which cars are in the database?"
    """
    name: Optional[str]
    brand: Optional[str]


# =========================
# Extraction helpers
# =========================

def make_stable_id(url: str) -> str:
    """Short deterministic ID based on URL."""
    return md5(url.encode("utf-8")).hexdigest()[:10]


def first_nonempty_lines(text: str, limit: int = 300) -> List[str]:
    """Return up to `limit` non-empty, stripped lines."""
    lines: List[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln:
            lines.append(ln)
        if len(lines) >= limit:
            break
    return lines


def extract_title(page_content: str) -> Optional[str]:
    """
    Extract title from a line like:
    'Polestar 2 (MY20) (2020-2021) price and specifications - EV Database'
    """
    for ln in first_nonempty_lines(page_content, limit=400):
        m = TITLE_RE.match(ln)
        if m:
            return m.group(1).strip()
    return None



# =========================
# Main shrink function
# =========================

def shrink_evdb_records(
    items: Sequence[Dict[str, Any]],
    *,
    keep_extra_specs: bool = False,
) -> List[SlimCarRecord]:
    """
    Convert big EVDB scrape records into small records.
    Drops huge `page_content` and keeps only identifiers needed for listing cars.
    """
    slim: List[SlimCarRecord] = []

    for it in items:
        meta = it.get("metadata") or {}
        page_content = it.get("page_content") or ""

        url = meta.get("source") or meta.get("loc") or ""
        if not url:
            # If there's no URL, skip (hard to dedupe / identify)
            continue

        title = extract_title(page_content)

        slim.append(
            SlimCarRecord(
                name=title,
                brand=meta.get("brand"),
            )
        )

    return slim


# =========================
# Output helpers
# =========================

def save_json_files(base_path: Path, records: Sequence) -> None:
    """
    Writes:
      - <base>.pretty.json readable in editors like PyCharm
    """
    payload = [asdict(r) for r in records]

    # Pretty (readable)
    (base_path.with_suffix(".json")).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    """# Compact (small)
    (base_path.with_suffix(".json")).write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )"""



def list_car_names(records: Sequence[SlimCarRecord]) -> List[str]:
    """Return sorted unique car names (skips None)."""
    return sorted({r.name for r in records if r.name})


# =========================
# CLI entry point
# =========================

def main() -> None:
    input_path = Path("combined.json")
    output_path = Path("database_info.json")

    data = json.loads(input_path.read_text(encoding="utf-8"))

    # If your file is already a list of EVDB records, this is correct:
    items = data

    slim = shrink_evdb_records(items, keep_extra_specs=False)
    save_json_files(output_path, slim)

    print(f"Wrote: {output_path} ({output_path.stat().st_size:,} bytes)")
    print("\nCars in DB:")
    for name in list_car_names(slim):
        print(f" - {name}")


if __name__ == "__main__":
    main()
