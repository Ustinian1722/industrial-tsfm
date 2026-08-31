from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_industrial_demo import REQUIRED_SECTIONS


def audit_demo(path: str | Path) -> None:
    path = Path(path).resolve()
    html = path.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        if section not in html:
            raise AssertionError(f"Demo is missing section: {section}")
    if 'id="demo-data"' not in html:
        raise AssertionError("Demo is missing embedded run data")
    marker = '<script id="demo-data" type="application/json">'
    payload_text = html.split(marker, 1)[1].split("</script>", 1)[0]
    payload = json.loads(payload_text)
    engine = payload.get("engine", {})
    if engine and engine.get("target_labels_used") is not False:
        raise AssertionError("Demo adaptation decision must declare target_labels_used=false")
    print(f"DEMO_AUDIT_OK {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a static Industrial TSFM demo")
    parser.add_argument("demo_path", type=Path)
    args = parser.parse_args()
    audit_demo(args.demo_path)


if __name__ == "__main__":
    main()
