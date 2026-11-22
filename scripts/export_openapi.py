import argparse
import json
from pathlib import Path

from dialogentis.api import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the API contract")
    parser.add_argument("--check", action="store_true", help="check the committed contract")
    args = parser.parse_args()
    target = Path("openapi.json")
    schema = json.dumps(create_app().openapi(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != schema:
            parser.exit(
                1, "openapi.json is stale. Run uv run --locked python scripts/export_openapi.py\n"
            )
        print("openapi.json is up to date")
    else:
        target.write_text(schema, encoding="utf-8")
        print("Updated openapi.json")
