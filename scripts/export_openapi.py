import json
from pathlib import Path

from dialogentis.api import create_app

if __name__ == "__main__":
    target = Path("openapi.json")
    target.write_text(json.dumps(create_app().openapi(), ensure_ascii=False, indent=2) + "\n")
    print("Updated openapi.json")
