"""Check bibliography, citations and declared evidence paths without network or writes."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/src"))
from ntqr.llm.scholarship import audit_research


def main():
    bibliography = (ROOT / "docs/manuscript/references.bib").read_text(encoding="utf-8")
    registry = json.loads((ROOT / "docs/research_map.json").read_text(encoding="utf-8"))
    paths = list((ROOT / "docs/manuscript").glob("*.md"))
    paths += list((ROOT / "scripts/manuscript/templates").glob("*.tmpl"))
    manuscripts = {
        str(path.relative_to(ROOT)): path.read_text(encoding="utf-8") for path in paths
    }
    evidence = {
        path
        for claim in registry["claims"]
        for path in claim.get("evidence_paths", [])
        if (ROOT / path).is_file()
    }
    result = audit_research(
        bibliography, manuscripts, registry, existing_paths=evidence
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
