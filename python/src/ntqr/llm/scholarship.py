"""Offline bibliography and claim-to-evidence integrity checks.

This parser supports the maintained brace-valued BibTeX subset, not the complete
BibTeX macro language. Online metadata verification is recorded separately.
"""

from __future__ import annotations
import re
from pathlib import PurePosixPath


def bibliography_entries(source):
    entries = {}
    for match in re.finditer(r"(?m)^@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,", source):
        kind, key = match.groups()
        if kind.lower() in {"comment", "preamble", "string"}:
            continue
        if key in entries:
            raise ValueError(f"duplicate bibliography key: {key}")
        depth = 1
        pos = match.end()
        while pos < len(source) and depth:
            if source[pos] == "{" and (pos == 0 or source[pos - 1] != "\\"):
                depth += 1
            elif source[pos] == "}" and (pos == 0 or source[pos - 1] != "\\"):
                depth -= 1
            pos += 1
        if depth:
            raise ValueError(f"unclosed bibliography entry: {key}")
        body = source[match.end() : pos - 1]
        fields = {}
        for field in re.finditer(r"(?m)^\s*([A-Za-z]+)\s*=\s*\{", body):
            level = 1
            end = field.end()
            while end < len(body) and level:
                if body[end] == "{" and body[end - 1] != "\\":
                    level += 1
                elif body[end] == "}" and body[end - 1] != "\\":
                    level -= 1
                end += 1
            if level:
                raise ValueError(f"unclosed field in {key}")
            fields[field.group(1).lower()] = body[field.end() : end - 1]
        entries[key] = {"kind": kind.lower(), **fields}
    return entries


def audit_research(bibliography, manuscripts, registry, *, existing_paths):
    entries = bibliography_entries(bibliography)
    errors = []
    citations = set()
    for filename, text in manuscripts.items():
        # Pandoc citations only; HTML IDs and figure/section references are separate.
        for group in re.findall(r"\[([^\]\n]*@[^\]\n]*)\]", text):
            for key in re.findall(r"@([A-Za-z][A-Za-z0-9_:-]*)", group):
                if not key.startswith(("fig:", "sec:", "eq:", "tbl:")):
                    citations.add(key)
        for group in re.findall(r"\\nocite\{([^}]+)\}", text):
            citations.update(
                key.strip() for key in group.split(",") if key.strip() != "*"
            )
    for key in sorted(citations - set(entries)):
        errors.append(f"missing_citation:{key}")
    for key, entry in entries.items():
        for field in ("title", "year"):
            if not entry.get(field):
                errors.append(f"missing_{field}:{key}")
        if not (entry.get("author") or entry.get("editor")):
            errors.append(f"missing_creator:{key}")
        if entry.get("doi") and not re.fullmatch(r"10\.\d{4,9}/\S+", entry["doi"]):
            errors.append(f"invalid_doi:{key}")
        if entry.get("url") and not entry["url"].startswith(("https://", "http://")):
            errors.append(f"invalid_url:{key}")
    verified = set()
    for reference in registry.get("verified_references", []):
        key = reference["key"]
        if key in verified:
            errors.append(f"duplicate_verified_reference:{key}")
        verified.add(key)
        if key not in entries:
            errors.append(f"unlisted_verified_reference:{key}")
        if not reference.get("source_url", "").startswith("https://"):
            errors.append(f"missing_primary_source:{key}")
        if not reference.get("verified_on"):
            errors.append(f"missing_verification_date:{key}")
    ids = set()
    for claim in registry.get("claims", []):
        cid = claim["id"]
        if cid in ids:
            errors.append(f"duplicate_claim:{cid}")
        ids.add(cid)
        if not claim.get("limitations"):
            errors.append(f"missing_limitations:{cid}")
        for key in claim.get("reference_keys", []):
            if key not in entries:
                errors.append(f"missing_claim_reference:{cid}:{key}")
        for path in claim.get("evidence_paths", []):
            pure = PurePosixPath(path)
            if pure.is_absolute() or ".." in pure.parts or path not in existing_paths:
                errors.append(f"missing_or_unsafe_evidence:{cid}:{path}")
    return {
        "schema_version": 1,
        "ok": not errors,
        "errors": sorted(errors),
        "bibliography_entries": len(entries),
        "cited_entries": len(citations),
        "verified_primary_records": len(verified),
        "mapped_claims": len(ids),
        "scope": "Offline structural integrity; neither bibliographic completeness nor scientific validity is implied.",
    }
