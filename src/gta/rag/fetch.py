"""Download Reactome and Gene Ontology source files, record a manifest, and build the document store."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from gta.rag import store

SOURCES: dict[str, str] = {
    "ReactomePathways.txt": "https://reactome.org/download/current/ReactomePathways.txt",
    "ReactomePathways.gmt.zip": "https://reactome.org/download/current/ReactomePathways.gmt.zip",
    "go-basic.obo": "https://purl.obolibrary.org/obo/go/go-basic.obo",
    "goa_human.gaf.gz": "https://current.geneontology.org/annotations/goa_human.gaf.gz",
}
USER_AGENT = "genomics-tool-agent (+https://github.com/santur90/genomics-tool-agent)"


def download(url: str, dest: Path) -> None:
    """Stream a URL to dest (via a .part file so interrupted downloads never look complete)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    part.replace(dest)


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _go_version(obo: Path) -> str:
    with open(obo, encoding="utf-8") as fh:
        for _ in range(20):
            m = re.match(r"data-version: (.+)", fh.readline())
            if m:
                return m.group(1).strip()
    return "unknown"


def fetch_all(
    dest: Path | None = None, force: bool = False, downloader: Callable[[str, Path], None] = download
) -> dict:
    """Download missing source files, extract the Reactome GMT, write manifest.json and build docs.jsonl."""
    dest = Path(dest or store.knowledge_dir())
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / store.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    for name, url in SOURCES.items():
        target = dest / name
        if force or not target.exists():
            downloader(url, target)
            manifest["files"][name] = {
                "url": url,
                "downloaded": datetime.now(UTC).date().isoformat(),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
    with zipfile.ZipFile(dest / "ReactomePathways.gmt.zip") as z:
        z.extract(store.REACTOME_GMT, dest)
    counts = store.build_store(dest)
    manifest["store"] = {
        **counts,
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "go_data_version": _go_version(dest / store.GO_OBO),
        "go_evidence_codes": sorted(store.EVIDENCE_CODES),
        "go_propagation": "is_a + part_of",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point shared by scripts/fetch_knowledge.py and `gta fetch-knowledge`."""
    import argparse

    ap = argparse.ArgumentParser(description="Download GO and Reactome and build the knowledge store.")
    ap.add_argument(
        "--dest", type=Path, help="target directory (default: GTA_KNOWLEDGE_DIR or ~/.cache/gta/knowledge)"
    )
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = ap.parse_args(argv)
    manifest = fetch_all(args.dest, force=args.force)
    s = manifest["store"]
    print(
        f"Knowledge store ready in {args.dest or store.knowledge_dir()}: {s['n_reactome']} Reactome pathways, {s['n_go']} GO terms"
    )
    return 0
