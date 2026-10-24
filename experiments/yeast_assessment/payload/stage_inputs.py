"""Stage exact user-supplied model inputs; do not fetch or redistribute models."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Manifest path leaves root: {relative}")
    return candidate


def stage(source_root, destination_root, groups=None):
    source_root = Path(source_root).expanduser().resolve()
    destination_root = Path(destination_root).expanduser().resolve()
    entries = json.loads((ROOT / "INPUT_MANIFEST.json").read_text())["entries"]
    if groups:
        entries = [entry for entry in entries if set(groups).intersection(entry["groups"])]
    operations = []
    # Validate the entire selected batch before copying anything.
    for entry in entries:
        source = contained(source_root, entry["source_path"])
        if not source.is_file() or sha(source) != entry["sha256"]:
            raise ValueError(f"Missing or SHA-256-mismatched input: {source}")
        for relative in entry["destinations"]:
            dest = contained(destination_root, relative)
            if dest.exists() and (not dest.is_file() or sha(dest) != entry["sha256"]):
                raise ValueError(f"Refusing to replace different destination: {dest}")
            operations.append((source, dest, entry["sha256"]))
    copied = 0
    for source, dest, expected in operations:
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            copied += 1
        if sha(dest) != expected:
            raise RuntimeError(f"Copy verification failed: {dest}")
    return {"source_entries": len(entries), "destinations": len(operations),
            "new_copies": copied, "all_sha256_verified": True,
            "destination_root": str(destination_root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--destination-root", type=Path, default=ROOT)
    parser.add_argument("--group", action="append", choices=["numerical", "temporal", "external", "recalibration", "legacy_reduced", "legacy_community"])
    args = parser.parse_args()
    print(json.dumps(stage(args.source_root, args.destination_root, args.group), indent=2))


if __name__ == "__main__":
    main()
