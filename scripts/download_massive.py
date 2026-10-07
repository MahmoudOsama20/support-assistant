#!/usr/bin/env python
"""Download, extract and verify MASSIVE 1.1 (ar-SA + en-US only).

Usage:
    python scripts/download_massive.py                  # download + verify
    python scripts/download_massive.py --verify-only    # verify existing files
    python scripts/download_massive.py --archive PATH   # use a manually downloaded archive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier.dataset import (  # noqa: E402
    DEFAULT_DATA_DIR,
    LOCALES,
    DatasetError,
    load_massive,
    locale_path,
    verify_dataset,
)

DATASET_VERSION = "1.1"
ARCHIVE_NAME = "amazon-massive-dataset-1.1.tar.gz"
DEFAULT_URL = f"https://amazon-massive-nlu-dataset.s3.amazonaws.com/{ARCHIVE_NAME}"
CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, dest: Path) -> None:
    """Stream the archive to dest via a .part file (never leaves a partial file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            next_report = 0.1
            while chunk := response.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                if total and done / total >= next_report:
                    print(f"  {done / total:5.0%}  ({done / 1e6:.0f} / {total / 1e6:.0f} MB)")
                    next_report += 0.1
    except (urllib.error.URLError, TimeoutError) as exc:
        part.unlink(missing_ok=True)
        raise DatasetError(
            f"Download failed: {exc}\n"
            f"Download {url} manually, then run:\n"
            f"  python scripts/download_massive.py --archive <path-to-{ARCHIVE_NAME}>"
        ) from exc
    part.replace(dest)


def extract_needed(archive: Path, data_dir: Path) -> None:
    """Extract only <locale>.jsonl for our locales (+ LICENSE if present).

    Files are streamed to data_dir by basename, so archive paths are never
    trusted for writing.
    """
    wanted = {f"{loc}.jsonl" for loc in LOCALES}
    found: set[str] = set()
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                p = PurePosixPath(member.name)
                is_data = p.name in wanted and p.parent.name == "data"
                is_license = p.name == "LICENSE" and len(p.parts) <= 2
                if not (is_data or is_license):
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                target = data_dir / p.name
                with source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                if is_data:
                    found.add(p.name)
                if found == wanted:
                    break
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise DatasetError(f"Could not read archive {archive}: {exc}") from exc

    missing = wanted - found
    if missing:
        raise DatasetError(
            f"Archive {archive} does not contain {sorted(missing)} under a 'data/' folder."
        )


def write_manifest(data_dir: Path, source: str, archive_sha: str) -> Path:
    manifest = {
        "dataset": "AmazonScience MASSIVE",
        "version": DATASET_VERSION,
        "source": source,
        "archive_sha256": archive_sha,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {
            f"{loc}.jsonl": sha256_file(locale_path(loc, data_dir)) for loc in LOCALES
        },
    }
    path = data_dir / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def ensure_dataset(args: argparse.Namespace, data_dir: Path) -> None:
    have_files = all(locale_path(loc, data_dir).is_file() for loc in LOCALES)
    if have_files and not args.force and args.archive is None:
        print(f"Dataset files already present in {data_dir} (use --force to re-download).")
        return

    if args.archive is not None:
        archive = args.archive.resolve()
        if not archive.is_file():
            raise DatasetError(f"--archive file not found: {archive}")
        source = f"local:{archive.name}"
    else:
        archive = data_dir / ARCHIVE_NAME
        if args.force or not archive.is_file():
            download_archive(args.url, archive)
        source = args.url

    archive_sha = sha256_file(archive)
    print(f"Archive SHA-256: {archive_sha}")
    if args.expected_sha256 and archive_sha != args.expected_sha256.lower():
        raise DatasetError(
            f"Archive SHA-256 mismatch: expected {args.expected_sha256}, got {archive_sha}"
        )

    extract_needed(archive, data_dir)
    manifest = write_manifest(data_dir, source, archive_sha)
    print(f"Extracted {', '.join(f'{l}.jsonl' for l in LOCALES)} -> {data_dir}")
    print(f"Wrote {manifest.relative_to(PROJECT_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--archive", type=Path, help="use a local archive instead of downloading")
    parser.add_argument("--expected-sha256", help="fail if the archive hash differs")
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    parser.add_argument("--verify-only", action="store_true", help="skip download, just verify")
    args = parser.parse_args()
    data_dir: Path = args.data_dir.resolve()

    try:
        if not args.verify_only:
            ensure_dataset(args, data_dir)
        records = load_massive(data_dir)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print()
    report = verify_dataset(records)
    print(report.format())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())