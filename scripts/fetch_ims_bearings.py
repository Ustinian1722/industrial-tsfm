"""Fetch and profile the public NASA/IMS bearing archive.

The archive is large and contains high-frequency vibration files. This helper
keeps the archive and extracted files local, records their provenance, and
does not turn the raw signals into benchmark features until the registered
loader applies its versioned feature contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ARCHIVE_URL = "https://data.nasa.gov/docs/legacy/IMS.zip"
CATALOG_URL = "https://catalog.data.gov/dataset/ims-bearings"
PROVENANCE_URL = (
    "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/"
    "pcoe/pcoe-data-set-repository/"
)
SUBSETS = ("1st_test", "2nd_test", "3rd_test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = Request(url, headers={"User-Agent": "IndustrialTSFM/0.1"})
            with urlopen(request, timeout=180) as response:
                while block := response.read(1024 * 1024):
                    temporary.write(block)
            if temporary_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty file from {url}")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    for member in members:
        path = Path(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe archive member: {member.filename}")
    return members


def _extract(archive_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_members(archive)
        for member in members:
            archive.extract(member, output_dir)


def _is_signal_file(path: Path) -> bool:
    name = path.name.lower()
    if name.startswith(("readme", "info", "license")):
        return False
    return path.suffix.lower() not in {".zip", ".json", ".md", ".yaml", ".yml"}


def _subset_dir(output_dir: Path, subset: str) -> Path:
    candidates = [output_dir / subset, output_dir / "IMS" / subset]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    for candidate in output_dir.rglob(subset):
        if candidate.is_dir() and candidate.name == subset:
            return candidate
    raise FileNotFoundError(f"Could not find extracted IMS subset {subset} under {output_dir}")


def _profile(output_dir: Path) -> dict[str, object]:
    subset_file_counts: dict[str, int] = {}
    signal_file_hashes: dict[str, str] = {}
    for subset in SUBSETS:
        subset_dir = _subset_dir(output_dir, subset)
        paths = sorted(
            (path for path in subset_dir.iterdir() if path.is_file() and _is_signal_file(path)),
            key=lambda path: path.name,
        )
        if not paths:
            raise RuntimeError(f"No IMS signal files found under {subset_dir}")
        subset_file_counts[subset] = len(paths)
        for path in paths:
            key = str(path.relative_to(output_dir)).replace("\\", "/")
            signal_file_hashes[key] = sha256(path)
    return {
        "subsets": list(SUBSETS),
        "subset_file_counts": subset_file_counts,
        "signal_file_sha256": signal_file_hashes,
        "feature_contract": (
            "Applied later by industrial_tsfm.data.ims_bearings: per-channel "
            "RMS/std/kurtosis/crest_factor averaged within each configured bearing group"
        ),
    }


def fetch(output_dir: Path, accept_source: bool, metadata_only: bool = False) -> None:
    if not accept_source:
        raise SystemExit(
            "The NASA/IMS archive is an external source. Re-run with --accept-source after "
            "reviewing the catalog, provenance, and local-only raw-data note."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    if metadata_only:
        payload = {
            "dataset": "nasa_ims_bearings",
            "source_kind": "public_nasa_catalog_distribution",
            "source_url": ARCHIVE_URL,
            "catalog_url": CATALOG_URL,
            "provenance_url": PROVENANCE_URL,
            "retrieved_at_utc": retrieved_at,
            "metadata_only": True,
            "license_note": (
                "The NASA catalog marks this public dataset as a U.S. government work. "
                "Keep the large raw archive local and cite the catalog/distribution."
            ),
        }
        (output_dir / "source_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"metadata {output_dir / 'source_manifest.json'}")
        return

    archive_path = output_dir / "IMS.zip"
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        print(f"fetch   {ARCHIVE_URL}")
        _download(ARCHIVE_URL, archive_path)
    else:
        print(f"exists  {archive_path}")
    extracted = True
    for subset in SUBSETS:
        try:
            _subset_dir(output_dir, subset)
        except FileNotFoundError:
            extracted = False
            break
    if not extracted:
        print(f"extract {archive_path}")
        _extract(archive_path, output_dir)
    else:
        print(f"exists  extracted subsets under {output_dir}")
    payload = {
        "dataset": "nasa_ims_bearings",
        "source_kind": "public_nasa_catalog_distribution",
        "source_url": ARCHIVE_URL,
        "catalog_url": CATALOG_URL,
        "provenance_url": PROVENANCE_URL,
        "retrieved_at_utc": retrieved_at,
        "archive_file": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "metadata_only": False,
        "license_note": (
            "The NASA catalog marks this public dataset as a U.S. government work. "
            "Keep the large raw archive local and cite the catalog/distribution."
        ),
        **_profile(output_dir),
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"manifest {output_dir / 'source_manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and profile NASA/IMS bearing data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/ims_bearings"))
    parser.add_argument("--accept-source", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write the provenance contract without downloading the approximately-GB archive",
    )
    args = parser.parse_args()
    fetch(args.output_dir, args.accept_source, args.metadata_only)


if __name__ == "__main__":
    main()
