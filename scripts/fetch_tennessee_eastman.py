"""Fetch the official Tennessee Eastman Challenge IDV archives.

The University of Washington archive uses a legacy ZIP compression method
that Python's stdlib ``zipfile`` does not decode on this Windows build.  The
helper therefore uses an installed ``unzip`` executable only for extraction,
while recording every archive and extracted-member hash in a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

DISTURBANCES = tuple(range(1, 16))
MEMBERS = ("y.dat", "u.dat", "r.dat", "t.dat")
BASE_URL = "https://depts.washington.edu/control/LARRY/TE/IDVs"
PROVENANCE_URL = "https://depts.washington.edu/control/LARRY/TE/download.html"
FORMAT_URL = "https://depts.washington.edu/control/LARRY/TE/IDVs/format.txt"


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
            with urlopen(request, timeout=120) as response:
                while block := response.read(1024 * 1024):
                    temporary.write(block)
            if temporary_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty file from {url}")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _find_unzip() -> str:
    for candidate in ("unzip", "unzip.exe"):
        executable = shutil.which(candidate)
        if executable:
            return executable
    raise RuntimeError(
        "The official TE archives use legacy ZIP method 6. Install an unzip executable "
        "and ensure it is on PATH; Python zipfile/PowerShell Expand-Archive are not enough."
    )


def _extract_member(unzip: str, archive: Path, member: str, destination: Path) -> None:
    completed = subprocess.run(
        [unzip, "-p", str(archive), member],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not extract {member} from {archive}: {detail}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(completed.stdout)


def _profile(output_dir: Path, disturbances: tuple[int, ...]) -> dict[str, object]:
    rows: dict[str, int] = {}
    member_hashes: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}
    for disturbance in disturbances:
        entity_dir = output_dir / f"idv{disturbance}"
        for member in MEMBERS:
            path = entity_dir / member
            if not path.exists():
                raise FileNotFoundError(f"Missing extracted member: {path}")
            values = np.loadtxt(path, dtype=float)
            if values.ndim != 2:
                raise RuntimeError(f"Expected a 2-D numeric member: {path}")
            key = f"idv{disturbance}/{member}"
            rows[key] = int(values.shape[0])
            shapes[key] = [int(value) for value in values.shape]
            member_hashes[key] = sha256(path)
    if set(rows.values()) != {301}:
        raise RuntimeError(f"Expected 301 time rows per TE member, got {sorted(set(rows.values()))}")
    return {
        "member_row_counts": rows,
        "member_shapes": shapes,
        "member_sha256": member_hashes,
        "n_disturbances": len(disturbances),
        "sampling_minutes": 10,
        "duration_hours": 50,
        "members": list(MEMBERS),
    }


def fetch(output_dir: Path, disturbances: tuple[int, ...], accept_source: bool) -> None:
    if not accept_source:
        raise SystemExit(
            "The TE archive is an external source. Re-run with --accept-source after reviewing "
            "the provenance and non-redistribution note."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    unzip = _find_unzip()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    archive_hashes: dict[str, str] = {}
    for disturbance in disturbances:
        archive_path = output_dir / f"idv{disturbance}.zip"
        url = f"{BASE_URL}/idv{disturbance}.zip"
        if not archive_path.exists() or archive_path.stat().st_size == 0:
            print(f"fetch   {url}")
            _download(url, archive_path)
        else:
            print(f"exists  {archive_path}")
        archive_hashes[archive_path.name] = sha256(archive_path)
        entity_dir = output_dir / f"idv{disturbance}"
        for member in MEMBERS:
            _extract_member(unzip, archive_path, f"idv{disturbance}/{member}", entity_dir / member)
            print(f"saved   {entity_dir / member} sha256={sha256(entity_dir / member)}")

    _write_manifest(
        output_dir,
        {
            "dataset": "tennessee_eastman_challenge_idv_archive",
            "source_kind": "official_university_archive",
            "source_url_template": f"{BASE_URL}/idv{{disturbance}}.zip",
            "provenance_url": PROVENANCE_URL,
            "format_url": FORMAT_URL,
            "retrieved_at_utc": retrieved_at,
            "disturbances": list(disturbances),
            "archive_sha256": archive_hashes,
            "extraction_tool": unzip,
            "license_note": (
                "The archive page provides access and provenance but does not state an SPDX "
                "license. Keep raw archives local, cite the University of Washington archive, "
                "and obtain permission before redistribution or commercial use."
            ),
            **_profile(output_dir, disturbances),
        },
    )


def _write_manifest(output_dir: Path, payload: dict[str, object]) -> None:
    (output_dir / "source_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official Tennessee Eastman IDV archives")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/tennessee_eastman")
    )
    parser.add_argument(
        "--disturbances",
        type=str,
        default=",".join(str(value) for value in DISTURBANCES),
        help="Comma-separated disturbance IDs; defaults to IDV1..IDV15",
    )
    parser.add_argument("--accept-source", action="store_true")
    args = parser.parse_args()
    disturbances = tuple(int(value) for value in args.disturbances.split(",") if value.strip())
    if not disturbances or any(value not in DISTURBANCES for value in disturbances):
        raise SystemExit(f"disturbances must be a non-empty subset of {DISTURBANCES}")
    fetch(args.output_dir, disturbances, args.accept_source)


if __name__ == "__main__":
    main()
