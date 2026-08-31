"""Fetch the UCI Gas Turbine CO/NOx files without checking raw data into git."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/551/"
    "gas%2Bturbine%2Bco%2Band%2Bnox%2Bemission%2Bdata%2Bset.zip"
)
MIRROR_URL = (
    "https://raw.githubusercontent.com/skforecast/skforecast-datasets/"
    "main/data/turbine_emission.csv"
)
YEARS = (2011, 2012, 2013, 2014, 2015)
FEATURE_COLUMNS = (
    "AT",
    "AP",
    "AH",
    "AFDP",
    "GTEP",
    "TIT",
    "TAT",
    "TEY",
    "CDP",
    "CO",
    "NOX",
)


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
            with urlopen(url, timeout=120) as response:
                while block := response.read(1024 * 1024):
                    temporary.write(block)
            if temporary_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty file from {url}")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_manifest(output_dir: Path, payload: dict[str, object]) -> None:
    (output_dir / "source_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _csv_profile(output_dir: Path) -> dict[str, object]:
    row_counts: dict[str, int] = {}
    file_hashes: dict[str, str] = {}
    for year in YEARS:
        path = output_dir / f"gt_{year}.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = {str(field).strip().upper() for field in (reader.fieldnames or [])}
            missing = set(FEATURE_COLUMNS) - fields
            if missing:
                raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
            row_counts[str(year)] = sum(1 for _ in reader)
        file_hashes[path.name] = sha256(path)
    return {
        "years": list(YEARS),
        "row_counts": row_counts,
        "total_rows": sum(row_counts.values()),
        "feature_columns": list(FEATURE_COLUMNS),
        "year_file_sha256": file_hashes,
    }


def _extract_official(archive_path: Path, output_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="gas_turbine_extract_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"Unsafe archive member: {member.filename}")
                archive.extract(member, temporary_path)
        for year in YEARS:
            matches = list(temporary_path.rglob(f"gt_{year}.csv"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected exactly one gt_{year}.csv in UCI archive")
            destination = output_dir / f"gt_{year}.csv"
            shutil.copyfile(matches[0], destination)
            print(f"saved   {destination} sha256={sha256(destination)}")


def _split_mirror(source_path: Path, output_dir: Path) -> None:
    year_rows: dict[int, list[dict[str, str]]] = {year: [] for year in YEARS}
    with source_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {str(field).strip().upper() for field in (reader.fieldnames or [])}
        required = set(FEATURE_COLUMNS) | {"DATETIME"}
        missing = required - fields
        if missing:
            raise RuntimeError(f"Mirror file is missing columns: {sorted(missing)}")
        for row in reader:
            timestamp = str(row.get("datetime", row.get("DATETIME", ""))).strip()
            try:
                year = int(timestamp[:4])
            except ValueError as error:
                raise RuntimeError(f"Invalid mirror timestamp: {timestamp!r}") from error
            if year not in year_rows:
                raise RuntimeError(f"Mirror contains an unexpected year: {year}")
            year_rows[year].append(
                {
                    column: str(row.get(column, row.get(column.lower(), ""))).strip()
                    for column in FEATURE_COLUMNS
                }
            )
    for year, rows in year_rows.items():
        if not rows:
            raise RuntimeError(f"Mirror contains no rows for {year}")
        destination = output_dir / f"gt_{year}.csv"
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved   {destination} sha256={sha256(destination)}")


def fetch(output_dir: Path, accept_uci: bool, accept_mirror: bool) -> None:
    if accept_uci == accept_mirror:
        raise SystemExit("Choose exactly one of --accept-uci or --accept-mirror.")
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    if accept_mirror:
        source_path = output_dir / "turbine_emission_skforecast.csv"
        if not source_path.exists() or source_path.stat().st_size == 0:
            print(f"fetch   {MIRROR_URL}")
            _download(MIRROR_URL, source_path)
            print(f"saved   {source_path} sha256={sha256(source_path)}")
        else:
            print(f"exists  {source_path} sha256={sha256(source_path)}")
        _split_mirror(source_path, output_dir)
        profile = _csv_profile(output_dir)
        _write_manifest(
            output_dir,
            {
                "dataset": "uci_gas_turbine_co_nox_551",
                "source_kind": "public_mirror",
                "source_url": MIRROR_URL,
                "upstream_source": UCI_URL,
                "retrieved_at_utc": retrieved_at,
                "source_sha256": sha256(source_path),
                "source_file": source_path.name,
                "mirror_note": "Combined CSV split by datetime year; source values are not modified.",
                **profile,
            },
        )
        return

    archive_path = output_dir / "gas_turbine_uci_551.zip"
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        print(f"fetch   {UCI_URL}")
        _download(UCI_URL, archive_path)
        print(f"saved   {archive_path} sha256={sha256(archive_path)}")
    else:
        print(f"exists  {archive_path} sha256={sha256(archive_path)}")
    _extract_official(archive_path, output_dir)
    _write_manifest(
        output_dir,
        {
            "dataset": "uci_gas_turbine_co_nox_551",
            "source_kind": "official_uci",
            "source_url": UCI_URL,
            "retrieved_at_utc": retrieved_at,
            "archive_sha256": sha256(archive_path),
            "archive_file": archive_path.name,
            **_csv_profile(output_dir),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/gas_turbine"))
    parser.add_argument("--accept-uci", action="store_true")
    parser.add_argument(
        "--accept-mirror",
        action="store_true",
        help="Use the public skforecast mirror and record its URL/hash in source_manifest.json",
    )
    args = parser.parse_args()
    fetch(args.output_dir, args.accept_uci, args.accept_mirror)


if __name__ == "__main__":
    main()
