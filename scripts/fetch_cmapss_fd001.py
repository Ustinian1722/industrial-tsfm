"""Fetch local C-MAPSS FD001 files without redistributing them in the repository."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

MIRROR_BASE = (
    "https://huggingface.co/datasets/DeveloperMindset123/"
    "CMAPSS_Jet_Engine_Simulated_Data/resolve/main"
)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(output_dir: Path, accept_mirror: bool, subsets: tuple[str, ...]) -> None:
    if not accept_mirror:
        raise SystemExit(
            "The NASA download is currently unavailable. Re-run with "
            "--accept-mirror to retrieve the unmodified public mirror."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset in subsets:
        subset = subset.upper()
        if subset not in SUBSETS:
            raise SystemExit(f"Unsupported subset {subset}; choose from {', '.join(SUBSETS)}")
        for filename in (f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"):
            destination = output_dir / filename
            if destination.exists() and destination.stat().st_size > 0:
                print(f"exists  {destination}  sha256={sha256(destination)}")
                continue
            url = f"{MIRROR_BASE}/{filename}"
            print(f"fetch   {url}")
            with urlopen(url, timeout=60) as response, destination.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
            print(f"saved   {destination}  sha256={sha256(destination)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--accept-mirror", action="store_true")
    parser.add_argument(
        "--subsets",
        type=str,
        default="FD001",
        help="Comma-separated C-MAPSS subsets (FD001,FD002,FD003,FD004)",
    )
    args = parser.parse_args()
    fetch(
        args.output_dir,
        args.accept_mirror,
        tuple(value.strip().upper() for value in args.subsets.split(",") if value.strip()),
    )


if __name__ == "__main__":
    main()
