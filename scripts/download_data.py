"""Download raw datasets into data/raw/ (never committed; see .gitignore).

Usage:
    uv run python scripts/download_data.py

Files are pinned to a specific release and verified by MD5, so every run
fetches exactly the same bytes. Files that already exist with the right
checksum are skipped. SweetenersDB is not handled yet.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ChemTastesDB v2.1 (2025-03-19), CC-BY-4.0: https://zenodo.org/records/15051366
CHEMTASTESDB_RECORD = "15051366"
CHEMTASTESDB_FILES = {
    "ChemTastesDB_database.xlsx": "301369521e51abf5540ed3bb2d08f4d5",
    "ChemTastesDB_references.xlsx": "a4ca4e509fc53ee1d1f9a8ae12bb11a4",
    "ChemTastesDB_readme.pdf": "2e14bd2252a2603cc2d45ff62e29a188",
}

# BitterDB 2019 release, CC-BY-NC-ND-4.0: https://bitterdb.agri.huji.ac.il/dbbitter.php
# Only the compound table is needed: it holds SMILES for every known bitter compound.
BITTERDB_URL = "https://bitterdb.agri.huji.ac.il/downloads/2019/?file={name}"
BITTERDB_FILES = {
    "BitterCompoundsPropA_2019.csv": "482e8e308fac817ad90d3bebb3071a5e",
}

# Some hosts (BitterDB) return an HTML error page to urllib's default
# "Python-urllib" User-Agent, so send a generic client name instead.
USER_AGENT = "tantalus-download/0.1"


def md5_of(path: Path) -> str:
    """Return the hex MD5 digest of a file."""
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path, expected_md5: str) -> None:
    """Download url to dest unless it already exists with the expected MD5.

    Downloads to a temporary ".part" file and only renames it into place after
    the checksum matches, so a failed or corrupted download never leaves a
    file that looks valid.
    """
    if dest.exists() and md5_of(dest) == expected_md5:
        print(f"ok (cached)  {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as resp, tmp.open("wb") as f:
        f.write(resp.read())
    actual = md5_of(tmp)
    if actual != expected_md5:
        tmp.unlink()
        raise ValueError(f"MD5 mismatch for {dest.name}: {actual} != {expected_md5}")
    tmp.replace(dest)
    print(f"downloaded   {dest.name}")


def download_chemtastesdb() -> None:
    """Fetch the pinned ChemTastesDB files into data/raw/chemtastesdb/."""
    out_dir = RAW_DIR / "chemtastesdb"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in CHEMTASTESDB_FILES.items():
        url = (
            f"https://zenodo.org/api/records/{CHEMTASTESDB_RECORD}/files/{name}/content"
        )
        fetch(url, out_dir / name, md5)


def download_bitterdb() -> None:
    """Fetch the pinned BitterDB compound table into data/raw/bitterdb/."""
    out_dir = RAW_DIR / "bitterdb"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in BITTERDB_FILES.items():
        fetch(BITTERDB_URL.format(name=name), out_dir / name, md5)


def main() -> int:
    """Download all supported datasets; return a process exit code."""
    try:
        download_chemtastesdb()
        download_bitterdb()
    except (OSError, ValueError) as exc:  # network/disk errors, MD5 mismatch
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
