"""Download and extract the prebuilt data/ archive from Google Drive.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --file-id <drive_file_id>

The Drive file ID defaults to the DATA_GDRIVE_ID env var, falling back to
the ID baked in below. The file must be shared as "Anyone with the link".
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE_ID = "19XqC0zkNVkAkzDHHAm6R6yUsIOhPGN6d"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", default=None, help="Google Drive file ID for data.zip")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded zip after extraction")
    args = parser.parse_args()

    import os

    file_id = args.file_id or os.environ.get("DATA_GDRIVE_ID", DEFAULT_FILE_ID)
    if file_id == "REPLACE_WITH_DRIVE_FILE_ID":
        raise SystemExit(
            "No Drive file ID set. Pass --file-id or set DATA_GDRIVE_ID."
        )

    zip_path = ROOT / "data.zip"
    print(f"Downloading data.zip from Google Drive (id={file_id})...")
    gdown.download(id=file_id, output=str(zip_path), quiet=False)

    data_dir = ROOT / "data"
    if data_dir.exists():
        print(f"Removing existing {data_dir}...")
        shutil.rmtree(data_dir)

    print(f"Extracting to {ROOT}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ROOT)

    if not args.keep_zip:
        zip_path.unlink()

    print("Done. data/ is ready.")


if __name__ == "__main__":
    main()
