"""Download and verify the raw churn datasets used in this project.

Usage:
    python -m src.download_data
    python -m src.download_data --dataset iranian
    python -m src.download_data --check-only
"""

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from src.config import (
    DATASET_PATH,
    IRANIAN_DATASET_PATH,
    ORANGE_DATASET_PATH,
)
from src.data_loader import get_file_sha256


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


DATASET_SPECS = {
    "ibm": {
        "label": "IBM Telco Customer Churn",
        "path": DATASET_PATH,
        "sha256": "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91",
        "download_type": "manual",
        "url": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
    },
    "iranian": {
        "label": "Iranian Churn (UCI id 563)",
        "path": IRANIAN_DATASET_PATH,
        "sha256": "90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1",
        "download_type": "zip",
        "url": "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip",
        "zip_member": "Customer Churn.csv",
    },
    "orange": {
        "label": "Orange Telecom Churn (mlcourse.ai mirror)",
        "path": ORANGE_DATASET_PATH,
        "sha256": "5d0aecc74b6feda887e9d2735363d51ff671b373153863562b7bbfb36464e0a4",
        "download_type": "direct",
        "url": "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/telecom_churn.csv",
    },
}


def verify_file_sha256(path, expected_sha256):
    """Check whether ``path`` exists and matches ``expected_sha256``.

    Returns:
        tuple[bool, str | None]: ``(is_valid, actual_sha256)``.
        ``actual_sha256`` is ``None`` when the file does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return False, None

    actual_sha256 = get_file_sha256(file_path)
    return actual_sha256.lower() == expected_sha256.lower(), actual_sha256


def download_bytes(url, timeout=60):
    """Download raw bytes from ``url`` using standard library HTTP."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "telco-churn-prediction/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def extract_csv_from_zip(zip_bytes, target_filename, dest_path):
    """Extract ``target_filename`` from an in-memory zip archive to ``dest_path``."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        members = archive.namelist()
        matched_member = next(
            (
                name
                for name in members
                if name == target_filename or Path(name).name == target_filename
            ),
            None,
        )
        if matched_member is None:
            raise FileNotFoundError(
                f"Expected {target_filename!r} inside zip archive; found {members}"
            )
        content = archive.read(matched_member)

    dest.write_bytes(content)
    return dest


def print_ibm_manual_instructions(spec):
    """Print clear manual download instructions for Kaggle IBM Telco dataset."""
    print(f"[download_data] {spec['label']}: Kaggle yeu cau dang nhap, khong tai tu dong.")
    print(f"  1. Truy cap: {spec['url']}")
    print("  2. Tai file CSV ve may va dat tai:")
    print(f"     {spec['path']}")
    print(f"  3. SHA-256 mong doi: {spec['sha256']}")
    print("  4. Chay lai `python -m src.download_data --check-only` de xac minh.")


def ensure_dataset(name, spec, force=False, check_only=False):
    """Verify and, when supported, download a single dataset.

    Returns:
        dict with keys ``dataset``, ``status``, ``path``, ``sha256_valid``, ``actual_sha256``.
    """
    path = Path(spec["path"])
    expected_hash = spec["sha256"]
    is_valid, actual_hash = verify_file_sha256(path, expected_hash)

    if is_valid and not force:
        print(f"[OK] {name}: da ton tai va khop SHA-256 ({path})")
        return {
            "dataset": name,
            "status": "cached_ok",
            "path": path,
            "sha256_valid": True,
            "actual_sha256": actual_hash,
        }

    if check_only:
        if actual_hash is None:
            print(f"[MISSING] {name}: chua co file tai {path}")
            status = "missing"
        else:
            print(
                f"[MISMATCH] {name}: SHA-256 khong khop!\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )
            status = "hash_mismatch"
        return {
            "dataset": name,
            "status": status,
            "path": path,
            "sha256_valid": False,
            "actual_sha256": actual_hash,
        }

    if spec["download_type"] == "manual":
        if actual_hash is not None:
            print(
                f"[MISMATCH] {name}: file hien co khong khop SHA-256!\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )
            status = "hash_mismatch"
        else:
            status = "manual_required"
        print_ibm_manual_instructions(spec)
        return {
            "dataset": name,
            "status": status,
            "path": path,
            "sha256_valid": False,
            "actual_sha256": actual_hash,
        }

    if actual_hash is not None and not is_valid:
        print(f"[WARN] {name}: file hien tai sai hash, dang tai lai tu {spec['url']}...")
    else:
        print(f"[DOWNLOAD] {name}: dang tai tu {spec['url']}...")

    raw_bytes = download_bytes(spec["url"])
    path.parent.mkdir(parents=True, exist_ok=True)

    if spec["download_type"] == "zip":
        extract_csv_from_zip(raw_bytes, spec["zip_member"], path)
    else:
        path.write_bytes(raw_bytes)

    is_valid, actual_hash = verify_file_sha256(path, expected_hash)
    if not is_valid:
        raise ValueError(
            f"SHA-256 mismatch after downloading {name} ({path}): "
            f"expected {expected_hash}, got {actual_hash}"
        )

    print(f"[OK] {name}: tai thanh cong va khop SHA-256 ({path})")
    return {
        "dataset": name,
        "status": "downloaded_ok",
        "path": path,
        "sha256_valid": True,
        "actual_sha256": actual_hash,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tai va kiem tra SHA-256 cho 3 bo du lieu Telco Churn."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASET_SPECS.keys()],
        default="all",
        help="Chon dataset can xu ly (mac dinh: all).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Chi kiem tra file hien co va SHA-256, khong tai tu mang.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tai lai ngay ca khi file da co va dung SHA-256.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Tra ve ma loi khac 0 neu bat ky dataset nao chua san sang (bao gom IBM).",
    )
    args = parser.parse_args(argv)

    targets = list(DATASET_SPECS.keys()) if args.dataset == "all" else [args.dataset]
    results = []
    has_error = False

    for name in targets:
        try:
            result = ensure_dataset(
                name,
                DATASET_SPECS[name],
                force=args.force,
                check_only=args.check_only,
            )
            results.append(result)
            if result["status"] == "hash_mismatch":
                has_error = True
            elif args.strict and not result["sha256_valid"]:
                has_error = True
        except (urllib.error.URLError, OSError, ValueError, zipfile.BadZipFile) as exc:
            print(f"[ERROR] {name}: {exc}")
            has_error = True

    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
