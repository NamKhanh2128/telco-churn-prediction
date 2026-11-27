"""Offline unit tests for src.download_data SHA-256 verification and extraction."""

import io
import sys
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.download_data import (
    DATASET_SPECS,
    ensure_dataset,
    extract_csv_from_zip,
    main as download_main,
    verify_file_sha256,
)


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


class TestSha256Verification:
    """Verify SHA-256 checks work offline without network access."""

    def test_verify_returns_true_for_matching_hash(self, tmp_path):
        payload = b"col1,col2\n1,0\n"
        file_path = tmp_path / "sample.csv"
        file_path.write_bytes(payload)

        valid, actual = verify_file_sha256(file_path, _sha256_bytes(payload))
        assert valid is True
        assert actual == _sha256_bytes(payload)

    def test_verify_is_case_insensitive_on_expected_hash(self, tmp_path):
        payload = b"a,b\n2,3\n"
        file_path = tmp_path / "upper.csv"
        file_path.write_bytes(payload)

        valid, _ = verify_file_sha256(file_path, _sha256_bytes(payload).upper())
        assert valid is True

    def test_verify_returns_false_for_mismatching_hash(self, tmp_path):
        payload = b"col1,col2\n1,0\n"
        file_path = tmp_path / "corrupt.csv"
        file_path.write_bytes(payload)

        valid, actual = verify_file_sha256(file_path, "0" * 64)
        assert valid is False
        assert actual == _sha256_bytes(payload)

    def test_verify_returns_false_and_none_when_missing(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        valid, actual = verify_file_sha256(missing, "a" * 64)
        assert valid is False
        assert actual is None


class TestEnsureDatasetOffline:
    """Verify caching, manual IBM handling, and zip extraction offline."""

    def test_cached_valid_file_skips_download(self, tmp_path, monkeypatch):
        payload = b"Churn\n1\n0\n"
        target = tmp_path / "iranian.csv"
        target.write_bytes(payload)
        spec = {
            "label": "Test Iranian",
            "path": target,
            "sha256": _sha256_bytes(payload),
            "download_type": "zip",
            "url": "https://example.invalid/iranian.zip",
            "zip_member": "Customer Churn.csv",
        }

        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("Network download must not run when SHA-256 matches")

        monkeypatch.setattr("src.download_data.download_bytes", _fail_if_called)
        result = ensure_dataset("iranian", spec)
        assert result["status"] == "cached_ok"
        assert result["sha256_valid"] is True

    def test_ibm_manual_missing_prints_instructions_and_verifies_hash_when_present(
        self, tmp_path, capsys
    ):
        payload = b"customerID,Churn\n0001,No\n"
        ibm_path = tmp_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        spec = {
            "label": "IBM Telco Customer Churn",
            "path": ibm_path,
            "sha256": _sha256_bytes(payload),
            "download_type": "manual",
            "url": DATASET_SPECS["ibm"]["url"],
        }

        missing_res = ensure_dataset("ibm", spec)
        out = capsys.readouterr().out
        assert missing_res["status"] == "manual_required"
        assert missing_res["sha256_valid"] is False
        assert "kaggle.com" in out

        # Now place the file manually and verify SHA-256 passes
        ibm_path.write_bytes(payload)
        present_res = ensure_dataset("ibm", spec)
        assert present_res["status"] == "cached_ok"
        assert present_res["sha256_valid"] is True

    def test_ibm_wrong_hash_detected(self, tmp_path):
        ibm_path = tmp_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        ibm_path.write_bytes(b"corrupted content")
        spec = {
            "label": "IBM Telco Customer Churn",
            "path": ibm_path,
            "sha256": "f" * 64,
            "download_type": "manual",
            "url": DATASET_SPECS["ibm"]["url"],
        }

        res = ensure_dataset("ibm", spec)
        assert res["status"] == "hash_mismatch"
        assert res["sha256_valid"] is False

    def test_extract_csv_from_zip(self, tmp_path):
        csv_bytes = b"Call Failure,Churn\n8,0\n"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("nested/Customer Churn.csv", csv_bytes)

        dest = tmp_path / "iranian-uci" / "Customer Churn.csv"
        extract_csv_from_zip(buf.getvalue(), "Customer Churn.csv", dest)
        assert dest.read_bytes() == csv_bytes

    def test_download_with_mocked_network_verifies_sha256(self, tmp_path, monkeypatch):
        csv_bytes = b"State,Churn\nKS,False\n"
        dest = tmp_path / "telecom_churn.csv"
        spec = {
            "label": "Orange",
            "path": dest,
            "sha256": _sha256_bytes(csv_bytes),
            "download_type": "direct",
            "url": "https://example.invalid/telecom_churn.csv",
        }
        monkeypatch.setattr("src.download_data.download_bytes", lambda _url: csv_bytes)

        res = ensure_dataset("orange", spec)
        assert res["status"] == "downloaded_ok"
        assert res["sha256_valid"] is True
        assert dest.read_bytes() == csv_bytes

    def test_download_corrupted_bytes_raises_value_error(self, tmp_path, monkeypatch):
        dest = tmp_path / "telecom_churn.csv"
        spec = {
            "label": "Orange",
            "path": dest,
            "sha256": "1" * 64,
            "download_type": "direct",
            "url": "https://example.invalid/telecom_churn.csv",
        }
        monkeypatch.setattr("src.download_data.download_bytes", lambda _url: b"bad")

        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            ensure_dataset("orange", spec)

    def test_cli_check_only_returns_nonzero_on_hash_mismatch(self, tmp_path, monkeypatch):
        bad_file = tmp_path / "bad.csv"
        bad_file.write_bytes(b"wrong")
        fake_specs = {
            "orange": {
                "label": "Orange",
                "path": bad_file,
                "sha256": "a" * 64,
                "download_type": "direct",
                "url": "https://example.invalid/orange.csv",
            }
        }
        monkeypatch.setattr("src.download_data.DATASET_SPECS", fake_specs)
        exit_code = download_main(["--dataset", "orange", "--check-only"])
        assert exit_code == 1
