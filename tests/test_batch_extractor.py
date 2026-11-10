"""Unit tests for DriveGuard Batch Extractor & Subject Splitter.
Verifies batch processing pipeline and leakage-free dataset splitting.
"""

from pathlib import Path
import pytest
from app.batch_extractor import extract_batch, generate_subject_independent_split
from app.config import FEATURES_DIR


def test_batch_extraction_and_splitting(tmp_path):
    out_dir = tmp_path / "features"
    res = extract_batch(max_samples_per_class=5, output_dir=out_dir, verbose=False)

    assert res["status"] == "success"
    assert res["success_count"] > 0
    assert (out_dir / "hog_features.csv").exists()
    assert (out_dir / "extraction_report.json").exists()

    # Test subject splitting
    split_res = generate_subject_independent_split(
        features_csv_path=out_dir / "hog_features.csv",
        output_split_dir=out_dir / "splits",
        test_subject="001",
    )

    assert split_res["strategy"] == "Leave-One-Subject-Out (LOSO)"
    assert Path(split_res["train_file"]).exists()
    assert Path(split_res["test_file"]).exists()
