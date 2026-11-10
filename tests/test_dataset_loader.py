"""Unit tests for DriveGuard Dataset Loader.
Verifies NTHU-DDD folder parsing, label integrity, and metadata extraction.
"""

from pathlib import Path
import pytest
from app.config import CLASS_LABELS
from app.dataset_loader import DatasetSample, NTHUDatasetLoader


def test_dataset_sample_representation():
    sample = DatasetSample(
        file_path="NTHU-DDD/drowsy/001_glasses_sleepyCombination_1003_drowsy.jpg",
        label="drowsy",
        label_id=1,
        subject_id="001",
        scenario="glasses_sleepyCombination",
        frame_index=1003,
    )
    d = sample.to_dict()
    assert d["label"] == "drowsy"
    assert d["label_id"] == 1
    assert d["subject_id"] == "001"
    assert d["frame_index"] == 1003
    assert d["scenario"] == "glasses_sleepyCombination"


def test_dataset_loader_scanning():
    loader = NTHUDatasetLoader()
    scan_res = loader.scan(max_samples_per_class=20)
    assert scan_res["status"] == "success"
    assert scan_res["total_samples"] >= 20
    assert scan_res["counts"]["drowsy"] > 0
    assert scan_res["counts"]["notdrowsy"] > 0


def test_label_mapping_integrity():
    loader = NTHUDatasetLoader()
    loader.scan(max_samples_per_class=10)
    for sample in loader.samples:
        assert sample.label in CLASS_LABELS
        assert sample.label_id == CLASS_LABELS[sample.label]
        # Ground truth check: drowsiness is strictly binary 1 or 0
        assert sample.label_id in (0, 1)


def test_metadata_parsing():
    loader = NTHUDatasetLoader()
    subj, scen, idx = loader._parse_filename_metadata("002_noglasses_slowBlinkWithNodding_1219_drowsy.jpg")
    assert subj == "002"
    assert "slowBlinkWithNodding" in scen
    assert idx == 1219


def test_corrupted_file_handling(tmp_path):
    # Create empty fake unreadable file
    fake_dir = tmp_path / "drowsy"
    fake_dir.mkdir(parents=True)
    bad_file = fake_dir / "bad_image.jpg"
    bad_file.write_text("corrupted content")

    loader = NTHUDatasetLoader(dataset_root=tmp_path)
    loader.scan()
    sample = loader.samples[0]
    img = sample.load_image()
    # cv2.imread returns None on non-image text file
    assert img is None
