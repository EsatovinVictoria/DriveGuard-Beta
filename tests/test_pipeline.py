"""Integration tests for DriveGuard full computer vision pipeline.
Verifies execution from raw input to feature vector generation and failure propagation.
"""

from pathlib import Path
import numpy as np
from app.dataset_loader import NTHUDatasetLoader
from app.feature_export import FeatureExporter
from app.pipeline import DriveGuardPipeline


def test_full_pipeline_on_dataset_sample():
    loader = NTHUDatasetLoader()
    loader.scan(max_samples_per_class=1)
    sample = loader.get_sample_by_index(0)
    assert sample is not None

    img = sample.load_image()
    assert img is not None

    pipeline = DriveGuardPipeline()
    res = pipeline.process_frame(img, include_visualizations=True)

    assert res["status"] == "success"
    assert res["stage_failed"] is None
    assert res["processing_time_ms"] > 0
    assert res["face_detection"]["detected"] is True
    assert res["eye_localization"]["detected"] is True
    assert res["roi"]["dimensions"] == (24, 64)
    assert res["hog_features"]["feature_dimension"] > 0

    # Test export handoff
    exporter = FeatureExporter()
    added = exporter.add_sample(
        sample_id=sample.frame_index or 1,
        source_file=sample.source_file,
        label=sample.label,
        label_id=sample.label_id,
        feature_vector=res["hog_features"]["feature_vector"],
        subject_id=sample.subject_id,
    )
    assert added is True
    exp_res = exporter.export_csv()
    assert exp_res["status"] == "success"
    assert Path(exp_res["features_csv"]).exists()


def test_pipeline_failure_on_empty_frame():
    pipeline = DriveGuardPipeline()
    empty = np.array([], dtype=np.uint8)
    res = pipeline.process_frame(empty)

    assert res["status"] == "failed"
    assert res["stage_failed"] == "input_validation"
    assert res["hog_features"] == {}
