"""DriveGuard Batch Feature Extractor & Subject-Independent Splitter.
Automates large-scale feature extraction across NTHU-DDD frames and prepares
leakage-free train/test splits based on subject identities for Raditya's Naïve Bayes modeling.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from app.config import (
    FEATURES_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
)
from app.dataset_loader import NTHUDatasetLoader
from app.feature_export import FeatureExporter
from app.pipeline import DriveGuardPipeline


def extract_batch(
    max_samples_per_class: Optional[int] = None,
    subject_filter: Optional[List[str]] = None,
    scenario_filter: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute batch extraction across dataset with auditing and progress reporting."""
    loader = NTHUDatasetLoader()
    scan_info = loader.scan(max_samples_per_class=max_samples_per_class)
    if verbose:
        print(f"DriveGuard Batch: Scanned {scan_info['total_samples']} samples from {scan_info['dataset_root']}")

    filtered_samples = loader.samples
    if subject_filter:
        filtered_samples = [s for s in filtered_samples if s.subject_id in subject_filter]
    if scenario_filter:
        filtered_samples = [s for s in filtered_samples if s.scenario in scenario_filter]

    if verbose:
        print(f"DriveGuard Batch: {len(filtered_samples)} samples matched filter criteria.")

    pipeline = DriveGuardPipeline()
    exporter = FeatureExporter(output_dir=output_dir)

    start_batch = time.perf_counter()
    success_count = 0
    fail_count = 0

    for idx, sample in enumerate(filtered_samples):
        img = sample.load_image()
        if img is None:
            exporter.add_failure(
                sample_id=sample.frame_index or sample.source_file,
                source_file=sample.source_file,
                label=sample.label,
                reason="Failed to decode image from disk",
                subject_id=sample.subject_id,
            )
            fail_count += 1
            continue

        res = pipeline.process_frame(img, include_visualizations=False)
        if res["status"] == "success":
            exporter.add_sample(
                sample_id=sample.frame_index or sample.source_file,
                source_file=sample.source_file,
                label=sample.label,
                label_id=sample.label_id,
                feature_vector=res["hog_features"]["feature_vector"],
                subject_id=sample.subject_id,
                scenario=sample.scenario,
                frame_index=sample.frame_index,
                processing_time_ms=res["processing_time_ms"],
            )
            success_count += 1
        else:
            exporter.add_failure(
                sample_id=sample.frame_index or sample.source_file,
                source_file=sample.source_file,
                label=sample.label,
                reason=res.get("message", "Pipeline execution failure"),
                subject_id=sample.subject_id,
            )
            fail_count += 1

        if verbose and (idx + 1) % 25 == 0:
            elapsed = time.perf_counter() - start_batch
            fps = (idx + 1) / elapsed
            print(f"  Processed {idx + 1}/{len(filtered_samples)} frames ({success_count} success, {fail_count} failed) - {fps:.1f} fps")

    # Export CSV and JSON
    export_summary = exporter.export_csv()
    total_time = round(time.perf_counter() - start_batch, 2)

    if verbose:
        print(f"DriveGuard Batch Complete in {total_time}s!")
        print(f"  Successfully extracted: {success_count} samples")
        print(f"  Failed samples: {fail_count}")
        print(f"  HOG Feature CSV: {export_summary.get('features_csv')}")

    return {
        "status": "success",
        "total_time_seconds": total_time,
        "processed_count": len(filtered_samples),
        "success_count": success_count,
        "fail_count": fail_count,
        "export_summary": export_summary,
    }


def generate_subject_independent_split(
    features_csv_path: Optional[Path] = None,
    output_split_dir: Optional[Path] = None,
    test_subject: str = "006",
) -> Dict[str, Any]:
    """Split extracted features CSV into train and test sets strictly based on Subject ID.

    Guarantees zero data leakage between video sequences.
    """
    csv_path = Path(features_csv_path) if features_csv_path else FEATURES_DIR / "hog_features.csv"
    out_dir = Path(output_split_dir) if output_split_dir else FEATURES_DIR / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Feature file not found at: {csv_path}")

    import csv
    train_rows: List[List[str]] = []
    test_rows: List[List[str]] = []
    header: List[str] = []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        subj_idx = header.index("subject_id")

        for row in reader:
            subj = row[subj_idx]
            if subj == test_subject:
                test_rows.append(row)
            else:
                train_rows.append(row)

    train_csv = out_dir / f"train_features_excl_{test_subject}.csv"
    test_csv = out_dir / f"test_features_subj_{test_subject}.csv"

    with open(train_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_rows)

    with open(test_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(test_rows)

    split_manifest = {
        "source_features": str(csv_path),
        "strategy": "Leave-One-Subject-Out (LOSO)",
        "test_subject": test_subject,
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "train_file": str(train_csv),
        "test_file": str(test_csv),
    }

    manifest_path = out_dir / f"split_manifest_subj_{test_subject}.json"
    with open(manifest_path, mode="w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    return split_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriveGuard Batch Feature Extractor")
    parser.add_argument("--samples", type=int, default=50, help="Max samples per class (default: 50)")
    parser.add_argument("--subject", type=str, default=None, help="Filter by subject ID (e.g. 001)")
    parser.add_argument("--split-test-subject", type=str, default="006", help="Generate train/test split leaving out subject (default: 006)")
    args = parser.parse_args()

    print(f"Executing batch extraction (limit: {args.samples} per class)...")
    res = extract_batch(max_samples_per_class=args.samples, subject_filter=[args.subject] if args.subject else None)

    if res["success_count"] > 0:
        print(f"Generating Subject-Independent Split (Test Subject: {args.split_test_subject})...")
        split_info = generate_subject_independent_split(test_subject=args.split_test_subject)
        print(f"Split created: Train={split_info['train_samples']} samples, Test={split_info['test_samples']} samples")
